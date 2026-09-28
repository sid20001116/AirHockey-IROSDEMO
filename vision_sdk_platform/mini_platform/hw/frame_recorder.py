"""Optional background mp4 recorder for the raw camera feed -- off by default, and by
construction unable to cost the rig a frame or a millisecond.

``run_hw_match --record-video`` tees every frame the camera publishes into a
:class:`FrameRecorder`, which encodes on its own thread. A recording is worth a lot when a log
says "the arm went for the wrong spot" and no column can say whether the puck was actually where
the estimate thought it was -- but a recorder on this rig is only worth having if it CANNOT slow
the capture path, because at 30 fps every lost frame is a lost EKF correction. Three properties
keep it out of the way:

  * frames are teed from the camera's **grab thread**, never from the 50 Hz control loop, so
    ``draw_action``, the planner and the RT joint stream never touch recording at all;
  * the producer only does a bounded ``put_nowait`` of a **reference** -- no copy, no encode, no
    I/O -- and DROPS the frame when the queue is full, so a disk stall can never back-pressure
    the grab loop into losing camera frames. It costs video, not detections, and the drops are
    counted and reported at close rather than hidden;
  * the encode (0.9 ms/frame at 640x360, measured) runs on a daemon thread, and OpenCV releases
    the GIL inside ``VideoWriter.write``, so it is not competing for the interpreter either.

Handing over a reference rather than a copy is safe because ``cv2.VideoCapture.retrieve()``
allocates a fresh array per call and nothing downstream mutates the published BGR frame (the
consumer in ``real_camera.read_rgb_stamped`` copies before converting). While this queue holds a
reference the buffer cannot be recycled under it.

The sidecar ``*_frames.npz`` is what makes the video evidence rather than a home movie: ``seq``
is the camera ``frame_seq`` of each WRITTEN video frame, and ``frame_seq`` is also a column in
the ``--log`` telemetry, so video frame *i* joins to the exact control tick that saw it. An mp4
has one constant frame rate and a real camera does not, so without those stamps the most an
analysis could say about frame *i* is "roughly i/30 s in".
"""

import os
import queue
import threading
import time

import numpy as np

#: Frames the hand-off queue may hold before the recorder starts dropping them. 60 is ~2 s of
#: slack at 30 fps and ~55 MB at 640x480x3 -- enough to ride out a page-cache flush, and bounded
#: so a stalled disk costs a measured amount of memory instead of all of it.
QUEUE_FRAMES = 60

#: Container fourcc. ``mp4v`` (MPEG-4 Part 2) is the one that opens on this rig's OpenCV build
#: with no external ffmpeg binary; ``avc1`` needs an H.264 encoder that is not present (it tries
#: ``h264_v4l2m2m``, finds no device and refuses to open).
FOURCC = "mp4v"


class FrameRecorder:
    """Encode submitted BGR frames to an mp4 on a background thread.

    Args:
        path: output ``.mp4``. Parent directories are created.
        fps: frame rate written into the container. Pass what the camera GRANTED
            (``V4L2Camera.props['fps']``), not what was asked for -- the mp4 timebase is
            constant, so this is what sets playback speed. :meth:`summary` reports the rate
            actually delivered next to it.
        queue_frames: hand-off queue depth (see :data:`QUEUE_FRAMES`).
    """

    def __init__(self, path, fps=30.0, queue_frames=QUEUE_FRAMES, fourcc=FOURCC):
        self.path = os.path.abspath(path)
        self.fps = float(fps) if fps and float(fps) > 0 else 30.0
        self.fourcc = str(fourcc)
        self.size = None                 # (w, h) of the first frame, i.e. what was encoded
        self.written = 0                 # frames in the mp4
        self.dropped = 0                 # submitted but refused: the queue was full
        self.error = None                # set once, then the loop drains and discards
        #: Wall clock paired with the ``perf_counter`` stamps, so the sidecar's monotonic times
        #: can be turned back into times of day: ``t_unix = t0_unix + (t - t0_perf)``.
        self.t0_perf = time.perf_counter()
        self.t0_unix = time.time()
        self._q = queue.Queue(maxsize=int(queue_frames))
        self._w = None
        self._seq = []
        self._t = []
        self._stop = threading.Event()
        self._closed = False
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._thread = threading.Thread(target=self._write_loop, name="frame-recorder",
                                        daemon=True)
        self._thread.start()

    @property
    def sidecar_path(self):
        """Where the per-frame ``seq``/``t`` arrays are written (see the module docstring)."""
        return os.path.splitext(self.path)[0] + "_frames.npz"

    # ------------------------------------------------------------------ producer side
    def submit(self, frame_bgr, seq=None, t=None):
        """Queue one BGR frame. Never blocks, never raises -- drops instead.

        Called from the camera's grab thread, so the two things it must not do are block (that
        would cost camera frames) and raise (that would kill the grab loop). A full queue means
        the encoder is behind; the frame is counted in :attr:`dropped` and thrown away.
        """
        if self._closed or frame_bgr is None:
            return
        try:
            self._q.put_nowait((frame_bgr, seq, t))
        except queue.Full:
            self.dropped += 1

    # ------------------------------------------------------------------ writer thread
    def _open(self, frame):
        import cv2

        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*self.fourcc), self.fps,
                                 (int(w), int(h)))
        if not writer.isOpened():
            self.error = (f"cv2.VideoWriter refused {self.path} "
                          f"({w}x{h} @ {self.fps:g} fps, fourcc {self.fourcc})")
            print(f"WARNING: video recording is OFF for this run -- {self.error}\n"
                  f"  The run itself is unaffected; frames are being discarded.")
            return
        self._w = writer
        self.size = (int(w), int(h))

    def _write_loop(self):
        while not (self._stop.is_set() and self._q.empty()):
            try:
                frame, seq, t = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.error is not None:
                continue            # drain and discard: never back-pressure the grab loop
            try:
                if self._w is None:
                    self._open(frame)
                    if self._w is None:
                        continue
                self._w.write(frame)
            except Exception as e:                      # a full disk is the realistic one
                self.error = f"{type(e).__name__}: {e}"
                print(f"WARNING: video recording stopped after {self.written} frames -- "
                      f"{self.error}\n  The run itself is unaffected.")
                continue
            self.written += 1
            self._seq.append(-1 if seq is None else int(seq))
            self._t.append(np.nan if t is None else float(t))

    # ------------------------------------------------------------------ shutdown
    @property
    def measured_fps(self):
        """Rate the frames were actually CAPTURED at, from their stamps (None if unknowable)."""
        if len(self._t) < 2:
            return None
        span = self._t[-1] - self._t[0]
        return None if not np.isfinite(span) or span <= 0 else (len(self._t) - 1) / span

    def close(self):
        """Flush the queued tail, finalise the mp4, write the sidecar, print one summary.

        Idempotent. Blocks for as long as the encoder needs to drain what is already queued
        (~50 ms for a full queue), which is bought at shutdown and costs the run nothing.
        """
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=30.0)
        if self._w is not None:
            self._w.release()
            self._w = None
        self._save_sidecar()
        print(self.summary())

    def _save_sidecar(self):
        if not self._seq:
            return
        np.savez(self.sidecar_path,
                 seq=np.asarray(self._seq, np.int64), t=np.asarray(self._t, float),
                 fps_written=float(self.fps),
                 fps_measured=float(self.measured_fps or np.nan),
                 dropped=int(self.dropped), written=int(self.written),
                 t0_perf=float(self.t0_perf), t0_unix=float(self.t0_unix),
                 resolution=np.asarray(self.size or (0, 0), int),
                 mp4=os.path.basename(self.path))

    def summary(self):
        """The one block printed at close: what was written, and whether to trust its timing."""
        if self.written == 0:
            why = f" ({self.error})" if self.error else " (no frames arrived)"
            return f"video: nothing written to {self.path}{why}"
        span = self._t[-1] - self._t[0] if np.isfinite(self._t[-1] - self._t[0]) else np.nan
        mb = os.path.getsize(self.path) / 1e6 if os.path.exists(self.path) else 0.0
        w, h = self.size
        out = [f"video: {self.written} frames, {span:.1f} s of table, {w}x{h}, {mb:.1f} MB\n"
               f"  -> {self.path}"]
        meas = self.measured_fps
        if meas is not None and abs(meas - self.fps) > 0.05 * self.fps:
            # The container carries ONE rate, so a camera that delivered something else plays
            # back at the wrong speed. Say so with the factor, rather than let someone time a
            # swing off the video and get a number that is quietly 20% wrong.
            out.append(f"  NB written at {self.fps:g} fps but the camera delivered {meas:.1f} -- "
                       f"playback runs {meas / self.fps:.2f}x real time. Use the sidecar stamps "
                       f"for any timing.")
        elif meas is not None:
            out.append(f"  {meas:.1f} fps delivered vs {self.fps:g} written: playback is real time.")
        out.append(f"  per-frame stamps + camera frame_seq -> {self.sidecar_path}\n"
                   f"  (join video frame i to the --log telemetry on frame_seq)")
        if self.dropped:
            out.append(f"  WARNING: {self.dropped} frames dropped by the recorder "
                       f"({100.0 * self.dropped / (self.dropped + self.written):.1f}%) -- the "
                       f"encoder or the disk fell behind. The RUN was not affected (the camera "
                       f"and control loop never wait on it); the video is missing those frames.")
        if self.error:
            out.append(f"  WARNING: recording ended early -- {self.error}")
        return "\n".join(out)
