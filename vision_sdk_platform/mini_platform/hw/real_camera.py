"""Real fixed camera: V4L2 capture + a ChArUco-calibrated pinhole model.

Deliberately mujoco-free so the geometry can be unit-tested without the sim stack.

The "tty device" the user described is a UVC USB webcam that enumerates at ``/dev/videoN``
(V4L2), not a serial port -- every camera path in this repo uses ``cv2.VideoCapture`` /
V4L2. Frames are returned as ``(H, W, 3)`` uint8 **RGB** (converted from OpenCV's BGR) to
match ``vision/detector.py::detect_puck``'s contract.

``RealPinholeCamera`` reproduces the method surface of ``vision/camera.py::PinholeCamera``
that ``PuckVisionSensor.measure()`` consumes (``pos``, ``cx``/``cy``/``f``, ``pixel_to_plane``,
``fd_jacobian``, ``project``), but with **calibrated** intrinsics + distortion and a **fixed
measured** camera->world extrinsic. OpenCV camera convention: +z forward, +x right, +y down.
"""

import os
import time
from dataclasses import dataclass

import numpy as np

# The puck's top face height above the playing surface (z=0 world). Detections are
# back-projected onto this exactly-known plane. Kept identical to vision/camera.py.
PUCK_TOP_Z = 0.007


def fourcc_str(v):
    """Decode a ``CAP_PROP_FOURCC`` value into its four characters (``''`` when unset)."""
    n = int(v)
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4)) if n else ""


#: V4L2 control ids (linux/videodev2.h). The exposure MENU, the manual exposure time, and -- the
#: one that actually costs frames -- AUTO_PRIORITY: with it set, the driver is allowed to drop the
#: frame rate to buy a longer exposure. Measured on the rig's Sonix KD-USB: a camera that
#: advertises 640x480 MJPG at 30 fps delivered 20, with auto_priority = 1.
V4L2_CID_EXPOSURE_AUTO = 0x009A0901
V4L2_CID_EXPOSURE_ABSOLUTE = 0x009A0902
V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0x009A0903
V4L2_EXPOSURE_MANUAL = 1
V4L2_EXPOSURE_APERTURE_PRIORITY = 3


def v4l2_control(device, cid, value=None):
    """Read (``value=None``) or write one V4L2 control directly; returns the EFFECTIVE value.

    Why not OpenCV: there is no ``CAP_PROP`` for ``exposure_auto_priority`` at all, and that is
    the control that trades frame rate for exposure. And ``CAP_PROP_AUTO_EXPOSURE`` goes straight
    through to this menu on the V4L2 backend, where the valid values are 1 (manual) and 3 (auto)
    -- the 0.25 / 0.75 convention some backends use writes 0 here, which the driver rejects, so
    the exposure silently stays on AUTO and every manual exposure after it is ignored.

    Needs no root and no ``v4l-utils`` (neither is available on this rig). Writes then reads back,
    so a caller can see whether the value actually took rather than assuming it did. Returns None
    when the control, the device or the platform will not do it -- never raises.
    """
    try:
        import fcntl
    except Exception:
        return None                              # not Linux
    import struct

    g = (3 << 30) | (8 << 16) | (0x56 << 8) | 27      # VIDIOC_G_CTRL
    s = (3 << 30) | (8 << 16) | (0x56 << 8) | 28      # VIDIOC_S_CTRL
    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except Exception:
        return None
    try:
        buf = bytearray(8)
        if value is not None:
            struct.pack_into("<Ii", buf, 0, int(cid), int(value))
            try:
                fcntl.ioctl(fd, s, buf, True)
            except OSError:
                pass                             # fall through and report what it really is
        struct.pack_into("<Ii", buf, 0, int(cid), 0)
        try:
            fcntl.ioctl(fd, g, buf, True)
        except OSError:
            return None
        return int(struct.unpack_from("<Ii", bytes(buf))[1])
    finally:
        os.close(fd)


# ----------------------------------------------------------------------------- capture
class V4L2Camera:
    """A UVC webcam opened via OpenCV V4L2, returning RGB frames.

    ``resolution`` is (width, height). MJPG is requested because most UVC webcams only
    reach full resolution / frame rate in MJPG, not raw YUYV (same choice as ``cam_tests.py``
    and the existing capture code).
    """

    def __init__(self, index=0, resolution=(640, 360), fourcc="MJPG", threaded=True,
                 fps=None, exposure=None, auto_exposure=None, auto_priority=None):
        import cv2

        self._cv2 = cv2
        self.index = index
        req_w, req_h = resolution
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        # Driver queue depth. Measured on this rig (Sonix KD-USB, MJPG 640x480 @ 30): a depth of
        # ONE throws away 4 frames in 10 -- 19.6 fps delivered instead of 30.0, inter-frame gaps
        # bimodal at 33 and 66 ms -- because a frame arriving while this loop sits BETWEEN reads
        # has no queued buffer to land in. It is the depth and not the JPEG cost: a bare `grab()`
        # with the decode removed still managed only 15.0. Depth 2 delivers a flat 30.0 with zero
        # doubled gaps. Freshness, which is why this was ever 1, is now held by the drain in
        # `_grab_loop` -- draining to the newest frame is what actually guarantees the newest
        # frame, whereas a one-deep queue merely guarantees losing frames. Best-effort: not every
        # V4L2 backend honours it.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open camera index {index}. Try `ls /dev/video*` and "
                f"`v4l2-ctl --list-devices` to find the right index.")
        self._cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Frame RATE and exposure (CONTROLLER_FIX_PLAN Fix 8b). Below ~20 detections/s this EKF
        # cannot track a struck puck at all -- at 11/s a 2.8 m/s puck moves ~250 mm between
        # frames, the NIS gate rejects the measurement as an outlier and the estimate visibly
        # freezes -- and no planner tuning fixes that. The usual cause is not the sensor but
        # AUTO-EXPOSURE: in dim light the driver lengthens the integration time and silently
        # halves the delivered rate (`exposure_auto_priority`). Asking for a rate the camera
        # cannot do is harmless -- V4L2 picks the nearest -- so ask, then record what was
        # actually granted, because the request and the grant are routinely different numbers.
        if fps is not None:
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        # The period the driver GRANTED, for the drain threshold below. Read back rather than
        # derived from the request: this camera's UVC descriptors top out at 30, so asking for
        # 60 or 120 is granted silently as 30 (verified at 320x240, 640x480 and 1280x720).
        _granted = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._period = (1.0 / _granted) if _granted > 0 else (1.0 / 30.0)
        #: The device node, for the controls OpenCV cannot reach (see :func:`v4l2_control`).
        self.device = index if isinstance(index, str) else f"/dev/video{int(index)}"
        # Exposure goes through V4L2 directly, not through OpenCV: the menu value has to be 1 or
        # 3 here, and auto_priority has no CAP_PROP at all. ORDER MATTERS -- switch the menu to
        # MANUAL first, or the absolute exposure that follows is ignored.
        if auto_exposure is not None:
            v = {0.25: V4L2_EXPOSURE_MANUAL, 0.75: V4L2_EXPOSURE_APERTURE_PRIORITY}.get(
                float(auto_exposure), int(auto_exposure))
            v4l2_control(self.device, V4L2_CID_EXPOSURE_AUTO, v)
        if exposure is not None:
            v4l2_control(self.device, V4L2_CID_EXPOSURE_ABSOLUTE, int(exposure))
        if auto_priority is not None:
            v4l2_control(self.device, V4L2_CID_EXPOSURE_AUTO_PRIORITY, int(auto_priority))
        #: What the driver reports AFTER the requests above -- the honest answer to "what is this
        #: camera actually set to", for the banner and `--stage cam-probe`. The exposure values
        #: are read BACK from the device, so a request the driver refused shows up as the old
        #: value rather than as the value that was asked for.
        self.props = dict(fps=float(cap.get(cv2.CAP_PROP_FPS)),
                          fourcc=fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)),
                          width=self.width, height=self.height,
                          exposure_auto=v4l2_control(self.device, V4L2_CID_EXPOSURE_AUTO),
                          exposure_absolute=v4l2_control(self.device,
                                                         V4L2_CID_EXPOSURE_ABSOLUTE),
                          exposure_auto_priority=v4l2_control(
                              self.device, V4L2_CID_EXPOSURE_AUTO_PRIORITY))

        # Async grab: cap.read() blocks until the next frame (~1/fps, up to ~33 ms). In the
        # 50 Hz control loop that stall silences the RT joint stream mid-tick and the arm
        # moves in stall-then-burst jerks. A background thread keeps grabbing so read_rgb()
        # returns the latest frame instantly and the control loop never blocks on the camera.
        self._threaded = threaded
        self._latest_bgr = None
        self._seq = 0            # monotonic count of DISTINCT frames (for frame-novelty gating)
        self._latest_t = None    # perf_counter() when that frame came back from the driver
        self._fps_n = 0          # frames delivered since the first one, for `delivered_fps`
        self._fps_t0 = None
        self._skipped = 0        # arrived but drained unseen as stale (see `frames_skipped`)
        #: Optional frame tee (`hw/frame_recorder.FrameRecorder`); set before the grab thread
        #: starts so a recorder attached at construction cannot miss the first frames.
        self._recorder = None
        if threaded:
            import threading
            self._lock = threading.Lock()
            self._running = True
            self._thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._thread.start()

    @property
    def resolution(self):
        return (self.width, self.height)

    @property
    def frame_time(self):
        """``time.perf_counter()`` of the newest grabbed frame (None before the first one).

        Prefer :meth:`read_rgb_stamped`, which returns the frame and its stamp atomically."""
        return self._latest_t

    @property
    def frame_seq(self):
        """Monotonic counter of DISTINCT camera frames grabbed (threaded) / read (sync).

        Lets a detector consumer detect each frame ONCE and coast between genuine new frames,
        instead of re-detecting the same buffer every loop (which, with the fast cv2 backend,
        made the EKF fuse a stale frame ~every control tick and blew up the velocity estimate)."""
        return self._seq

    @property
    def delivered_fps(self):
        """Frames per second the DRIVER actually delivered, or None before two frames.

        The number Fix 8b is about, and the one nothing downstream can infer: a low detection
        rate reads identically whether the camera sent nothing or the colour gate found nothing,
        and those call for opposite fixes. Measured in the grab loop itself, so it counts what
        arrived, not what the control loop managed to consume.
        """
        if self._fps_t0 is None or self._fps_n < 2:
            return None
        elapsed = time.perf_counter() - self._fps_t0
        return None if elapsed <= 0 else (self._fps_n - 1) / elapsed

    @property
    def frames_skipped(self):
        """Frames the driver delivered that the consumer never saw, dropped as stale by the drain.

        Non-zero means the control loop stalled long enough for the queue to back up (the arm
        would otherwise have been handed a frame from that stall). It is the honest companion to
        :attr:`delivered_fps`: that counts what the camera SENT, this counts what was too old to
        be worth showing, and ``frame_seq`` counts what was actually published.
        """
        return self._skipped

    def attach_recorder(self, recorder):
        """Tee every PUBLISHED frame to ``recorder`` (``None`` to stop teeing).

        The tee sits in the grab thread, not in the control loop, and the recorder's ``submit``
        is non-blocking and drops rather than waits -- so recording cannot cost a camera frame
        or a control tick. See ``hw/frame_recorder.py``. The frame is handed over WITHOUT a copy:
        ``retrieve()`` allocates a fresh array per call and nothing mutates a published frame,
        and while the recorder holds a reference the buffer cannot be recycled under it.

        Closing the camera closes an attached recorder (finalising the mp4), so the mp4 is
        written on every exit path that closes the sensor -- including Ctrl-C and a HardwareFault.
        """
        self._recorder = recorder

    def _grab_loop(self):
        # A grab that returns in a small fraction of the frame period was served from a buffer
        # ALREADY sitting in the queue -- i.e. a backlog -- so keep grabbing (cheap: no decode)
        # until one blocks waiting on the sensor, and decode only that newest frame. This is what
        # keeps the latency at one frame with a queue deeper than one: the control loop can stall
        # for 275 ms (this rig's measured `draw_ms` max) and still be handed the CURRENT frame
        # rather than a queue of stale ones. Drained frames did arrive, so they still count
        # toward `delivered_fps`; `frames_skipped` records how many the consumer never saw.
        fresh = 0.4 * self._period
        while self._running:
            t_call = time.perf_counter()
            ok = self._cap.grab()
            t = time.perf_counter()
            if not ok:
                time.sleep(0.005)   # transient drop -- back off briefly, don't busy-spin
                continue
            drained = 0
            while self._running and (t - t_call) < fresh:
                t_call = t
                if not self._cap.grab():
                    break
                t = time.perf_counter()
                drained += 1
            ok, frame = self._cap.retrieve()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest_bgr = frame
                self._seq += 1
                seq = self._seq
                self._latest_t = t
                if self._fps_t0 is None:
                    self._fps_t0 = t
                self._fps_n += 1 + drained
                self._skipped += drained
            # Outside the lock on purpose: `read_rgb_stamped` blocks the 50 Hz control thread on
            # it, so nothing that is not strictly publishing belongs inside it.
            if self._recorder is not None:
                self._recorder.submit(frame, seq, t)

    def read_rgb(self):
        """Latest (H, W, 3) uint8 RGB frame, or ``None`` if none has arrived yet.

        Threaded mode returns the most recent grabbed frame without blocking; synchronous
        mode does a direct (blocking) read. Either way the frame is copied so callers can
        mutate it freely.
        """
        return self.read_rgb_stamped()[0]

    def read_rgb_stamped(self):
        """``(rgb, seq, t)`` for the latest frame: the image, its ``frame_seq``, and the
        ``time.perf_counter()`` at which the driver returned it.

        Reading the three together **under one lock** is the point: a caller that polls for a
        new ``frame_seq`` and then reads the frame separately can straddle a grab and pair an
        image with the wrong timestamp. Frame timing is what turns a roll capture into a
        velocity (`hw/capture_puck_rolls.py`), so a stamp off by one frame period is a ~3%
        velocity error at 30 fps -- and the damping fit is a fit to exactly that decay.
        Returns ``(None, seq, None)`` when no frame has arrived yet.
        """
        if self._threaded:
            with self._lock:
                frame = None if self._latest_bgr is None else self._latest_bgr.copy()
                seq, t = self._seq, self._latest_t
            if frame is None:
                return None, seq, None
            return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB), seq, t
        ok, frame = self._cap.read()
        t = time.perf_counter()
        if not ok or frame is None:
            return None, self._seq, None
        self._seq += 1
        self._latest_t = t
        if self._fps_t0 is None:
            self._fps_t0 = t
        self._fps_n += 1
        if self._recorder is not None:
            self._recorder.submit(frame, self._seq, t)
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB), self._seq, t

    def close(self):
        if self._threaded and getattr(self, "_thread", None) is not None:
            self._running = False
            self._thread.join(timeout=1.0)
            self._thread = None
        if getattr(self, "_cap", None) is not None:
            self._cap.release()
            self._cap = None
        # After the grab thread is joined, so nothing can submit into a closing recorder.
        rec, self._recorder = getattr(self, "_recorder", None), None
        if rec is not None:
            rec.close()


# --------------------------------------------------------------------------- calibration
@dataclass
class CameraCalibration:
    """Everything the real sensor needs to turn a pixel into a world point.

    K            3x3 intrinsics (OpenCV), pixels.
    dist         distortion [k1, k2, p1, p2, k3].
    R_wc         3x3 world-from-camera rotation (columns map camera axes into world).
    C_world      (3,) camera center in the world/table frame, metres.
    resolution   (width, height) the intrinsics were calibrated at.
    """

    K: np.ndarray
    dist: np.ndarray
    R_wc: np.ndarray
    C_world: np.ndarray
    resolution: tuple

    def save(self, path):
        np.savez(path, K=self.K, dist=self.dist, R_wc=self.R_wc,
                 C_world=self.C_world, resolution=np.asarray(self.resolution, dtype=int))

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=False)
        res = tuple(int(x) for x in d["resolution"])
        return CameraCalibration(K=d["K"], dist=d["dist"], R_wc=d["R_wc"],
                                 C_world=d["C_world"], resolution=res)


def load_intrinsics(path):
    """Read (K, dist) from a ``cam_tests.py`` ``calibration.npz``."""
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["K"], float), np.asarray(d["dist"], float).ravel()


# --------------------------------------------------------------------------- pinhole model
class RealPinholeCamera:
    """Calibrated fixed pinhole with the same surface as ``vision/camera.py::PinholeCamera``.

    Extrinsics are constant (a fixed camera viewing the table), so ``pos``/``rot`` are plain
    attributes rather than live reads. Distortion is removed per pixel via
    ``cv2.undistortPoints`` before back-projection.
    """

    def __init__(self, K, dist, R_wc, C_world, resolution, puck_top_z=PUCK_TOP_Z):
        import cv2

        self._cv2 = cv2
        self.K = np.asarray(K, dtype=float)
        self.dist = np.asarray(dist, dtype=float).ravel()
        self.R_wc = np.asarray(R_wc, dtype=float)
        self.R_cw = self.R_wc.T
        self._pos = np.asarray(C_world, dtype=float)
        self.width, self.height = resolution
        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])
        # Single focal length for the bias-correction basis (fx ~= fy); the bias model is
        # re-fit on the real feed anyway, and defaults to zero until it is.
        self.f = self.fx
        self.puck_top_z = puck_top_z

    @property
    def pos(self):
        """Camera center, world frame (metres)."""
        return self._pos

    @property
    def rot(self):
        """World-from-camera rotation."""
        return self.R_wc

    def pixel_to_ray(self, u, v):
        """Unit world-frame direction through pixel (u, v), lens distortion removed."""
        pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
        und = self._cv2.undistortPoints(pts, self.K, self.dist)  # -> normalized (x_n, y_n)
        xn, yn = float(und[0, 0, 0]), float(und[0, 0, 1])
        d_cam = np.array([xn, yn, 1.0])          # OpenCV camera frame: +z forward
        d_world = self.R_wc @ d_cam
        return d_world / np.linalg.norm(d_world)

    def intersect_z_plane(self, d, z_p=None):
        """World point where ray ``d`` from ``pos`` meets plane z=z_p, or None.

        Assumes a camera above the plane looking down (fixed table view): a ray must point
        downward (d[2] < 0) and the hit must be in front of the camera.
        """
        if z_p is None:
            z_p = self.puck_top_z
        p = self._pos
        if d[2] >= -1e-9:
            return None
        t = (z_p - p[2]) / d[2]
        if t < 0:
            return None
        return p + t * d

    def pixel_to_plane(self, u, v, z_p=None):
        return self.intersect_z_plane(self.pixel_to_ray(u, v), z_p)

    def fd_jacobian(self, u, v, z_p=None, eps=0.5):
        """2x2 d(X_xy)/d(u,v) on the plane by central differences, or None if a probe misses."""
        pts = [self.pixel_to_plane(uu, vv, z_p)
               for uu, vv in ((u + eps, v), (u - eps, v), (u, v + eps), (u, v - eps))]
        if any(q is None for q in pts):
            return None
        return np.column_stack([(pts[0][:2] - pts[1][:2]) / (2 * eps),
                                (pts[2][:2] - pts[3][:2]) / (2 * eps)])

    def project(self, X):
        """World point -> pixel (u, v) with distortion, or None when behind the camera."""
        Xc = self.R_cw @ (np.asarray(X, dtype=float) - self._pos)
        if Xc[2] <= 0:
            return None
        img, _ = self._cv2.projectPoints(
            Xc.reshape(1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        return float(img[0, 0, 0]), float(img[0, 0, 1])
