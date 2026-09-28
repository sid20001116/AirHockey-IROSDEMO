"""Real-camera puck sensor.

``RealPuckSensor`` subclasses ``vision/sensor.py::PuckVisionSensor`` so it inherits the
``measure()`` pipeline and ``bias_correction`` **verbatim** -- only the pixel source
(``render``) and the camera model change. Because ``measure()`` returns ``pos_xy`` in the
world frame, and here the ``RealPinholeCamera`` extrinsic is calibrated into the *same*
world/table frame that ``env_info['robot']['base_frame']`` is expressed in,
``MiniBaselineAgent.get_puck_measurement`` transforms world->robot->virtual unchanged.

Importing this module pulls in mujoco (through ``vision.sensor``), exactly as the sim vision
code does -- fine in the cp310 runtime env. The distortion/geometry math lives in the
mujoco-free ``real_camera`` module so it stays independently testable.

Bias defaults to zero: the sim's bias model corrected MuJoCo-render artefacts (near-wall
pixels, perspective disc-centroid shift). A real, undistorted, calibrated feed needs its own
re-fit; re-fit and bake ``sigma_px``/``bias_*`` after calibration (the static-sweep procedure
in ``tools/vision_eval.py --mode static`` is the template).
"""

import threading
import time

import numpy as np

from mini_platform.vision.sensor import PuckVisionSensor
from mini_platform.hw.real_camera import RealPinholeCamera


class RealPuckSensor(PuckVisionSensor):
    """``PuckVisionSensor`` backed by a real fixed camera + calibration.

    Args:
        camera_capture: object with ``.read_rgb() -> (H,W,3) uint8 RGB | None`` and
            ``.close()`` (e.g. :class:`~mini_platform.hw.real_camera.V4L2Camera`). May also
            offer ``.read_rgb_stamped() -> (rgb, seq, t)``, which is what lets a measurement
            carry its capture time (Fix 8c); without it measurements have ``stamp=None``.
        calib: :class:`~mini_platform.hw.real_camera.CameraCalibration`.
        sigma_px, sigma_floor, sigma_border, bias_radial, bias_image: measurement-noise /
            bias parameters, same meaning as on ``PuckVisionSensor``. Defaults keep the sim's
            noise scale but zero the (render-specific) bias until re-fit.
    """

    def __init__(self, camera_capture, calib, puck_profile=None,
                 sigma_px=0.186, sigma_floor=1e-3, sigma_border=4e-3,
                 bias_radial=(0.0, 0.0), bias_image=(0.0, 0.0), table=None,
                 mask_margin=0.015, mask_goal_depth=0.060):
        # NB: intentionally does NOT call super().__init__ (which builds a MuJoCo
        # PinholeCamera). We set the same attributes measure()/bias_correction rely on.
        self._capture = camera_capture
        self.resolution = tuple(calib.resolution)          # (width, height)
        self.camera = RealPinholeCamera(calib.K, calib.dist, calib.R_wc,
                                        calib.C_world, self.resolution)
        #: The real feed goes through detect_puck's OpenCV backend: cvtColor/inRange +
        #: floodFill + connectedComponents instead of the pure-numpy rgb_to_hsv + scipy
        #: ndimage, which is ~10x faster at full resolution (see vision/detector.py). cv2
        #: is already a hard dependency of the capture path (V4L2Camera).
        self._detect_backend = "cv2"
        #: Real-puck colour gate (tune with mini_platform.hw.tune_puck). None falls back to
        #: the sim red-puck default, which is wrong for a real puck -- pass a tuned profile.
        self._puck_profile = puck_profile
        self.sigma_px = sigma_px
        self.sigma_floor = sigma_floor
        self.sigma_border = sigma_border
        self.bias_radial = tuple(bias_radial)
        self.bias_image = tuple(bias_image)
        self._renderer = None  # unused; kept so close()/render() shape matches the parent
        #: Capture stamp of the frame `render()` last pulled, copied onto each measurement by
        #: the inherited `measure()`. Set here because this class deliberately does not call
        #: super().__init__ (which would build a MuJoCo camera).
        self._frame_stamp = None
        #: World-frame table geometry for the play-area mask (keys length/width/goal_width/
        #: radius), or None to leave frames untouched. See :meth:`_build_table_mask`.
        self._table = table
        self._mask_margin = float(mask_margin)
        self._mask_goal_depth = float(mask_goal_depth)
        self._mask = None              # built lazily on the first frame (needs its size)
        self._mask_tried = False
        #: The UNMASKED frame, kept for ``--view``: the operator needs to see what the camera
        #: saw, including whatever the mask removed, or a mis-aimed mask is invisible.
        self._last_raw_frame = None
        #: Fraction of the image the mask keeps, for the banner. None until the mask is built;
        #: stays None when masking is off or failed open.
        self.mask_keep_frac = None

    def _build_table_mask(self, shape):
        """A pixel mask of the play area, built once from the table geometry.

        ``detect_puck`` returns the largest blob passing the colour gate, so any red object
        bigger than the visible puck replaces it for that frame -- on the 2026-09-15 logs an
        object just past the far end won 158 frames. Nothing downstream undoes that cleanly,
        because by then it is an ordinary-looking measurement in a plausible place. Zeroing the
        pixels that are not table is the one point where the object can be removed BEFORE it
        competes for the blob.

        The outline is the puck-CENTRE boundary (the rim inset by a puck radius) grown by
        ``mask_margin``, with a tongue through each goal mouth so a shot on target is not
        clipped at the end line. Points are taken at puck-top height in the world frame -- the
        same plane the detector back-projects onto -- and pushed through the full calibrated
        projection, distortion included, so the mask lands where the puck lands. The edges are
        sampled rather than drawn corner-to-corner because distortion bends straight lines.

        Fails OPEN. Missing geometry, no cv2, or any outline point that will not project leaves
        the frame untouched: a mask that is wrong blinds the detector completely, which is a
        far worse failure than the object it was meant to remove.
        """
        t = self._table
        if t is None:
            return None
        try:
            import cv2
        except Exception:
            return None
        half_l = float(t["length"]) / 2
        half_w = float(t["width"]) / 2
        r_p = float(t["radius"])
        m = self._mask_margin
        x0, x1 = -half_l + r_p - m, half_l - r_p + m
        y0, y1 = -half_w + r_p - m, half_w - r_p + m
        ym = float(t["goal_width"]) / 2 + r_p                 # half-width of a mouth tongue
        xg0, xg1 = x0 - self._mask_goal_depth, x1 + self._mask_goal_depth
        z = float(getattr(self.camera, "puck_top_z", 0.007))

        # Closed outline: the -y long edge, out through the +x mouth, back along +y, out
        # through the -x mouth (our own goal), and home.
        corners = [(x0, y0), (x1, y0),
                   (x1, -ym), (xg1, -ym), (xg1, ym), (x1, ym),
                   (x1, y1), (x0, y1),
                   (x0, ym), (xg0, ym), (xg0, -ym), (x0, -ym)]
        uv = []
        for i in range(len(corners)):
            ax, ay = corners[i]
            bx, by = corners[(i + 1) % len(corners)]
            for s in np.linspace(0.0, 1.0, 12, endpoint=False):
                p = self.camera.project(np.array([ax + (bx - ax) * s, ay + (by - ay) * s, z]))
                if p is None:
                    return None               # part of the table is behind the camera
                uv.append(p)

        h, w = shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.round(np.asarray(uv, float)).astype(np.int32)], 255)
        keep = float((mask > 0).mean())
        if keep < 0.02:
            return None                       # degenerate projection: fail open
        self.mask_keep_frac = keep
        return mask

    def render(self):
        """Latest RGB frame (H, W, 3), with everything that is not the play area zeroed.

        A dropped read yields a black frame, which the detector reports as 'no puck' ->
        ``measure()`` returns None (a clean dropout). The unmasked frame is kept on
        ``_last_raw_frame`` for ``--view``.
        """
        # Stamped read: the frame and the instant it arrived, under ONE lock. Taking the stamp
        # separately can straddle a grab and pair an image with the wrong time, and the whole
        # point of the stamp is that it is the frame's own (Fix 8c).
        # The documented capture contract is read_rgb()/close(), so the stamped form is OPTIONAL:
        # a capture without it still works, it simply cannot say how old its frames are and the
        # age logs as NaN. Demanding it here would break every stand-in and every older capture
        # for a diagnostic.
        stamped = getattr(self._capture, "read_rgb_stamped", None)
        if stamped is not None:
            frame, _seq, stamp = stamped()
        else:
            frame, stamp = self._capture.read_rgb(), None
        self._frame_stamp = stamp
        if frame is None:
            w, h = self.resolution
            return np.zeros((h, w, 3), dtype=np.uint8)
        self._last_raw_frame = frame
        if self._table is not None:
            if not self._mask_tried:
                self._mask_tried = True
                self._mask = self._build_table_mask(frame.shape)
            if self._mask is not None:
                frame = np.where(self._mask[:, :, None] > 0, frame, 0)
        return frame

    @property
    def frame_seq(self):
        """Distinct-camera-frame counter from the capture (or None if it has none), so a
        threaded consumer can detect each frame once and coast between frames."""
        cap = getattr(self, "_capture", None)
        return getattr(cap, "frame_seq", None) if cap is not None else None

    @property
    def frame_age(self):
        """Seconds since the driver returned the newest frame, or None if it cannot be known.

        Logged per control tick (run_hw_match step 0c) so a log can separate "the camera
        delivered nothing this tick" from "it delivered a frame with no puck in it" -- the two
        look identical in a measurement column that only ever says None, and they call for
        opposite fixes (camera rate vs the colour gate)."""
        cap = getattr(self, "_capture", None)
        t = getattr(cap, "frame_time", None) if cap is not None else None
        return None if t is None else max(0.0, time.perf_counter() - t)

    def close(self):
        if getattr(self, "_capture", None) is not None:
            self._capture.close()
            self._capture = None


class ThreadedPuckSensor:
    """Runs an inner puck sensor's ``measure()`` on a background thread so the control loop
    never blocks on vision.

    ``measure()`` on the real camera is ``render() + detect_puck()``. With the OpenCV
    detector backend (``RealPuckSensor._detect_backend == "cv2"``) that is a few ms at
    640x480, down from ~40 ms on the pure-numpy path (whose RGB->HSV segmentation
    dominated). Even the fast detector is decoupled here: called inline from the agent's
    ``draw_action`` (otherwise ~0.3 ms/tick), a slow vision frame throttles the whole
    plan+stream loop below 50 Hz, and then the RT trajectory streamer
    (:class:`~mini_platform.hw.run_hw_match._JointStreamer`) re-anchors so seldom that every
    re-anchor becomes a visible velocity discontinuity: the table-shaking jerk. The cv2
    speedup widens the margin; the decoupling keeps a stray slow frame from ever reaching it.

    A background thread calls the inner ``measure()`` as fast
    as the camera + detector allow; the control thread's ``measure()`` returns the newest
    frame's result **exactly once** (then ``None`` until a newer frame lands). Returning
    ``None`` between camera frames makes :meth:`PuckTracker.step` predict-only-coast, so the
    EKF still runs at the full control rate and corrects once per frame -- no double-fusing a
    stale detection. All other attributes (``camera``, ``_puck_profile``, ``_last_frame`` /
    ``_last_detection`` for ``--view``) delegate to the inner sensor.
    """

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()
        self._latest = None          # newest measurement (PuckMeasurement or None on no-puck)
        self._seq = 0                # bumped once per NEW camera frame (per pass if no frame_seq)
        self._consumed_seq = 0       # last seq the control thread has taken
        self._last_frame_seq = None  # camera frame_seq last detected (None => detect every pass)
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            # Frame-novelty gate: with the fast cv2 backend a free-running detect loop re-detects
            # the SAME camera buffer many times per control tick, so measure() returned non-None
            # every tick and the EKF fused a stale frame every tick -> the velocity estimate blew
            # up (est speed p99 22, max 52 m/s on hardware). Detect each camera frame ONCE; between
            # genuine new frames sleep and leave _seq alone, so the control thread gets None and
            # PuckTracker.step coasts (predict-only) -- restoring the smoothing the numpy path had.
            # When the inner sensor exposes no frame_seq (e.g. a test stand-in), fall back to the
            # original detect-every-pass behaviour.
            fs = getattr(self._inner, "frame_seq", None)
            if fs is not None and fs == self._last_frame_seq:
                time.sleep(0.002)
                continue
            try:
                m = self._inner.measure()
            except Exception:
                # A transient capture/detect hiccup must not kill the vision thread; a genuine
                # camera failure surfaces as sustained no-detection (measure() -> None), which
                # the tracker already handles as a dropout.
                m = None
            with self._lock:
                self._latest = m
                self._seq += 1
            # Mark the most recent frame seen as processed (post-detect read, so a frame that
            # arrived mid-detect is not re-detected next loop).
            self._last_frame_seq = getattr(self._inner, "frame_seq", None)

    def measure(self):
        """Newest measurement once per completed frame; ``None`` when none is fresh.

        A ``None`` here means "no new frame this tick" -- identical, for the EKF, to a genuine
        no-detection: :meth:`PuckTracker.step(None)` predicts forward without correcting.
        """
        with self._lock:
            if self._seq == self._consumed_seq:
                return None
            self._consumed_seq = self._seq
            return self._latest

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._inner.close()

    def __getattr__(self, name):
        # Only reached for attributes not set on the wrapper itself (camera, _puck_profile,
        # _last_frame, _last_detection, render, ...). Guard _inner so an access before
        # __init__ finishes can't recurse.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)
