"""Offscreen-render lifecycle and puck measurement extraction (vision assignment Part 1).

One persistent `mujoco.Renderer`, created lazily on first use so the sensor can be built
on machines without a GL context; the camera name is passed per `update_scene`, so a
second (wrist) camera could share the renderer with no new abstraction. Callers hand in
the live SCENE handles (`env.base_env._model` / `._data`) -- the robot model inside
`env_info` is an arm-only FK scratchpad with no table, puck or cameras.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from mini_platform.envs.env_base import PUCK_RADIUS
from mini_platform.estimation.kalman_filter import OBS_COV
from mini_platform.vision.camera import PinholeCamera
from mini_platform.vision.detector import detect_puck

#: The blue marker's centre offset from the puck centre along the puck's local +x
#: (assets/mini_table.xml, site puck_marker pos). Sets the yaw lever arm.
MARKER_OFFSET = 0.0155


@dataclass
class PuckMeasurement:
    """One camera measurement of the puck, world frame."""

    pos_xy: np.ndarray    # (2,) puck centre
    yaw: float | None     # None when the marker is not measurable
    R_xy: np.ndarray      # 2x2 position covariance, world axes
    R_yaw: float | None   # yaw variance; None iff yaw is None
    n_pixels: int         # red blob area, px
    clipped: bool = False  # blob touched the image border (R carries sigma_border)
    #: ``time.perf_counter()`` at which the CAMERA delivered the frame this came from, or None
    #: when there is no separate capture instant (the sim renders on demand). What makes the
    #: measurement's AGE knowable downstream: by the time the EKF sees a real one it is a camera
    #: period + detection + thread hand-off old, and fusing it as though it described the puck
    #: NOW is a straight position error -- 50 ms of age on a 2 m/s puck is 100 mm, pulled into
    #: the estimate and then forward-integrated into the intercept (Fix 8c).
    stamp: float | None = None


class PuckVisionSensor:
    """Renders a named camera and turns the frame into a `PuckMeasurement`.

    `sigma_px` (centroid noise, px) and the two `bias_*` pairs (centroid bias model)
    are calibrated by `tools/vision_eval.py --mode static` (Part 2).
    """

    # Calibrated by `vision_eval --mode static` (21x9 grid x 4 yaws, 640x360,
    # mini-defend, mallet at home; artifacts/vision/static_sweep.json). sigma_px is the
    # median per-cell jitter scatter over ground-sample distance (rendering is noise-
    # free, so this is pure centroid quantization). The centroid bias model replaces
    # the assignment's scalar kappa, which the sweep falsified (residual yaw RMSE
    # 0.114 rad vs the 0.1 bar): the world-radial term is AFFINE in horizontal camera
    # distance (the visible wall band shrinks with viewing elevation), and an
    # image-radial term linear in the normalized pixel offsets captures the projected
    # disc's perspective centroid shift. bias_image is in normalized image coordinates
    # (dimensionless), which makes the world-space correction resolution-invariant.
    # Fitted jointly by least squares over the unoccluded in-frustum grid at the
    # CURRENT camera pose (0.3341 m above the surface = 0.55 m above the floor, the
    # physical rig height); residual position RMSE ~0.6 mm centre / ~1.4 mm corners,
    # yaw RMSE ~0.05 rad. Re-fit after any change to the puck geometry or the camera
    # pose.
    # sigma_border: extra per-axis noise (m) when the red bbox touches the image
    # border. A clipped blob loses pixels on one side, so its centroid is biased
    # toward the visible part and jumps as the puck moves -- measured 3.8 mm jitter
    # scatter at the one grid cell whose blob clips (vs ~0.1 mm unclipped), and the
    # bias grows with clip depth, so the measurement is downweighted, not discarded.
    def __init__(self, model, data, camera="table_side_left", resolution=(640, 360),
                 sigma_px=0.186, sigma_floor=1e-3, sigma_border=4e-3,
                 bias_radial=(0.0005, -0.0142), bias_image=(0.00746, 0.00400)):
        self._model = model
        self._data = data
        self._camera_name = camera
        self.resolution = resolution
        self.camera = PinholeCamera(model, data, camera, resolution)
        self.sigma_px = sigma_px
        self.sigma_floor = sigma_floor
        self.sigma_border = sigma_border
        self.bias_radial = tuple(bias_radial)
        self.bias_image = tuple(bias_image)
        #: Colour gate for detect_puck. None -> the sim red-puck default. The hardware
        #: sensor sets this to a tuned HSV profile; the sim path leaves it None.
        self._puck_profile = None
        #: detect_puck backend. The sim uses "numpy" (the reference path these bias/sigma
        #: constants were fitted against); the hardware sensor overrides this to "cv2" for
        #: the ~10x-faster OpenCV segmentation (see vision/detector.py).
        self._detect_backend = "numpy"
        #: Last frame + detection from measure(), cached so a live viewer can show exactly
        #: what the agent acted on without triggering a second camera capture.
        self._last_frame = None
        self._last_detection = None
        self._renderer = None
        #: Capture stamp of the frame `render()` most recently produced, or None when the source
        #: has no such notion (this sim renderer). `measure()` copies it onto the measurement.
        self._frame_stamp = None

    def bias_correction(self, P_xy, u_px, v_px, J):
        """World-space centroid bias predicted at back-projected point `P_xy` seen at
        pixel (u_px, v_px) with plane Jacobian `J`. Subtract from the raw point. Also
        used by the static sweep to express its refit in the same basis."""
        cam = self.camera
        u_hat = P_xy - cam.pos[:2]
        hd = float(np.linalg.norm(u_hat))
        u_hat = u_hat / hd
        un = (u_px - cam.cx) / cam.f
        vn = (v_px - cam.cy) / cam.f
        return ((self.bias_radial[0] + self.bias_radial[1] * hd) * u_hat
                + J @ (cam.f * np.array([self.bias_image[0] * un,
                                         self.bias_image[1] * vn])))

    def render(self):
        """RGB frame (H, W, 3) from the named camera at the sensor's resolution."""
        if self._renderer is None:
            w, h = self.resolution
            self._renderer = mujoco.Renderer(self._model, h, w)  # height first
        self._renderer.update_scene(self._data, camera=self._camera_name)
        return self._renderer.render()

    def measure(self):
        """`PuckMeasurement` from the current scene, or None when no puck is detected.

        Pipeline: render -> segment -> back-project both centroids onto the puck-top
        plane -> yaw from the marker offset direction (the marker sits on the puck's
        local +x, so atan2 of the offset IS puck_yaw, world frame) -> wall-pixel bias
        correction -> pose-dependent covariance from the FD Jacobian.
        """
        frame = self.render()
        det = detect_puck(frame, self._puck_profile, backend=self._detect_backend)
        self._last_frame = frame          # cached for a live viewer (see run_hw_match --view)
        self._last_detection = det
        if det is None:
            return None
        cam = self.camera
        P = cam.pixel_to_plane(*det.centroid)
        J = cam.fd_jacobian(*det.centroid)
        if P is None or J is None:
            return None

        # A blob clipped by the image border keeps its measurement but at inflated
        # uncertainty (see the sigma_border comment above).
        u0, v0, u1, v1 = det.bbox
        w_px, h_px = self.resolution
        border_var = (self.sigma_border ** 2
                      if u0 <= 0 or v0 <= 0 or u1 >= w_px - 1 or v1 >= h_px - 1
                      else 0.0)

        # Remove the calibrated centroid bias (near-side wall pixels + perspective
        # disc-centroid shift; see the constructor comment) BEFORE the yaw
        # computation: the marker has no visible wall (it sits 0.1 mm proud of the red
        # top), so a ~5 mm-biased centre against an unbiased marker would skew the
        # 15.5 mm yaw lever arm by up to ~0.33 rad (measured: yaw RMSE 0.29 rad with
        # no correction at all).
        pos_xy = P[:2] - self.bias_correction(P[:2], *det.centroid, J)

        yaw = None
        R_yaw = None
        if det.marker_centroid is not None:
            Pm = cam.pixel_to_plane(*det.marker_centroid)
            Jm = cam.fd_jacobian(*det.marker_centroid)
            if Pm is not None and Jm is not None:
                off = Pm[:2] - pos_xy
                d = float(np.linalg.norm(off))
                # A marker whose back-projected offset is not roughly the physical
                # 15.5 mm is a misdetection (blue leak, partial blob) -- discard yaw.
                if 0.3 * PUCK_RADIUS <= d <= 1.1 * PUCK_RADIUS:
                    yaw = float(np.arctan2(off[1], off[0]))
                    R_c = (self.sigma_px ** 2 * (J @ J.T)
                           + (self.sigma_floor ** 2 + border_var) * np.eye(2))
                    R_m = self.sigma_px ** 2 * (Jm @ Jm.T) + self.sigma_floor ** 2 * np.eye(2)
                    R_yaw = float((np.trace(R_c) + np.trace(R_m)) / (2 * MARKER_OFFSET ** 2))
                    # Floor at the EKF's fitted yaw observation variance: the filter
                    # was never tuned for a yaw more certain than its own R.
                    R_yaw = max(R_yaw, float(OBS_COV[2, 2]))

        R_xy = (self.sigma_px ** 2 * (J @ J.T)
                + (self.sigma_floor ** 2 + border_var) * np.eye(2))
        return PuckMeasurement(pos_xy, yaw, R_xy, R_yaw, det.area,
                               clipped=border_var > 0, stamp=self._frame_stamp)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
