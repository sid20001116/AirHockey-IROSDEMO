"""Log real puck rolls from the table camera -- the raw data the mini-table EKF refit needs.

The EKF's puck-motion constants (`estimation/kalman_filter.py`: ``damping``, the rim
``b_params``/``n_params``/``theta_params``, ``OBS_COV``) are still fits to the FULL-SIZE KUKA
Air-Hockey Challenge table. On the mini table they bias ``PuckTracker.get_prediction`` and the
bias **compounds with travel time**, which is exactly the "arm is late when the puck is far"
symptom. This tool records the three kinds of motion that identify them, and
``tools/fit_mini_table.py`` turns the recordings into a ``mini_table_ekf.npz`` that
``run_hw_match --ekf-params`` loads.

**No arm, no sim.** Camera and calibration only: no ``metal_sdk``, no CAN, no mujoco (see the
dependency split in ``hw/__init__.py``). Roll the puck by hand; nothing moves under power.

    conda activate metal_hw            # any env with numpy + scipy + cv2 will do
    cd /home/siddarth/metal-python-ros2

    # 1. straight glides -- damping.   ~15-20 long rolls, varied speeds and directions,
    #    each one clear of every rail.
    python -m mini_platform.hw.capture_puck_rolls --tag straight --out rolls_straight.npz \\
        --calib hw_calib.npz --puck-profile puck_profile.json --camera-index 4 --view

    # 2. wall bounces -- restitution.  ~10 into EACH rail (both long rails + the far short
    #    rail; skip the goal mouth), mostly perpendicular plus a few oblique.
    python -m mini_platform.hw.capture_puck_rolls --tag bounce --out rolls_bounce.npz ...

    # 3. static puck -- measurement noise. >=30 s at ~3 spots (centre + two corners); move
    #    the puck between spots, the fitter separates them.
    python -m mini_platform.hw.capture_puck_rolls --tag static --out rolls_static.npz ...

Then::

    python -m mini_platform.tools.fit_mini_table rolls_*.npz --out mini_table_ekf.npz

What is logged, and why it is the same number the EKF sees
---------------------------------------------------------
Per **distinct camera frame** (never the same buffer twice -- see ``read_rgb_stamped``):
the driver timestamp, the world xy of the puck centre, yaw when the marker resolves, the blob
area, the pixel centroid, and the 2x2 plane Jacobian at that pixel.

World xy here is ``camera.pixel_to_plane(centroid)``, which is what ``--stage vision --view``
displays AND -- on hardware -- exactly what the EKF is fed: ``RealPuckSensor`` inherits
``PuckVisionSensor.measure()``, whose only extra step is ``bias_correction``, and the real
sensor's ``bias_radial``/``bias_image`` are **(0, 0)** (that model corrected MuJoCo render
artefacts and was never re-fitted for the real feed), so the correction is identically zero.
The pixel centroid and Jacobian are logged so the fitter can recover ``sigma_px`` -- the
sensor's own noise scale, which is what actually sets the per-measurement ``R`` on hardware --
and reconstruct that ``R`` when replaying (``tools/ekf_replay.py``).

The world frame is the calibration's: origin at the table centre, +x toward the far end. The
EKF's frame V is that frame translated by ``(V_SETBACK, 0)`` and nothing else, so velocities,
damping and rim-local bounce ratios are numerically identical in both -- see
``estimation/mini_env.py``. ``--base-frame`` is therefore deliberately NOT an option here: the
robot's mounting pose cancels out of the world -> V map and cannot affect these fits.
"""

import argparse
import os
import time

import numpy as np

from mini_platform.estimation.mini_env import PUCK_RADIUS, TABLE_LENGTH, TABLE_WIDTH
from mini_platform.hw.real_camera import CameraCalibration, RealPinholeCamera, V4L2Camera
from mini_platform.vision.detector import PuckColorProfile, detect_puck

#: What the capture is for. The fitter reads it out of the npz and routes the samples:
#: ``straight`` -> damping, ``bounce`` -> restitution, ``static`` -> measurement noise.
TAGS = ("straight", "bounce", "static")

#: Speed (m/s) above which the puck counts as "rolling", for the live roll counter and the
#: end-of-run summary. Only cosmetic -- the fitter segments the log itself, with its own
#: (stricter) gates.
ROLL_SPEED = 0.15

#: A roll must last this long (s) to be counted/announced -- filters out a nudge.
ROLL_MIN_DURATION = 0.20


def _measure(camera, det, puck_radius=PUCK_RADIUS):
    """``(xy, yaw, J)`` from one detection, mirroring ``PuckVisionSensor.measure``.

    ``yaw`` is None unless the blue marker resolved at a plausible lever arm (the same
    ``0.3-1.1`` puck-radius gate the sim sensor applies, which rejects blue leak and partial
    blobs). ``J`` is the 2x2 d(world xy)/d(pixel) Jacobian, or None where the finite-difference
    probe leaves the plane.
    """
    P = camera.pixel_to_plane(*det.centroid)
    J = camera.fd_jacobian(*det.centroid)
    if P is None or J is None:
        return None, None, None
    xy = P[:2]
    yaw = None
    if det.marker_centroid is not None:
        Pm = camera.pixel_to_plane(*det.marker_centroid)
        if Pm is not None:
            off = Pm[:2] - xy
            d = float(np.linalg.norm(off))
            if 0.3 * puck_radius <= d <= 1.1 * puck_radius:
                yaw = float(np.arctan2(off[1], off[0]))
    return xy, yaw, J


class RollWatcher:
    """Live feedback: current speed, and a count of the rolls captured so far.

    Deliberately simple (a 2-point difference over the last two frames, hysteresis on
    ``ROLL_SPEED``) -- this exists so the operator can see that a roll registered and how fast
    it was, not to measure anything. The fit does its own segmentation offline.
    """

    def __init__(self):
        self.rolls = []             # (t_start, duration, peak_speed, path_length)
        self.speed = 0.0
        self._prev = None           # (t, xy) of the previous detected frame
        self._cur = None            # in-progress roll accumulator

    def update(self, t, xy):
        if xy is None:
            self._prev = None       # a dropout breaks the difference, not the roll
            return None
        if self._prev is not None:
            dt = t - self._prev[0]
            step = float(np.linalg.norm(xy - self._prev[1]))
            if 1e-4 < dt < 0.5:
                self.speed = step / dt
                if self._cur is not None:
                    self._cur["peak"] = max(self._cur["peak"], self.speed)
                    self._cur["path"] += step
        self._prev = (t, np.asarray(xy, float))

        done = None
        if self.speed >= ROLL_SPEED and self._cur is None:
            self._cur = {"t0": t, "peak": self.speed, "path": 0.0}
        elif self.speed < ROLL_SPEED and self._cur is not None:
            dur = t - self._cur["t0"]
            if dur >= ROLL_MIN_DURATION:
                done = (self._cur["t0"], dur, self._cur["peak"], self._cur["path"])
                self.rolls.append(done)
            self._cur = None
        return done


def _annotate(cv2, frame_rgb, det, xy, lines):
    """Detection overlay (green box + centroid cross) with a status block. Returns BGR."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if det is None:
        cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)      # red border = dropout
        cv2.putText(bgr, "no detection", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        u0, v0, u1, v1 = det.bbox
        cv2.rectangle(bgr, (int(u0), int(v0)), (int(u1), int(v1)), (0, 255, 0), 1)
        cv2.drawMarker(bgr, (int(round(det.centroid[0])), int(round(det.centroid[1]))),
                       (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
        txt = "world xy: --" if xy is None else \
            f"world xy: {xy[0]:+.3f} {xy[1]:+.3f} m   px: {det.area}"
        cv2.putText(bgr, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y = 52
    for line in lines:
        cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        y += 24
    return bgr


def _off_table(xy, margin=0.05):
    """True when a detection lands implausibly far outside the playing surface.

    A stale/rotated extrinsic shows up here immediately -- the detector is happily locked on
    the puck while the back-projection puts it a metre off the table. Worth a loud warning
    before 20 minutes of rolls are recorded against the wrong frame.
    """
    return (abs(xy[0]) > TABLE_LENGTH / 2 + margin) or (abs(xy[1]) > TABLE_WIDTH / 2 + margin)


def capture(camera, calib_camera, profile, args):
    """Poll the camera, detect once per NEW frame, and collect the log rows."""
    rows = {k: [] for k in ("t", "x", "y", "yaw", "area", "u", "v", "J", "seq")}
    watcher = RollWatcher()
    win, cv2 = None, None
    if args.view:
        try:
            import cv2 as _cv2
            _cv2.namedWindow(args.window, _cv2.WINDOW_NORMAL)
            cv2, win = _cv2, args.window
        except Exception as e:                     # headless box / no GTK build
            print(f"  (--view unavailable: {type(e).__name__}: {e} -- continuing headless)")

    print(f"capturing '{args.tag}'. Roll the puck by hand. "
          f"{'q/ESC in the window or ' if win else ''}Ctrl-C to stop"
          f"{f' (auto-stop after {args.duration:.0f}s)' if args.duration else ''}.")
    last_seq, t0, warned_off = None, None, False
    n_det = n_frames = 0
    try:
        while True:
            frame, seq, t_grab = camera.read_rgb_stamped()
            if frame is None or seq == last_seq:
                # No genuinely new frame: sleep well under a frame period and poll again. The
                # capture must never log the same buffer twice -- a duplicated sample looks
                # like zero displacement and would drag every velocity estimate down.
                time.sleep(0.001)
                continue
            last_seq = seq
            n_frames += 1
            if t0 is None:
                t0 = t_grab
            t = t_grab - t0

            det = detect_puck(frame, profile, backend=args.backend)
            xy = yaw = J = None
            if det is not None:
                xy, yaw, J = _measure(calib_camera, det)
            if xy is not None:
                n_det += 1
                if not warned_off and _off_table(xy):
                    warned_off = True
                    print(f"\n  !! detection back-projects to {np.round(xy, 3)} m, well OUTSIDE "
                          f"the {TABLE_LENGTH:.3f} x {TABLE_WIDTH:.3f} m table. The extrinsic "
                          f"(--calib) is probably stale -- re-run hw.calibrate_robot before "
                          f"trusting this capture.")
            rows["t"].append(t)
            rows["x"].append(np.nan if xy is None else xy[0])
            rows["y"].append(np.nan if xy is None else xy[1])
            rows["yaw"].append(np.nan if yaw is None else yaw)
            rows["area"].append(0 if det is None else int(det.area))
            rows["u"].append(np.nan if det is None else float(det.centroid[0]))
            rows["v"].append(np.nan if det is None else float(det.centroid[1]))
            rows["J"].append(np.full(4, np.nan) if J is None else np.asarray(J, float).ravel())
            rows["seq"].append(int(seq))

            done = watcher.update(t, xy)
            if done is not None:
                print(f"\n  roll {len(watcher.rolls):2d}: {done[1]:4.2f} s, peak "
                      f"{done[2]:4.2f} m/s, travelled {done[3] * 100:5.1f} cm")

            status = (f"  t {t:6.1f}s  frames {n_frames:5d}  det {100.0 * n_det / n_frames:5.1f}%"
                      f"  speed {watcher.speed:5.2f} m/s  rolls {len(watcher.rolls):2d}   ")
            print(status, end="\r", flush=True)

            if win is not None:
                lines = [f"tag: {args.tag}   rolls: {len(watcher.rolls)}   "
                         f"speed: {watcher.speed:.2f} m/s",
                         f"frames: {n_frames}   detected: {100.0 * n_det / max(n_frames, 1):.0f}%"]
                cv2.imshow(win, _annotate(cv2, frame, det, xy, lines))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            if args.duration and t >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if win is not None:
            cv2.destroyAllWindows()
    print()
    return rows, watcher, n_frames, n_det


def save(path, rows, args, calib, profile_path, n_frames, n_det):
    """Write the capture npz (the fitter's input format)."""
    t = np.asarray(rows["t"], float)
    out = dict(
        t=t,
        xy=np.stack([np.asarray(rows["x"], float), np.asarray(rows["y"], float)], axis=1),
        yaw=np.asarray(rows["yaw"], float),
        area=np.asarray(rows["area"], int),
        uv=np.stack([np.asarray(rows["u"], float), np.asarray(rows["v"], float)], axis=1),
        J=np.asarray(rows["J"], float).reshape(-1, 2, 2),
        seq=np.asarray(rows["seq"], int),
        # --- provenance: what this was, and against which calibration ---
        tag=np.array(args.tag),
        calib_path=np.array(os.path.abspath(args.calib)),
        puck_profile_path=np.array("" if profile_path is None else os.path.abspath(profile_path)),
        resolution=np.asarray(calib.resolution, int),
        camera_index=np.array(args.camera_index),
        frame_rate=np.array(len(t) / (t[-1] - t[0]) if len(t) > 1 else np.nan),
        detect_rate=np.array(n_det / max(n_frames, 1)),
    )
    np.savez(path, **out)
    print(f"saved {len(t)} frames ({100.0 * n_det / max(n_frames, 1):.1f}% detected) -> {path}")


class RollCapture:
    """One capture npz, loaded. Owns the on-disk format together with :func:`save`.

    ``t`` (n,) seconds from the first frame, ``xy`` (n, 2) world metres, ``yaw`` (n,) rad,
    ``uv`` (n, 2) pixel centroid, ``J`` (n, 2, 2) d(world)/d(pixel), ``seq`` (n,) camera frame
    counter, ``good`` (n,) bool = a detection landed on that frame. Rows with ``good == False``
    are dropouts and are kept, not dropped: the gap they leave is what tells the fitter that
    two samples are not adjacent in time.
    """

    def __init__(self, path, d):
        self.path = str(path)
        self.t = np.asarray(d["t"], float)
        self.xy = np.asarray(d["xy"], float)
        self.yaw = np.asarray(d["yaw"], float)
        self.uv = np.asarray(d["uv"], float)
        self.J = np.asarray(d["J"], float)
        self.seq = np.asarray(d["seq"], int)
        self.area = np.asarray(d["area"], int)
        self.tag = str(d["tag"])
        self.resolution = tuple(int(x) for x in d["resolution"])
        self.calib_path = str(d["calib_path"]) if "calib_path" in d.files else ""
        self.good = np.all(np.isfinite(self.xy), axis=1)

    def __len__(self):
        return len(self.t)

    @property
    def fps(self):
        return (len(self.t) - 1) / (self.t[-1] - self.t[0]) if len(self.t) > 1 else float("nan")


def load_capture(path):
    """Read a capture npz written by this tool -> :class:`RollCapture`."""
    d = np.load(path, allow_pickle=False)
    missing = [k for k in ("t", "xy", "yaw", "uv", "J", "seq", "area", "tag", "resolution")
               if k not in d.files]
    if missing:
        raise ValueError(f"{path}: missing {missing} -- not a capture_puck_rolls npz "
                         f"(got {list(d.files)})")
    return RollCapture(path, d)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, choices=TAGS,
                    help="what this capture is for (routes it in the fitter)")
    ap.add_argument("--out", required=True, help="output npz")
    ap.add_argument("--calib", required=True, help="hw_calib.npz (camera->world extrinsic)")
    ap.add_argument("--puck-profile", default=None,
                    help="tuned colour gate from hw.tune_puck (STRONGLY recommended: the sim "
                         "default gate mis-detects a real puck)")
    ap.add_argument("--camera-index", type=int, default=4)
    ap.add_argument("--resolution", default=None,
                    help="WxH override; defaults to the calibration's resolution (which is "
                         "what the intrinsics are valid at -- only override to debug)")
    ap.add_argument("--backend", default="cv2", choices=["cv2", "numpy", "auto"],
                    help="detector backend; cv2 is what the hardware sensor uses")
    ap.add_argument("--duration", type=float, default=None, help="auto-stop after N seconds")
    ap.add_argument("--view", action="store_true", help="live annotated window (q/ESC quits)")
    ap.add_argument("--window", default="puck roll capture")
    args = ap.parse_args(argv)

    calib = CameraCalibration.load(args.calib)
    resolution = tuple(calib.resolution)
    if args.resolution:
        w, h = args.resolution.lower().split("x")
        resolution = (int(w), int(h))
        if resolution != tuple(calib.resolution):
            print(f"WARNING: capturing at {resolution} but the intrinsics were calibrated at "
                  f"{tuple(calib.resolution)} -- world xy will be WRONG. For debugging only.")

    profile = None
    if args.puck_profile:
        profile = PuckColorProfile.load(args.puck_profile)
        print(f"puck colour profile: {args.puck_profile} (mode={profile.mode}, "
              f"marker {'on' if profile.marker_enabled else 'OFF -> no yaw/spin data'})")
    else:
        print("WARNING: no --puck-profile; using the SIM red-puck gate, which will mis-detect a "
              "real puck. Tune one with: python -m mini_platform.hw.tune_puck")

    camera = V4L2Camera(index=args.camera_index, resolution=resolution)
    print(f"camera {args.camera_index} at {camera.resolution}; extrinsic from {args.calib} "
          f"(camera {np.round(calib.C_world, 3)} m in world)")
    calib_camera = RealPinholeCamera(calib.K, calib.dist, calib.R_wc, calib.C_world,
                                     camera.resolution)
    try:
        rows, watcher, n_frames, n_det = capture(camera, calib_camera, profile, args)
    finally:
        camera.close()

    if not rows["t"]:
        raise SystemExit("no frames captured -- is the camera index right?")
    save(args.out, rows, args, calib, args.puck_profile, n_frames, n_det)

    if args.tag == "static":
        xy = np.stack([np.asarray(rows["x"], float), np.asarray(rows["y"], float)], axis=1)
        good = np.all(np.isfinite(xy), axis=1)
        if good.sum() > 10:
            sd = np.std(xy[good], axis=0) * 1000
            print(f"  static scatter over the whole capture: {sd[0]:.2f} / {sd[1]:.2f} mm "
                  f"(x/y, all spots pooled -- the fitter splits them per spot)")
    else:
        print(f"  {len(watcher.rolls)} rolls detected live "
              f"(the fitter re-segments with stricter gates, so this is a floor).")


if __name__ == "__main__":
    main()
