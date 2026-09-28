"""Robot-referenced camera + base_frame calibration -- derive BOTH from the arm itself.

Why this exists: the board method (`calibrate_extrinsics`) produced an extrinsic with the world
X-axis INVERTED, sending the defender off-board. Root cause: the rig as then built was mirror-
imaged in y vs the sim's `base_frame` (facing the far end from the arm, the arm's own rail was on
the LEFT; the sim had it on the RIGHT). No rigid camera pose can reconcile a mirrored rig with
the sim frame, so a board fit throws the contradiction into an axis flip. A real arm can't be
mirror-imaged, so we must MEASURE its actual mounted pose rather than guess a flipped base_frame.

That specific mirror is gone since the 2026-09-10 re-mount -- the arm now stands on the table's
own axis at the middle of a short end, so there is no left/right to get wrong and only WHICH
short end sets the chirality. Measuring the pose is still the right answer, and this tool is
still the one to run after any re-mount: the base moved AND rotated, and `--stage set-origin`
corrects translation only.

Approach (all with the arm in GRAVITY COMPENSATION -- back-drivable, you move it by hand, no IK,
no planning):

  Phase 1  HAND-EYE:  GRIP THE PUCK in the gripper (it is exactly what puck_profile detects --
                      no separate marker needed). Move the arm by hand to several spread poses
                      (vary height too). At each, the joint encoders + FK give the puck in the
                      ROBOT BASE frame (it sits at the grasp point, so --tip-offset ~ 0), and the
                      camera gives its pixel. solvePnP on those 3D<->2D pairs recovers the camera
                      pose in the base frame.

  Phase 2  TABLE:     swap the puck for the MALLET, then touch three physical features, all
                      within easy reach from the END mount (``--touches goal``):

                        GOAL MOUTH  robot xy ~ (0.13, 0.00) -- mallet centred in your own goal
                                    mouth, its centre on the goal line
                        LEFT RAIL   robot xy ~ (0.35, +0.23) -- mallet edge on the left rail
                        RIGHT RAIL  robot xy ~ (0.35, -0.23) -- mallet edge on the right rail

                      +z is the arm's own; +y is rail to rail (500 mm baseline); +x = y x z. The
                      origin's y is the rail midpoint (the centre line, by symmetry) and its x is
                      the goal line placed at -L/2 -- the table model the planner uses -- so
                      reach-probe reads 0 at the goal mouth by construction. No touch HEIGHT
                      enters the frame, and nothing needs a tape measure. See
                      ``world_from_goal_and_rails`` for why each earlier choice failed:
                      a tape-measured centre touch landed 92 mm short, and rails riding up 36 mm
                      tilted a fitted plane 10.2 deg. ``--surface-drop`` is the mallet's grasp
                      height above the surface.

Compose the two -> a consistent ``hw_calib.npz`` (camera->world) AND ``base_frame.npz``
(world<-base) that agree. Pass the latter to the runtime with ``--base-frame``.

Known rig prior (APPROXIMATE; used only as a coarse sanity check on the result): the camera is
at the middle of one LONG rail, ~0.54 m above the surface, looking across and down the table at
~45 deg -- the last solve put it at world (-0.016, +0.307, 0.537). The arm stands at the middle
of the -x SHORT end, so from the arm the camera is ~0.69 m away, ~26 deg to its LEFT. The
recovered camera position is printed against this so a grossly wrong solve is obvious.

SAFETY: the camera rail is a PHYSICAL OBSTACLE and nothing here or in the play loop knows about
it -- run_hw_match's reach/play stages command joint targets clamped only to joint limits. At the
side mount the rail sat ~30 cm in front of the base, well inside the envelope; from the end mount
it is ~0.69 m away and 0.57 m above the base, at the edge of reach rather than the middle of it.
Re-check the clearance by hand before trusting it. This tool itself keeps the arm in gravity
compensation throughout (you move it by hand -- the arm never drives itself). Keep a hand near
the E-stop and start with short ``--duration`` runs.

Usage (arm powered, can0 up, cp310 runtime)::

    python -m mini_platform.hw.calibrate_robot \
        --intrinsics calibration.npz --camera-index 0 --resolution 640x480 \
        --puck-profile puck_profile.json \
        --out hw_calib_new.npz --out-base base_frame_new.npz

Then verify:  ... run_hw_match --stage vision --view --calib hw_calib.npz --base-frame base_frame.npz
"""

import argparse
import os

import numpy as np

# Same self-contained vendor URDF run_hw_match defaults to (relative meshes, no ROS sourcing).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_URDF = os.path.join(_REPO_ROOT, "metal_sdk", "example", "urdf", "metal_with_gripper.urdf")

from mini_platform import make
from mini_platform.core.kinematics import forward_kinematics
from mini_platform.envs.env_base import GRASP_HEIGHT_ABOVE_SURFACE, N_JOINTS
from mini_platform.hw.real_camera import CameraCalibration, load_intrinsics


# ------------------------------------------------------------------- homogeneous helpers
def homog(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float).ravel()
    return T


def inv_homog(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ------------------------------------------------------------------------ pure math (tested)
def solve_hand_eye(ee_base_pts, img_pts, K, dist):
    """PnP: 3D marker points in the ROBOT BASE frame <-> pixels -> camera pose in base frame.

    Returns (T_base_cam (4x4, pose of camera in base), reproj_err_px). Needs >= 4 non-coplanar
    points; 6-8 spread over position AND height give a solid solve.
    """
    import cv2

    obj = np.asarray(ee_base_pts, float).reshape(-1, 3)
    img = np.asarray(img_pts, float).reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed on the hand-eye correspondences")
    R_cb, _ = cv2.Rodrigues(rvec)                  # base -> camera
    T_cam_base = homog(R_cb, tvec.ravel())
    reproj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(reproj.reshape(-1, 2) - img, axis=1)
    return inv_homog(T_cam_base), err              # pose of camera IN base


def world_from_goal_and_rails(goal_b, left_b, right_b, half_length, surface_drop):
    """World frame from the OWN GOAL MOUTH + the two LONG RAILS (grasp points, base frame).

    Built for the END mount. Every input is a physical feature the arm reaches easily, and
    nothing depends on touch HEIGHT, which is where the previous protocols lost it:

      * +z is the ARM's own +z. The arm stands upright and the table is level to it --
        ee_height.npz puts seven flat-mallet spots across the table within 1.9 mm. A normal
        FITTED through three touches is at the mercy of any one of them riding up a rail: on
        2026-09-15 both rail touches did, by ~36 mm, which tilted the frame 10.2 deg and, once
        planarised, moved the base 33 mm down-table.
      * +y is LEFT - RIGHT, horizontal. 500 mm apart, so 2 mm of touch error is ~0.2 deg of
        yaw, and a rail cannot be mistaken for the down-table direction.
      * +x = y x z, from the arm down the table. The sign is checked against the arm itself:
        the table lies in front of the base, so the rails' midpoint must have positive x.
      * ORIGIN y is the rails' MIDPOINT -- the centre line by symmetry, whatever the mallet
        radius, because the two edge offsets cancel.
      * ORIGIN x is the goal touch + ``half_length``. The mallet centred in its own goal mouth
        sits ON the goal line, which the planner puts at x = -half_length, so this uses the
        same table model as everything downstream and reach-probe reads 0 at the goal mouth by
        construction. It replaces a tape-measured table-centre touch at 0.64 m of reach, past
        where a vertical wrist solves; on 2026-09-15 that touch was recorded 92 mm short.
      * ORIGIN z is the goal touch height minus ``surface_drop`` (grasp height, mallet flat).

    Returns (T_base_world, info). ``info`` is diagnostics only -- none of it enters the frame.
    """
    g = np.asarray(goal_b, float)
    l = np.asarray(left_b, float)
    r = np.asarray(right_b, float)
    z = np.array([0.0, 0.0, 1.0])

    y = (l - r) * np.array([1.0, 1.0, 0.0])
    span = float(np.linalg.norm(y))
    if span < 0.05:
        raise RuntimeError(f"the two rail touches are only {span*1000:.0f} mm apart -- they must "
                           "be on OPPOSITE long rails")
    y = y / span
    x = np.cross(y, z)                              # right-handed: x cross y = z

    mid = 0.5 * (l + r)
    flipped = bool(np.dot(x, mid) < 0)              # the table must be in FRONT of the base
    if flipped:
        x, y = -x, -y

    origin = (np.dot(g, x) + half_length) * x + np.dot(mid, y) * y + (g[2] - surface_drop) * z

    n_fit = np.cross(l - g, r - g)
    n_fit = n_fit / max(np.linalg.norm(n_fit), 1e-12)
    info = {
        "rail_span": span,
        "flipped": flipped,
        "goal_off_centre": float(np.dot(g, y) - np.dot(mid, y)),
        "left_rise": float(l[2] - g[2]),
        "right_rise": float(r[2] - g[2]),
        "touch_plane_tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(n_fit[2]))))),
    }
    return homog(np.column_stack([x, y, z]), origin), info


def sim_camera_relative_to_base(camera="table_side_left", scene="mini_single.xml"):
    """Where the SIM puts the table camera relative to arm 1's base, in the base's own frame.

    Hand-eye alone recovers exactly this -- no table touch enters it -- so it is the one check
    that can catch a bad hand-eye solve before phase 2 buries it inside a world frame.
    """
    import mujoco
    from mini_platform.envs.env_base import ASSET_DIR
    m = mujoco.MjModel.from_xml_path(os.path.join(ASSET_DIR, scene))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "metal_1/base_link")
    return d.xmat[bid].reshape(3, 3).T @ (d.cam_xpos[cid] - d.xpos[bid])


def check_hand_eye(T_base_cam, reproj_px, ee_base_pts, expected_rel, tol_plan=0.15,
                   up_range=(0.25, 0.95)):
    """Validate the hand-eye solve on its own, against the sim's camera placement.

    Forward and left are set by where the arm and camera are mounted, so they are compared to
    the sim with a tolerance. UP is not: the real base stands far higher than the model's (~0.04
    m below the surface against 0.21), so the camera only has to be clearly ABOVE the base.
    On 2026-09-15 this solve put the camera 0.43 m BELOW the base -- which nothing caught,
    because the table touches then wrapped it in a world frame. Returns a plain bool.
    """
    rel = np.asarray(T_base_cam, float)[:3, 3]
    exp = np.asarray(expected_rel, float)
    checks = [
        ("camera forward of base", abs(rel[0] - exp[0]) <= tol_plan,
         f"{rel[0]:+.3f} m   (sim {exp[0]:+.3f})"),
        ("camera left of base", abs(rel[1] - exp[1]) <= tol_plan,
         f"{rel[1]:+.3f} m   (sim {exp[1]:+.3f})"),
        ("camera above base", up_range[0] <= rel[2] <= up_range[1],
         f"{rel[2]:+.3f} m   (must be {up_range[0]:.2f}..{up_range[1]:.2f})"),
        ("reprojection mean", float(np.mean(reproj_px)) <= 3.0,
         f"{np.mean(reproj_px):.2f} px (max {np.max(reproj_px):.2f})"),
    ]
    print("\n---------------- hand-eye: camera relative to the arm base ----------------")
    for name, ok, txt in checks:
        print(f"  {name:24s} {txt:40s} {'ok' if ok else '<-- FAIL'}")
    z_spread = float(np.ptp(np.asarray(ee_base_pts, float)[:, 2]))
    if z_spread < 0.08:
        print(f"  WARNING: the captures span only {z_spread*1000:.0f} mm in height. Near-coplanar "
              "points let PnP converge to a mirrored camera -- vary height by 10 cm or more.")
    ok = all(c[1] for c in checks)
    if not ok:
        print("  The hand-eye solve is wrong on its own, before any table touch. Recollect phase 1:")
        print("  10+ poses, spread in HEIGHT as well as across the table, puck boxed green each time.")
    return bool(ok)


def report_vs_sim(base_frame, env_info, tol_xy=0.08, tol_yaw=10.0):
    """Check the recovered base pose against the mount the SIM is built for, and say so loudly.

    ``env_info['robot']['base_frame']`` is read straight out of the MJCF, so this compares the
    calibration against the layout every downstream consumer (table geometry, EKF bounce model,
    V frame, tactic ranges) actually assumes. Only xy and yaw are compared -- the real base sits
    at a different HEIGHT than the model and that is expected and handled by --ee-height.

    A 90 deg frame rotation shows up here immediately, which is the whole point.
    """
    T_sim = np.asarray(env_info["robot"]["base_frame"][0], float)
    yaw = lambda T: np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    d_xy = float(np.linalg.norm(base_frame[:2, 3] - T_sim[:2, 3]))
    d_yaw = abs((yaw(base_frame) - yaw(T_sim) + 180) % 360 - 180)
    ok = bool(d_xy <= tol_xy and d_yaw <= tol_yaw)   # plain bool: numpy's is not `is True`
    print("\n---------------- vs the SIM mount (assets/arm1.xml) ----------------")
    print(f"  measured base xy {np.round(base_frame[:2, 3], 4)}  yaw {yaw(base_frame):+7.2f} deg")
    print(f"  sim      base xy {np.round(T_sim[:2, 3], 4)}  yaw {yaw(T_sim):+7.2f} deg")
    print(f"  delta            {d_xy*1000:6.0f} mm        {d_yaw:7.2f} deg    "
          f"{'PASS' if ok else '<-- FAIL'}")
    if not ok:
        print("  The calibration does NOT describe the mount the sim is built for. Do not use it.")
        if d_yaw > 45:
            print(f"  A {d_yaw:.0f} deg error means the world axes are swapped: +x landed on the")
            print("  table's WIDTH instead of its LENGTH. Re-do phase 2 -- with --touches goal")
            print("  this cannot happen unless the two rail touches were on the same rail.")
    return ok


def world_from_touches(centre_b, far_b, side_b, surface_drop):
    """Define the world frame from three mallet-touch points (grasp positions in the base frame).

    origin = table centre projected down to the SURFACE, +x toward the far point, +z = table
    normal (up), +y = z x x. Right-handed by construction, with the rig's real chirality --
    whichever physical rail +y lands on is simply where it is. ``surface_drop`` shifts the
    grasp-height plane down to the surface (the grasp frame sits that far above the mallet head).

    Returns T_base_world (4x4, pose of the world frame expressed in the base frame).
    """
    c = np.asarray(centre_b, float)
    f = np.asarray(far_b, float)
    s = np.asarray(side_b, float)
    n = np.cross(f - c, s - c)
    n = n / np.linalg.norm(n)
    if n[2] < 0:                                    # make +z point up (base z is up: arm upright)
        n = -n
    x = (f - c) - np.dot(f - c, n) * n              # far direction, flattened into the plane
    x = x / np.linalg.norm(x)
    y = np.cross(n, x)                              # right-handed: x cross y = n(=z)
    R_base_world = np.column_stack([x, y, n])
    origin = c - surface_drop * n                   # grasp-height centre -> surface centre
    return homog(R_base_world, origin)


def compose(T_base_world, T_base_cam):
    """(T_base_world, T_base_cam) -> (R_wc, C_world, base_frame).

    base_frame is env's 'world<-base' (pose of base in world); R_wc/C_world are the camera pose
    in world for CameraCalibration.
    """
    base_frame = inv_homog(T_base_world)            # world <- base  (pose of base in world)
    T_world_cam = inv_homog(T_base_world) @ T_base_cam
    R_wc = T_world_cam[:3, :3]
    C_world = T_world_cam[:3, 3]
    return R_wc, C_world, base_frame


# --------------------------------------------------------------------------- collection (hw)
def _fk_marker_base(env_info, q, tip_offset):
    """Marker position in the robot base frame: FK grasp point + tool-frame offset."""
    rm = env_info["robot"]["robot_model"]
    rd = env_info["robot"]["robot_data"]
    pos, rot = forward_kinematics(rm, rd, np.asarray(q, float)[:N_JOINTS])
    return np.asarray(pos, float) + rot @ np.asarray(tip_offset, float)


def _annotate(cv2, frame_rgb, det, lines=None):
    """Overlay the live puck detection (green bbox+centroid, or a red 'no puck' border)."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if det is None:
        cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
        cv2.putText(bgr, "NO PUCK", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        u0, v0, u1, v1 = det.bbox
        cv2.rectangle(bgr, (int(u0), int(v0)), (int(u1), int(v1)), (0, 255, 0), 2)
        cu, cv = int(round(det.centroid[0])), int(round(det.centroid[1]))
        cv2.drawMarker(bgr, (cu, cv), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(bgr, f"puck ({cu},{cv})  area {det.area}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y = 52
    for ln in (lines or []):
        cv2.putText(bgr, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        y += 24
    return bgr


def collect_hand_eye(arm, cam, profile, env_info, tip_offset, cv2, win, min_pts=6):
    from mini_platform.vision.detector import detect_puck

    print("\n== PHASE 1: hand-eye ==\nGrip the PUCK. Move the arm BY HAND (gravity comp) to a "
          "spread of poses -- vary height and wrist angle -- keeping the puck detected (GREEN "
          f"box) in the live view. Capture each, done when >= {min_pts}. Keep clear of the rail.")
    ee_b, px = [], []

    def _capture(det):
        if det is None:
            print("    NO PUCK detected -- reposition until the box is green.")
            return
        ee_b.append(_fk_marker_base(env_info, arm.q(), tip_offset))
        px.append(np.asarray(det.centroid, float))
        print(f"    captured #{len(ee_b)}: base {np.round(ee_b[-1], 3)}  px {np.round(px[-1], 1)}")

    if cv2 is None:                                    # headless fallback: stdin-driven
        while True:
            s = input(f"  [{len(ee_b)} captured] Enter=capture / d=done: ").strip().lower()
            if s == "d" and len(ee_b) >= min_pts:
                break
            if s == "d":
                print(f"    need >= {min_pts}.")
                continue
            frame = cam.read_rgb()
            _capture(None if frame is None else detect_puck(frame, profile, backend="cv2"))
        return np.array(ee_b), np.array(px)

    while True:                                        # live-view: keys in the window
        frame = cam.read_rgb()
        det = None if frame is None else detect_puck(frame, profile, backend="cv2")
        if frame is not None:
            cv2.imshow(win, _annotate(cv2, frame, det,
                                      [f"HAND-EYE  captured {len(ee_b)} (need >= {min_pts})",
                                       "SPACE/ENTER = capture    d = done    q = quit"]))
        key = cv2.waitKey(20) & 0xFF
        if key in (32, 13):                            # SPACE / ENTER
            _capture(det)
        elif key == ord("d"):
            if len(ee_b) >= min_pts:
                break
            print(f"    need >= {min_pts} before done.")
        elif key in (ord("q"), 27):                    # q / ESC
            raise KeyboardInterrupt("aborted from the view window")
    return np.array(ee_b), np.array(px)


def collect_touch(arm, env_info, name, cam, profile, cv2, win):
    from mini_platform.vision.detector import detect_puck

    if cv2 is None:
        input(f"  Rest the gripped tool (VERTICAL, on the surface) at {name}, then Enter... ")
    else:
        print(f"  Rest the gripped tool (VERTICAL, on the surface) at {name}; SPACE=capture, q=quit.")
        while True:
            frame = cam.read_rgb()
            if frame is not None:
                det = detect_puck(frame, profile, backend="cv2")
                cv2.imshow(win, _annotate(cv2, frame, det,
                                          [f"TOUCH: {name}", "SPACE/ENTER = capture    q = quit"]))
            key = cv2.waitKey(20) & 0xFF
            if key in (32, 13):
                break
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("aborted from the view window")
    rm = env_info["robot"]["robot_model"]
    rd = env_info["robot"]["robot_data"]
    pos, _ = forward_kinematics(rm, rd, arm.q()[:N_JOINTS])
    print(f"    grasp point (base): {np.round(pos, 3)}")
    return np.asarray(pos, float)


# ------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--camera-index", type=int, default=4)
    ap.add_argument("--resolution", default="640x480")
    ap.add_argument("--puck-profile", default=None)
    ap.add_argument("--task", default="mini-defend-sdk")
    ap.add_argument("--can-id", default="can0")
    ap.add_argument("--urdf", default=DEFAULT_URDF,
                    help="arm URDF (default: the vendor metal_with_gripper.urdf, as run_hw_match)")
    ap.add_argument("--arm-end-type", type=int, default=1)
    ap.add_argument("--tip-offset", default="0,0,0",
                    help="marker/puck offset from the grasp ('ee') frame, tool coords m 'x,y,z' "
                         "(a puck gripped at the grasp point is ~0,0,0)")
    ap.add_argument("--touches", choices=["goal", "far"], default="goal",
                    help="how phase 2 defines the table frame. 'goal' (default; the one for the "
                         "current END mount): the mallet in your own goal mouth plus its edge "
                         "against each long rail. +z is the arm's own, +y is rail-to-rail, the "
                         "origin's y is the rail midpoint and its x is the goal line -- all "
                         "within easy reach, and no touch height moves the frame. 'far' is the "
                         "legacy centre/far-end/side protocol, which from this mount produced a "
                         "90 deg rotated frame (2026-09-14).")
    ap.add_argument("--surface-drop", type=float, default=GRASP_HEIGHT_ABOVE_SURFACE,
                    help="grasp-frame height above the playing surface with the MALLET resting "
                         f"flat (default {GRASP_HEIGHT_ABOVE_SURFACE:.3f} m is the SIM mallet -- "
                         "measure yours with calipers, jaw centre to mallet face). Sets world "
                         "z = 0 at the surface, which the oblique camera's plane intersection "
                         "depends on: every mm wrong here is ~a mm of puck position error.")
    ap.add_argument("--force", action="store_true",
                    help="write the calibration even if a check fails. Without it a failed run "
                         "leaves --out / --out-base untouched (raw captures are always saved).")
    ap.add_argument("--view", action=argparse.BooleanOptionalAction, default=True,
                    help="live annotated camera window during capture (default ON; --no-view "
                         "for headless). Lets you confirm the PUCK (not a reflection/the gripper) "
                         "is what's detected before each capture.")
    ap.add_argument("--mock", action="store_true", help="skip the arm (math smoke test only)")
    ap.add_argument("--out", default="hw_calib.npz")
    ap.add_argument("--out-base", default="base_frame.npz")
    args = ap.parse_args(argv)

    w, h = (int(x) for x in args.resolution.lower().split("x"))
    K, dist = load_intrinsics(args.intrinsics)
    tip_offset = np.array([float(x) for x in args.tip_offset.split(",")], float)
    env_info = make(args.task).env_info

    from mini_platform.hw.real_camera import V4L2Camera
    from mini_platform.hw.sdk_backend import MetalArm, MockMetalArm
    from mini_platform.vision.detector import PuckColorProfile

    profile = PuckColorProfile.load(args.puck_profile) if args.puck_profile else None
    cam = V4L2Camera(index=args.camera_index, resolution=(w, h))
    arm = (MockMetalArm() if args.mock else
           MetalArm(can_id=args.can_id, urdf_path=args.urdf, arm_end_type=args.arm_end_type))

    cv2mod, win = None, None
    if args.view:
        try:
            import cv2 as cv2mod
            win = "metal robot calibration"
            cv2mod.namedWindow(win, cv2mod.WINDOW_NORMAL)
        except Exception as e:
            print(f"(live view unavailable: {e} -- continuing headless; use --no-view to silence)")
            cv2mod = None

    try:
        arm.init()
        arm.set_mode_gravity()          # back-drivable: you position the arm by hand throughout

        ee_b, px = collect_hand_eye(arm, cam, profile, env_info, tip_offset, cv2mod, win)
        T_base_cam, he_err = solve_hand_eye(ee_b, px, K, dist)
        print(f"\nhand-eye reprojection px: mean={he_err.mean():.2f} max={he_err.max():.2f}")

        print("\n== PHASE 2: table frame ==")
        if args.touches == "goal":
            print("  Swap the gripped PUCK for the MALLET first -- --surface-drop is the mallet's")
            print(f"  grasp height ({args.surface_drop:.3f} m). Every touch: mallet FLAT on the playing")
            print("  surface. The grasp point prints in the BASE frame after each -- check it:")
            print("     GOAL MOUTH  robot xy ~ (0.13,  0.00)  mallet CENTRED in your own goal mouth,")
            print("                 its centre ON the goal line (the line joining the rim ends).")
            print("                 Sets the origin along the table: the planner's goal line is")
            print("                 x = -L/2, so reach-probe reads 0 exactly here.")
            print("     LEFT RAIL   robot xy ~ (0.35, +0.23)  mallet EDGE against the inner face of")
            print("                 the long rail on the arm's LEFT (the camera side).")
            print("     RIGHT RAIL  robot xy ~ (0.35, -0.23)  mallet edge against the OPPOSITE rail.")
            print("  Keep the mallet on the surface at the rails. If it rides up that is reported,")
            print("  but no touch height moves the frame any more. Rail order is self-checked.\n")
            g = collect_touch(arm, env_info, "your own GOAL MOUTH (centre on the line)", cam, profile, cv2mod, win)
            f = collect_touch(arm, env_info, "the LEFT rail (arm's left)", cam, profile, cv2mod, win)
            s = collect_touch(arm, env_info, "the RIGHT rail (opposite side)", cam, profile, cv2mod, win)
        else:
            print("  Legacy protocol, for a mount where the far end is reachable. At the END mount")
            print("  it is 1.135 m out against a 0.740 m reach -- use --touches goal there.")
            g = collect_touch(arm, env_info, "the TABLE CENTRE", cam, profile, cv2mod, win)
            f = collect_touch(arm, env_info, "a point toward the FAR end (centre line)", cam, profile, cv2mod, win)
            s = collect_touch(arm, env_info, "a point to ONE SIDE (toward a rail)", cam, profile, cv2mod, win)
    finally:
        try:
            arm.catch_and_close()
        except Exception:
            pass
        cam.close()
        if cv2mod is not None:
            cv2mod.destroyAllWindows()

    # Raw captures are written no matter what, so a bad run can be diagnosed offline instead of
    # reconstructed from terminal scrollback.
    raw_path = os.path.splitext(args.out_base)[0] + "_raw.npz"
    np.savez(raw_path, hand_eye_base=ee_b, hand_eye_px=px, hand_eye_err_px=he_err,
             T_base_cam=T_base_cam, touches=np.array([g, f, s]), touch_protocol=args.touches,
             surface_drop=args.surface_drop, K=K, dist=dist)
    print(f"\nraw captures -> {raw_path}")

    checks_ok = check_hand_eye(T_base_cam, he_err, ee_b, sim_camera_relative_to_base())

    if args.touches == "goal":
        T_base_world, info = world_from_goal_and_rails(
            g, f, s, half_length=env_info["table"]["length"] / 2, surface_drop=args.surface_drop)
        expect_span = env_info["table"]["width"] - 2 * env_info["mallet"]["radius"]
        span_ok = bool(abs(info["rail_span"] - expect_span) <= 0.03)
        checks_ok = checks_ok and span_ok
        print("\n---------------- table touches ----------------")
        print(f"  rail separation   {info['rail_span']:.4f} m  (expect {expect_span:.4f} = width - 2 x "
              f"mallet radius)  {'ok' if span_ok else '<-- FAIL'}")
        print(f"  goal off centre   {info['goal_off_centre']*1000:+.0f} mm lateral  (does not move the "
              f"origin -- the rail midpoint sets y)")
        print(f"  rails vs goal z   left {info['left_rise']*1000:+.0f} mm, right "
              f"{info['right_rise']*1000:+.0f} mm  (mallet riding up -- reported, not used)")
        print(f"  a plane fitted through the touches would tilt {info['touch_plane_tilt_deg']:.1f} deg; "
              f"ignored -- the arm's own +z is used")
        if info["flipped"]:
            print("  rails were given RIGHT-then-LEFT; +y flipped to match the arm. No action needed.")
    else:
        T_base_world = world_from_touches(g, f, s, surface_drop=args.surface_drop)
    R_wc, C_world, base_frame = compose(T_base_world, T_base_cam)

    calib = CameraCalibration(K=K, dist=dist, R_wc=R_wc, C_world=C_world, resolution=(w, h))

    print("\n================ ROBOT-REFERENCED CALIBRATION ================")
    print(f"camera centre (world m) : {np.round(C_world, 4)}   height {C_world[2]:+.3f}")
    print(f"arm base   (world m)    : {np.round(base_frame[:3, 3], 4)}")
    cam_ok = bool(C_world[2] > 0.2)
    if not cam_ok:
        print("  <-- FAIL: the camera is at or below the table surface.")
    checks_ok = checks_ok and cam_ok
    checks_ok = report_vs_sim(base_frame, env_info) and checks_ok

    if not checks_ok and not args.force:
        print(f"\nNOT SAVED. At least one check failed, and writing would overwrite {args.out} and "
              f"{args.out_base} with a calibration known to be wrong. Recollect, or pass --force to "
              f"write it anyway. Raw captures: {raw_path}")
        raise SystemExit(2)

    calib.save(args.out)
    np.savez(args.out_base, T=base_frame)
    print(f"\nsaved camera -> {args.out}")
    print(f"saved base_frame (world<-base) -> {args.out_base}")
    print(f"\nVerify:  python -m mini_platform.hw.run_hw_match --stage reach-probe "
          f"--base-frame {args.out_base}   (dist-to-goal ~0 with the mallet in the goal mouth)")


if __name__ == "__main__":
    main()
