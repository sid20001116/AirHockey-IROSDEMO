"""Fit the arm's base frame from TELEOP touches at tape-measured table points.

Mark a few spots on the table, measure each one's world coordinates with a tape, rest the mallet
on each by hand, and this fits where the arm's base is -- x, y and yaw -- plus the length from the
jaw point to the mallet's contact, which the kinematic model does not know. No camera, no board,
no hand-eye: every number comes from a tape measure and the arm's own encoders.

WHY THE TOOL LENGTH IS FITTED. Forward kinematics (``link="ee"``) reports the JAW point, where the
gripper holds the mallet handle. The mallet touches the table further down the tool axis. With the
wrist vertical the two share xy; tilted, they do not -- and from the end mount the wrist sits ~40
deg off vertical at both the goal line (leaning back toward the base) and the table centre (leaning
out). On 2026-09-15 that alone made a taped 0.508 m read 0.391 m, and every "touch was short" and
"rails are too close" result that day was this, not operator error. The contact is modelled as
``jaw + L * tool_axis``; touches at DIFFERENT tilts make L observable, and the per-point residuals
show whether the model holds. Pass ``--tool-length`` to fix L instead (e.g. once measured).

Frame convention (as everywhere else): origin at the table CENTRE, +x toward the far end (away from
arm 1), +y toward the arm's LEFT rail (the camera side), z up with z = 0 on the playing surface.
The own goal line is x = -0.508 on this 1016 x 508 mm table.

Spread the marks: some near the goal (the wrist leans back), some near the centre (it leans out),
and some off to the sides (they pin yaw). Five to eight is plenty::

    source /opt/ros/humble/setup.bash
    python -m mini_platform.hw.calibrate_teleop \\
        --points "-0.508,0  -0.40,0.15  -0.40,-0.15  -0.25,0  -0.10,0.15  -0.10,-0.15  0,0" \\
        --out-base base_frame_teleop.npz

Then use the SAME taped marks for the camera, so both share one world frame::

    python -m mini_platform.hw.calibrate_points --intrinsics calibration.npz --camera-index 0 \\
        --puck-profile puck_profile.json --points "<same list>" --out hw_calib_teleop.npz

The saved frame carries ``tool_length``; ``run_hw_match --stage reach-probe`` reads it and reports
the mallet CONTACT, so the readout is right at any wrist tilt.
"""

import argparse
import os

import numpy as np

#: Largest per-point fit residual (mm) a calibration may keep before it is refused.
MAX_RESIDUAL_MM = 8.0


# ------------------------------------------------------------------------ pure math (tested)
def tilt_deg(jaw_rot):
    """Tool-axis angle off vertical (deg). The jaw frame's +z is the tool axis, pointing DOWN."""
    return float(np.degrees(np.arccos(np.clip(-np.asarray(jaw_rot, float)[2, 2], -1.0, 1.0))))


def contact_point(jaw_pos, jaw_rot, tool_length):
    """Mallet contact in the robot frame: the jaw point moved ``tool_length`` down the tool axis."""
    return np.asarray(jaw_pos, float) + float(tool_length) * np.asarray(jaw_rot, float)[:, 2]


def _world_xy(tx, ty, yaw, L, jaw_pos, jaw_rot):
    c, s = np.cos(yaw), np.sin(yaw)
    p = jaw_pos[:, :2] + L * jaw_rot[:, :2, 2]
    return np.column_stack([c * p[:, 0] - s * p[:, 1] + tx, s * p[:, 0] + c * p[:, 1] + ty])


def fit_base_frame(jaw_pos, jaw_rot, world_xy, tool_length=None, mallet_radius=0.025,
                   initial_tool_length=0.06):
    """Least-squares base pose (x, y, yaw) -- and tool length, unless given -- from touches.

    ``jaw_pos`` (N,3) / ``jaw_rot`` (N,3,3) are FK of the jaw point in the ROBOT frame;
    ``world_xy`` (N,2) the taped table coordinates of where the mallet rested.

    The base height comes from the same touches: each contact's height, less the lift a tilted
    mallet gets from resting on its rim (``mallet_radius * sin(tilt)``), is the playing surface in
    the robot frame. Returns (T world<-base 4x4, upright; info dict).
    """
    from scipy.optimize import least_squares

    jaw_pos = np.asarray(jaw_pos, float).reshape(-1, 3)
    jaw_rot = np.asarray(jaw_rot, float).reshape(-1, 3, 3)
    world_xy = np.asarray(world_xy, float).reshape(-1, 2)
    fit_L = tool_length is None
    need = 3
    if len(world_xy) < need:
        raise ValueError(f"need at least {need} touches, got {len(world_xy)}")

    # the tool length is only observable if the tool axis POINTS differently across the touches
    axis_xy = jaw_rot[:, :2, 2]
    axis_spread = float(np.max(np.linalg.norm(axis_xy[:, None, :] - axis_xy[None, :, :], axis=2)))
    if fit_L and axis_spread < 0.3:
        raise ValueError(
            f"the wrist barely changed tilt across these touches (tool-axis spread {axis_spread:.2f}) "
            "so the tool length cannot be fitted. Add marks near the goal AND near the centre, or "
            "pass --tool-length.")

    L0 = initial_tool_length if fit_L else float(tool_length)
    p = jaw_pos[:, :2] + L0 * axis_xy                      # 2-D Kabsch at L0 for the start point
    pc, wc = p.mean(0), world_xy.mean(0)
    H = (p - pc).T @ (world_xy - wc)
    yaw0 = float(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
    R0 = np.array([[np.cos(yaw0), -np.sin(yaw0)], [np.sin(yaw0), np.cos(yaw0)]])
    t0 = wc - R0 @ pc

    if fit_L:
        sol = least_squares(lambda x: (_world_xy(*x, jaw_pos, jaw_rot) - world_xy).ravel(),
                            [t0[0], t0[1], yaw0, L0])
        tx, ty, yaw, L = sol.x
    else:
        sol = least_squares(lambda x: (_world_xy(*x, L0, jaw_pos, jaw_rot) - world_xy).ravel(),
                            [t0[0], t0[1], yaw0])
        (tx, ty, yaw), L = sol.x, L0

    contacts = jaw_pos + L * jaw_rot[:, :, 2]
    tilts = np.array([tilt_deg(R) for R in jaw_rot])
    surface_z = contacts[:, 2] - mallet_radius * np.sin(np.radians(tilts))

    T = np.eye(4)
    T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    T[:3, 3] = [tx, ty, -float(np.mean(surface_z))]
    residual = np.linalg.norm(_world_xy(tx, ty, yaw, L, jaw_pos, jaw_rot) - world_xy, axis=1)
    return T, {
        "tool_length": float(L),
        "tool_length_fitted": fit_L,
        "yaw_deg": float(np.degrees(yaw)),
        "residual_m": residual,
        "tilt_deg": tilts,
        "surface_z_spread_m": float(np.ptp(surface_z)),
        "axis_spread": axis_spread,
    }


def parse_points(s):
    """'x,y  x,y ...' (metres, table-centre frame) -> (N,2)."""
    vals = [float(v) for v in s.replace(",", " ").split()]
    if len(vals) % 2:
        raise SystemExit("--points must be pairs 'x,y x,y ...'")
    return np.array(vals, float).reshape(-1, 2)


# ----------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", required=True,
                    help="tape-measured world points 'x,y x,y ...' (m, table-centre frame)")
    ap.add_argument("--tool-length", type=float, default=None,
                    help="jaw point -> mallet contact along the tool axis (m). Default: FIT it "
                         "(needs touches at different wrist tilts).")
    ap.add_argument("--out-base", default="base_frame_teleop.npz")
    ap.add_argument("--max-residual-mm", type=float, default=MAX_RESIDUAL_MM)
    ap.add_argument("--force", action="store_true", help="save even if the fit checks fail")
    ap.add_argument("--read-only", action="store_true",
                    help="do not take the arm: only read it, while ANOTHER process (e.g. "
                         "run_hw_match --stage grav) holds it in gravity compensation")
    ap.add_argument("--seconds", type=float, default=1.0, help="steady window per touch")
    ap.add_argument("--task", default="mini-defend-sdk")
    ap.add_argument("--can-id", default="can0")
    ap.add_argument("--arm-end-type", type=int, default=1)
    ap.add_argument("--urdf", default=None, help="arm URDF for the SDK (default: the vendor one)")
    args = ap.parse_args(argv)

    from mini_platform import make
    from mini_platform.hw.calibrate_robot import DEFAULT_URDF
    from mini_platform.hw.capture_home import STEADY_BAND, capture
    from mini_platform.hw.sdk_backend import MetalArm

    world = parse_points(args.points)
    if len(world) < 3:
        raise SystemExit("need at least 3 points (5-8 spread across the own half is better)")
    env_info = make(args.task).env_info
    r_m = env_info["mallet"]["radius"]
    half_l = env_info["table"]["length"] / 2

    arm = MetalArm(can_id=args.can_id, urdf_path=args.urdf or DEFAULT_URDF,
                   arm_end_type=args.arm_end_type)
    jaw_pos, jaw_rot, qs, used = [], [], [], []
    arm.init()
    try:
        if not args.read_only:
            arm.set_mode_gravity()           # back-drivable: you place the mallet by hand
        print("Rest the MALLET on each mark -- flat if the wrist lets it, otherwise on its rim --")
        print("hold still, press ENTER. 's' skips a mark, 'q' finishes early.\n")
        i = 0
        while i < len(world):
            x, y = world[i]
            ans = input(f"[{i+1}/{len(world)}] mallet on world ({x:+.3f}, {y:+.3f}), ENTER: ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                i += 1
                continue
            q, pos, rot, spread = capture(arm, env_info, seconds=args.seconds)
            if float(spread.max()) > STEADY_BAND:
                print(f"   moved {np.degrees(spread.max()):.2f} deg during the window -- hold still and retry")
                continue
            print(f"   jaw robot xy ({pos[0]:+.4f}, {pos[1]:+.4f})  z {pos[2]:+.4f}  tilt {tilt_deg(rot):4.1f} deg")
            jaw_pos.append(pos); jaw_rot.append(rot); qs.append(q); used.append((x, y))
            i += 1
    finally:
        if not args.read_only:
            try:
                arm.catch_and_close()        # hand the arm to gravity comp on the way out
            except Exception:
                pass

    jaw_pos, jaw_rot, qs, used = map(np.asarray, (jaw_pos, jaw_rot, qs, used))
    raw_path = os.path.splitext(args.out_base)[0] + "_raw.npz"
    np.savez(raw_path, jaw_pos=jaw_pos, jaw_rot=jaw_rot, q=qs, world_xy=used)
    print(f"\nraw touches -> {raw_path}")
    if len(used) < 3:
        raise SystemExit(f"only {len(used)} touches recorded; need at least 3.")

    try:
        T, info = fit_base_frame(jaw_pos, jaw_rot, used, tool_length=args.tool_length,
                                 mallet_radius=r_m)
    except ValueError as e:
        raise SystemExit(f"fit refused: {e}")

    res_mm = info["residual_m"] * 1000
    print("\n================ TELEOP BASE-FRAME FIT ================")
    print(f"  {'mark (world)':>18s}  {'fitted contact':>18s}  {'residual':>9s}  {'tilt':>6s}")
    L = info["tool_length"]
    for (x, y), pw, rot, e in zip(used, jaw_pos, jaw_rot, res_mm):
        c = contact_point(pw, rot, L)
        w = T[:3, :3] @ c + T[:3, 3]
        print(f"  ({x:+.3f}, {y:+.3f})   ({w[0]:+.3f}, {w[1]:+.3f})   {e:6.1f} mm  {tilt_deg(rot):5.1f}")
    print(f"\n  residual          rms {np.sqrt(np.mean(res_mm**2)):.1f} mm, max {res_mm.max():.1f} mm")
    print(f"  tool length       {L*1000:.1f} mm jaw -> contact "
          f"({'fitted' if info['tool_length_fitted'] else 'fixed'}; the sim mallet is 60 mm)")
    print(f"  base world        ({T[0,3]:+.4f}, {T[1,3]:+.4f}, {T[2,3]:+.4f})   yaw {info['yaw_deg']:+.2f} deg")
    print(f"  own goal line     robot x {-half_l - T[0,3]:+.4f} (straight out from the base)")
    print(f"  surface heights   spread {info['surface_z_spread_m']*1000:.1f} mm across the touches")

    ok = bool(res_mm.max() <= args.max_residual_mm)
    if info["tool_length_fitted"] and not (0.0 <= L <= 0.25):
        print(f"  <-- FAIL: a {L*1000:.0f} mm tool length is not physical")
        ok = False
    if not ok:
        print(f"  <-- FAIL: residual above {args.max_residual_mm:.0f} mm. A mark was mis-measured or "
              "the mallet was not on it; the worst row above is the suspect.")
        if not args.force:
            print(f"\nNOT SAVED (pass --force to save anyway). Raw touches: {raw_path}")
            raise SystemExit(2)

    np.savez(args.out_base, T=T, tool_length=L, residual_m=info["residual_m"])
    print(f"\nsaved base_frame (world<-base) + tool_length -> {args.out_base}")
    print(f"\nCheck it:  python -m mini_platform.hw.run_hw_match --stage reach-probe "
          f"--base-frame {args.out_base}\n           (mallet in the goal mouth -> dist-to-goal ~0, "
          f"at any wrist tilt)")
    print(f"Camera:    python -m mini_platform.hw.calibrate_points --intrinsics calibration.npz "
          f"--camera-index 0 --puck-profile puck_profile.json --points \"{args.points}\" "
          f"--out hw_calib_teleop.npz")


if __name__ == "__main__":
    main()
