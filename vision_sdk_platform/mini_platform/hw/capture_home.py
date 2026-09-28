"""Capture the park pose from the arm itself: teleop to the spot you want, hold, run this.

Reads the held joint configuration over a short window, checks it is actually steady, runs FK
and prints the mallet position in the ROBOT BASE frame -- the frame the arm measures in, which
is the frame the default park is stored in (``baseline_mini.HOME_PARK_XY``). Nothing here
commands the arm or changes its control mode: ``Init()`` only opens the bus, ``q()`` is a pure
read, and the context manager (which would hand the arm to gravity compensation on the way out)
is deliberately not used, so whatever mode is holding the arm keeps holding it.

    source /opt/ros/humble/setup.bash        # metal_sdk links rclcpp
    python -m mini_platform.hw.capture_home

Storing the park in the ROBOT frame is what makes the capture durable: it is the arm's own FK
that measured it, so the park stays at the spot you teleoped to even if ``base_frame`` is later
recalibrated. Only its table coordinates move, and those are derived, not stored.
"""
import argparse
import time

import numpy as np

from mini_platform.core.kinematics import forward_kinematics
from mini_platform.envs.env_base import N_JOINTS
from mini_platform.hw.sdk_backend import MetalArm

#: Joint-motion budget (rad) across the sampling window for the pose to count as "held". ~0.29
#: deg: tight enough that a drifting or hand-guided arm is rejected, loose enough for the
#: encoder noise and the small sag a gravity-comp hold sits in.
STEADY_BAND = 5e-3


def capture(arm, env_info, seconds=1.0, rate_hz=50.0):
    """Sample the held configuration, verify it is steady, and return (q, ee_position_robot)."""
    n = max(2, int(seconds * rate_hz))
    qs = []
    for _ in range(n):
        qs.append(np.asarray(arm.q(), float)[:N_JOINTS])
        time.sleep(1.0 / rate_hz)
    qs = np.array(qs)
    spread = qs.max(axis=0) - qs.min(axis=0)
    q = qs.mean(axis=0)

    rm, rd = env_info["robot"]["robot_model"], env_info["robot"]["robot_data"]
    pos, rot = forward_kinematics(rm, rd, q)
    return q, np.asarray(pos, float), np.asarray(rot, float), spread


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can-id", default="can0")
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--arm-end-type", type=int, default=1)
    ap.add_argument("--task", default="mini-hit-sdk")
    ap.add_argument("--seconds", type=float, default=1.0,
                    help="sampling window (s) the pose must be held steady for")
    ap.add_argument("--out", default=None,
                    help="also save the capture (q, ee position, rotation) to this .npz")
    ap.add_argument("--force", action="store_true",
                    help="report the pose even if it moved more than the steady band")
    ap.add_argument("--ee-height", type=float, default=None,
                    help="the play plane this rig runs at (m), as passed to run_hw_match. Only "
                         "affects the z sanity line: without it the comparison is against the "
                         "SIM nominal, which is metres away from any real table.")
    args = ap.parse_args()

    import mini_platform
    env_info = mini_platform.make(args.task).env_info

    arm = MetalArm(can_id=args.can_id, urdf_path=args.urdf, arm_end_type=args.arm_end_type)
    arm.init()          # opens the bus only -- no mode change, no command
    print(f"reading the held pose for {args.seconds:.1f}s -- keep holding it.")
    q, pos, rot, spread = capture(arm, env_info, seconds=args.seconds)

    moved = float(spread.max())
    if moved > STEADY_BAND and not args.force:
        raise SystemExit(
            f"the arm moved {np.degrees(moved):.2f} deg during the window (joint "
            f"{int(np.argmax(spread))}, band {np.degrees(STEADY_BAND):.2f} deg) -- it was not "
            f"being held still. Settle it and re-run, or pass --force to take it anyway.")

    # Wrist tilt off vertical: the tool axis is the rotation's third column; a vertical mallet
    # points straight down, so compare against -z.
    tilt = np.degrees(np.arccos(np.clip(-rot[:, 2] @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0)))
    z_play = (env_info["robot"]["ee_desired_height"] if args.ee_height is None
              else float(args.ee_height))
    z_src = "sim nominal -- pass --ee-height for the real one" if args.ee_height is None else \
            "--ee-height"

    print(f"\nheld steady to {np.degrees(moved):.3f} deg.")
    print(f"  q (rad)      : {np.round(q, 4).tolist()}")
    print(f"  mallet, ROBOT frame xy: [{pos[0]:+.4f}, {pos[1]:+.4f}] m "
          f"({np.linalg.norm(pos[:2]):.3f} m out from the base)")
    print(f"  z            : {pos[2]:+.4f} m vs the play plane {z_play:+.4f} ({z_src}), "
          f"{(pos[2] - z_play) * 1000:+.0f} mm\n"
          f"                 -- only xy is stored; the park's z is pinned to the play plane")
    print(f"  wrist tilt   : {tilt:.1f} deg off vertical")
    print(f"\nto make this the default park, set in agents/baseline_mini.py:")
    print(f"  HOME_PARK_XY = np.array([{pos[0]:.4f}, {pos[1]:.4f}])")

    if args.out:
        np.savez(args.out, q=q, pos=pos, rot=rot, tilt_deg=tilt)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
