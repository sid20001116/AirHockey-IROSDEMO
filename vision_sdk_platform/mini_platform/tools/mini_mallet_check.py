"""T0.2 acceptance -- verify the gripped mallet geometrically and dynamically.

The mallet is a FREE BODY held only by friction between the gripper fingers and its 10 mm
handle cube. Nothing about it is guaranteed by construction, so everything here is measured:

  1. Geometry at the home pose: where the handle sits relative to the jaws, whether the head
     rests on the table, and whether the head overlaps the puck's z-extent.
  2. Static hold: does the grip carry the mallet at all, and how far does it settle.
  3. Rebound: a puck driven at the parked mallet comes back out rather than tunnelling.
  4. Renders of the home pose.

    MUJOCO_GL=glfw python -m mini_platform.tools.mini_mallet_check
"""

import os

import mujoco
import numpy as np

from mini_platform.envs.env_base import (GRASP_HEIGHT, GRASP_OFFSET,
                                                             GRIP_FORCE, HANDLE_HALF,
                                                             MALLET_HALF_HEIGHT, MALLET_RADIUS,
                                                             PUCK_RADIUS)
from mini_platform.core.challenge_wrapper import AirHockeyChallengeWrapper

from mini_platform import ARTIFACT_DIR

OUT_DIR = os.path.join(ARTIFACT_DIR, "reports", "img")


def geometry_check(env):
    be = env.base_env
    m, d = be._model, be._data

    head = d.geom("metal_1/ee")
    head_bottom = float(head.xpos[2]) - MALLET_HALF_HEIGHT
    puck_top = 2 * float(m.geom("puck").size[1])
    slip, tilt, jaw = be.grip_state(1)

    print("== geometry at the home pose ==")
    print(f"  grasp point z             {GRASP_HEIGHT:.4f} m (offset {GRASP_OFFSET} below the wrist)")
    print(f"  handle half-width         {HANDLE_HALF * 1000:.1f} mm; jaws at {jaw * 1000:.2f} mm")
    print(f"  mallet head               r {MALLET_RADIUS * 1000:.0f} mm, {2 * MALLET_HALF_HEIGHT * 1000:.0f} mm tall")
    print(f"  head bottom z             {head_bottom:+.5f} m (0 == resting on the surface)")
    print(f"  head z span               {head_bottom:+.4f} .. {head_bottom + 2 * MALLET_HALF_HEIGHT:+.4f}")
    print(f"  puck z span               {0.0:+.4f} .. {puck_top:+.4f}")
    print(f"  mallet tilt / slip        {tilt:.3f} deg / {slip * 1000:.2f} mm")
    print(f"  contact distance d_c      {PUCK_RADIUS + MALLET_RADIUS:.5f} m")
    print(f"  grip force                {GRIP_FORCE} N per jaw")

    covers = head_bottom <= 0.001 and (head_bottom + 2 * MALLET_HALF_HEIGHT) >= puck_top
    seated = jaw > 0.8 * HANDLE_HALF and tilt < 1.0 and slip < 0.002
    print(f"  head covers puck height:  {covers}")
    print(f"  mallet seated in jaws:    {seated}")
    return covers and seated


def hold_check(env, steps=500):
    """Hold the arm still and confirm the grip carries the mallet."""
    be = env.base_env
    obs = env.reset()
    q = obs[env.env_info["joint_pos_ids"]]
    action = np.vstack([q, np.zeros_like(q)])

    worst_slip, worst_tilt = 0.0, 0.0
    for _ in range(steps):
        obs, r, d, info = env.step(action)
        slip, tilt, jaw = be.grip_state(1)
        worst_slip = max(worst_slip, slip)
        worst_tilt = max(worst_tilt, tilt)
        if jaw < 1e-3:
            print("\n== static hold ==\n  DROPPED -- jaws shut with no mallet between them")
            return False

    print(f"\n== static hold ({steps} steps) ==")
    print(f"  max slip in the jaws      {worst_slip * 1000:.2f} mm")
    print(f"  max tilt                  {worst_tilt:.2f} deg")
    print("  still gripped:            True")
    return worst_slip < 0.01


def rebound_check(env, speeds=(0.5, 1.0, 2.0, 2.5)):
    """Head-on rebound over the speed range the tasks actually generate.

    `mini-defend` tops out at 2.53 m/s. Measured head-on tunnelling threshold is between
    3.0 and 3.5 m/s at the 1 ms timestep -- only ~1.2x headroom, so re-measure this if a
    planner ever commands faster shots. MuJoCo has no continuous collision detection.
    """
    be = env.base_env
    m, d = be._model, be._data
    print("\n== puck rebound off the gripped mallet ==")
    ok = True
    d_c = PUCK_RADIUS + MALLET_RADIUS

    for v in speeds:
        obs = env.reset()
        q = obs[env.env_info["joint_pos_ids"]]
        action = np.vstack([q, np.zeros_like(q)])

        mallet_xy = d.body("mallet_1").xpos[:2].copy()
        d.joint("puck_x").qpos = mallet_xy[0] + 0.20
        d.joint("puck_y").qpos = mallet_xy[1]
        d.joint("puck_x").qvel = -v
        # `mini-hit` resets the puck with a random lateral and spin velocity; leaving those in
        # makes this a head-on test only by accident, and the result varies run to run.
        d.joint("puck_y").qvel = 0.0
        d.joint("puck_yaw").qvel = 0.0
        mujoco.mj_forward(m, d)

        min_gap, v_out = np.inf, 0.0
        for _ in range(120):
            env.step(action)
            puck = np.array([d.joint("puck_x").qpos[0], d.joint("puck_y").qpos[0]])
            min_gap = min(min_gap, float(np.linalg.norm(puck - d.body("mallet_1").xpos[:2])))
            v_out = float(d.joint("puck_x").qvel[0])
            if v_out > 0.01:
                break

        tunnelled = min_gap < 0.5 * d_c
        good = v_out > 0 and not tunnelled
        ok = ok and good
        print(f"  v_in {v:4.1f} -> v_out {v_out:+6.3f} m/s   closest {min_gap:.4f} m (d_c {d_c:.4f})   "
              f"{'ok' if good else 'TUNNELLED' if tunnelled else 'NO REBOUND'}")
    return ok


def render(env):
    be = env.base_env
    env.reset()
    os.makedirs(os.path.normpath(OUT_DIR), exist_ok=True)

    from PIL import Image
    renderer = mujoco.Renderer(be._model, 720, 1280)

    renderer.update_scene(be._data, camera="table_side_left")
    Image.fromarray(renderer.render()).save(
        os.path.join(os.path.normpath(OUT_DIR), "mini_home_side_camera.png"))

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [-0.45, 0.0, 0.05]
    cam.distance, cam.azimuth, cam.elevation = 0.32, 150.0, -15.0
    renderer.update_scene(be._data, camera=cam)
    Image.fromarray(renderer.render()).save(
        os.path.join(os.path.normpath(OUT_DIR), "mini_home_mallet_closeup.png"))
    print(f"\nwrote 2 renders to {os.path.normpath(OUT_DIR)}")


def main():
    env = AirHockeyChallengeWrapper(env="mini-hit", interpolation_order=3)
    m = env.base_env._model
    print(f"mini_single/double: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody}\n")
    env.reset()

    g = geometry_check(env)
    h = hold_check(env)
    r = rebound_check(env)
    try:
        render(env)
    except Exception as e:
        print(f"\nrender skipped: {type(e).__name__}: {e}")

    print(f"\nT0.2 acceptance: geometry {'PASS' if g else 'FAIL'}, "
          f"static hold {'PASS' if h else 'FAIL'}, rebound {'PASS' if r else 'FAIL'}")


if __name__ == "__main__":
    main()
