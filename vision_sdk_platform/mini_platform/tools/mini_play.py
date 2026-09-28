"""Play the mini platform live in the MuJoCo viewer.

    python -m mini_platform.tools.mini_play                      # mini-hit, 10 episodes
    python -m mini_platform.tools.mini_play --task mini-defend
    python -m mini_platform.tools.mini_play --task mini-prepare --episodes 3 --paused
    python -m mini_platform.tools.mini_play --task mini-hit-sdk-2p --episodes 50

The last one is the match to watch: both arms run a real `MiniBaselineAgent` (rather than the
scripted lissajous wander) on the METAL SDK's own low-level joint law.

On-screen controls (`tools/match_controls.py`, `--no-controls` to hide): a clickable strip at
the bottom-left -- PLAY, PAUSE, RESET (new episode, does not consume one from `--episodes`)
and HIT PUCK, which kicks a stalled puck back into play. Keys `r` and `x` do the last two.
The strip also reads out puck speed, so you can see a stall rather than guess at one.

Viewer controls (mushroom_rl's MujocoViewer): SPACE pause/resume, left-drag orbit, right-drag
pan, scroll zoom, `C` contact forces, `T` transparency, `0`-`8` toggle geom groups (group 1 is
the visual meshes, so `1` strips the arm down to its collision geometry), `S`/`H` menu, `E`
reference frames, `S`/`F` speed, `TAB` cycle camera, `ESC` quit.

Live stats are printed once per episode -- watching alone will not tell you whether the mallet
is staying flat or creeping out of the gripper's jaws.

Offline tool: never import from the runtime path.
"""

import argparse

import mujoco
import numpy as np
from mushroom_rl.utils.mujoco.viewer import MujocoViewer

from mini_platform.agents.baseline_mini import MiniBaselineAgent
from mini_platform.core.challenge_wrapper import AirHockeyChallengeWrapper
from mini_platform.tools import match_controls

TASKS = ["mini-hit", "mini-defend", "mini-prepare",
         "mini-hit-sdk", "mini-defend-sdk", "mini-prepare-sdk",
         "mini-hit-2p", "mini-hit-sdk-2p"]


def patch_viewer_for_mujoco3():
    """mushroom_rl 1.10.1 calls mjv_moveCamera with MuJoCo 2.x's signature.

    `mjv_moveCamera(m, action, reldx, reldy, scene, cam)` lost its `scene` argument in MuJoCo
    3.x, so every mouse drag and scroll raises TypeError and kills the run. The viewer opens
    and renders fine -- it only breaks the moment you touch it, which makes it look like a
    crash rather than an incompatibility.

    Rebinding the two callbacks here keeps the fix out of site-packages.
    """
    import glfw

    def scroll(self, window, x_offset, y_offset):
        mujoco.mjv_moveCamera(self._model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0, 0.05 * y_offset, self._camera)

    def mouse_move(self, window, x_pos, y_pos):
        if not self.button_left and not self.button_right and not self.button_middle:
            return

        dx, dy = x_pos - self.last_x, y_pos - self.last_y
        self.last_x, self.last_y = x_pos, y_pos
        width, height = glfw.get_window_size(self._window)

        mod_shift = (glfw.get_key(self._window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                     glfw.get_key(self._window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

        if self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(self._model, action, dx / width, dy / height, self._camera)

    MujocoViewer.scroll = scroll
    MujocoViewer.mouse_move = mouse_move


# Framed on the mini table, which is small enough that the viewer's default 15 m distance puts
# it in the middle of the skybox.
CAMERA_PARAMS = {
    "static": {"distance": 2.0, "elevation": -35.0, "azimuth": 135.0,
               "lookat": np.array([-0.2, 0.0, 0.0])},
    "follow": {"distance": 1.2, "elevation": -35.0, "azimuth": 135.0},
    "top_static": {"distance": 1.6, "elevation": -89.0, "azimuth": 90.0,
                   "lookat": np.array([0.0, 0.0, 0.0])},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mini-hit", choices=TASKS)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--paused", action="store_true", help="start paused; SPACE to run")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-controls", action="store_true",
                    help="hide the on-screen PLAY/PAUSE/RESET/HIT PUCK strip")
    ap.add_argument("--nudge-speed", type=float, default=1.0,
                    help="puck speed (m/s) the HIT PUCK button imparts")
    ap.add_argument("--planner", choices=["bezier", "vpsto"], default="bezier",
                    help="strike planner for the agent: bezier (default) or vpsto (needs cma)")
    args = ap.parse_args()

    patch_viewer_for_mujoco3()

    if args.seed is not None:
        np.random.seed(args.seed)

    viewer_params = {"camera_params": CAMERA_PARAMS,
                     "start_paused": args.paused,
                     "width": 1280, "height": 720}
    if not args.no_controls:
        viewer_params["custom_render_callback"] = match_controls.draw_active

    env = AirHockeyChallengeWrapper(
        args.task, interpolation_order=3, viewer_params=viewer_params)
    agent = MiniBaselineAgent(env.env_info, agent_id=1, planner=args.planner)
    controls = (None if args.no_controls else
                match_controls.MatchControls(env, nudge_speed=args.nudge_speed))

    print(f"{args.task}: {args.episodes} episodes, horizon {env.info.horizon}. "
          f"SPACE pauses, ESC quits.\n")
    print(f"{'ep':>3} {'steps':>6} {'success':>8} {'puck vmax':>10} {'tilt p99':>9} {'slip':>10} {'violations':>11}")

    successes = 0
    ep = 0
    try:
        while ep < args.episodes:
            obs = env.reset()
            agent.episode_start()
            tilt, v_max, viol, success, slip = [], 0.0, 0, 0, 0.0
            aborted = False

            for t in range(env.info.horizon):
                obs, r, absorbing, info = env.step(agent.draw_action(obs))
                env.render()

                if controls is not None and controls.take_reset():
                    aborted = True
                    break

                success = max(success, info["success"])
                v_max = max(v_max, float(np.linalg.norm(env.base_env.get_puck(obs)[1][:2])))
                slip_m, tilt_deg, _ = env.base_env.grip_state(1)
                tilt.append(tilt_deg)
                slip = max(slip, slip_m * 1000.0)
                viol += sum(1 for v in info["constraints_value"].values() if np.any(v > 0))

                if absorbing:
                    break

            if aborted:
                # A hand reset is not a result: it does not consume an --episodes slot.
                print(f"{'--':>3} {t + 1:>6} {'reset':>8}")
                continue

            ep += 1
            successes += success
            print(f"{ep:>3} {t + 1:>6} {'yes' if success else '-':>8} "
                  f"{v_max:>9.2f}m/s {np.percentile(tilt, 99):>8.1f}d {slip:>8.2f}mm {viol:>11}")
    except KeyboardInterrupt:
        print("\ninterrupted")

    print(f"\nsuccess {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
