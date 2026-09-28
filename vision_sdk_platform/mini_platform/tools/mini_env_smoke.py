"""T0.4 acceptance gate -- exercise every mini env with the retargeted baseline.

Reports, per task: episode outcomes, worst constraint value, whether the mallet actually
tracks the puck, and -- since the mallet is held by a friction grip rather than bolted on --
how far it has slipped and tipped in the jaws. A mallet that tilts or walks out of the grip
silently invalidates every contact-planning number downstream (optimal_control.md 3.1 assumes
a horizontal contact normal).

Offline tool: never import from the runtime path.

    python -m mini_platform.tools.mini_env_smoke [--episodes 20]
    python -m mini_platform.tools.mini_env_smoke --episodes 60 --tasks mini-defend,mini-defend-sdk

`drops` is the sharpest instrument here -- read it before anything else. It counts EPISODES in
which the mallet left the jaws, which is what actually matters. `slip_mm_p99` does not measure
that: it is dominated by *when* a drop happened, since a drop at t=50 contaminates the
remaining 450 samples. Measured on this rig, 912 mm p99 corresponded to 3 dropped episodes and
10 mm p99 to 1 -- a 90x spread in the percentile for a 3x difference in the thing you care
about.
"""

import argparse

import numpy as np

from mini_platform.agents.baseline_mini import MiniBaselineAgent
from mini_platform.core.challenge_wrapper import AirHockeyChallengeWrapper

TASKS = ["mini-hit", "mini-defend", "mini-prepare"]

#: Slip beyond which the mallet is considered out of the grip rather than shifted in it. The
#: handle is a 10 mm cube, so 20 mm means it has cleared the jaws entirely.
DROP_SLIP_M = 0.020


def run(task, n_episodes, seed=0):
    np.random.seed(seed)
    env = AirHockeyChallengeWrapper(env=task, interpolation_order=3)
    agent = MiniBaselineAgent(env.env_info, agent_id=1)
    horizon = env.info.horizon

    worst = {}
    successes = 0
    steps = []
    tilt_deg = []
    slip_mm = []
    jaw_mm = []
    ee_travel = []
    puck_speed_max = []
    drops = 0

    for _ in range(n_episodes):
        obs = env.reset()
        agent.episode_start()
        ee_start = env.base_env.get_ee()[0].copy()
        ee_extreme = 0.0
        v_max = 0.0
        success = 0
        slip_ep_max = 0.0

        for t in range(horizon):
            obs, r, absorbing, info = env.step(agent.draw_action(obs))

            for k, v in info["constraints_value"].items():
                worst[k] = max(worst.get(k, -np.inf), float(np.max(v)))
            success = max(success, info["success"])

            # The mallet is held by friction, so how far it has shifted in the jaws and how
            # far it has tipped are measured, never assumed.
            slip, tilt, jaw = env.base_env.grip_state(1)
            tilt_deg.append(tilt)
            slip_mm.append(slip * 1000.0)
            jaw_mm.append(jaw * 1000.0)
            slip_ep_max = max(slip_ep_max, slip)

            ee = env.base_env.get_ee()[0]
            ee_extreme = max(ee_extreme, float(np.linalg.norm(ee[:2] - ee_start[:2])))
            v_max = max(v_max, float(np.linalg.norm(env.base_env.get_puck(obs)[1][:2])))

            if absorbing:
                break

        successes += success
        steps.append(t + 1)
        ee_travel.append(ee_extreme)
        puck_speed_max.append(v_max)
        drops += int(slip_ep_max > DROP_SLIP_M)

    return {
        "task": task,
        "episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "drops": drops,
        "mean_steps": float(np.mean(steps)),
        "worst_constraint": worst,
        "ee_travel_mean_m": float(np.mean(ee_travel)),
        "mallet_tilt_deg_p99": float(np.percentile(tilt_deg, 99)),
        "mallet_tilt_deg_max": float(np.max(tilt_deg)),
        "slip_mm_max": float(np.max(slip_mm)),
        "slip_mm_p99": float(np.percentile(slip_mm, 99)),
        "jaw_mm_min": float(np.min(jaw_mm)),
        "puck_speed_max_mean": float(np.mean(puck_speed_max)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--tasks", type=str, default=",".join(TASKS),
                    help="comma-separated task names, e.g. mini-defend,mini-defend-sdk")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        r = run(task, args.episodes, seed=args.seed)
        print(f"\n=== {r['task']}  ({r['episodes']} episodes) ===")
        print(f"  success rate      {r['success_rate']:.2f}")
        print(f"  mallet drops      {r['drops']}/{r['episodes']}  "
              f"(episodes where slip > {DROP_SLIP_M * 1000:.0f} mm)")
        print(f"  mean ep length    {r['mean_steps']:.0f} steps")
        print(f"  mallet moved      {r['ee_travel_mean_m']:.3f} m (mean max displacement)")
        print(f"  puck peak speed   {r['puck_speed_max_mean']:.2f} m/s")
        print(f"  mallet tilt       p99 {r['mallet_tilt_deg_p99']:.2f} deg, "
              f"max {r['mallet_tilt_deg_max']:.2f} deg")
        print(f"  grip slip         p99 {r['slip_mm_p99']:.2f} mm, max {r['slip_mm_max']:.2f} mm")
        print(f"  jaw opening       min {r['jaw_mm_min']:.2f} mm (handle half-width 5.00)")
        print("  worst constraint (>0 == violation):")
        for k, v in r["worst_constraint"].items():
            flag = "  <-- VIOLATED" if v > 0 else ""
            print(f"      {k:20s} {v:+.4f}{flag}")


if __name__ == "__main__":
    main()
