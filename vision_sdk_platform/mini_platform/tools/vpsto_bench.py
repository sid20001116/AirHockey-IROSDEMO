"""Headless Stage-3 validation bench for the (Cartesian) VPSTO strike planner.

Drives the real agent through the sim with --planner bezier vs vpsto and reports the metrics the
VPSTO planner is meant to improve, with NO viewer (so it is fast and scriptable):
  * success rate, table tilt, mallet slip;
  * ee-z stays on the play plane (the joint-space formulation's failure mode -- the Cartesian one
    should keep |z - plane| within the +-20 mm ee_constr);
  * ee_constr violations;
  * implied per-tick joint velocity of the streamed command (the "147 rad/s -> throttle -> lag"
    metric): VPSTO + QP should stay within the joint limit.

Run in metal_hw:
    python -m mini_platform.tools.vpsto_bench [--episodes 6] [--task mini-hit] [--seed 0]
"""
import argparse
import time

import numpy as np

import mini_platform
from mini_platform.agents.baseline_mini import MiniBaselineAgent
from mini_platform.core.kinematics import forward_kinematics


def run(task, planner, episodes, seed, horizon=500):
    np.random.seed(seed)
    env = mini_platform.make(task)
    be = env.base_env
    agent = MiniBaselineAgent(env.env_info, agent_id=1, planner=planner)
    opt = agent.traj_generator.optimizer
    eeh = env.env_info['robot']['ee_desired_height']

    succ = 0
    zdev = []           # |ee_z - plane| of each commanded config (mm)
    jvel = []           # implied per-tick joint velocity of the command (rad/s)
    viol = 0
    tilt99 = []
    slipmax = 0.0
    t_solve = []
    q_prev = None

    for ep in range(episodes):
        obs = env.reset()
        agent.reset()
        for t in range(horizon):
            t0 = time.perf_counter()
            a = agent.draw_action(obs)
            t_solve.append((time.perf_counter() - t0) * 1e3)
            qc = np.asarray(a[0], float)
            z = forward_kinematics(opt.robot_model, opt.robot_data, qc)[0][2]
            zdev.append(abs(z - eeh) * 1000.0)
            if q_prev is not None:
                jvel.append(np.max(np.abs(qc - q_prev)) / env.env_info['dt'])
            q_prev = qc
            obs, r, done, info = env.step(a)
            for v in info['constraints_value'].values():
                if np.any(np.asarray(v) > 0):
                    viol += 1
            s, ti, _ = be.grip_state(1)
            slipmax = max(slipmax, s)
            if 'success' in info:
                succ = max(succ, 0)
            if done:
                succ += int(info.get('success', 0) > 0)
                break
        tilt99.append(ti)

    def p(a, q):
        return float(np.percentile(a, q)) if len(a) else float('nan')
    return dict(planner=planner, succ=succ, episodes=episodes,
                zdev_p99=p(zdev, 99), zdev_max=max(zdev) if zdev else float('nan'),
                jvel_p99=p(jvel, 99), jvel_max=max(jvel) if jvel else float('nan'),
                viol=viol, tilt99=p(tilt99, 99), slip_mm=slipmax * 1000.0,
                draw_p99=p(t_solve, 99), draw_max=max(t_solve) if t_solve else float('nan'))


def _row(m):
    return (f"  {m['planner']:>7}  success {m['succ']}/{m['episodes']}  "
            f"ee-z|dev| p99={m['zdev_p99']:.1f} max={m['zdev_max']:.1f} mm (tol 20)  "
            f"ee_constr viol={m['viol']}\n"
            f"           implied joint vel p99={m['jvel_p99']:.2f} max={m['jvel_max']:.2f} rad/s  "
            f"tilt p99={m['tilt99']:.0f}d  slip={m['slip_mm']:.2f}mm  "
            f"draw p99={m['draw_p99']:.1f} max={m['draw_max']:.0f} ms")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="mini-hit")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print(f"== VPSTO bench: {args.task}, {args.episodes} episodes, seed {args.seed} ==")
    for planner in ("bezier", "vpsto"):
        m = run(args.task, planner, args.episodes, args.seed)
        print(_row(m))
    print("  (VPSTO target: ee-z within 20 mm, 0 ee_constr violations, implied joint vel within "
          "the limit, success >= bezier.)")


if __name__ == "__main__":
    main()
