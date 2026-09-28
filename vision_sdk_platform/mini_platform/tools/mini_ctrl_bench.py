"""Closed-loop identity bench: what does each joint's servo actually do?

The end-to-end gate (`mini_env_smoke`) measures task outcomes, which are noisy and confound
the controller with the planner. This measures the controller alone, per joint, and pins it to
numbers you can diff:

  * gravity hold  -- command a constant pose for 2 s and see whether the arm stays there.
                     Fails immediately if the feedforward's sign is wrong or the MJCF's joint
                     damping was edited.
  * step response -- fit natural frequency and damping ratio from a 0.05 rad step.
  * saturation    -- what fraction of substeps sit pinned at the torque ceiling.

Sampling is per SUBSTEP (1 kHz) via `controller_record`, not per env step (50 Hz), because a
9.4 Hz closed loop has only ~5 samples per period at 50 Hz.

Offline tool: never import from the runtime path.

    python -m mini_platform.tools.mini_ctrl_bench
    python -m mini_platform.tools.mini_ctrl_bench --tasks mini-defend,mini-defend-sdk
"""

import argparse

import numpy as np

from mini_platform.core.challenge_wrapper import AirHockeyChallengeWrapper

#: `controller_record` block layout differs between the two controllers. Both begin with
#: [q_ref, q, qd_ref, qd, qdd_ref]; the SDK one appends its command and torque blocks.
LEGACY_BLOCKS = 5
SDK_BLOCKS = 10


def _record(env, n_joints):
    """controller_record -> dict of (T, n_joints) arrays, one per logged block."""
    rec = np.asarray(env.controller_record)
    names = ["q_ref", "q", "qd_ref", "qd", "qdd_ref",
             "cmd_q", "cmd_qd", "cmd_tau", "tau", "tau_out"]
    n_blocks = rec.shape[1] // n_joints
    out = {names[i]: rec[:, i * n_joints:(i + 1) * n_joints]
           for i in range(min(n_blocks, len(names)))}
    if rec.shape[1] > SDK_BLOCKS * n_joints:            # trailing [substep, latched] scalars
        out["substep"] = rec[:, -2]
        out["latched"] = rec[:, -1]
    return out


def _drive(task, q_target, n_steps, seed=0):
    """Hold a constant joint-position command and return the per-substep record."""
    np.random.seed(seed)
    env = AirHockeyChallengeWrapper(env=task, interpolation_order=3, debug=True)
    base = env.base_env
    n = base.n_robot_joints

    env.reset()
    base.controller_record = []
    action = np.vstack([q_target, np.zeros(n)])
    for _ in range(n_steps):
        env.step(action)

    r = _record(base, base._num_env_joints)
    # For the 2-agent hit env only the first agent's block is ours.
    return {k: (v[:, :n] if v.ndim == 2 else v) for k, v in r.items()}, base


def _fit_second_order(y, delta, dt):
    """Fit (f_n [Hz], zeta) from a unit step response.

    Underdamped: read the damping ratio off the overshoot and the natural frequency off the
    peak time -- both closed-form, no optimiser, no initial guess.
        OS   = exp(-pi*z / sqrt(1-z^2))      ->  z   = -ln(OS) / sqrt(pi^2 + ln^2(OS))
        t_p  = pi / (w_n * sqrt(1-z^2))      ->  w_n = pi / (t_p * sqrt(1-z^2))
    Overdamped responses have no peak; report zeta >= 1 and estimate w_n from the 10-90 % rise
    time using the critically-damped relation t_r ~ 3.36 / w_n.
    """
    u = y / delta
    k = int(np.argmax(u))
    peak = u[k]

    if peak > 1.0 + 1e-4 and k > 0:
        os_ = peak - 1.0
        ln = np.log(os_)
        zeta = -ln / np.sqrt(np.pi ** 2 + ln ** 2)
        t_p = k * dt
        wn = np.pi / (t_p * np.sqrt(max(1.0 - zeta ** 2, 1e-9)))
        return wn / (2 * np.pi), zeta, os_

    try:
        i10 = int(np.argmax(u >= 0.1))
        i90 = int(np.argmax(u >= 0.9))
        t_r = max((i90 - i10) * dt, dt)
        return (3.36 / t_r) / (2 * np.pi), 1.0, 0.0
    except ValueError:
        return float("nan"), float("nan"), 0.0


def gravity_hold(task, seconds=2.0, settle=0.2):
    n_steps = int(seconds / 0.02)
    env = AirHockeyChallengeWrapper(env=task, interpolation_order=3, debug=True)
    base = env.base_env
    q0 = base.init_state.copy()
    rec, base = _drive(task, q0, n_steps)

    dt = base._timestep
    i0 = int(settle / dt)
    q, qd = rec["q"][i0:], rec["qd"][i0:]

    slip, tilt, jaw = base.grip_state(1)
    ee = base.get_ee()[0]
    return {
        "droop_rad": float(np.abs(q - q0).max()),
        "droop_per_joint": np.abs(q - q0).max(axis=0),
        "qd_max": float(np.abs(qd).max()),
        "ee_z_err_m": float(abs(ee[2] - base.env_info["robot"]["ee_desired_height"])),
        "slip_mm": slip * 1000.0,
        "tilt_deg": tilt,
        "jaw_mm": jaw * 1000.0,
    }


def step_response(task, delta=0.05, seconds=1.0):
    n_steps = int(seconds / 0.02)
    env = AirHockeyChallengeWrapper(env=task, interpolation_order=3)
    base = env.base_env
    n = base.n_robot_joints
    q0 = base.init_state.copy()
    lo, hi = base.env_info["robot"]["joint_pos_limit"]

    rows = []
    for j in range(n):
        # Step away from whichever limit is closer, so the command never clips.
        d = delta if (q0[j] + delta) < hi[j] - 1e-3 else -delta
        target = q0.copy()
        target[j] += d

        rec, b = _drive(task, target, n_steps)
        dt = b._timestep
        f_n, zeta, os_ = _fit_second_order(rec["q"][:, j] - q0[j], d, dt)

        tau = rec.get("tau_out", rec.get("q"))
        ceil = getattr(b, "_tmax_out", None)
        if ceil is None:                                  # legacy controller
            ceil = np.abs(b.robot_model.actuator_ctrlrange[:, 1])
        sat = float(np.mean(np.abs(tau[:, j]) >= ceil[j] * (1 - 1e-6))) if tau.shape[1] > j else float("nan")

        err = np.abs(rec["q_ref"][:, j] - rec["q"][:, j])
        rows.append({"joint": j + 1, "f_n_hz": f_n, "zeta": zeta, "overshoot": os_,
                     "sat_frac": sat, "err_p99_rad": float(np.percentile(err, 99))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, default="mini-defend,mini-defend-sdk")
    ap.add_argument("--delta", type=float, default=0.05)
    args = ap.parse_args()

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        print(f"\n=== {task} ===")

        g = gravity_hold(task)
        flag = "  <-- FAIL" if g["droop_rad"] >= 2e-3 else ""
        print("  gravity hold (2 s, constant command):")
        print(f"      max |q - q0|    {g['droop_rad']:.5f} rad   (want < 0.002){flag}")
        print(f"      per joint       " + " ".join(f"{v:.5f}" for v in g["droop_per_joint"]))
        print(f"      max |qd|        {g['qd_max']:.5f} rad/s")
        print(f"      ee z error      {g['ee_z_err_m'] * 1000:.2f} mm")
        print(f"      grip            slip {g['slip_mm']:.2f} mm, tilt {g['tilt_deg']:.2f} deg, "
              f"jaw {g['jaw_mm']:.2f} mm")

        print(f"  step response ({args.delta} rad):")
        print("      J     f_n [Hz]   zeta   overshoot   sat%   err_p99 [rad]")
        for r in step_response(task, delta=args.delta):
            print(f"      J{r['joint']}    {r['f_n_hz']:7.2f}  {r['zeta']:6.3f}   "
                  f"{r['overshoot'] * 100:7.1f}%  {r['sat_frac'] * 100:5.2f}   {r['err_p99_rad']:.5f}")


if __name__ == "__main__":
    main()
