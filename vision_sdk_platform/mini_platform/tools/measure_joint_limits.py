"""T0.3 -- measure the metal arm's joint velocity and acceleration limits.

`mini_platform/assets/arm1.xml` declares `actuatorfrcrange`, `damping` and `armature` but **no**
`q_dot_max`. Section 3.3 of `optimal_control.md` (the max-mallet-speed LP) is *defined* by
`q_dot_max`, and `env_info['robot']['joint_vel_limit']` must not be copied from the KUKA.

There is no servo datasheet for this arm, so both limits are **measured in MuJoCo**:

  q_dot_max[i]  saturate joint i with its actuator force from the centre of its own range
                (the rest of the arm PD-held at a sampled posture) and integrate until the
                joint reaches its stop or 3 s elapses; the peak velocity is the limit.
                Report the worst case over sampled arm configurations.
  q_ddot_max[i] acceleration produced by the same force from rest, i.e. the acceleration at
                t=0; again the worst (smallest) case over configurations.

Joint i starts at the *centre* of its range rather than at the sampled value: starting next
to a stop and pushing into it measures the stop, not the servo. Note that with damping=2 the
travel available (1.5-2.8 rad from centre) is shorter than the distance needed to reach the
true asymptotic velocity, so these are range-limited peaks -- which is the operative limit
for a joint that has to reverse inside its own travel anyway.

Both therefore inherit `damping="2"` / `armature="0.1"` from the MJCF, which are CAD
placeholders (see optimal_control_implementation.md 0.2). Re-run this after H6 sim-to-real
system ID.

Offline tool: never import from the runtime path.

    python -m mini_platform.tools.measure_joint_limits
"""

import json
import os

import mujoco
import numpy as np

from mini_platform import ARTIFACT_DIR, ASSET_DIR

ARM_XML = os.path.join(ASSET_DIR, "metal_arm_only.xml")
OUT = os.path.join(ARTIFACT_DIR, "sysid", "metal_arm_limits.json")

N_JOINTS = 6
SETTLE_S = 3.0
N_CONFIGS = 64
SEED = 0


def _sample_configs(model, rng, n):
    """Random in-range configurations plus the all-zeros pose."""
    lo, hi = model.jnt_range[:N_JOINTS, 0], model.jnt_range[:N_JOINTS, 1]
    qs = rng.uniform(lo, hi, size=(n, N_JOINTS))
    return np.vstack([np.zeros((1, N_JOINTS)), qs])


def _hold_others(model, data, q_hold, j, tau_max):
    """Gravity-compensating PD on every joint except j, so the arm keeps its posture."""
    kp, kd = 200.0, 20.0
    tau = data.qfrc_bias[:N_JOINTS] - kp * (data.qpos[:N_JOINTS] - q_hold) - kd * data.qvel[:N_JOINTS]
    tau = np.clip(tau, -tau_max, tau_max)
    data.ctrl[:N_JOINTS] = tau
    return tau


def measure(model, data, configs):
    tau_max = np.abs(model.actuator_ctrlrange[:N_JOINTS]).max(axis=1)
    centre = model.jnt_range[:N_JOINTS].mean(axis=1)
    n_steps = int(SETTLE_S / model.opt.timestep)

    vel = np.full((len(configs), N_JOINTS), np.inf)
    acc = np.full((len(configs), N_JOINTS), np.inf)

    for c, q_cfg in enumerate(configs):
        for j in range(N_JOINTS):
            q0 = q_cfg.copy()
            q0[j] = centre[j]
            lo, hi = model.jnt_range[j]

            for sign in (1.0, -1.0):
                mujoco.mj_resetData(model, data)
                data.qpos[:N_JOINTS] = q0
                mujoco.mj_forward(model, data)
                _hold_others(model, data, q0, j, tau_max)
                data.ctrl[j] = sign * tau_max[j]
                mujoco.mj_forward(model, data)

                # |q_ddot| from rest is the acceleration limit at this configuration.
                acc[c, j] = min(acc[c, j], abs(data.qacc[j]))

                peak = 0.0
                for _ in range(n_steps):
                    _hold_others(model, data, q0, j, tau_max)
                    data.ctrl[j] = sign * tau_max[j]
                    mujoco.mj_step(model, data)
                    peak = max(peak, abs(data.qvel[j]))
                    if data.qpos[j] <= lo + 1e-3 or data.qpos[j] >= hi - 1e-3:
                        break
                vel[c, j] = min(vel[c, j], peak)

    return vel, acc


def main():
    model = mujoco.MjModel.from_xml_path(os.path.normpath(ARM_XML))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(SEED)

    configs = _sample_configs(model, rng, N_CONFIGS)
    vel, acc = measure(model, data, configs)

    # Worst case over configurations -- effective inertia varies a lot on a 6-DOF arm and a
    # limit that only holds at the home pose is not a limit.
    q_dot_max = vel.min(axis=0)
    q_ddot_max = acc.min(axis=0)

    result = {
        "provenance": "measured in MuJoCo (no servo datasheet available)",
        "model": os.path.relpath(os.path.normpath(ARM_XML),
                                 os.path.normpath(os.path.join(os.path.dirname(ARM_XML), "..", "..", "..", "..", ".."))),
        "method": ("per joint: saturate actuator force from rest at each of "
                   f"{len(configs)} configurations (1 home + {N_CONFIGS} uniform-random, seed "
                   f"{SEED}), both signs; q_dot_max = worst-case peak velocity over "
                   f"{SETTLE_S}s, q_ddot_max = worst-case |qacc| at t=0"),
        "caveat": ("inherits damping=2 / armature=0.1 from the MJCF, which are CAD "
                   "placeholders, not measured -- re-run after sim-to-real system ID (H6)"),
        "n_joints": N_JOINTS,
        "joint_vel_limit_rad_s": q_dot_max.tolist(),
        "joint_acc_limit_rad_s2": q_ddot_max.tolist(),
        "actuator_force_limit_Nm": np.abs(model.actuator_ctrlrange[:N_JOINTS]).max(axis=1).tolist(),
        "per_config_vel_median": np.median(vel, axis=0).tolist(),
        "per_config_acc_median": np.median(acc, axis=0).tolist(),
    }

    os.makedirs(os.path.dirname(os.path.normpath(OUT)), exist_ok=True)
    with open(os.path.normpath(OUT), "w") as f:
        json.dump(result, f, indent=2)

    np.set_printoptions(precision=3, suppress=True)
    print("q_dot_max  (rad/s)   worst:", q_dot_max, " median:", np.median(vel, axis=0))
    print("q_ddot_max (rad/s^2) worst:", q_ddot_max, " median:", np.median(acc, axis=0))
    print("written to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
