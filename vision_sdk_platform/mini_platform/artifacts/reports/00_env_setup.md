# P1 — Python environment

**Status: PASS.**

## What was built

No conda env on the machine had `mushroom_rl`, `osqp` or `nlopt`, so `import
air_hockey_challenge` failed outright (implementation guide §1.1). Created a dedicated env
rather than mutating `base`:

```bash
conda create -n ahc python=3.11 -y
conda activate ahc
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    "numpy<2" "mujoco==3.3.1" "mushroom-rl==1.10.1" "osqp>=0.6.2.post8" \
    "nlopt>=2.7.1" "PyYAML>=6.0"
pip install dm_control
```

| package | version | note |
|---|---|---|
| python | 3.11 | |
| mujoco | **3.3.1** | the `requirements.txt` pin, not the 3.8.1 in `base` |
| numpy | **1.26.4** | pinned `<2`; `mushroom-rl` 1.10.1 predates NumPy 2 |
| mushroom-rl | 1.10.1 | |
| osqp / nlopt / scipy | 1.0.4 / 2.9.1 / 1.17.1 | |
| dm_control | latest | `mushroom_rl.environments.mujoco` imports `dm_control.mjcf` at module load — not obvious from `requirements.txt`, and the first thing that breaks after a minimal install |

`torch`, `jax`, `dm_control`'s full stack, `sbx` and the Huawei OBS SDK from
`requirements.txt` were **not** installed: nothing in Phases 0–4 needs them, and the git
dependencies are slow and failure-prone. Add `torch` when Phase 6 (EBM) starts.

Resolves open decision 5: **mujoco 3.3.1 + numpy<2**, in a dedicated env. The 3.8.1 in `base`
is left alone.

## Smoke test (guide §1.3) — initially FAILED, now passes

The documented smoke test failed on the first `env.step`:

```
AssertionError: Unexpected action shape. Expected (2, 7) but got (2, 9)
```

**Pre-existing, in the working tree, not introduced here.** `environments/iiwas/env_base.py:157`
was changed from `iiwa_only.xml` to `metal_only.xml`, and `metal_only.xml` carried the two
passive `striker_joint_*` joints, giving `nq = 9` where the robot-only model contract requires
`nq == nv == nu == n_joints == 7`. Two places depend on that contract:

- `position_control_wrapper.PositionControl._controller` assigns a 7-vector to
  `robot_data.qpos` and calls `mj_mulM` with a 7-vector,
- `utils.kinematics.inverse_kinematics` returns the whole `qpos`, which becomes the action.

**Fixed** by making `metal_only.xml` match `iiwa_only.xml`'s contract: `striker_joint_link` is
now a leaf body with no joints, and the two striker motors are gone. The universal joint and
the mallet geometry stay in the *scene* models (`metal1.xml` / `metal2.xml`), which is where
they belong. All four full-size envs construct, reset and step again:

| env | obs | worst constraint after 20 steps |
|---|---|---|
| hit | (23,) | −0.0154 |
| defend | (20,) | −0.0154 |
| prepare | (20,) | −0.0154 |
| tournament | (46,) | −0.0154 |

This is why the mini platform's robot-only model is a separate `metal_arm_only.xml` with
exactly 6 joints rather than a reuse of `arm1.xml` (which has 8).

## For the next session

- `conda activate ahc` before anything.
- Rendering: `MUJOCO_GL=glfw` works. `osmesa` and `egl` both fail on this machine.
- `mushroom_rl` does not enforce wall-clock, so nothing here is a real-time measurement.
