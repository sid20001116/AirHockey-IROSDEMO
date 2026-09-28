# Session handoff — `mini_platform`

Two sessions of work, newest first. **Part A** is the METAL SDK controller port (this session).
**Part B** is Phase 0, which built the platform itself and is unchanged except where Part A
supersedes it — those places are marked.

Companion documents:

- [`../README.md`](../README.md) — layout, run commands, per-file description of every folder
- [`../artifacts/reports/04_sdk_controller_port.md`](../artifacts/reports/04_sdk_controller_port.md)
  — the controller port in full: measurements, deviations, open risks
- [`../artifacts/reports/02_joint_limits.md`](../artifacts/reports/02_joint_limits.md) — joint
  limits, original and re-measured
- [`optimal_control_plan.md`](optimal_control_plan.md) /
  [`optimal_control_implementation.md`](optimal_control_implementation.md) — the task board

---

# Part A — the METAL SDK's low-level controller (2026-08-17/18)

## A0. Read this before touching anything

```bash
conda activate ahc
cd mini_platform && pip install -e .

pytest tests/ -q                                                    # 17 tests, ~10 s
python -m mini_platform.tools.mini_ctrl_bench --tasks mini-defend,mini-defend-sdk
python -m mini_platform.tools.mini_env_smoke --episodes 60 --tasks mini-defend,mini-defend-sdk
python -m mini_platform.tools.mini_play --task mini-hit-sdk-2p      # needs a display
```

**Three traps that cost time this session:**

1. **`mini_platform/` is still not under version control.** It is untracked in the parent repo
   and has no `.git` of its own. There is no `git checkout` rollback for anything in here. This
   was item 1 of Part B's next-steps and it is still not done — it is now more urgent, because
   this session edited the MJCF. **Do `git init` or `git add mini_platform/` before your first
   edit.**
2. **`conda run -n ahc python - <<'EOF'` silently does nothing.** `conda run` does not forward
   stdin, so a heredoc piped into `python -` executes zero lines and exits 0. Several patches
   this session appeared to succeed and had not run. Write the script to a file first, then
   `conda run -n ahc python thatfile.py`. `python -c "..."` is fine.
3. **The gate is not bit-reproducible** (unchanged from Part B) — `planning/optimizer.py:155`
   terminates on wall clock. Note Part B blamed `optimizer.py:96`/`set_maxtime`; that call site
   (`solve_hit_config`) is **dead code** with zero callers. The live path is
   `solve_hit_config_ik_null` at `:155`. Pinning it to an iteration budget is a one-line change,
   worth doing only if you intend to judge hit success.

## A1. What was asked, and what was built

> "i want their low level contoller in terms of gains, gravcomp since its super stable and i
> dont need to tune. However, i do want to keep the planner logic here"

Then, later: raise the MJCF torque ceilings to the SDK's, and make both robots active in
`mini-hit`.

Kept untouched: the FSM, the Bezier planner, the OSQP null-space IK, the `(2, 6)` `[pos; vel]`
action at 50 Hz, `PositionControl`'s cubic interpolator, the observation pipeline, the 3 N grip.
`core/position_control.py` is byte-identical — the legacy controller still runs.

## A2. Which SDK, and why it is not the obvious one

Two SDKs are vendored under `mini_platform/`. **They are different arms, not copies.**

| | `metal-python-ros/metal_sdk` (C++) | `maker-arm-sdk` (Python) |
|---|---|---|
| Motors | DM / Damiao | RobStride RS00/RS02 |
| Rate | 400 Hz | 200 Hz |
| Setpoint | quintic trajectory | slew-rate ramp, 9 rad/s |
| `kp` | 300, 500, 500, 400, 25, 20 | 60, 150, 90, 30, 30, 30 |
| Gravity/Coriolis/friction FF | **yes** | **none — `tau_ff` hard 0** |
| Gripper | linear, 0–102.5 mm stroke table | revolute motor 7 |

`metal_sdk` is the right one on four independent grounds:

1. Gravity compensation was the requirement, and `maker-arm-sdk` has none: every command is
   `send_mit(pos, 0.0, kp, kd, 0.0)`. Its stability comes from a slow slew limiter.
2. The MJCF's joint ranges match `config/motor_config.cpp` **exactly** on all six joints.
3. The MJCF's two ±0.05 m prismatic fingers match the metal gripper's 102.5 mm **linear** stroke
   table; the maker gripper is a single revolute motor.
4. The metal SDK has a gripper mode, and the physical arm has a gripper.

The C++ SDK **cannot be executed here** — no `.so` is checked in, building needs ROS 2 Humble +
orocos-KDL + trac_ik, and `Init()` opens a real CAN socket and blocks on motor replies. It was
ported as a specification. All constants live in one data module so switching to the maker
values is a one-line change.

## A3. The law

Per joint, every 2.5 ms (400 Hz, `can_manager.cpp:996-998`) the host packs
`(q_des, qd_des, kp, kd, tau_ff)` into one CAN frame; the **motor firmware** closes the loop at
its own kHz rate:

```
tau    = kp*(q_des - q) + kd*(qd_des - qd) + tau_ff      clipped to ±tmax
tau_ff = Kg·G(q) + Kc·C(q,q̇) + b·clamp(q̇)               with |q̇| < 0.05 → 0
```

`kp = [300, 500, 500, 400, 24.908, 19.902]` (J5/J6 are the 12-bit-truncated values the motor
actually receives), `kd = [5, 5, 5, 4, 0.199, 0.199]`, `Kg = [1.2, 1.15, 1.1, 1.15, 1, 1]`,
`Kc = [1.1, 1.15, 1, 1, 1.1, 1.1]`, `b = [0.15, 0.3025, 0.52, 0.42, 0.015, 0.015]` with `q̇`
clamped ±0.8 on J3 and ±1.2 on J4. Every one carries a `file:line` citation in `sdk_params.py`.

**Implementation points that are easy to get wrong:**

- **The 400 Hz hold.** 2.5 ms does not divide the 1 ms substep, so a phase accumulator latches
  at substeps `[2,4,7,9,12,14,17,19]` — 8 latches per 20 ms step, gaps `2,3,2,3,2,3,2,3` ms,
  and the last latch lands exactly on the planner's segment endpoint. `_hold_mask` asserts that
  `hold_hz·dt` is an integer so a future `dt` change cannot silently alias.
- **`next(self.traj)` is called unconditionally**, on every substep, latched or not. Gating it
  on the hold mask desynchronises the generator's 20-yield contract with mushroom's 20 calls,
  truncating every segment to its first 40 % — silently, with plausible numbers.
- **`G` and `C` are split**, because `Kg ≠ Kc` and MuJoCo only exposes the combined bias:
  `G := qfrc_bias at qvel=0`, `C := qfrc_bias(q,q̇) − G`. The pipeline
  `mj_kinematics → mj_comPos → mj_comVel → mj_rne` is bit-identical to `mj_forward` and 3.6×
  faster. **`mj_rne` reads `d.cvel`, not `d.qvel`, so `mj_comVel` must be re-run after writing
  `qvel`** or it silently returns `G` twice.
- **Raw `qvel`, not `get_joints(obs)`.** The observation pipeline runs a 0.274 IIR at the 1 ms
  rate — a 51 Hz lag. Motor firmware sees unfiltered encoder velocity.
- **`_simulation_pre_step` is deliberately NOT overridden.** `AirHockeyBase`'s writes the 3 N
  grip every substep and does not call `super()`. Shadowing it drops the mallet with no error.
- **`+b·q̇` cancels drag, it is not damping.** Flipping that sign doubles drag instead. Guarded
  by test T3.

## A4. Three named deviations from the SDK

All in `sdk_params.py`, all opt-out.

1. **`inertia_ff=True`** — the SDK defines `M(q)·q̈` (`kdl_solver.cpp:329`) and never calls it.
   Here it is load-bearing, because the sim's inertia is 84–99.5 % armature on J4–J6. Measured
   on defend, n=60: **with → 3/60 drops, tilt p99 29°, `ee_constr` −0.0019; without → 7/60
   drops, tilt p99 92°, `ee_constr` +0.0193.**
2. `qd_cmd` is clamped before quantisation; the SDK wraps an out-of-range value through
   `float_to_uint`. Reproducing an integer wrap would inject garbage, not fidelity.
3. Output torque is clipped explicitly to make saturation observable rather than silent.

**Not ported:** the SDK's own 400 Hz quintic re-planner and its ~27 ms transport lag. You keep
your planner. `TRAJ_LIMITS`/`POINT_LIMITS` are recorded in `sdk_params.py` so the omission is
explicit and reversible.

## A5. Files

**Created**

| path | role |
|---|---|
| `core/sdk_params.py` | Every SDK constant with a `file:line` citation, `SdkConfig`, `nrt_config()`/`follow_config()`. numpy only. |
| `core/sdk_control.py` | `SdkJointControl` — the 400 Hz host tick + per-substep firmware law. 188 lines of code; the rest is provenance comment. |
| `tools/mini_ctrl_bench.py` | Controller identity independent of the planner: gravity hold, step response fitted to `f_n`/ζ, saturation duty. Samples at 1 kHz via `controller_record`. |
| `tests/test_sdk_control.py` | T1–T8: latch pattern, `G`/`C`/`M·q̈` bit-identity, the law on a `q̇` straddling every branch, quantisation, per-agent gain tiling, the 20/8 pairing, grip survival, position clamp. |
| `tests/test_two_player.py` | The two-player wiring, incl. behaviourally that the opponent leaves READY. |
| `artifacts/reports/04_sdk_controller_port.md` | The full record. |

**Modified:** `envs/position_control.py` (SDK env classes + opponent wiring), `envs/__init__.py`
and `__init__.py` (task registration), `tools/mini_env_smoke.py` (`--tasks`, `drops` counter),
`tools/mini_play.py` (task list), `assets/arm{1,2}.xml` + `assets/metal_arm_only.xml` (torque
ceilings), `envs/env_base.py` (joint limits), `envs/env_hitting.py` (comment only), README and
reports.

**Eight tasks now exist.** `mini-{hit,defend,prepare}` (legacy PD),
`mini-{hit,defend,prepare}-sdk` (SDK law), `mini-hit-2p` / `mini-hit-sdk-2p` (both arms active).
Everything above the controller is identical between a task and its `-sdk` sibling, so the
comparison is a task-name change.

## A6. Stage 2 — torque ceilings raised to the SDK's `tmax`

The MJCF disagreed with the SDK on four joints. Now `[30, 120, 30, 30, 20, 20]` N·m on both
actuator `ctrlrange` and joint `actuatorfrcrange` in all three assets. `joint7` (gripper) stays
±3 — it is a linear finger force in N, not one of the SDK's joint torques.

**This forced a re-measurement**, because the old limits were measured under the old saturated
force. `JOINT_VEL_LIMIT` is now the **elementwise min of two independent bounds**:

| | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---|---|---|---|---|
| measured (this plant) | 10.345 | 18.876 | 10.100 | 13.657 | 9.640 | 9.960 |
| SDK `vmax` (the motor) | 10 | 10 | 10 | 10 | 30 | 30 |
| **shipped** | 10.0 | 10.0 | 10.0 | 10.0 | 9.640 | 9.960 |
| binding | SDK | SDK | SDK | SDK | plant | plant |

J1/J3 re-measured bit-identically — their torque did not change, so their limits must not
either. That is the check that the harness is sound. `JOINT_ACC_LIMIT` stays purely measured:
the SDK bounds acceleration only inside its unported time allocator.

**Re-run `tools/measure_joint_limits.py` after ANY change to actuator force, damping or
armature**, and re-baseline both controllers.

## A7. Measured results — final state

`tools/mini_env_smoke.py`, seed 0. Read `drops` (episodes where the mallet left the jaws), not
`slip_mm_p99` — the percentile is dominated by *when* a drop happened.

| task | controller | success | drops | tilt p99 | `ee_constr` | `joint_vel_constr` |
|---|---|---|---|---|---|---|
| defend (n=60) | legacy | 1.00 | 4/60 | 69.7° | −0.0104 ✓ | −3.019 ✓ |
| defend (n=60) | sdk | 1.00 | 4/60 | 78.5° | −0.0144 ✓ | −2.320 ✓ |
| prepare (n=60) | legacy | 0.38 | 0/60 | 33.5° | −0.0121 ✓ | −5.607 ✓ |
| prepare (n=60) | sdk | 0.32 | 2/60 | 34.3° | −0.0104 ✓ | −4.034 ✓ |
| hit (n=25) | legacy | 0.20 | 2/25 | 90.1° | −0.0158 ✓ | −3.617 ✓ |
| hit (n=25) | sdk | 0.04 | 3/25 | 38.2° | −0.0078 ✓ | −3.680 ✓ |
| hit-2p (n=25) | legacy | 0.00 | 2/25 | 90.1° | — | — |
| hit-2p (n=25) | sdk | 0.00 | 2/25 | 34.8° | — | — |

**Every constraint violation on the rig is gone, for both controllers.** That is the headline,
and it is Stage 2's doing, not Stage 1's.

Closed-loop identity (`mini_ctrl_bench`), which is the regression anchor — re-run it after any
MJCF edit and these should not move:

| | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---|---|---|---|---|
| legacy `f_n` [Hz] / ζ | 3.88 / 1.00 | 7.33 / 1.00 | 3.41 / 1.00 | 3.32 / 1.00 | 3.24 / 1.00 | 2.69 / 1.00 |
| **sdk `f_n` / ζ** | 6.18 / 0.577 | 4.96 / 0.451 | 5.89 / 0.431 | 7.92 / 0.587 | 2.21 / 0.742 | 1.95 / 0.792 |
| sdk saturation | 0.70 % | 0.20 % | 1.00 % | 0.70 % | 0.50 % | 0.50 % |

Gravity hold, 2 s at constant command: legacy **0.00124 rad**, sdk **0.00035 rad** — 3.5×
better, and the single sharpest number in the suite. It fails immediately if a feedforward sign
is wrong or the MJCF's damping was edited.

## A8. Two-player `mini-hit`

`mini-hit` follows the organizers' `IiwaPositionHit`: the opponent traces a lissajous and never
plays the puck. Measured, that makes it an **accidental goalkeeper** — it parks 5.9 cm off its
own goal line and sweeps 1.33× the goal width, on a path deterministic in time (identical to
10 µm across seeds). So shot *timing* decides whether you score, and since the controller sets
the timing, this is a systematic confound, not zero-mean noise. `mini-hit` is the weakest of the
three tasks for judging a controller.

`mini-hit-2p` / `mini-hit-sdk-2p` put a real `MiniBaselineAgent(agent_id=2)` on the other side.
The plumbing already existed: `_preprocess_action` hands the opponent `np.split(obs, 2)[1]`,
which `env_double._modify_observation` has already expressed in robot 2's own base frame.
Two things had to change: the opponent gets **cubic interpolation (order 3)**, not −1 (a real
agent emits a proper `(pos, vel)` trajectory; −1 suits the wander whose velocity is a finite
difference), and **`episode_start()` fires per episode** or the FSM carries state across.

You can also pass your own: `make("mini-hit", opponent_agent=my_agent)`.

It genuinely contests — over 25 episodes:

| | opponent touches puck | puck reaches opponent goal | scored |
|---|---|---|---|
| `mini-hit` | 1/25 | 14/25 | 5/25 |
| `mini-hit-2p` | 6/25 | 3/25 | 0/25 |

The attacker aims about as well either way (7 vs 6 shots in the goal-width corridor); it loses
on the last stretch. **⚠ A rally is impossible by construction**: `AirHockeyHit.is_absorbing`
ends the episode the moment the puck crosses back into your half moving toward you, so play
stops at the opponent's first return. 2P episodes run ≈225 steps vs ≈320. A rally task needs a
different termination rule; not built.

## A9. Corrections to claims made during this session

1. **"The torque ceiling mismatch has to be fixed or the ported gains are meaningless."** Too
   strong. A prototype measured raising them as *increasing* mallet drops (1/60 → 4/60), which
   is why it was staged separately. It was then done on your instruction, and the final result
   is better than that prototype suggested — all constraint violations cleared — but at the cost
   in A10.1.
2. **The design pass predicted the SDK controller would cut mallet drops 3×.** It did not
   reproduce: 3/60 vs 3/60 on defend before Stage 2, 4/60 vs 4/60 after. The `inertia_ff` A/B
   *did* reproduce almost exactly, so the discrepancy is in the comparison against legacy, not
   in the law.
3. **The predicted closed-loop ζ table was 20–28 % off on J2–J4**, because it assumed an ideal
   step with no saturation and no 20 ms command ramp. The measured table in A7 is the reference.
4. **Part B blamed `optimizer.py:96` for non-determinism.** That call site is dead; the live one
   is `:155`.

## A10. Open risks

1. **The SDK controller's wrist-tilt advantage was erased by Stage 2**: defend tilt p99 went
   29° → 78.5°, now slightly worse than legacy's 69.7°. Mechanism: J5/J6 velocity limits roughly
   doubled (4.84 → 9.64, 4.99 → 9.96 rad/s), so the planner can whip the wrist twice as fast,
   and the friction-held mallet's orientation follows the wrist. **The targeted lever, unmeasured:
   cap `JOINT_VEL_LIMIT[4:6]` near 4.9 while keeping the raised torque.**
2. **The SDK controller no longer has a clear task-level edge over the legacy one.** Its real
   remaining advantages are: nothing to tune, 3.5× better gravity hold, and being the law the
   hardware runs. The constraint fix that distinguished it in Stage 1 is now available to both.
3. **`damping=2` / `armature=0.1` are CAD placeholders and the closed loop depends on them** —
   they supply a large share of total damping and most of J4–J6's inertia. "No tuning" is now
   true of the *controller*; it is not true of the *plant*. This is the first thing to fix if you
   ever do sim-to-real system ID.
4. **`mini-hit-2p` scores 0/25 for both controllers**, so it currently has no discriminating
   power for a controller A/B. It is a better *game*; `mini-defend` remains the better instrument.
5. **Joint sign and zero are assumed identity.** Nothing in `metal-python-ros` establishes a sign
   map (`GetJointPosition()` returns the motor angle verbatim), but `test/calib_direction.py`
   exists because J2 was suspect. `sdk_params.JOINT_SIGN`/`JOINT_OFFSET` are there so a
   calibration result is a one-line change.
6. **Sim-vs-hardware direction is known**: the sim arm lags more at speed (plant damping is 4–133×
   the SDK's `b`) and lacks the SDK re-planner's ~27 ms transport lag. These partially cancel —
   do not read sim tracking numbers as a hardware prediction.

## A11. What I would do next, in order

1. **Put `mini_platform/` under version control.** Still not done, now urgent.
2. **Measure the J5/J6 velocity cap** in A10.1 — one constant, re-run the gate, and you learn
   whether the wrist tilt is recoverable without giving back the J4 fix.
3. **Decide whether `mini-hit` should keep the wander opponent at all.** If the answer is "score
   past a live defender", `mini-hit-2p` is the task; if it is "measure striking", add a
   shot-quality metric (would the puck have crossed the goal line unobstructed?) so the
   goalkeeper lottery stops dominating.
4. **The wrist-orientation term in `_solve_aqp`** (Part B, §7, still the deepest issue). Position-
   only IK is why the mallet tilts at all; the controller port did not touch it.
5. Everything in Part B §11 items 2, 4, 5, 6 still stands — puck-drag fit, servo datasheet
   (partially closed: J1–J4 now come from the motor spec), stall strip, and **H2 before the
   camera→EKF swap**.

---

# Part B — Phase 0: the mini platform runs

**Date:** 2026-08-13/14 · **Branch:** `qualifying-2025`

`mini_platform/` is now a **standalone package** — it imports nothing from the parent Robot Air
Hockey Challenge repo. The framework and the organizers' baseline control stack (EKF,
optimiser, tactic FSM) are vendored under `core/`, `estimation/`, `planning/`, `fsm/`, with
provenance and per-file modifications in [`../THIRD_PARTY.md`](../THIRD_PARTY.md). See
[`../README.md`](../README.md) for layout and limitations; this file is the narrative of how it
got there and what is still missing.

One session's work. Going in, `import air_hockey_challenge` failed on every conda env on the
machine and nothing in Python referenced `mini_platform/assets/`. Coming out, `mini-hit` / `mini-defend` /
`mini-prepare` are live environments with a gripped mallet, and a retargeted `BaselineAgent`
plays all three.

Companion documents, all updated in this session:

- [`optimal_control_plan.md`](optimal_control_plan.md) — the *what* and *why*
- [`optimal_control_implementation.md`](optimal_control_implementation.md) — task breakdown,
  APIs, gates, session protocol (**start here on a fresh session**)
- `mini_platform/artifacts/reports/00_env_setup.md`, `02_joint_limits.md`, `03_mallet_and_mini_envs.md` —
  the numbers behind everything below

---

## 1. Start here

```bash
conda activate ahc
cd mini_platform && pip install -e .      # standalone; nothing else on the path is needed

python -m mini_platform.tools.mini_env_smoke --episodes 25   # the acceptance gate
python -m mini_platform.tools.mini_play --task mini-hit      # watch it play (needs a display)
```

Expected from the gate: success ≈ **0.12–0.16 / 1.00 / 0.24–0.28** (hit / defend / prepare),
no divergence, one small `ee_constr` violation on defend.
⚠ **Superseded by Part A, A7.** Stage 2 raised the torque ceilings and re-derived the joint
limits, which moved both controllers: defend's `ee_constr` violation is gone, prepare is
≈0.38, hit ≈0.20. Use A7's table as the reference. §7 explains why those numbers are what
they are.

⚠ **The gate is not bit-reproducible.** *(Part A, A0 trap 3 corrects the file:line below — the
`set_maxtime` call site named here is dead code; the live one is `optimizer.py:155`.)*
It fixes the numpy seed, but the organizers' optimizer
terminates on wall-clock (`baseline/baseline_agent/optimizer.py:96` `set_maxtime(5e-3)`, plus a
`time.time()` budget at line 151), so results move with machine load — two back-to-back
25-episode runs gave hit 0.16 and 0.12. Treat it as a distribution: a few points of drift is
noise, a collapse to 0 or a constraint blow-up is a real regression.

Rendering needs `MUJOCO_GL=glfw`. `osmesa` and `egl` both fail on this machine.

---

## 2. Python environment (was blocking everything)

No conda env had `mushroom_rl`, `osqp` or `nlopt`. Created a dedicated one rather than
mutating `base`:

```bash
conda create -n ahc python=3.11 -y
conda activate ahc
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    "numpy<2" "mujoco==3.3.1" "mushroom-rl==1.10.1" "osqp>=0.6.2.post8" \
    "nlopt>=2.7.1" "PyYAML>=6.0"
pip install dm_control        # mushroom_rl.environments.mujoco imports dm_control.mjcf
```

`mujoco==3.3.1` is the `requirements.txt` pin (not the 3.8.1 in `base`); `numpy<2` because
`mushroom-rl` 1.10.1 predates NumPy 2. `torch`, `jax`, `sbx` and the Huawei OBS SDK were
**not** installed — nothing in Phases 0–4 needs them. Add `torch` when Phase 6 (EBM) starts.

Resolves open decision 5.

---

## 3. Files

### New

| path | what |
|---|---|
| `mini_platform/envs/` | `__init__`, `env_base`, `env_single`, `env_double`, `env_hitting`, `env_defend`, `env_prepare` |
| `mini_platform/assets/mallet1.xml`, `mallet2.xml` | the free-body mallets |
| `mini_platform/assets/metal_arm_only.xml` | robot-only kinematics model, exactly 6 joints |
| `mini_platform/assets/__init__.py` | so the package path resolves |
| `mini_platform/agents/baseline_mini.py` | `BaselineAgent` retargeted to the mini platform |
| `mini_platform/tools/measure_joint_limits.py` | T0.3 |
| `mini_platform/tools/mini_mallet_check.py` | T0.2 acceptance — geometry, grip, rebound, renders |
| `mini_platform/tools/mini_env_smoke.py` | T0.4 acceptance gate |
| `mini_platform/tools/mini_play.py` | live viewer |
| `mini_platform/artifacts/sysid/metal_arm_limits.json` | measured joint limits |
| `mini_platform/artifacts/reports/*.md`, `mini_platform/artifacts/reports/img/` | reports and renders |

### Vendored (new — this is what made the package standalone)

MIT, © 2025 Donat, Liu, Guenster, Tateo. Every file carries a provenance header; local changes
are marked `MINI:` in-line and tabulated in `../THIRD_PARTY.md`.

| here | from | changed |
|---|---|---|
| `core/challenge_wrapper.py` | `framework/air_hockey_challenge_wrapper.py` | task table from `mini_platform.envs`; `check_success` uses table fractions |
| `core/constraints.py` | `constraints/constraints.py` | `LinkConstraint.z_lb` from `env_info` |
| `core/kinematics.py` | `utils/kinematics.py` | `link_to_xml_name` reduced to the metal arm |
| `core/position_control.py` | `environments/position_control_wrapper.py` | mixin only; iiwa classes dropped |
| `core/agent_base.py`, `core/transformations.py` | same | unchanged |
| `estimation/kalman_filter.py` | `baseline/.../kalman_filter.py` | unchanged — **the EKF** |
| `planning/{optimizer,trajectory_generator,cubic_linear_planner}.py` | `baseline/...` | unchanged |
| `planning/bezier_planner.py` | `baseline/.../bezier_planner_new.py` | `matplotlib` import moved into `__main__` — it forced the `tkAgg` backend at module scope and broke headless imports |
| `fsm/{system_state,tactics}.py` | `baseline/...` | unchanged |
| `agents/baseline_agent.py` | `baseline/.../baseline_agent.py` | unchanged |

`utils/universal_joint_plugin.py` is deliberately **not** vendored — a gripped mallet has no
universal joint.

Packaging: `pyproject.toml` (package dir mapped to `.`, assets included as package data),
`requirements.txt`, `THIRD_PARTY.md`, `LICENSE.third_party`, `.gitignore`. Verified by building
a wheel and confirming the MJCF and STL assets ship inside it.

### Changes to the parent repo: none

Earlier revisions of this work patched four files in `air_hockey_challenge/`. **All of those
have been reverted** — vendoring removed the need. All four full-size envs were re-verified
after the revert.

The one exception is `air_hockey_challenge/environments/data/iiwas/metal_only.xml`, which is
**kept**: it carried the striker joints, giving the robot-only model `nq = 9` where the contract
is 7, and that broke *every* full-size env on its first step. That is a full-size bug, unrelated
to the mini rig. `scripts/place_wrist_cam.py` was also repointed at `mini_platform/assets/`.

## 4. The bug that was blocking the full-size envs

`environments/iiwas/env_base.py:157` had been switched from `iiwa_only.xml` to
`metal_only.xml`, and `metal_only.xml` carried the two passive `striker_joint_*` joints. The
robot-only model contract is **`nq == nv == nu == n_joints`**, because:

- `PositionControl._controller` assigns an `n_joints`-vector to `robot_data.qpos` and calls
  `mj_mulM` with one;
- `utils.kinematics.inverse_kinematics` returns the entire `qpos`, which becomes the action.

Every full-size env raised `Expected (2, 7) but got (2, 9)` on its first step. Fixed by making
`striker_joint_link` a leaf body with no joints and dropping the two striker motors — the
universal joint and mallet belong in the *scene* model (`metal1.xml`), not the robot model.
All four full-size envs construct, reset and step again.

This is also why the mini robot-only model is a separate 6-joint `metal_arm_only.xml` rather
than a reuse of `arm1.xml` (which has 8).

---

## 5. T0.3 — joint limits (measured, and the provenance matters)

`mini_platform/artifacts/sysid/metal_arm_limits.json`, via `tools/measure_joint_limits.py`.

⚠️ **Superseded.** The torque ceilings were later raised to the METAL SDK's `tmax` and these
were re-measured. Current values and the `min(measured, SDK vmax)` rule are in
`artifacts/reports/02_joint_limits.md`; the shipped `JOINT_VEL_LIMIT` is now
`[10.0, 10.0, 10.0, 10.0, 9.640, 9.960]`. Table kept for provenance.

| joint | force (N·m) | q̇_max worst (rad/s) | q̈_max worst (rad/s²) |
|---|---|---|---|
| 1 | 30 | 10.345 | 39.191 |
| 2 | 100 | 17.492 | 234.942 |
| 3 | 30 | 10.100 | 133.904 |
| 4 | 10 | 4.395 | 78.972 |
| 5 | 10 | 4.841 | 97.059 |
| 6 | 10 | 4.986 | 99.266 |

Method: per joint, placed at the **centre of its own range** (starting next to a stop measures
the stop, not the servo) with the rest of the arm PD-held, actuator force saturated in both
directions, over 65 configurations; worst case reported.

> ⚠ **These are 4–12× the KUKA's rated limits (1.48–2.36 rad/s).** The MJCF models no servo
> speed limit and its `damping="2"` / `armature="0.1"` are CAD placeholders. A 0.5 kg-class arm
> outrunning a KUKA is not credible. §3.3's LP — hence every bank-shot result — is *defined* by
> these numbers. **Open decision 4 is not closed. Get the servo datasheet.**

Joint 1 is the acceleration bottleneck by 6×, and is also the joint that showed the largest
tracking error under the position controller.

---

## 6. Two pre-existing MJCF bugs, both silently fatal

Both predate this session and affect anything using `mini_platform/assets/`, including the RL track.

**1. `base_link` ↔ `link1` permanently interpenetrated by 6.25 mm** (`arm1.xml`, `arm2.xml`).
They are mating CAD parts. MuJoCo's parent-child contact filter does **not** remove the pair:
the filter skips any pair where one body is welded to the world, and `base_link` has no joint,
so it is. The contact resolved at **~2.5×10⁹ N** every step. Symptom: joint 1 lagged its
command by 0.18 rad — a 2.4 cm cartesian error, larger than `d_c` — so the mallet passed *near*
the puck without ever touching it, **at any controller gain**. Fixed with an explicit
`<exclude>`. A 20 000-configuration sweep afterwards shows only genuine folded-pose
self-collisions (3–8% of random configurations), which is correct behaviour.

**2. Puck drag ~8× too high** (`mini_platform/assets/mini_table.xml`) — **already diagnosed** in `../../docs/rl_dr_train_plan.md` §10; fixed here, but with a different value (`6.448e-4`) than that section proposes (`0.00124`). See the discrepancy table there. `damping="0.005"` was carried over from the
full-size table unchanged. Damping is a *force* per unit velocity, so what must match between
tables is `damping / mass`:

| | mass | damping | deceleration |
|---|---|---|---|
| full-size | 0.01 kg | 0.005 | 0.5 s⁻¹ |
| mini (before) | 0.0012896 kg | 0.005 | **3.878 s⁻¹** |
| mini (after) | 0.0012896 kg | **6.448e-4** | 0.5 s⁻¹ |

At 3.878 s⁻¹ the puck's total travel is capped at `v₀/3.878` — 0.24 m at 0.94 m/s, under a
quarter of the table. No shot could ever reach the goal. Yaw damping corrected the same way
(`2e-6` → `6.595e-8`). Fixing this alone took `mini-hit` from 0.00 to 0.40 and `mini-prepare`
from 0.00 to 0.80. **Provisional** — T1.1 replaces it with a fit.

---

## 7. T0.2 / T0.4 — the mallet is gripped, and what that costs

### What was built

The original two-finger gripper is **intact** — `link7` / `link8`, their slide joints and the
mimic `<equality>` are exactly as they were. The mallet is a **free body** held by friction
alone: no weld, no equality, no joint to the arm.

| part | geometry |
|---|---|
| head | ⌀50 mm × 30 mm, resting on the table, **frictionless** |
| post | ⌀10 mm × 25 mm |
| handle | **10 mm cube**, µ = 2.0 |

- `r_m = 0.025`, **`d_c = r_p + r_m = 0.040990`**
- Grasp point **0.065 m** below the wrist, `GRASP_HEIGHT` = `ee_desired_height` = **0.060 m**
- `joint7` is a `<motor>` held at a constant **3 N** by `env_base._simulation_pre_step`;
  `joint8` follows through the equality. A kp=50 position servo cannot do this — it would need
  0.06 m of tracking error to reach 3 N, more than the jaw's whole travel.
- The gripper is deliberately **outside the action space**: the agent drives six arm joints.

The grasp offset was measured off the finger meshes, not guessed: the inner faces track the
joint travel exactly (inner face at y = |q7|) only in the z band 0.060–0.070, where the fingers
are ~18 mm wide. Nearer the knuckle they sit 2–3 mm proud and the cube would grip on a corner.

**Verified** (`tools/mini_mallet_check.py`): jaws close on 5.00 mm against a 5.00 mm handle
half-width, tilt 0.000°, slip 0.46 mm over 500 held steps, head bottom at z = −0.00002.
Restitution `e ≈ 0.30–0.66` over 0.5–4 m/s.

### The finding that matters most

Success over 25 episodes/task is **0.12–0.16 / 1.00 / 0.24–0.28** (see the reproducibility
caveat in §1). That is not the grip failing — it is
what the grip *exposed*:

> **A gripped mallet is only as flat as the wrist holding it, and nothing controls the wrist.**

`optimizer._solve_aqp` is position-only differential IK; it never constrains wrist orientation.
Measured wrist deviation from vertical reaches **30°+** in normal play. Two consequences:

1. **Edge strikes.** §3.1 assumes a horizontal contact normal. A head tilted 30° contacts the
   puck on its rim. This is what costs hit and prepare.
2. **Ejection.** Traced frame by frame on `mini-defend` ep3: the arm drives the grasp point
   below the mallet's resting height, `head_z` falls from 0.0150 to **0.0033** — 12 mm into a
   rigid table — the table pushes back, the handle cube rotates and pries the jaws open
   (4.9 → 6.4 → **8.1 mm**, and 8.1 ≈ the cube's 7.07 mm half-diagonal), and the mallet leaves
   the gripper. ~2/25 episodes on defend.

**This is not fixable in the MJCF.** Measured drop rates over 12 defend episodes:

| grip force | handle µ | defend drops | hit drops |
|---|---|---|---|
| 3 N | 2.0 | 2/12 | 0/12 |
| 3 N | 4.0 | 1/12 | 0/12 |
| 6 N | 2.0 | 2/12 | 1/12 |
| 10 N | 2.0 | 2/12 | 1/12 |
| 10 N | 4.0 | 1/12 | 0/12 |

Clamping force is **irrelevant** — 3 N and 10 N are identical — because the cube is levered
out by rotation, not pulled out by slipping. Raising the carry height (0/4/8 mm) removes the
defend drops at 8 mm but costs hit and leaves the mallet dangling unsupported.

**The fix is T4.2**, which must now add a **wrist-orientation term alongside its z-pin**, not
just the z-pin. The same root cause produces the one surviving constraint violation
(`ee_constr` component 4, `z > z_lb`, by ~5 mm on defend).

### Superseded

An earlier revision of this session deleted the gripper and bolted a compliant rod-and-
universal-joint mallet to the wrist (`r_m = 0.024327`, `d_c = 0.040317`, scores 0.38/1.00/0.82).
That was a misreading of the plan doc's "ends at a gripper, gripping nothing" as an instruction
rather than a description. **It is gone.** Its higher scores came from the universal joint
hiding the wrist-tilt problem, and its restitution was far worse (`e ≈ 0.02–0.10`). If you see
`striker_joint_link` / `striker_mallet` referenced for the mini platform anywhere, it is stale
— the only survivor is the massless `striker_joint_link` frame in `metal_arm_only.xml`, which
is the IK target and deliberately has no mallet attached.

---

## 7b. Sensing: there is none yet

Worth stating plainly, because it changes how every success rate above should be read.

**No camera is in the runtime path.** The agent's 21-dim observation is pure state — puck
(x, y, yaw, ẋ, ẏ, ẏaw) read straight off the sim's puck joints, plus 6 joint positions and 6
joint velocities. A grep for any renderer or camera reference across `mini_platform/envs/`,
`mini_platform/agents/` and `baseline/baseline_agent/` comes back empty.

So `PuckTracker` is filtering **simulator ground truth** against a constant `OBS_COV`. It is
doing tracking, not sensing. The current setup is therefore strictly easier than the real rig.

Cameras that exist in the MJCF but are wired to nothing:

| scene | cameras |
|---|---|
| `mini_single` | `table_side_left` + `metal_1/wrist_cam` |
| `mini_double` | `table_side_left` + `metal_1/wrist_cam` + `metal_2/wrist_cam` |

**One** table-side camera, at (0, 0.298046, 0.25), 45° down, fovy 80° @ 1920×1080 (f ≈ 643.5 px).
H2's mirrored `table_side_right` at y = −0.298046 is **not built**.

When perception replaces ground truth (the stated intent), the seam is a single line:
`baseline_agent.draw_action` calls `self.state.update_observation(q, dq, self.get_puck_pos(obs))`
— swap that third argument. Four things will bite:

1. `OBS_COV` (`kalman_filter.py:17`) is a constant diagonal, fitted full-size. It has to become
   the anisotropic rotated `R(x, y)` built from the camera pose above (**T2.2**).
2. **Latency is not modelled at all** — the measurement currently arrives on the same timestep
   it is generated. **T2.3** wants delay+jitter injected in sim *before* hardware.
3. **One side camera plus a ⌀50 mm mallet is an occlusion problem precisely at contact** — the
   mallet parks between the camera and the puck exactly when the measurement matters most. This
   got materially worse when the mallet went from a thin rod to a solid head, and it is the
   concrete argument for **H2**.
4. **H1 (bottom-edge detection) is the cheapest win** — it removes the parallax term, which
   peaks at 30 mm at the far corner and is the largest single entry in the mini error budget.

---

## 8. Retargeted baseline

`mini_platform/agents/baseline_mini.py`. **`baseline/` is not edited.** Only three things in it are
full-size literals rather than `env_info` lookups, and only those are overridden:

- `optimizer.py:23` `anchor_weights` (7 entries) → `np.ones(6)`. With ~1 DOF of redundancy the
  null space is essentially one-dimensional, so the KUKA's `[10,1,10,1,10,10,1]` pattern has no
  structure to express.
- `trajectory_generator.py:40` `+ [1.51, 0]` → the actual base offset.
- `tactics.py:402` `goal_pos = [2.49, 0]` → `table_length/2 + base_offset` = 1.129.

Agent ranges are derived as **fractions of the own half**, not by scaling the baseline's
absolute values — that scaling is invalid across these two base placements (guide §0.3).
`hit_range` = [0.251, 0.511], capped at the measured 0.541 strike limit.

Its *fitted* constants are still full-size fits (`kalman_filter.py`'s `damping`, `b_params`,
`OBS_COV`), so its predictions on the mini table are biased and **the success rates are a lower
bound**. Refitting is T1.1–T1.3.

---

## 9. Tooling notes

**`mushroom_rl` 1.10.1's viewer is broken against mujoco 3.x.** It calls
`mjv_moveCamera(m, action, dx, dy, scene, cam)`; `scene` was removed in 3.x, so the *first
mouse drag or scroll* raises `TypeError` and kills the run. The window opens and renders fine,
which makes it look like a random crash. `tools/mini_play.py::patch_viewer_for_mujoco3()`
rebinds `scroll` and `mouse_move` — `site-packages` is untouched, per the guide's rule.

The viewer's default 15 m camera distance also puts the mini table in the skybox;
`mini_play.CAMERA_PARAMS` fixes the framing.

---

## 10. State of the task board

| Task | Status | Notes |
|---|---|---|
| P1 env setup | **done** | conda `ahc` |
| T0.1 workspace study | **partial** | §0.3's reach numbers taken as given and used to pick the mount. **Stall-strip rate still unmeasured** — that, not reach, decides whether the mount changes |
| T0.2 mallet MJCF | **done** | gripper grips a free mallet |
| T0.3 joint vel limits | **done** | ⚠ provenance caveat, §5 |
| T0.4 mini envs | **done** | 0.12–0.16 / 1.00 / 0.24–0.28 |
| T1.1–T1.3 | **next** | puck drag is currently an analytical rescale, not a fit |
| everything else | not started | |

### Not obtained this session

A full-size `BaselineAgent` reference run (25 eps × 3 tasks) was launched to give a comparison
baseline. It ran for over an hour without completing — the full-size `hit` env's default
opponent calls `inverse_kinematics` every control step, which is slow — and was killed. **There
are no full-size comparison numbers.** If you want them, run fewer episodes or use
`hit_single`, which has no scripted opponent.

---

## 10b. Corrections to earlier revisions of this document

Recorded because the first two changed a number someone might act on.

1. **Tunnelling threshold: 3.0–3.5 m/s head-on, not ~6 m/s.** `mini-defend` generates up to
   2.53 m/s, so headroom is **~1.2×**, not the 2.3× first reported. The old figure came from a
   bug in `mini_mallet_check.rebound_check`: it reset the puck's *forward* velocity but not its
   lateral and spin velocity, so the "4 m/s" case was a glancing blow off a still-moving puck.
   The tool now zeroes all three and is deterministic run to run.
2. **The acceptance gate is not reproducible** (see §1). Earlier revisions quoted single
   numbers as if they were fixed.
3. **The puck-drag bug was not found here.** `docs/rl_dr_train_plan.md` §10 had already
   diagnosed it ("~7.8× too draggy… **Still present on disk**"). It was *fixed* here, but with a
   different value than that section proposes — `6.448e-4` (τ = 2.0 s) versus `0.00124`
   (τ ≈ 1.04 s). The difference is one assumption: whether puck speeds scale with the table.
   The mini task envs do scale them, so the on-disk value is self-consistent with them; an RL
   track feeding unscaled speeds wants the other. **Neither is a fit** — T1.1 settles it. The
   comparison table now lives in that section of the RL plan so the two tracks cannot silently
   diverge.
4. The earlier "compliant rod-mounted mallet" revision is **superseded** — see §7's *Superseded*
   note. Its higher scores came from the universal joint hiding the wrist-tilt problem.

---

## 11. What I would do next, in order

1. **`git add mini_platform/`** — there is no history behind any of the MJCF work.
   *(Still true, and now more urgent — see Part A, A0 trap 1.)*
2. **T1.1** (free-slide drag fit). The puck drag currently in `mini_table.xml` is an analytical
   rescale of the full-size value, chosen to make the deceleration coefficient match. It is
   defensible but it is not a fit, and every prediction downstream depends on it. Note
   `kalman_filter.py`'s fitted 0.2125 s⁻¹ already disagrees with the MJCF's 0.5 s⁻¹ at full
   size, so decide which one is ground truth.
3. **Pull T4.2's wrist-orientation term forward**, or at least prototype it. It is currently
   listed as Phase 4 work, but §7 shows it is what is capping every success rate now. A cheap
   experiment: add an orientation row to `_solve_aqp` and re-run the gate. If hit and prepare
   jump, the diagnosis is confirmed and the rest of Phase 1–3 gets a much better substrate.
4. **Chase the servo datasheet** (§5). Everything in §3.3 rests on numbers that are currently
   an artefact of placeholder damping.
5. **T0.1's stall-strip rate**, whenever the mount comes up for review.
6. **When perception replaces ground truth** (stated intent — camera into the EKF rather than
   GT): do **H2** first, not last. One side camera plus the new solid mallet occludes the puck
   at contact, and the mirrored camera is nearly free in sim. Then T2.2's `R(x, y)`, then T2.3's
   latency injection. §7b has the seam and the four gotchas.
