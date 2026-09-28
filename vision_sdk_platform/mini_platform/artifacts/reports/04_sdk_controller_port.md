# 04 — Porting the METAL SDK's low-level controller

## Why

The controller the mini platform inherited from the Air Hockey Challenge is a PD with an
inverse-dynamics feedforward, and its gains (`p=[150,400,200,80,60,40]`, `d=[10,20,12,5,3,2]`)
were guessed: they came from the `<position>` servo `kp` values the MJCF used to carry, with
`d ≈ kp/16`. They have never been validated against hardware. A 4×p/2×d sweep moved hit
0.38 → 0.55 and prepare 0.82 → 0.55 — net zero, and fixed none of the constraint violations.

The real arm already has a control law that is stable without tuning, in
`metal-python-ros/metal_sdk` (C++). This report records porting it.

## Which SDK

Two SDKs are vendored under `mini_platform/`. They are **not** copies of each other.

| | `metal_sdk` (C++) | `maker-arm-sdk` (Python) |
|---|---|---|
| Motors | DM / Damiao | RobStride RS00/RS02 |
| Control rate | 400 Hz | 200 Hz |
| Setpoint shaping | quintic trajectory | slew-rate ramp, 9 rad/s |
| `kp` | 300, 500, 500, 400, 25, 20 | 60, 150, 90, 30, 30, 30 |
| `qd_des` | trajectory velocity | hard 0 |
| Gravity/Coriolis/friction FF | **yes** | **none — `tau_ff` is hard 0** |
| Gripper | linear, 0–102.5 mm stroke table | revolute motor 7 |

`metal_sdk` is the one this rig models, on four independent grounds:

1. Gravity compensation was the requirement. `maker-arm-sdk` never sends a feedforward torque:
   every command is `send_mit(pos, 0.0, kp, kd, 0.0)`.
2. `assets/arm1.xml`'s joint ranges match `config/motor_config.cpp` **exactly** on all six
   joints: `−2.79/2.79`, `−3.14/0`, `0/3.14`, `−2.145/1.413`, `±1.482`, `±2.529`.
3. The MJCF's two ±0.05 m prismatic fingers match the metal gripper's 102.5 mm **linear**
   stroke table. The maker arm's gripper is a single revolute motor.
4. The metal SDK has an explicit gripper mode; the physical arm has a gripper.

The C++ SDK cannot be executed here: no `.so` is checked in, building needs ROS 2 Humble +
orocos-KDL + trac_ik, and `Init()` opens a real CAN socket and blocks on motor replies. It was
ported as a specification, not called as a library.

## The law

Per joint, every 2.5 ms (400 Hz, `can_manager.cpp:996-998`) the host packs
`(q_des, qd_des, kp, kd, tau_ff)` into one CAN frame. The **motor firmware** closes the loop at
its own kHz rate:

```
tau    = kp*(q_des - q) + kd*(qd_des - qd) + tau_ff       clipped to ±tmax
tau_ff = Kg·G(q) + Kc·C(q,q̇) + b·clamp(q̇)                |q̇| < 0.05 → 0
```

`Kg = [1.2, 1.15, 1.1, 1.15, 1.0, 1.0]`, `Kc = [1.1, 1.15, 1.0, 1.0, 1.1, 1.1]`,
`b = [0.15, 0.3025, 0.52, 0.42, 0.015, 0.015]` (`kdl_solver.cpp:290, 291, 441`). Every
constant carries a `file:line` citation in `core/sdk_params.py`.

The 400 Hz host tick does not divide the 1 ms sim substep, so a phase accumulator latches at
substeps `[2, 4, 7, 9, 12, 14, 17, 19]` — 8 latches per 20 ms env step, gaps
`2,3,2,3,2,3,2,3` ms, mean exactly 2.5 ms, and the final latch lands exactly on the planner's
segment endpoint. The PD term is evaluated every substep against **raw** `qpos`/`qvel`, because
the firmware's `kd` sees unfiltered encoder velocity, not the observation pipeline's 51 Hz IIR.

## Deviations from the SDK

Three, all opt-out, all documented in `core/sdk_params.py`.

1. **`inertia_ff=True`.** The SDK defines `M(q)·q̈` (`kdl_solver.cpp:329`) and never calls it.
   Measured here, it is the difference between the port working and not — see below. The sim's
   `M` is 84–99.5 % armature on J4–J6, so this term is much larger here than on the real arm.
2. `qd_cmd` is clamped before quantisation. `dm_motor_writer.cpp:53` does not clamp velocity,
   so an out-of-range value wraps through `float_to_uint`. Reproducing an integer wrap would
   inject garbage, not fidelity.
3. Output torque is clipped to `min(SDK tmax, MJCF ctrlrange)`. MuJoCo clips to `ctrlrange`
   anyway; doing it explicitly makes the saturation observable instead of silent.

**Not ported:** the SDK's own 400 Hz quintic re-planner and its ~27 ms transport lag. The
platform's planner, FSM and optimizer are untouched, and the `(2, 6)` 50 Hz action contract is
unchanged.

## Measured

Both controllers are registered, so every comparison below is a task-name change and nothing
else. `drops` counts **episodes** in which the mallet left the jaws (slip > 20 mm); it is the
metric to read, not `slip_mm_p99`, which is dominated by *when* a drop happened.

### End-to-end, `tools/mini_env_smoke.py`, seed 0

| task | controller | success | drops | tilt p99 | `ee_constr` | `joint_vel_constr` |
|---|---|---|---|---|---|---|
| defend (n=60) | legacy | 1.00 | 3/60 | 90.0° | **+0.0037 ✗** | −0.960 |
| defend (n=60) | **sdk** | 1.00 | 3/60 | **29.0°** | **−0.0019 ✓** | **+0.255 ✗** |
| prepare (n=60) | legacy | 0.37 | 0/60 | 33.6° | −0.0121 | −2.156 |
| prepare (n=60) | **sdk** | 0.33 | 1/60 | 32.5° | −0.0104 | −0.956 |
| hit (n=25) | legacy | 0.12 | 1/25 | 90.0° | −0.0136 | −1.020 |
| hit (n=25) | **sdk** | 0.08 | 3/25 | **47.6°** | −0.0129 | −0.080 |

What this says, stated conservatively:

- **The one standing constraint violation is fixed.** Defend's `ee_constr` z-violation
  (reported in `03_mallet_and_mini_envs.md`) goes `+0.0037 → −0.0019`.
- **A new violation appears**: `joint_vel_constr` on J4, `+0.255`. Nothing clamps on it —
  constraints only report — but anything scoring violations will now flag J4.
- **Mallet tilt improves substantially** where it was worst: defend 90.0° → 29.0°, hit
  90.0° → 47.6°. This is the failure mode `03_mallet_and_mini_envs.md` documents as breaking
  the horizontal-contact-normal assumption.
- **Drop rate is unchanged.** 3/60 vs 3/60 on defend, and hit went 1/25 → 3/25. At these
  sample sizes none of those differences is distinguishable from zero.
- **Success rates are unchanged within noise.** Defend is 1.00 for both. Prepare 0.37 → 0.33
  and hit 0.12 → 0.08 are each within one standard error (Wald SE at p≈0.1, n=25 is ~0.065).
  **Do not judge hit success at n ≤ 200** — see `README.md` §5 on gate reproducibility.

### The `inertia_ff` deviation, defend n=60, seed 0

| | drops | tilt p99 | `ee_constr` | `joint_vel_constr` |
|---|---|---|---|---|
| `inertia_ff=True` (shipped) | 3/60 | 29.0° | **−0.0019 ✓** | +0.255 |
| `inertia_ff=False` (literal SDK) | 7/60 | 92.3° | **+0.0193 ✗** | +0.991 |

Removing the term more than doubles the drop rate, triples the tilt, and re-opens the
`ee_constr` violation the port was supposed to close. The deviation is kept, and named.

### Closed-loop identity, `tools/mini_ctrl_bench.py`

Gravity hold, 2 s at a constant command — the sharpest single number in the suite, and the one
that fails loudest if the feedforward sign is wrong or the MJCF's joint damping is edited:

| controller | max \|q − q₀\| | max \|q̇\| | ee z error | grip slip |
|---|---|---|---|---|
| legacy | 0.00124 rad | 0.00157 rad/s | 0.06 mm | 0.41 mm |
| **sdk** | **0.00035 rad** | **0.00085 rad/s** | 0.04 mm | 0.30 mm |

Step response, 0.05 rad:

| | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---|---|---|---|---|
| legacy f_n [Hz] / ζ | 3.88 / 1.00 | 6.86 / 1.00 | 3.43 / 1.00 | 3.00 / 1.00 | 3.18 / 1.00 | 2.69 / 1.00 |
| **sdk f_n [Hz] / ζ** | 6.18 / 0.577 | 4.99 / 0.432 | 5.92 / 0.418 | 6.77 / 0.511 | 2.37 / 0.755 | 2.07 / 0.802 |
| sdk overshoot | 10.9 % | 22.2 % | 23.6 % | 15.4 % | 2.7 % | 1.5 % |
| sdk saturation | 0.70 % | 1.00 % | 1.00 % | 2.60 % | 1.70 % | 1.70 % |

The legacy controller is critically damped on every joint (ζ = 1.000, zero overshoot). The SDK
law is deliberately underdamped on J1–J4 and slower on J5/J6. These measured values are the
reference: re-run the bench after any MJCF edit and they should not move.

⚠️ A first-principles prediction (linearised, ideal step, no saturation) gave J2 6.57 Hz,
J3 7.50 Hz, J4 9.39 Hz. The measured values are 20–28 % lower on those three joints. The
prediction ignores the 20 ms cubic command ramp, the 1–2.6 % torque saturation and inter-joint
coupling, all of which lower apparent `f_n` and raise apparent ζ. **The measured table is the
reference, not the prediction.**

## Stage 2 — torque ceilings raised to the SDK's `tmax`

Done as a second, separate change, on the instruction to treat the SDK as the source for
numbers. `assets/arm1.xml`, `arm2.xml` and `metal_arm_only.xml`: J2 100→120, J4 10→30,
J5/J6 10→20 N·m on both the actuator `ctrlrange` and the joint `actuatorfrcrange`. J1/J3
already agreed at 30. The gripper's `joint7` stays at ±3 — it is a linear finger force in N,
not one of the SDK's joint torques.

`tools/measure_joint_limits.py` was re-run (mandatory: the old limits were measured under the
old saturated force) and `JOINT_VEL_LIMIT` recomputed as `min(measured, SDK vmax)`. See
`02_joint_limits.md`.

### It did what it was meant to

| | J2 sat | J4 sat | J4 err p99 | J5/J6 sat |
|---|---|---|---|---|
| before | 1.00 % | 2.60 % | 0.04053 rad | 1.70 % |
| after | 0.20 % | **0.70 %** | **0.02615 rad** | 0.50 % |

**Every constraint violation on the rig is now gone, for both controllers.** The J4
`joint_vel_constr` violation the SDK port introduced (+0.255) is at −2.32, and the legacy
controller's long-standing defend `ee_constr` violation (+0.0037) is at −0.0104 — the raised
limits changed the planner's bounds, so the legacy controller benefited too.

### End-to-end after Stage 2, seed 0

| task | controller | success | drops | tilt p99 | `ee_constr` | `joint_vel_constr` |
|---|---|---|---|---|---|---|
| defend (n=60) | legacy | 1.00 | 4/60 | 69.7° | −0.0104 ✓ | −3.019 ✓ |
| defend (n=60) | sdk | 1.00 | 4/60 | 78.5° | −0.0144 ✓ | −2.320 ✓ |
| prepare (n=60) | legacy | 0.38 | 0/60 | 33.5° | −0.0121 ✓ | −5.607 ✓ |
| prepare (n=60) | sdk | 0.32 | 2/60 | 34.3° | −0.0104 ✓ | −4.034 ✓ |
| hit (n=25) | legacy | 0.20 | 2/25 | 90.1° | −0.0158 ✓ | −3.617 ✓ |
| hit (n=25) | sdk | 0.04 | 3/25 | 38.2° | −0.0078 ✓ | −3.680 ✓ |

### What it cost, stated plainly

1. **The SDK controller's tilt advantage on defend evaporated: 29.0° → 78.5°.** Before Stage 2
   it was 29° against the legacy controller's 90°; now it is 78.5° against 69.7°, i.e. slightly
   *worse* than legacy. The likely mechanism is J5/J6: their velocity limit went 4.84 → 9.64
   and 4.99 → 9.96 rad/s, so the planner is free to whip the wrist twice as fast, and the
   mallet's orientation follows the wrist because it is held by friction, not welded.
2. **Drops rose slightly for both** (3/60 → 4/60 on defend). Indistinguishable at this n.
3. **`mini-hit-sdk` success fell to 0.04** (1/25) while legacy rose to 0.20 (5/25). Fisher
   exact p ≈ 0.19 — not significant, and hit is documented as unjudgeable below n ≈ 200. It
   should not be read as a result, but it should not be ignored either.
4. **The SDK controller no longer has a clear task-level edge over the legacy one.** Its
   remaining, real advantages are: nothing to tune, 3.5× better gravity hold, and being the law
   the hardware actually runs. Those were the point of the exercise; the constraint fix that
   distinguished it in Stage 1 is now available to both controllers.

**If the wrist tilt matters more than the extra J5/J6 speed**, the targeted lever is to cap
`JOINT_VEL_LIMIT[4:6]` back at ~4.9 rad/s while keeping the raised torque ceilings — that
retains the J4 saturation fix and the constraint fixes, and gives back the wrist restraint.
This has not been measured.

### Closed-loop identity after Stage 2

| | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---|---|---|---|---|
| legacy f_n [Hz] / ζ | 3.88 / 1.00 | 7.33 / 1.00 | 3.41 / 1.00 | 3.32 / 1.00 | 3.24 / 1.00 | 2.69 / 1.00 |
| **sdk f_n [Hz] / ζ** | 6.18 / 0.577 | 4.96 / 0.451 | 5.89 / 0.431 | 7.92 / 0.587 | 2.21 / 0.742 | 1.95 / 0.792 |
| sdk saturation | 0.70 % | 0.20 % | 1.00 % | 0.70 % | 0.50 % | 0.50 % |

Gravity hold is unchanged by Stage 2 (legacy 0.00124 rad, sdk 0.00035 rad) — as expected, since
it is a static test well below any torque ceiling.

## Known issues this port does not fix

1. ~~**J4's velocity constraint is violated by ~6 %.**~~ **Fixed by Stage 2** — now at −2.32.
2. **J5's tracking is worse**, because the MJCF's placeholder `damping=2` acts against
   `kp = 25`. On hardware the equivalent term (`b₅ = 0.015`) is 130× smaller. J5 is the ee's
   y-axis, so this compounds the wrist-tilt problem rather than relieving it.
3. **`damping=2` and `armature=0.1` are CAD placeholders, and the closed loop depends on
   them.** They supply a large fraction of the total damping and most of J4–J6's inertia.
   "No tuning" is now true of the *controller*; it is not true of the *plant*.
4. ~~**J4 saturates ~2.6 % of substeps.**~~ **Fixed by Stage 2** — 0.70 %, with a 35 % better
   tracking error. But see *What it cost* above: the extra J5/J6 speed that came with it made
   the wrist tilt worse on defend.
5. **Joint sign and zero are assumed identity.** Nothing in `metal-python-ros` establishes a
   sign map — `GetJointPosition()` returns the motor angle verbatim — but
   `test/calib_direction.py` exists because J2 was suspect. `sdk_params.JOINT_SIGN` /
   `JOINT_OFFSET` are there so a calibration result is a one-line change.
