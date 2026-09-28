# T0.3 — metal-arm joint velocity and acceleration limits

**Status: PASS, with a provenance caveat that matters.**

Deliverable: `mini_platform/artifacts/sysid/metal_arm_limits.json`, produced by
`mini_platform/tools/measure_joint_limits.py`. Consumed by
`mini_platform/envs/env_base.py` as `JOINT_VEL_LIMIT` / `JOINT_ACC_LIMIT`.

## Provenance: MEASURED IN MUJOCO, not from a datasheet

There is no servo datasheet for this arm, so the guide's fallback was used. **These numbers
describe the MJCF, not the hardware.** `arm1.xml` declares `damping="2"` and `armature="0.1"`
on every joint — CAD placeholders per §0.2 — and declares no servo speed limit at all. The
measurement therefore returns the *torque-and-inertia* limit of the model, which is an upper
bound on the real arm and, as the table below shows, an implausibly generous one.

Open decision 4 in `optimal_control_plan.md` is **not** resolved by this. Everything in §3.3
(the max-mallet-speed LP, hence every bank shot) rests on these numbers. Get the datasheet.

## Method

Per joint, at each of 65 configurations (the home pose + 64 uniform-random, seed 0):

- the joint is placed at the **centre of its own range** — starting next to a stop and pushing
  into it measures the stop, not the servo — while the rest of the arm is held at the sampled
  posture by a gravity-compensating PD;
- its actuator force is saturated in each direction and the model integrated for 3 s or until
  the joint reaches a stop;
- `q̇_max` = peak velocity, `q̈_max` = `|qacc|` at t = 0.

> Values were re-measured after `striker_joint_link` was reduced from a 0.0129 kg body to a
> 0.001 kg grasp frame (the mallet became a separate free body). That shifts the arm's inertia
> slightly, so figures drift <1% from earlier revisions of this report. The tool is otherwise
> deterministic for a fixed seed and model.

Reported value is the **worst case over configurations**, not the mean: effective inertia
varies by a factor of ~3 across the workspace, and a limit that only holds at the home pose is
not a limit.

With `damping="2"` the joints do not have room to reach their asymptotic velocity inside their
own travel (1.5–2.8 rad from centre), so these are range-limited peaks. For an arm that has to
reverse inside its own travel anyway, that is the operative limit.

## Results

⚠️ **Superseded.** The table below was measured against the MJCF's original torque ceilings.
Those were raised to the SDK's `tmax` — see *Re-measurement* below and
`artifacts/reports/04_sdk_controller_port.md`. Kept for provenance.

| joint | force limit (N·m) | q̇_max worst (rad/s) | q̇_max median | q̈_max worst (rad/s²) | q̈_max median |
|---|---|---|---|---|---|
| 1 | 30 | **10.345** | 14.047 | **39.191** | 149.0 |
| 2 | 100 | **17.492** | 21.469 | **234.942** | 252.5 |
| 3 | 30 | **10.100** | 11.860 | **133.904** | 156.8 |
| 4 | 10 | **4.395** | 4.945 | **78.972** | 85.7 |
| 5 | 10 | **4.841** | 5.020 | **97.059** | 97.4 |
| 6 | 10 | **4.986** | 4.994 | **99.266** | 99.5 |

## Re-measurement after the torque ceilings were raised to the SDK's `tmax`

The MJCF's actuator `ctrlrange` / joint `actuatorfrcrange` disagreed with the hardware SDK on
four joints. They now match `config/motor_config.cpp`: J2 100→120, J4 10→30, J5/J6 10→20
(J1/J3 already agreed at 30). Same method, same 65 configurations.

| joint | force limit (N·m) | q̇_max worst (rad/s) | q̇_max median | q̈_max worst (rad/s²) | q̈_max median |
|---|---|---|---|---|---|
| 1 | 30 *(unchanged)* | **10.345** | 14.047 | **39.191** | 149.0 |
| 2 | **120** | **18.876** | 23.353 | **283.664** | 304.0 |
| 3 | 30 *(unchanged)* | **10.100** | 11.860 | **133.904** | 156.8 |
| 4 | **30** | **13.657** | 14.003 | **261.759** | 271.8 |
| 5 | **20** | **9.640** | 9.802 | **194.396** | 194.9 |
| 6 | **20** | **9.960** | 9.966 | **198.820** | 199.0 |

J1 and J3 are bit-identical to the previous run, which is the sanity check: their torque did
not change, so their limits must not either. J4 more than tripled — it had been running a
30 N·m motor against a 10 N·m ceiling.

### What `JOINT_VEL_LIMIT` is now

Not this table. It is the **elementwise minimum** of this table and the SDK's own `vmax`
(`motor_config.cpp:13, 34, 55, 76, 97, 118`), because either bound can bind:

| | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---|---|---|---|---|
| measured (this plant) | 10.345 | 18.876 | 10.100 | 13.657 | 9.640 | 9.960 |
| SDK `vmax` (the motor) | 10 | 10 | 10 | 10 | 30 | 30 |
| **shipped** | **10.0** | **10.0** | **10.0** | **10.0** | **9.640** | **9.960** |
| binding | SDK | SDK | SDK | SDK | plant | plant |

The SDK binds on J1–J4; the plant binds on J5/J6, which cannot get near their 30 rad/s motor
limit. `JOINT_ACC_LIMIT` stays purely measured — the SDK bounds acceleration only inside its
own trajectory time allocator, which is deliberately not ported.

This resolves the credibility problem raised below in *The caveat, quantified*: J1–J4 are no
longer set by an unmodelled MuJoCo plant, they are set by the actual motor spec.

## The caveat, quantified

The KUKA's *rated* limits are 85–135 °/s, i.e. **1.48–2.36 rad/s**. This model comes out
**4–12× faster**. A 0.5 kg-class hobby arm being an order of magnitude faster than a KUKA is
not credible; it is the absence of any servo speed model in the MJCF.

Two consequences to keep in view:

1. `joint_vel_limit` feeds `PositionControl._enforce_safety_limits` and
   `JointVelocityConstraint`. At these values neither binds, so the constraint report will
   look clean whether or not the commands are achievable. In the T0.4 runs the observed worst
   `joint_vel_constr` was −1.16 (hit) — i.e. the agent used well under the limit anyway, so
   nothing is currently hiding behind it.
2. §3.3's LP maximises `s` subject to `|q̇ᵢ| ≤ q̇_max,ᵢ`. Inflated limits inflate the whole
   velocity table uniformly, which mostly rescales rather than reshapes it — but the
   *relative* profile across joints (joint 2 is 1.7× joint 1 and 4× joints 4–6) is set by the
   force ranges and is more trustworthy than the absolute magnitudes.

Joint 1 is the acceleration bottleneck by a wide margin (38.9 rad/s² worst case, 6× less than
joint 2). It is also the joint that showed the largest tracking error under the position
controller in T0.4.

Re-run this tool after H6 (sim-to-real system ID) replaces `damping` / `armature`.
