# T0.2 + T0.4 — striker mallet and the mini environments

**Status: both PASS.** `AirHockeyChallengeWrapper(env="mini-hit" | "mini-defend" |
"mini-prepare")` constructs, resets, renders, and a retargeted `BaselineAgent` plays all three
tasks for 1000+ steps with no divergence and no constraint explosion.

**The mallet is gripped, not welded** — the original two-finger gripper is intact and holds a
separate free-body mallet by friction alone. Read "The wrist tilt is now load-bearing" below
before reading the success rates: they are much lower than the bolted-on version's, for a
reason that is a controller gap (T4.2), not a modelling defect.

Two modelling bugs in the existing MJCF were found and fixed along the way; both would have
silently invalidated everything downstream. See "Bugs found" below — read that section even if
you skip the rest.

---

## T0.2 — the mallet, gripped

**The gripper is kept. The mallet is a separate free body that the gripper actually grips —
friction only, no weld, no equality, no joint.** `metal_1/link7` / `link8`, their slide joints
and the mimic `<equality>` are exactly as they were in the original `arm1.xml`.

*(An earlier revision of this work deleted the gripper and bolted a mallet to the wrist on a
rod. That was wrong — the plan doc's "ends at a gripper, gripping nothing" is a description,
not an instruction to remove it. The measurements in that revision are superseded below.)*

**Mallet** — `mini_platform/assets/mallet1.xml`, `mallet2.xml`. Geometry as specified:

| part | geometry | local z (world z, grasp point at 0.060) |
|---|---|---|
| handle | **10 mm cube** | −0.005 .. +0.005 (0.065 .. 0.055) |
| post | ⌀10 mm | +0.005 .. +0.030 (0.055 .. 0.030) |
| head | **⌀50 mm × 30 mm** | +0.030 .. +0.060 (0.030 .. 0.000, resting on the table) |

- `r_m` = **0.025 m**, so **`d_c = r_p + r_m = 0.040990 m`**.
- Mass 0.035 kg; body origin is at the handle centre, which is the grasp point.
- The 25 mm post exists because the fingertips reach 0.0264 below the grasp point and would
  otherwise sit inside the 30 mm head.

**Grasp point at 0.065 m below `gripper_base`**, measured off the finger meshes rather than
guessed: the inner faces track the joint travel exactly (inner face at y = |q7|) only in the
z band 0.060–0.070, and the fingers are ~18 mm wide there, so a 10 mm cube seats squarely.
Nearer the knuckle the faces sit 2–3 mm proud and the cube would grip on a corner.
`ee_desired_height` = `GRASP_HEIGHT` = **0.060 m** — set by the mallet's own geometry, not chosen.

**Grip.** joint7's actuator changed from `<position>` to `<motor>` and is driven at a constant
**3 N** (its `actuatorfrcrange` limit) every substep by `env_base._simulation_pre_step`;
joint8 follows through the equality. A position servo cannot do this job: at kp = 50 it would
need 0.06 m of tracking error to reach 3 N, more than the jaw's entire travel. The gripper is
deliberately outside the action space — the agent drives six arm joints and the grip is simply
held shut.

Friction is split by purpose: the **handle** is µ = 2.0 at priority 3 (beats the fingers'
default 1.0 — this is what holds the mallet up), the **head** is frictionless at priority 2
(beats the table's 0.08 and the puck's default, so it slides freely and imparts no tangential
impulse, `μ_m ≈ 0`, §3.1).

The mallet is **not** excluded from the table: it rests on the surface, held laterally by the
grip, the way a real one is. That support also keeps it upright — a 65 mm mallet on a 10 mm
handle is a long lever for a friction grip to hold unaided.

`env_base.grip_state(agent)` returns `(slip, tilt_deg, jaw)` so the grip is measured, never
assumed. `robot_1/ee_pos` now reads the **mallet**, i.e. where it actually is, not where the
arm commanded it; the two differ by the slip.

`UniversalJointPlugin` is no longer used by the mini env — there is no universal joint to
drive. It keeps the generalisation added earlier (explicit name lists + gains, iiwa defaults),
which is harmless and leaves the full-size envs untouched.

### Acceptance (`mini_platform/tools/mini_mallet_check.py`) — PASS

- Compiles: `nq=33 nv=31 nu=14 nbody=28` (6 arm + 2 gripper + 6 free mallet + 3 puck, ×2 arms).
- Home pose: jaws at 5.00 mm on a 5.00 mm handle half-width, mallet tilt **0.000°**, slip
  **0.00 mm**, head bottom at z = −0.00002 (resting on the surface), head spans 0 .. 0.030 and
  covers the puck's 0 .. 0.0101.
- Static hold, 500 steps: max slip **0.46 mm**, max tilt **0.00°**, still gripped.
- Render: `mini_platform/artifacts/reports/img/mini_home_mallet_closeup.png` shows the fingers clamped on the
  handle cube with the head flat on the table.
- Rebound, puck fired at the parked mallet:

| v_in (m/s) | v_out | e | closest approach (d_c = 0.0410) |
|---|---|---|---|
| 0.5 | +0.152 | **0.30** | 0.0400 |
| 1.0 | +0.331 | **0.33** | 0.0401 |
| 2.0 | +0.865 | **0.43** | 0.0411 |
| 4.0 | +2.639 | **0.66** | 0.0296 |

**Restitution is far better than the rod-and-universal-joint version** (e ≈ 0.30–0.66 against
0.02–0.10), because the mallet is now a rigid free body backed by the table rather than a
compliant pendulum absorbing the impact. It is still speed-dependent, so T1.3 must still fit
`e_m` as a function of impact speed — but from a much healthier starting point.

**Tunnelling threshold: 3.0–3.5 m/s head-on** at the 1 ms timestep. `mini-defend` generates up
to 2.53 m/s, so headroom is only **~1.2×**. An earlier revision of this report claimed ~6 m/s
and 2.3× headroom; that was wrong — `rebound_check` reset the puck's *forward* velocity but not
its lateral and spin velocity, so the "4 m/s" case was a glancing blow, not a head-on hit. The
tool now zeroes all three and is deterministic run to run. MuJoCo has no continuous collision
detection, so the only fix is a smaller timestep.

---

## T0.4 — the mini environments

New package `mini_platform/envs/`, mirroring `environments/iiwas/`:
`env_base`, `env_single`, `env_double`, `env_hitting`, `env_defend`, `env_prepare`.
`PositionControlMetal` + `MetalPosition{Hit,Defend,Prepare,Double}` in
`position_control_wrapper.py`; `mini-hit` / `mini-defend` / `mini-prepare` registered in
`AirHockeyChallengeWrapper.env_dict`.

`env_info` is built from mini constants, with joint ranges read off the compiled `MjModel`:

```
table   length 1.016  width 0.508  goal_width 0.122351
puck    radius 0.0159904
mallet  radius 0.025
robot   n_joints 6   ee_desired_height 0.060   universal_height 0.060
        link_constraint_height 0.035   control_frequency 50
        joint_vel_limit / joint_acc_limit  <- artifacts/sysid/metal_arm_limits.json
```

Robot-only model: **`mini_platform/assets/metal_arm_only.xml`**, a new file with exactly 6 joints and a
massless `striker_joint_link` frame at the grasp point (0.065 below the wrist). `arm1.xml`
could not be reused — it has 8 joints, and the robot-model contract is
`nq == nv == nu == n_joints` (see `00_env_setup.md`). The mallet has no place in the robot's
kinematic chain at all: the planner commands where the *grasp point* goes, and whether the
mallet follows is a question for the physics, answered by `grip_state()`.

### Generalisations to shared code (all backward-compatible)

| file | change |
|---|---|
| `utils/kinematics.py` | `link_to_xml_name` now picks the naming scheme from the model (`iiwa_1/` vs `metal_1/`) instead of hard-coding iiwa names. For the metal arm, link `"7"` — what `LinkConstraint` calls the wrist — resolves to `gripper_base`, since there is no seventh joint. |
| `utils/universal_joint_plugin.py` | takes `arm_joint_names`, `striker_joint_names`, `Kp`, `Kd`; the hard-coded `i*7` slice is now `i*n_joints`. Defaults reproduce the iiwa exactly. |
| `constraints/constraints.py` | `LinkConstraint.z_lb` reads `env_info['robot']['link_constraint_height']`, default 0.25. The iiwa's 0.25 m is meaningless on an arm whose gripper sits ~0.125 m above its base. |
| `framework/air_hockey_challenge_wrapper.py` | `check_success`'s `-0.8 / -0.2 / 0.39105` are now the same *fractions* of the table (0.8214 / 0.2053 of the half-length, 0.7535 of the half-width). Identical behaviour at full size; on the mini table the literals were off the playing surface. |

### Task parameters

Puck start ranges use the **same fractions of the table** as the full-size envs, so the tasks
are the same shape. Speeds scale linearly with the table (×0.505226) so the puck takes the
same time to cross.

| | full-size | mini |
|---|---|---|
| hit `hit_range` x | [−0.7, −0.2] | [−0.365, −0.104] |
| hit init speed | (0, 0.5) | (0, 0.253) |
| defend `start_range` x | [0.4, 0.75] | [0.209, 0.391] |
| defend init speed | (1, 5) | (0.505, 2.526) |
| prepare side / bottom x | [−0.8, −0.2] / [−0.94, −0.8] | [−0.417, −0.104] / [−0.490, −0.417] |

The far end of `hit_range` (−0.104) sits just outside the ~0.08 m strip in front of the
centreline the arm cannot reach (§0.3), so no episode starts with an unplayable puck.

`hit_range` **for the agent** (`agent_params`) is a different quantity — see below.

### Controller gains

`PositionControlMetal`: `p_gain = [150, 400, 200, 80, 60, 40]` (the MJCF's own `<position>`
servo `kp` values, per the guide's instruction), `d_gain = [10, 20, 12, 5, 3, 2]` ≈ kp/16,
`i_gain = 0`. Explicitly **not** `PositionControlIIWA`'s [1500, …], which are sized for a
320 N·m KUKA — against 10–100 N·m actuators they would sit clipped.

A 4× p / 2× d variant was measured over 40 episodes per task: it improves hit (0.38 → 0.55)
and *costs* prepare (0.82 → 0.55), for the same total. Neither setting eliminates the
`ee_constr` overshoot below. The less-tuned, better-justified setting was kept; revisit once
the Phase 1 refit makes the success rates mean something.

### Acceptance (`mini_platform/tools/mini_env_smoke.py`, 25 episodes/task)

| task | success | worst `ee_constr` | mallet tilt p99 | grip slip p99 | drops |
|---|---|---|---|---|---|
| mini-hit | **0.16** | −0.015 | 90° * | 595 mm * | ~1/25 |
| mini-defend | **1.00** | +0.001 | 30° | 612 mm * | ~2/25 |
| mini-prepare | **0.24** | −0.016 | 32° | 5.1 mm | 0/25 |

\* the huge tilt/slip percentiles are dominated by the episodes where the mallet is dropped —
once it is lying on the table, "tilt" and "slip" are measuring a loose object. In episodes
where the grip holds, slip settles at **3–6 mm** and tilt tracks the wrist.

The env is stable — no divergence, no NaNs, no runaway constraints — but **play is much worse
than with the earlier bolted-on compliant mallet (0.38 / 1.00 / 0.82)**, and the cause is a
single mechanism worth stating precisely, because it is not a bug in the grip.

### The wrist tilt is now load-bearing, and nothing controls it

With a compliant universal joint, the mallet hung plumb no matter what the wrist did. Gripped
rigidly, **the mallet inherits the wrist's orientation** — and the baseline's differential IK
(`optimizer._solve_aqp`) is *position-only*: it never constrains wrist orientation. Measured
wrist deviation from vertical reaches **30°+** in normal play. That has two consequences:

1. **Strike geometry is wrong.** §3.1 assumes a horizontal contact normal. A head tilted 30°
   contacts the puck on an edge. This is what costs the hit and prepare success rates.
2. **It ejects the mallet.** Traced step by step on `mini-defend` ep3: the arm drives the
   grasp point *below* the mallet's resting height, `head_z` falls from its resting 0.0150 to
   **0.0033** — 12 mm into a rigid table — the table pushes back, the handle cube rotates and
   pries the jaws open (4.9 → 6.4 → **8.1 mm**, and 8.1 ≈ the cube's 7.07 mm half-diagonal),
   and the mallet leaves the gripper. Two frames later the jaws are shut on nothing.

This is the **same root cause as the `ee_constr` z-violation** already recorded (the mallet
dipping 5.3 mm below its lower bound on defend). A compliant mount absorbed it; a true grip
does not. The fix is **T4.2**'s z-pin row plus a wrist-orientation term — now with a far
sharper justification than "the constraint reads slightly positive".

**Parameter sweep — the grip is not the problem.** Drops over 12 defend episodes:

| grip force | handle µ | defend drops | hit drops | worst slip (held episodes) |
|---|---|---|---|---|
| 3 N | 2.0 | 2/12 | 0/12 | 64 mm |
| 3 N | 4.0 | 1/12 | 0/12 | 25 mm |
| 6 N | 2.0 | 2/12 | 1/12 | 64 mm |
| 10 N | 2.0 | 2/12 | 1/12 | 64 mm |
| 10 N | 4.0 | 1/12 | 0/12 | 25 mm |

Clamping force is **irrelevant** — 3 N and 10 N give identical drop rates — because the cube
is levered out by rotation, not pulled out by slipping. Friction helps somewhat. Raising the
carry height was also tried (0 / 4 / 8 mm of clearance): 8 mm removes the defend drops
entirely but costs hit (the head bottom rises to 8 mm against a 10.1 mm puck, leaving 2 mm of
overlap) and leaves the mallet dangling unsupported. **None of these are the fix; the
controller is.**

### Retargeted baseline

`mini_platform/agents/baseline_mini.py`. `baseline/` is **not** edited. Only three things in it
are full-size literals rather than `env_info` lookups, and only those are overridden:

- `optimizer.py:23` `anchor_weights` (7 entries) → `np.ones(6)`. With ~1 DOF of redundancy the
  null space is essentially one-dimensional, so the KUKA's `[10,1,10,1,10,10,1]` pattern has
  no structure to express.
- `trajectory_generator.py:40` `+ [1.51, 0]` → the actual base offset from `base_frame`.
- `tactics.py:402` `goal_pos = [2.49, 0]` → `table_length/2 + base_offset` = 1.129.

Agent-side ranges, derived as fractions of the **own half** (`goal_line = 0.113`,
half-length 0.508, robot frame):

| param | full-size | fraction of own half | mini |
|---|---|---|---|
| `hit_range` | [0.8, 1.3] | 0.271 – 0.784 | **[0.251, 0.511]** |
| `defend_range` | [0.8, 1.0] | 0.271 – 0.476 | [0.251, 0.355] |
| `defend_width` | 0.45 | 0.867 of half-width | 0.220 |
| `x_home` | [0.65, 0, 0.1645] | — | [0.171, 0, 0.032587] |

`hit_range`'s upper bound is capped at the **measured** strike limit of 0.541 m (§0.3), not
obtained by scaling [0.8, 1.3] — that scaling is invalid across these two base placements. The
cap does not bind here (0.511 < 0.541).

The baseline's *fitted* constants are still full-size fits — `kalman_filter.py`'s `damping =
[0.2125, 0.2562]`, `b_params`, `OBS_COV`. Its geometry is fully `env_info`-driven, but its
physics is not, so its predictions on the mini table are biased and **the success rates above
are a lower bound**. Refitting is T1.1–T1.3.

---

## Bugs found in the existing MJCF

### 1. `base_link` ↔ `link1` permanently interpenetrated (arm1.xml, arm2.xml)

The two bodies' collision meshes overlap by **6.25 mm at every configuration** — they are
mating CAD parts. MuJoCo's parent-child contact filter does **not** remove this pair: the
filter skips any pair in which one body is welded to the world, and `base_link` has no joint,
so it is. The result was a permanent deep contact resolving at **~2.5×10⁹ N**, which the arm
then had to fight through every trajectory. Symptom: joint 1 lagged its command by 0.18 rad
(a 2.4 cm cartesian error, larger than `d_c`), so the mallet passed *near* the puck without
ever contacting it, at any controller gain.

Fixed with an explicit `<exclude>` in each arm file. A 20 000-configuration sweep after the
fix shows only genuine folded-pose self-collisions (each in 3–8% of random configurations),
which is the desired behaviour. Every other adjacent pair in the chain is filtered normally.

This bug predates this work and affects any use of `mini_platform/assets/`, including the RL track.

### 2. Puck drag ~8× too high (mini_table.xml)

> **Credit: this one was already diagnosed** in `docs/rl_dr_train_plan.md` §10 ("puck is ~7.8×
> too draggy… **Still present on disk**"). It was fixed here, but with a **different value**
> than that section proposes — `6.448e-4` vs `0.00124`. The discrepancy is documented in that
> section now; it comes down to whether puck speeds scale with the table. T1.1 settles it.

`damping="0.005"` was carried over from `table.xml` unchanged. Damping is a *force* per unit
velocity, so what has to match between tables is `damping / mass`:

| | mass | damping | deceleration coeff |
|---|---|---|---|
| full-size | 0.01 kg | 0.005 | 0.5 s⁻¹ |
| mini (before) | 0.0012896 kg | 0.005 | **3.878 s⁻¹** |
| mini (after) | 0.0012896 kg | **6.448e-4** | 0.5 s⁻¹ |

At 3.878 s⁻¹ a puck's total travel is capped at `v₀/3.878` — **0.24 m at 0.94 m/s**, under a
quarter of the table. No shot could ever reach the goal, and `mini-hit` scored 0.00 with the
puck visibly being struck. Yaw damping was corrected the same way (`2e-6/5e-6 = 0.4 s⁻¹` →
`6.595e-8` against Izz `1.64871e-07`).

After the fix, `mini-hit` went 0.00 → 0.40 and `mini-prepare` 0.00 → 0.80 with no other
change. **Provisional**: T1.1 replaces this with a fit against the real table, and note that
`kalman_filter.py`'s fitted 0.2125 s⁻¹ disagrees with the MJCF's 0.5 s⁻¹ at full size too.

---

## What the next session needs to know

- `conda activate ahc`; `MUJOCO_GL=glfw` for rendering.
- Tools: `mini_platform/tools/{measure_joint_limits,mini_mallet_check,mini_env_smoke}.py`.
  All offline-only.
- **Not yet done from Phase 0:** T0.1's formal workspace study. §0.3's numbers were taken as
  given (they are stated as already measured), and the mount was chosen from them. The
  **stall-strip rate** — how often a puck comes to rest in the ~0.08 m unreachable strip — is
  still unmeasured, and that, not the reach figure, is what decides whether the mount needs
  changing. `mini-prepare` at 0.82 is weak evidence it is tolerable.
- The `joint_vel_limit` provenance caveat in `02_joint_limits.md` is the biggest open risk to
  Phase 3.
- Backups of the pre-T0.2 `arm1.xml` / `arm2.xml` / `mini_single.xml` / `mini_double.xml` are
  in the session scratchpad, not in the repo — `mini_platform/assets/` is untracked by git, so there is
  no history to fall back on. Consider `git add`ing it.
