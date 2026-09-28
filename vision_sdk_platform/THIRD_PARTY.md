# Third-party code vendored into `mini_platform`

This package is standalone: it does not import the parent
[Robot Air Hockey Challenge](https://github.com/AirHockeyChallenge/air_hockey_challenge)
repository. To make that possible, part of that project is vendored here.

**License:** MIT — Copyright (c) 2025 Atalay Donat, Puze Liu, Jonas Guenster, Davide Tateo.
Full text in [`LICENSE.third_party`](LICENSE.third_party).

Every vendored file carries a provenance header naming its original path. Local changes are
marked `MINI:` in-line.

## Framework — from `air_hockey_challenge/`

| here | original | modified? |
|---|---|---|
| `core/agent_base.py` | `framework/agent_base.py` | no |
| `core/challenge_wrapper.py` | `framework/air_hockey_challenge_wrapper.py` | **yes** — task table comes from `mini_platform.envs`; `check_success` uses table *fractions* rather than full-size literals (`-0.8`, `-0.2`, `0.39105`), which fall outside this table |
| `core/constraints.py` | `constraints/constraints.py` | **yes** — `LinkConstraint.z_lb` reads `env_info['robot']['link_constraint_height']`; the iiwa's 0.25 m is meaningless on an arm whose gripper sits ~0.125 m above its base |
| `core/kinematics.py` | `utils/kinematics.py` | **yes** — `link_to_xml_name` reduced to the metal arm; `"7"` resolves to `gripper_base` (no seventh joint), `"ee"` is the grasp frame |
| `core/transformations.py` | `utils/transformations.py` | no |
| `core/position_control.py` | `environments/position_control_wrapper.py` | **yes** — only the `PositionControl` mixin is kept; `PositionControlIIWA` and the `IiwaPosition*` tasks are dropped |

`utils/universal_joint_plugin.py` is deliberately **not** vendored: the mallet is gripped, so
there is no passive universal joint to drive.

## Control stack — from `baseline/baseline_agent/`

The organizers' IROS-2021 baseline. This is where the EKF, the trajectory optimizer and the
tactic FSM come from.

| here | original | modified? |
|---|---|---|
| `estimation/kalman_filter.py` | `kalman_filter.py` | no — `PuckTracker`, `SystemModel`, `AirHockeyTable`. **Its fitted constants are still full-size** (see README) |
| `planning/optimizer.py` | `optimizer.py` | no — OSQP null-space differential IK |
| `planning/bezier_planner.py` | `bezier_planner_new.py` | **yes** — `matplotlib` was imported at module scope and forced the `tkAgg` backend, breaking any headless import; it is now imported inside `__main__`, where it is actually used |
| `planning/cubic_linear_planner.py` | `cubic_linear_planner.py` | no |
| `planning/trajectory_generator.py` | `trajectory_generator.py` | no |
| `fsm/system_state.py` | `system_state.py` | no — `TACTICS` enum, hysteresis counters |
| `fsm/tactics.py` | `tactics.py` | no — `Init`/`Ready`/`Prepare`/`Defend`/`Repel`/`Smash` |
| `agents/baseline_agent.py` | `baseline_agent.py` | no |

`agents/baseline_mini.py` is **new** — it subclasses the vendored baseline and overrides the
three places that hold full-size constants. The vendored files above are left byte-faithful
apart from import paths so that upstream changes stay easy to diff in.
