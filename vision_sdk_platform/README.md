# Vision puck tracking + SDK low-level controller

A self-contained MuJoCo air-hockey rig (1.016 × 0.508 m table, 6-DOF "metal arm" gripping
a mallet by friction) carrying two opt-in replacements for parts of the default control
stack:

| | replaces | switched on by | lives in |
|---|---|---|---|
| **Vision puck tracking** | the simulator's ground-truth puck pose feeding the agent's EKF | an agent kwarg | `mini_platform/vision/`, `estimation/kalman_filter.py`, `fsm/system_state.py`, `agents/baseline_mini.py` |
| **SDK low-level controller** | the challenge stack's hand-tuned joint PD | the task name (`-sdk` suffix) | `mini_platform/core/sdk_control.py`, `core/sdk_params.py`, `envs/position_control.py` |

They are independent — use either, both, or neither. Both default to **off**, and with both
off the package is byte-identical to the pre-vision baseline.

**Self-contained.** Nothing here imports the parent Robot Air Hockey Challenge repo. The
framework and baseline are vendored under `core/`, `estimation/`, `planning/` and `fsm/`,
with provenance and modifications recorded in [`THIRD_PARTY.md`](THIRD_PARTY.md) (MIT,
© 2025 Donat, Liu, Guenster, Tateo).

---

## Contents

```
pyproject.toml               install; the optional-dependency slicing is the one open decision
requirements.txt             the same pins, for a non-pip workflow
mini_platform/
  vision/                    camera model, detector, sensor          <- vision tracking
  estimation/kalman_filter.py  EKF: variable-dim update, NIS gate    <- vision tracking
  estimation/mini_env.py     the table + frames, mujoco-free, for the offline EKF tools
  hw/capture_puck_rolls.py   camera-only roll logger (EKF refit input)
  tools/fit_mini_table.py    fits damping / restitution / OBS_COV from those rolls
  tools/ekf_replay.py        scores get_prediction() offline: old vs refit constants
  fsm/                       tactic state machine + puck blackboard
  core/sdk_control.py        the SDK's joint law, ported to MuJoCo   <- SDK controller
  core/sdk_params.py         every SDK constant, with file:line cites <- SDK controller
  core/                      vendored framework (wrapper, constraints, kinematics, PD)
  envs/                      the tasks, incl. every `-sdk` variant
  agents/                    the baseline agent + its mini retarget
  planning/                  Bezier / cubic planners, OSQP null-space IK
  tools/                     offline evaluation; never imported from the runtime path
  assets/                    MJCF scenes and meshes — source of truth for all geometry
  artifacts/reports/         how each constant was measured
tests/                       50 tests
docs/
  vision_handoff.md          vision implementation record: every measured number,
                             all design deviations, calibration provenance
  vision_puck_assignment.md  the assignment the vision work answers
  handoff.md                 Part A = the SDK controller port; Part B = the rig
```

---

## 1. Install

```bash
conda create -n ahc python=3.11 -y && conda activate ahc
cd vision_sdk_platform
pip install -e '.[tools]'          # drop [tools] if you only need the controllers
pip install pytest                 # for the test suite
```

`pip install -e .` alone is enough to **run** both controllers — their runtime is
`mujoco` + `numpy` + `scipy` and nothing else. The extras are only for the offline
evaluation tools and the test suite; see the comments in `pyproject.toml`.

> The editable install matters: without it, `import mini_platform` only resolves when your
> working directory is this folder.

`mujoco` is pinned to 3.3.1 and `numpy` to `<2` deliberately: `mushroom-rl` 1.10.1 predates
NumPy 2, and newer mujoco wheels pull NumPy 2 in, which breaks `nlopt`.

### GL context (vision only)

Rendering needs an offscreen GL context. The SDK controller needs none.

```bash
export MUJOCO_GL=egl        # headless-safe, and what this rig is verified on
```

`glfw` also works on a desktop; `osmesa` is broken here. A cosmetic
`Exception ignored ... EGLError` may print at interpreter exit — the exit code stays 0 and
the pixels are correct. Anything in `docs/` claiming "egl fails" is stale.

---

## 2. Quick start

### Vision tracking

```python
import mini_platform
from mini_platform.agents.baseline_mini import MiniBaselineAgent

env = mini_platform.make("mini-hit")
agent = MiniBaselineAgent(env.env_info, agent_id=1,
                          puck_obs_source="vision",   # default "gt"
                          vision_env=env.base_env)    # sensor needs the LIVE scene

obs = env.reset()
agent.episode_start()
for _ in range(env.info.horizon):
    obs, reward, absorbing, info = env.step(agent.draw_action(obs))
    if absorbing:
        break
agent._sensor.close()                                 # releases the mujoco.Renderer
```

`vision_env` must be `env.base_env`, not `env_info`: the model inside `env_info` is an
arm-only FK scratchpad with no table, puck or cameras. Omit both kwargs and the agent is
byte-identical to the pre-vision baseline.

### SDK low-level controller

```python
import mini_platform
env = mini_platform.make("mini-defend-sdk")   # instead of "mini-defend"
```

Task names: `mini-hit-sdk`, `mini-defend-sdk`, `mini-prepare-sdk`, `mini-hit-sdk-2p`.
Everything above the controller is identical to the non-`-sdk` sibling — same planner, same
FSM, same optimizer, same `(2, 6)` action at 50 Hz — so switching the name is a controlled
A/B on exactly one variable.

### Both at once

```python
env = mini_platform.make("mini-defend-sdk")
agent = MiniBaselineAgent(env.env_info, agent_id=1,
                          puck_obs_source="vision", vision_env=env.base_env)
```

---

## 3. Vision tracking

### The pipeline

```
mujoco.Renderer (640x360, camera "table_side_left")
   -> detector.detect_puck        chromaticity-RATIO red mask -> fill_holes ->
                                  largest component -> area/bbox gates -> subpixel
                                  centroid; blue yaw marker searched STRICTLY inside
                                  the filled red component
   -> sensor.measure()            back-project through PinholeCamera onto the puck-top
                                  plane -> calibrated bias correction -> yaw from the
                                  15.5 mm marker lever arm -> pose-dependent covariance
                                  R_xy = sigma_px^2 * J J^T + sigma_floor^2 * I
   -> EKF (estimation/kalman_filter.py)   variable-dim update (2-D when yaw is
                                  unavailable), NIS gate, dropout handling
   -> fsm/system_state.py         puck state the tactics read
```

Ratios, never absolute thresholds: the puck material has `specular=0.5`, so highlights push
all channels toward white — that escapes an absolute gate but cleanly fails a ratio test,
producing a hole in the mask (refilled) rather than a false blob.

### Sensor API

```python
from mini_platform.vision.sensor import PuckVisionSensor

sensor = PuckVisionSensor(env.base_env._model, env.base_env._data,
                          camera="table_side_left", resolution=(640, 360))
m = sensor.measure()          # -> PuckMeasurement | None  (None = dropout)
sensor.close()
```

`PuckMeasurement` fields: `pos_xy` (2, world frame), `yaw` (`None` when the marker is not
measurable), `R_xy` (2×2 world-axis covariance), `R_yaw`, `n_pixels`, `clipped` (blob
touched the image border → `R` carries an extra `sigma_border = 4 mm` per axis; the
measurement is downweighted, not discarded).

Measurements are in the **world** frame. `MiniBaselineAgent` converts W → robot → "V" frame
internally; if you consume the sensor directly you own that conversion.

### Measured accuracy

Static sweep, 640×360, camera at the physical rig height (0.3341 m above the surface):

| metric | result | bar |
|---|---|---|
| detection (unoccluded, in-frustum) | 100% | 100% |
| position RMSE, centre / far corners | 0.77 / 1.07 mm | < 3 / < 8 mm |
| yaw availability | 85.0% | ≥ 60% |
| yaw RMSE where available | 0.040 rad | < 0.1 rad |
| predicted √tr(R) / empirical | 1.00–1.28× | 0.5–2× |

Closed loop, 20 episodes, seed 0: detection 100%, EKF position RMSE 2.17 mm (`mini-hit`) /
4.13 mm (`mini-defend`), success rates matching ground truth within the 0.10 bar. The side
camera covers 97.7% of the surface; out-of-frustum steps are classified and reported
separately, not silently averaged in.

### Calibration

Four constants — `sigma_px`, `bias_radial`, `bias_image`, `sigma_border` — are baked into
`PuckVisionSensor.__init__` defaults with a provenance comment. **Re-fit after any change to
the puck geometry, the camera pose, or the marker:**

```bash
MUJOCO_GL=egl python -m mini_platform.tools.vision_eval --mode static
# bake the printed absolute fits into vision/sensor.py -> re-run -> deltas ~0, bars PASS
```

`--resolution 960x540` is the one-parameter escalation if yaw availability ever drops below
60% after a scene change (the bias model transfers; `sigma_px` must be re-fit).

---

## 4. SDK low-level controller

### What it reproduces

Per joint, `mini_platform/core/sdk_control.py` runs the METAL arm SDK's
`NRT_JOINT_POSITION` law:

```
HOST, every 2.5 ms (400 Hz), zero-order held until the next tick:
    qd_ff  = qd_meas with |qd| < 0.05 rad/s zeroed
    tau_ff = Kg*G(q) + Kc*C(q,qd_ff) + b*clamp(qd_ff)
    tau_ff = clip(tau_ff, -tmax, +tmax)
    q_des  = clip(q_ref, position_min, position_max)
    -> quantised into one CAN frame (16-bit q; 12-bit qd/tau/kp/kd)

MOTOR FIRMWARE, every 1 ms substep:
    tau = kp*(q_des - q) + kd*(qd_des - qd) + tau_ff
    tau = clip(tau, -tmax, +tmax)
```

Every constant in `core/sdk_params.py` carries a `file:line` citation into the SDK's C++
(`metal_sdk/native/`, not shipped here). Nothing is tuned, fitted or scaled. The gains are
2–5× stiffer than the legacy PD on J1–J4 and 2–2.4× softer on J5/J6, with 2–4× less `kd`
everywhere — only sane because the feedforward removes gravity, Coriolis and viscous drag
before the PD ever sees them.

`SdkJointControl` replaces `PositionControl._controller` and `_enforce_safety_limits`, and
nothing else. The interpolator, the action-shape contract and `reset()` are the vendored
ones.

### Configuring it

Kwargs flow through `make()` to the environment class:

```python
import mini_platform
from mini_platform.core import sdk_params

# the leader-follower gain set (RT_JOINT_POSITION) instead of the NRT default
env = mini_platform.make("mini-defend-sdk", gain_set="follow")

# or a fully explicit config
cfg = sdk_params.nrt_config(inertia_ff=False,   # DEVIATION 1 off -> the literal SDK law
                            quantize=False,     # skip the CAN quantisation
                            hold_hz=None,       # re-latch every substep (upper bound, not hw)
                            safety="ros")       # add the legacy k=20 governor
env = mini_platform.make("mini-defend-sdk", sdk_config=cfg)
```

`SdkConfig` also exposes every gain/limit array (`kp`, `kd`, `tmax`, `vmax`, `pmax`,
`pos_min/max`, `k_gravity`, `k_coriolis`, `b_viscous`, `visc_vel_min/max`, `vel_deadband`)
if you want to perturb one.

### The three deviations from the SDK

All of them are in `core/sdk_params.py`, all opt-out:

1. **`inertia_ff=True`** — the SDK computes `M(q)·q̈` and never calls it. On this plant
   removing it takes defend mallet drops from 1/60 to 7/60. The sim's `M` is 84–99.5%
   armature on J4–J6, so the term is far larger here than on the real arm.
   `inertia_ff=False` for the literal SDK law.
2. **`qd_cmd` is clamped to ±`vmax` before quantisation.** The SDK does not clamp, so an
   out-of-range value wraps through `float_to_uint`. Reproducing an integer wrap would
   inject garbage, not fidelity.
3. **Output torque clipped to `min(SDK tmax, the MJCF's actuator ctrlrange)`.** MuJoCo clips
   `data.ctrl` regardless; doing it here only makes the saturation observable.

**Not ported:** the SDK's own 400 Hz quintic re-planner and its ~27 ms transport lag. The
planner and `PositionControl`'s cubic interpolator keep the 50 Hz `(2, 6)` contract.
Recorded but unused constants (`TRAJ_LIMITS`, `POINT_LIMITS`, `B_COULOMB_DISABLED`) are in
`sdk_params.py` so the omission is explicit and reversible.

### Not verified against hardware

`JOINT_SIGN` / `JOINT_OFFSET` are identity because nothing in the SDK establishes a joint
sign or zero map — `GetJointPosition()` returns the motor angle verbatim, so the SDK itself
assumes identity. A calibration run that contradicts this is a one-line change in
`sdk_params.py`.

Full A/B against the legacy PD:
[`mini_platform/artifacts/reports/04_sdk_controller_port.md`](mini_platform/artifacts/reports/04_sdk_controller_port.md).

---

## 5. Verify

```bash
# Everything (~90 s). Rendering tests skip cleanly where there is no GL.
MUJOCO_GL=egl pytest tests/ -q                      # expect 50 passed

# SDK controller only -- no display, no planner, ~2 s
pytest tests/test_sdk_control.py -q                 # 9 tests

# Vision only
MUJOCO_GL=egl pytest tests/test_vision_detector.py tests/test_vision_integration.py -q
pytest tests/test_kalman_variable_dim.py tests/test_puck_dynamics.py -q
```

| file | what it pins |
|---|---|
| `test_sdk_control.py` | the 400 Hz latch pattern, `G`/`C`/`M·q̈` bit-identity against `mj_forward`, the feedforward formula across every velocity branch, CAN quantisation, per-agent gain tiling, the position clamp, that the 3 N grip survives |
| `test_vision_detector.py` | camera round-trip, projection-vs-blob, mask behaviour, per-pose accuracy |
| `test_kalman_variable_dim.py` | 2-D/3-D update, the NIS gate, the re-lock, and the **bitwise ground-truth invariance pin** |
| `test_vision_integration.py` | agent-level wiring, incl. the W→R→V frame composition |
| `test_puck_dynamics.py` | the 50×7 mm puck's damping constants and fitted deceleration slopes |
| `test_ekf_params.py` | the `--ekf-params` load hook: off-by-default is the vendored constants *on the same objects*, on-when-asked reaches all five consumers |
| `test_fit_mini_table.py` | the refit recovers a known damping / restitution / noise from synthetic rolls, and never invents the spin row it cannot measure |
| `test_ekf_replay.py` | the offline prediction score: runtime-faithful cadence, no scoring across dropouts, and capture→fit→load→replay end to end |
| `test_mini_env.py` | that the mujoco-free copy of the table geometry has not drifted from `env_base` / the live `env_info` |
| `test_two_player.py`, `test_match_controls.py` | the two-player variants and the viewer's control strip |

The GL vision tests take `--show-vision`, which pops the annotated camera view at each
rendered assertion (off by default, so CI is untouched):

```bash
MUJOCO_GL=egl pytest tests/test_vision_detector.py tests/test_vision_integration.py \
    --show-vision -q
```

---

## 6. The evaluation tools

Outputs land in `mini_platform/artifacts/vision/` (gitignored, regenerable).

```bash
# 1. Snapshot: 5 teleported poses -> PNGs + red-pixel counts. Expect "snapshot: PASS".
MUJOCO_GL=egl python -m mini_platform.tools.vision_eval --mode snapshot

# 2. Static accuracy map: 21x9 grid x 4 yaws + calibration refit (~1 min).
#    Expect all 6 bars PASS (the table in section 3).
MUJOCO_GL=egl python -m mini_platform.tools.vision_eval --mode static

# 3. Closed loop, same-seed GT vs vision agent (~25 min). Expect "closedloop: PASS".
MUJOCO_GL=egl python -m mini_platform.tools.vision_eval --mode closedloop \
    --episodes 20 --tasks mini-hit,mini-defend --seed 0

# 4. Live view: the vision agent playing, camera feed annotated -- green outline + box
#    around the detected puck, green centroid cross, yellow marker cross, red border on
#    dropout frames. Needs matplotlib and a display.
MUJOCO_GL=egl python -m mini_platform.tools.vision_eval --mode live \
    --task mini-defend --episodes 3

# 5. Headless recording: same annotations, no window. With ffmpeg on PATH frames are
#    STREAMED to a real-time-paced MP4 at 50 fps (constant memory, fine for 100+
#    episodes); without ffmpeg it falls back to an in-RAM GIF.
MUJOCO_GL=egl python -m mini_platform.tools.vision_eval --mode live \
    --task mini-hit --episodes 100 --record
```

SDK controller side:

```bash
# Per-joint closed-loop identity: gravity hold, step response (f_n / zeta), saturation duty
python -m mini_platform.tools.mini_ctrl_bench --tasks mini-defend,mini-defend-sdk

# Acceptance gate: success rate, mallet drops, constraint violations, grip slip
python -m mini_platform.tools.mini_env_smoke --episodes 60 --tasks mini-defend,mini-defend-sdk

# Watch both arms play on the SDK law (needs a display)
python -m mini_platform.tools.mini_play --task mini-hit-sdk-2p --episodes 50
```

Viewer controls: `SPACE` pause · left-drag orbit · right-drag pan · scroll zoom · `C`
contact forces · `T` transparency · `1` hide visual meshes · `TAB` cycle camera · `ESC`
quit. The bottom-left strip gives PLAY / PAUSE / RESET / HIT PUCK (keys `r`, `x`).

---

## 7. Gotchas

**The live window can never be smooth.** Each frame waits for the planner's own
wall-clock-bounded solve (10–100 ms on replans). Use `--record` for smooth playback.

**`mini-defend` is not byte-reproducible, even against itself.** `planning/optimizer.py`
bounds every solve by wall clock (`opt.set_maxtime(5e-3)` plus a `time.time()` bound in the
null-space loop), so identical output needs an idle machine. `mini-hit` / `mini-prepare` do
reproduce byte-identically. Any A/B of agent behaviour — vision vs GT, SDK vs PD — is only
clean under matched machine load. The hard guarantee lives in
`tests/test_kalman_variable_dim.py::test_gt_regression_bitwise`, which is load-independent.

**The NIS gate is keyed to the measurement channel.** `R=None` (trusted ground truth) takes
the original unconditional update, so GT invariance is structural; only measurements
carrying a covariance are gated. Gating the GT path fired 89 times over 20 episodes and
visibly changed GT behaviour.

**Yaw dropping to `None` is designed, not a fault.** The blue marker is gated strictly to
the filled puck component because the scene's blue-grey floor checker passes a plain blue
ratio test. Marker outlines that alias open yield `yaw=None` and the EKF takes its 2-D
branch. 85% availability against a 60% bar.

**Non-editable installs write into `site-packages`.** `ARTIFACT_DIR` is resolved relative to
the package, so `vision_eval` writes its outputs next to the installed code. Use the
editable install for any run that produces artifacts.

**`env_info['robot']['robot_model']` must satisfy `nq == nv == nu == n_joints`.** That is why
`assets/metal_arm_only.xml` exists with exactly 6 joints, instead of reusing `arm1.xml`
(8 joints, incl. the gripper). Violating that contract fails at the first step — loudly, but
only at the first step.
