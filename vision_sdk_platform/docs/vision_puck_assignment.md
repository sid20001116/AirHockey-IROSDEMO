# Assignment: Vision-Based Puck Tracking

Replace the simulator ground-truth puck pose with camera-based detection feeding the
existing EKF, using the fixed side camera as the primary sensor and (optionally) the
wrist camera to cover mallet occlusion.

**Spec:** red puck, 50 mm diameter × 7 mm height. Fixed camera at the middle of the
long rail (`table_side_left`), wrist camera on the gripper (`metal_1/wrist_cam`).
Detections feed `PuckTracker` (the EKF) instead of `mjData.qpos`.

All paths are relative to `mini_platform/` unless prefixed with `../`.

---

## 0. Background — read this before writing code

### What already exists (do not rebuild)

| Asset | Where | Notes |
|---|---|---|
| Side camera `table_side_left` | `assets/mini_table.xml:102-104` | Mid +y long rail, 0.25 m up, 45° down, fovy 80, 94.7% surface coverage |
| Wrist camera `metal_1/wrist_cam` | `assets/arm1.xml:85-87` | Straight down over mallet, ±8 cm footprint. **Auto-generated block between `WRIST-CAM BEGIN/END` sentinels — never hand-edit; regenerate with `../scripts/place_wrist_cam.py`** |
| Offscreen buffer sizing | `assets/mini_table.xml` `<visual><global offwidth="1920" offheight="1080"/>` | Any render resolution ≤ 1080p works |
| Working `mujoco.Renderer` examples | `../scripts/place_wrist_cam.py` (`measure()`), `tools/mini_mallet_check.py:127-146` | Named-camera rendering from a live env via `env.base_env._model/._data` |
| EKF with rim-bounce model | `estimation/kalman_filter.py` | State `[px,py,vx,vy,θ,ω]`; measurement currently hard-wired to `[x,y,yaw]` |
| Design notes that anticipated this task | `docs/handoff.md` §7b; `../docs/rl_dr_train_plan.md` §0.1–0.4 | Hazards T2.2 (pose-dependent R), H1 (parallax), H2 (occlusion) |

### The data path you are cutting into

```
mjData.qpos[puck_x/y/yaw]                     (world frame W)
  → obs[0:3] via observation_spec              envs/env_base.py:136-141
  → _modify_observation: W → robot base R      envs/env_single.py:63-79
  → BaselineAgent.draw_action                  agents/baseline_agent.py:71-72
  → MiniBaselineAgent.get_puck_pos: R → V      agents/baseline_mini.py:421-423
  → SystemState.update_observation             fsm/system_state.py:109
  → PuckTracker.step(measurement)              fsm/system_state.py:121   ← THE swap point
  → get_prediction → tactics → trajectory
```

Three planar frames: **W** (world = table center, surface top z=0), **R** (robot base,
measured on hardware, `env_info['robot']['base_frame']`), **V** (virtual end-mount,
`baseline_mini.py:113-121`). The EKF and all tactics live in V. A camera measurement
is born in W, so it must travel W → R (`AirHockeyBase._puck_2d_in_robot_frame`,
`envs/env_base.py:379-400`) → V (`self._r2v`, a `PlanarXform`, `baseline_mini.py:77-110`).

Two facts that shape the design:
- `get_puck_vel` is **never called** at runtime — the agent's velocity knowledge comes
  entirely from the EKF state. Swapping only the position measurement is sufficient.
- `is_absorbing`/`check_success` read the **unmodified** obs, so leaving ground truth
  in the observation vector keeps env logic and eval logging intact for free.

Timing: physics dt 1 ms × 20 intermediate steps = **50 Hz control**, one measurement
per step (`envs/env_base.py:124,212`). Keep control at 50 Hz; the camera renders once
per control step.

### Libraries

- `mujoco` (offscreen `Renderer`) — installed 3.8.1, requirements pin 3.3.1
- `numpy`, `scipy.ndimage` — already dependencies
- **No OpenCV.** The four image ops needed (threshold, fill-holes, connected
  components, centroid) are a few numpy/scipy lines each
- `pillow` — optional, debug PNGs in tools only (repo convention)
- GL backend: `MUJOCO_GL=egl` headless / CI; `glfw` on a desktop. Library code never
  sets it; tools catch renderer-creation failure and print the hint. Don't mix the
  interactive viewer and offscreen rendering in one process

### New code layout

```
mini_platform/
  vision/
    __init__.py
    camera.py        # pinhole model: intrinsics, extrinsics, ray/plane, project, FD Jacobian
    detector.py      # red/blue segmentation → subpixel centroids
    sensor.py        # Renderer lifecycle + measure() → PuckMeasurement | None
  tools/
    vision_eval.py   # --mode snapshot | static | closedloop
  tests/
    test_puck_dynamics.py
    test_vision_detector.py
    test_kalman_variable_dim.py
```

Rule for every part: **ground truth stays the default; the sim runs exactly as today
at every commit.** Vision becomes reachable only behind a flag in Part 4.

---

## Part 0 — Resize the puck to 50 × 7 mm (no vision code)

The sim puck is currently 32 mm ⌀ × 10.1 mm, and its *visible* red surface is only a
3 mm wafer at the bottom (the collision geom is transparent, `rgba="0 0 0 0"`). You
will resize it and make the red site cover the full cylinder so the **top face**
becomes the detected surface — it back-projects onto the exactly-known plane
z = 0.007, which by construction removes the ~30 mm parallax hazard (handoff H1).

### Task

Edit `assets/mini_table.xml:75-86` and `envs/env_base.py:32`.

Scaling laws — the current inertial block is exactly a uniform cylinder
(ρ = 158.9 kg/m³); keep density:

- m′ = 0.0012896 · (0.025/0.0159904)² · (0.007/0.0101045) = **0.0021838 kg**
- Izz′ = ½ m′R′² = **6.8244e-7**;  Ixx′ = Iyy′ = m′(3R′² + H′²)/12 = **3.5014e-7**
- Damping: the invariant is **deceleration**, not force (the XML's own comment,
  `mini_table.xml:66-74`): linear 0.5 s⁻¹, yaw 0.4 s⁻¹ →
  `damping = 0.5·m′ =` **1.0919e-3** (puck_x, puck_y), `0.4·Izz′ =` **2.7298e-7**
  (puck_yaw). Never carry the literals; update the comment's numbers too.

```xml
<body name="puck">
    <geom pos="0 0 0.0035" name="puck" type="cylinder" rgba="0 0 0 0"
          size="0.025 0.0035" condim="4" priority="0"/>
    <!-- red site covers the FULL cylinder: top face = detection surface at z=0.007 -->
    <site pos="0 0 0.0035" name="puck_site" type="cylinder" material="red"
          size="0.025 0.0035"/>
    <!-- marker top sits 0.1 mm proud of the red top: renders on top, no z-fighting -->
    <site pos="0.0155 0 0.0035" name="puck_marker" type="cylinder" material="blue"
          size="0.008 0.0036"/>
    <inertial pos="0 0 0" mass="0.0021838"
              diaginertia="3.5014e-07 3.5014e-07 6.8244e-07"/>
</body>
```

`PUCK_RADIUS = 0.025` at `envs/env_base.py:32` is the **single Python constant**;
`env_info['puck']['radius']` (EKF bounce boundary, `fsm/tactics.py`, spawn ranges)
all flow from it.

### Definition of done

1. `python -m mini_platform.tools.mini_mallet_check` → geometry / hold / rebound all
   PASS (contact diameter becomes 0.050; no tunnelling at 2.5 m/s).
2. New `tests/test_puck_dynamics.py` pins the retune: load `mini_single.xml` raw, set
   `puck_x` qvel = 1.0, `mj_step` for 1 s, fitted slope of ln|v| = **−0.5 ± 5%**;
   same for yaw at ω₀ = 10 → **−0.4 ± 5%**.
3. `python -m mini_platform.tools.mini_env_smoke --episodes 20` on hit / defend /
   prepare, compared against the same command run *before* the change: success rates
   within noise, no new constraint violations.

---

## Part 1 — Camera model + rendering harness

### Task

Build `vision/camera.py` and the shell of `vision/sensor.py`.

**Sensor:** `PuckVisionSensor(model, data, camera="table_side_left",
resolution=(640, 360))` owns **one persistent `mujoco.Renderer`**, lazily created,
shared across cameras (`update_scene(data, camera=name)` per frame), with `close()`.
Callers pass `env.base_env._model` / `._data` (the `mini_mallet_check.py` pattern)
and call `measure()` after `env.step()` returns. Read the camera pose from
`data.cam_xpos` / `data.cam_xmat` **every frame** — constant for the fixed camera,
and later gives the wrist camera live kinematics for free.

**Resolution 640×360 by default** (constructor parameter). It is 1/9 the pixels of
the 1080p hardware spec (`rl_dr_train_plan.md`: 1080p is a hardware spec, not a
runtime resolution). At f = 180/tan(40°) ≈ 214.5 px the puck is ~28 px across
mid-table, ≥13 px at the far corner; the blue marker is 6–9 px mid-table and ~4 px at
the far corners (that's what the yaw fallback is for). If Part 2's yaw-availability
map disappoints, bump to 960×540 — one parameter.

**Camera math** (MuJoCo conventions: `fovy` is vertical, pixels square, the camera
looks along **−Z** of its frame, +Y is image-up):

```python
f  = (H / 2) / np.tan(np.radians(model.cam_fovy[cid]) / 2)  # from RENDER size,
cx, cy = (W - 1) / 2, (H - 1) / 2                           # not cam_resolution
p   = data.cam_xpos[cid]                        # camera center, world frame
Rwc = data.cam_xmat[cid].reshape(3, 3)          # cols: image-right, image-UP, backward

def pixel_to_ray(u, v):                         # v grows downward in the image
    d = Rwc @ np.array([(u - cx) / f, -(v - cy) / f, -1.0])
    return d / np.linalg.norm(d)

def intersect_z_plane(p, d, z_p=0.007):         # puck TOP plane
    t = (z_p - p[2]) / d[2]                     # reject if d[2] >= -1e-9 or t < 0
    return p + t * d                            # measurement = X[:2], world frame

def project(X):                                 # world point → pixel (tests, wrist gate)
    Xc = Rwc.T @ (X - p)                        # reject if Xc[2] >= 0 (behind camera)
    return cx + f * Xc[0] / -Xc[2], cy - f * Xc[1] / -Xc[2]
```

Also implement a finite-difference Jacobian `J = ∂X_xy/∂(u,v)` (2×2, central
differences at ±½ px — four extra ray-plane intersections) for Part 2's covariance.

Create `tools/vision_eval.py` with `--mode snapshot`: teleport the puck to 5 grid
positions, save PNGs to `artifacts/`, print red-pixel counts as a smoke check.

### Definition of done

- Snapshot PNGs show the red puck and blue marker at all 5 positions.
- pytest (module-scope env fixture per `tests/test_two_player.py`, GL-skip guard for
  the rendering half): round-trip `project(intersect_z_plane(pixel_to_ray(u,v)))`
  returns (u, v) to 1e-6; `project(gt_puck_top_center)` lands within 2 px of the
  rendered blob centroid at one mid-table pose.

---

## Part 2 — Detector, back-projection, and the static accuracy map

### Task

Implement `vision/detector.py` and complete `sensor.measure()` so it returns
`PuckMeasurement(pos_xy_world, yaw_or_None, R_xy_world, R_yaw, n_pixels)` or `None`.

**Segmentation — chromaticity ratios, not raw RGB, not HSV.** The puck material has
`specular=0.5`; highlights push pixels toward *white* (all channels rise together),
which escapes absolute thresholds like the `(R>110)&(G<90)&(B<90)` in
`place_wrist_cam.py` but cleanly fails a ratio test — producing a hole in the mask,
not a false blob. HSV buys nothing extra for a pure-red material at ~3× the cost.

```python
im = rgb.astype(np.int16)
R, G, B = im[..., 0], im[..., 1], im[..., 2]

red = (R > 60) & (R > 1.6 * G) & (R > 1.6 * B)
red = ndimage.binary_fill_holes(red)            # refill specular holes (centroid bias)
lbl, n = ndimage.label(red)
# keep the largest component; sanity gates: area >= 20 px,
# bbox major dimension in [4, 90] px — else return None (no detection)
ys, xs = np.nonzero(lbl == largest)
c_px = (xs.mean(), ys.mean())                   # subpixel centroid

# blue marker: search ONLY inside the dilated red bbox — the blue-grey floor
# checker passes a naive blue test; area >= 3 px, else yaw = None
blue = (B > 50) & (B > 1.5 * R) & (B > 1.2 * G)
```

**Back-projection:** both centroids → z = 0.007 plane;
`yaw = atan2(Pm[1] - P[1], Pm[0] - P[0])` (the marker sits on the puck's local +x, so
this matches `puck_yaw` exactly). Sanity: back-projected marker offset must be in
[0.3·R, 1.1·R] (7.5–27.5 mm) or discard the yaw.

**Wall-pixel correction:** side-wall pixels pull the centroid toward the camera.
Correct with `X += κ · R_puck · û` (û = horizontal camera→puck unit vector); κ is a
single scalar you calibrate from the static sweep below (default 0, expect 0.05–0.1).

**Pose-dependent covariance** (this is handoff hazard T2.2 — the ground-sample
distance runs 2.8 mm/px mid-table to ~12 mm/px at the far corners, so a constant R is
~4× wrong and mis-oriented):

```python
J = fd_jacobian(u, v)                        # 2×2, from Part 1
R_xy  = sigma_px**2 * (J @ J.T) + sigma_floor**2 * np.eye(2)   # world axes
R_yaw = (np.trace(R_c) + np.trace(R_m)) / (2 * d**2)           # d = 0.0155 m marker offset
R_yaw = max(R_yaw, 1.66e-3)                  # floor at OBS_COV's fitted yaw variance
```

Start σ_px = 1.0 px, σ_floor = 1 mm; recalibrate σ_px from the sweep's empirical
scatter.

**Static sweep** (`vision_eval.py --mode static`): teleport the puck over a dense
grid × several yaws (write `puck_x/puck_y/puck_yaw` qpos, `mj_forward` — the
`place_wrist_cam.measure()` pattern; mallet parked at home), record detection rate,
position error vs GT, yaw availability/error, and predicted-vs-empirical scatter.
Fit and report κ and σ_px.

### Definition of done

- Detection rate 100% on the unoccluded grid; position RMSE < 3 mm for |x| ≤ 0.3,
  < 8 mm at the far corners.
- Yaw available on ≥ 60% of the grid; RMSE < 0.1 rad where available.
- Predicted √tr(R) within 2× of empirical scatter in every cell.
- `tests/test_vision_detector.py` green: detection + <5 mm at ~8 canonical poses, yaw
  <0.15 rad at 4 yaws mid-table, `measure() is None` with the puck teleported
  off-frustum.

---

## Part 3 — EKF: variable-dimension update, gating, dropout (pure unit work)

No rendering here — this part is independent of Parts 1–2 and is pure numpy.

### Task

Modify `estimation/kalman_filter.py` and `fsm/system_state.py` (both use the `MINI:`
change-marker convention). **The default GT path must stay numerically identical.**

- `update(measurement, predicted_state, P, R=None)` — `R = self.R if R is None`.
  If `len(measurement) == 2`: use `H2 = self.H[:2]` and `R[:2, :2]`, xy innovation
  only. Else the existing 3-D path with the wrapped yaw innovation
  (`kalman_filter.py:194-202`).
- `step(measurement, R=None)` — if `measurement is None`: predict-only
  (`self.state, self.P = self.predict(...)`). Otherwise gate, then update:

```python
y  = innovation(measurement, predicted_state)        # 2- or 3-dim, yaw wrapped
S  = H @ P @ H.T + R
nis = y @ np.linalg.solve(S, y)
if nis > (13.82 if len(y) == 2 else 16.27):          # chi^2 at 99.9%
    self._reject_count += 1
    if self._reject_count >= 5:                      # diverged (occlusion + bounce):
        self.reset(np.append(measurement[:2],        # re-lock on raw measurement
                             measurement[2] if len(measurement) == 3 else 0.0))
    else:
        self.state, self.P = predicted_state, P      # predict-only this step
    return
self._reject_count = 0
# ... standard update
```

  Why the gate: the filter currently has **zero** outlier defense; a single
  mallet-edge misdetection would inject a phantom bounce.
- Reset semantics unchanged (`P = eye(6)` ≈ 1 m² variance is genuinely diffuse). When
  vision hasn't seen the puck at episode start, reset from a prior — own-half center
  in frame V ≈ `[0.367, 0, 0]`; the first real detection then passes the gate
  trivially and the near-unity gain snaps the state. No special-case logic needed.
- `fsm/system_state.py` — `update_observation(joint_pos, joint_vel, puck_state,
  meas_R=None)`: restart branch uses the prior when `puck_state is None`; line 121
  becomes `self.puck_tracker.step(puck_state, meas_R)`. The motion counters
  (`:124-136`) already read the filtered state — untouched.

### Definition of done — `tests/test_kalman_variable_dim.py`

(a) **GT regression:** a 200-step sequence of 3-D measurements with `R=None`
produces bitwise-identical state/P to the pre-change code. (b) A 2-D update leaves a
yaw-only innovation untouched. (c) `step(None)` grows tr(P) and moves the state
ballistically. (d) A 1 m outlier is rejected; 5 consecutive outliers force a re-lock
onto the new position. (e) Prior-reset converges within 3 steps of the first real
measurement. `cd mini_platform && pytest tests/ -q` fully green.

---

## Part 4 — Integration behind `puck_obs_source` + closed-loop evaluation

### Task

Wire the sensor into the agent — **agent-side**, the seam `handoff.md` §7b itself
designates. (Not env-side: `is_absorbing`/`check_success` read the unmodified obs so
env logic needs zero changes; GT stays in obs ids 0–5 for free eval logging; and only
vision agents pay the render cost — RL training and every existing tool keep the pure
env.)

```python
# agents/baseline_agent.py, draw_action (lines 71-72):
puck_meas, meas_R = self.get_puck_measurement(obs)          # MINI: vision hook
self.state.update_observation(self.get_joint_pos(obs), self.get_joint_vel(obs),
                              puck_meas, meas_R)

# BaselineAgent default — GT path byte-identical:
def get_puck_measurement(self, obs):
    return self.get_puck_pos(obs), None
```

`MiniBaselineAgent.__init__(..., puck_obs_source="gt", vision_env=None)`: on
`"vision"`, build `PuckVisionSensor(vision_env._model, vision_env._data)` and set
`agent_params['puck_prior']`. Then:

```python
def get_puck_measurement(self, obs):
    if self._puck_obs_source == "gt":
        return super().get_puck_measurement(obs)
    m = self._sensor.measure()
    if m is None:
        return None, None
    # frames: W → R → V, and rotate the covariance with the planar composite
    pos_R = AirHockeyBase._puck_2d_in_robot_frame(..., base_frame[0])   # W → R
    pos_V = self._r2v.pos2(pos_R); yaw_V = self._r2v.angle(m.yaw)       # R → V
    G = self._r2v.rot2 @ base_frame[0][:2, :2].T                        # Γ (planar)
    R_V = G @ m.R_xy_world @ G.T
    if m.yaw is None:
        return np.array([*pos_V]), R_V                                   # 2-D
    return np.array([*pos_V, yaw_V]), block_diag(R_V, m.R_yaw)           # 3-D
```

Puck **velocity** obs stays GT and is never read at runtime — the agent's velocity
knowledge comes entirely from the EKF, which is the point. v1 scope: single-agent
tasks (mini-hit / mini-defend / mini-prepare), agent_id 1.

**Closed-loop eval** (`vision_eval.py --mode closedloop`): per-step CSV of GT world
pose, detection/dropout, measurement, EKF state error (GT converted to V), occlusion
classification (dropout AND mallet within one mallet-radius of the camera–puck sight
line); summary RMSE / P95 / detection rate / dropout-run-length histogram / success
rate vs the GT agent on the same seeds.

### Definition of done

- `python -m mini_platform.tools.vision_eval --mode closedloop --episodes 20
  --tasks mini-hit,mini-defend`: detection > 97% outside occlusion segments; EKF
  position RMSE < 5 mm unoccluded; success rate within 0.10 of the GT agent on the
  same seeds.
- Integration pytest: vision agent runs 100 steps on mini-hit without exception, EKF
  position error < 2 cm on unoccluded steps (GL-skip guard).
- GT smoke run unchanged; full pytest suite green.

---

## Part 5 (extension, optional) — Wrist cam during side-cam occlusion

The ⌀50 mm near-black mallet head parks between the side camera and the puck exactly
at contact (handoff H2). The wrist cam's ±8 cm straight-down footprint covers
precisely that region. Mounting is done — this is sensor-fusion work only. (Reminder:
the camera block in `arm1.xml` is generated; retune only via
`../scripts/place_wrist_cam.py`.)

### Task

- Same detector, second named camera on the same `Renderer`; pose from
  `data.cam_xpos/cam_xmat` each frame (live kinematics for free).
- **Render gate** — don't pay double render cost every step: render the wrist cam
  only when the EKF-predicted puck (V → W) `project()`s inside the wrist image with
  margin and range < 0.35 m.
- **Trust gate** — stricter min blob area; back-projected point must lie within the
  footprint. The marker is ~90 px there, so yaw is nearly always available.
- **Sequential update** — one `predict`, then one `update()` per available
  measurement (independent noise). No new EKF math; let `step` accept a list.

### Definition of done

Occlusion-segment dropout run-lengths shrink (P95 gap < 3 steps vs side-only); EKF
error during contact windows measurably improves; side-only mode unchanged.

---

## Future work (explicitly out of scope)

Latency/jitter injection (handoff T2.3) · mirrored `table_side_right` camera (H2) ·
analytic projection Jacobians · calibrated σ_px refinement · two-agent vision · RL
observations from vision.

## Suggested order & effort

| Part | Depends on | Touches runtime path? | Rough effort |
|---|---|---|---|
| 0 — puck resize | — | XML + 1 constant | ½ day |
| 1 — camera model | 0 | no | ½–1 day |
| 2 — detector + accuracy map | 1 | no | 1–2 days |
| 3 — EKF changes | — (parallel with 1–2) | flag-neutral | 1 day |
| 4 — integration + eval | 2, 3 | yes, flag-gated | 1–2 days |
| 5 — wrist cam | 4 | opt-in | 1 day |

Part 0 must land first: every pixel-size gate, the marker dimensions, and the
z = 0.007 projection plane depend on the new puck. Parts 1–2 live entirely in tools
and tests. Part 3 lands its GT-regression pin *before* any behavior change. Part 4 is
the only runtime-path touch and defaults to GT.

**Final acceptance:** `MUJOCO_GL=egl python -m mini_platform.tools.vision_eval
--mode closedloop --episodes 20` meets the Part 4 bar, and
`cd mini_platform && pytest tests/ -q` is green.
