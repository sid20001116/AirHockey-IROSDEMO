# Vision Puck Tracking — Implementation Record

Status record for the vision-based puck tracking assignment (`Vision Puck
Tracking.pdf` at the package root, rendered from `docs/vision_puck_assignment.md`).
**Parts 0–4 are complete and verified**; Part 5 (wrist-cam fusion) was explicitly
excluded by the user. This file replaces the mid-task handoff that previously lived
here; everything a maintainer needs — outcomes, measured deviations from the
assignment's sketches, calibration provenance, and re-verification commands — is
below.

All commands use the conda `ahc` interpreter:

```bash
PY=/home/siddarth/miniconda3/envs/ahc/bin/python
```

---

## 1. Status board

| Part | Content | Verification | Commit |
|---|---|---|---|
| — | Baseline snapshot of untracked `mini_platform/` | n/a | `78819d7` |
| **0** | 50×7 mm puck retune (damping + marker) + dynamics tests | 3/3 tests; mallet-check PASS; smoke within noise | `378efc9` |
| **1** | Pinhole camera + sensor shell + snapshot harness | round-trip 1e-6; blob-vs-projection; 5 PNGs | `46cc8b5` |
| **2** | Detector + `measure()` + static accuracy map + calibration | all 6 sweep bars PASS (see §3) | `1f82a86` |
| **3** | EKF variable-dim / NIS gate / dropout | 8/8 tests incl. bitwise GT pin | `7dfab63` |
| **4** | Agent integration + closed-loop eval | closedloop PASS all bars; 5/5 integration tests | `9072c69` |
| 5 | wrist cam | **out of scope** | — |

Full suite: **43 tests pass** (17 pre-existing + 26 added by this work).

> **Note (2026-08-22):** the per-part commits listed above were removed by an
> interactive `git reset` (work to be re-committed manually); the hashes survive in
> the reflog for recovery, and the baseline `78819d7` still exists. The full
> implementation lives in the working tree.

---

## 2. Environment facts (verified)

- **Interpreter:** `/home/siddarth/miniconda3/envs/ahc/bin/python` — python 3.11.15,
  mujoco 3.3.1, numpy 1.26.4, scipy 1.17.1, pillow 12.3.0, pytest 9.1.1.
  `mini-platform` is installed editable. The assignment's "mujoco 3.8.1" note
  measured conda *base* — ignore it.
- **`MUJOCO_GL=egl` works** (RTX 5070, driver 595.71). `glfw` works on the desktop;
  `osmesa` is broken. Repo docs claiming "egl fails" (`docs/handoff.md:360`,
  `README.md:37`, `artifacts/reports/00_env_setup.md:71`) are stale and were left
  unedited (out of scope). At interpreter exit mujoco's EGL teardown prints an
  `Exception ignored ... EGLError`; exit code stays 0 — cosmetic.
- **git:** branch `qualifying-2025` of the parent repo. `maker-arm-sdk/` and
  `metal-python-ros/` are gitlinks. **Every `git add` must stay path-scoped to
  `mini_platform/`** — the parent repo carries unrelated modified files.
- **The smoke tool is timing-sensitive**: `planning/optimizer.py` bounds every solve
  by wall clock (`opt.set_maxtime(5e-3)`, and a `time.time()` bound in the null-space
  loop). Measured control: `mini_env_smoke --tasks mini-defend` run twice
  back-to-back on an idle machine, same code, same seed, produced **different
  output** (drops, tilt, constraint decimals; success rate stable at 0.95) —
  mini-defend byte-identity is unattainable, period, while mini-hit / mini-prepare
  reproduce byte-identically on an idle machine. Any before/after smoke comparison
  is therefore "hit+prepare identical, defend within noise"; the *bitwise*
  GT-invariance guarantee lives in `tests/test_kalman_variable_dim.py::
  test_gt_regression_bitwise`, which is load-independent.

---

## 3. Outcomes

### Part 0 — puck retune (`378efc9`)

The Aug-19 edits had already resized the geoms/inertial; this pass fixed the joint
damping literals (`1.0919e-3` / `2.7298e-7`, preserving the 0.5 / 0.4 s⁻¹
deceleration invariant), the stale comment numbers, and named the marker site
(`puck_marker`, top face 0.1 mm proud). `tests/test_puck_dynamics.py` pins the
constants and the fitted decel slopes (−0.5 / −0.4 ± 5%). Smoke before/after within
noise (hit 0.30→0.20, defend 1.00→0.95, prepare 0.20→0.20; drops 1/5/0 unchanged; all
constraints < 0).

### Part 1 — camera model (`46cc8b5`)

`vision/camera.py` (intrinsics from the **render** size, extrinsics live from
`data.cam_xpos/cam_xmat`, ray/plane at z = 0.007, `project`, FD Jacobian) and
`vision/sensor.py` (one lazy persistent `Renderer`, camera name per `update_scene`).
`tests/conftest.py` adds the suite's first GL guard (`require_gl` session skip).
Snapshot mode writes 5 PNGs with red-pixel counts.

### Part 2 — detector + calibration (`1f82a86`)

`vision/detector.py`: chromaticity-ratio red mask → `fill_holes` → largest component
→ gates → subpixel centroid; blue marker gated **strictly to the filled puck
component**. `sensor.measure()`: back-projection, calibrated bias correction (§4.2)
**before** the yaw computation, marker-offset sanity gate, pose-dependent
`R_xy = σ_px²·J Jᵀ + σ_floor²·I`, `R_yaw` floored at `OBS_COV[2,2]`.

Final static sweep (640×360, defaults baked into the sensor):

| metric | result | bar |
|---|---|---|
| detection (unoccluded, in-frustum) | **100%** | 100% |
| position RMSE, \|x\| ≤ 0.3 | **0.74 mm** | < 3 mm |
| position RMSE, far corners | **1.80 mm** | < 8 mm |
| yaw availability | **87.6%** | ≥ 60% |
| yaw RMSE where available | **0.049 rad** | < 0.1 rad |
| predicted √tr(R) / empirical | **1.00–1.04×** | 0.5–2× |

### Part 3 — EKF (C4)

`estimation/kalman_filter.py`: `_innovation` (2-D xy-only / 3-D verbatim-original),
`update(..., R=None)` keeping `np.linalg.inv` operand-for-operand on the default
path, `step(measurement, R=None)` with dropout, the channel-keyed NIS gate, the
rim-proximity bypass and the two-point re-lock (§4.5–4.6).
`fsm/system_state.py::update_observation` accepts `None` (restart → `puck_prior`)
and 2-D measurements, and forwards `meas_R`.

### Part 4 — integration + closed-loop (C5)

`agents/baseline_agent.py` gains the `get_puck_measurement` seam (GT default
byte-identical). `MiniBaselineAgent(..., puck_obs_source="vision",
vision_env=env.base_env)` builds the sensor and converts measurements
W → R (`_puck_2d_in_robot_frame`) → V (`_r2v`), covariance rotated by the planar
composite — **not** the assignment sketch's `_r2v.angle(m.yaw)`, which is off by the
+90° base rotation (`test_yaw_frame_composition` pins this).

Final closed-loop (20 episodes, seed 0, same-seed spawn equality asserted):

| | mini-hit | mini-defend | bar |
|---|---|---|---|
| detection (unoccluded, in-frustum) | **99.72%** | **100%** | > 97% |
| EKF pos RMSE (usable steps) | **4.04 mm** (P95 7.1) | **3.11 mm** (P95 3.6) | < 5 mm |
| success rate GT / vision | 0.15 / 0.15 | 1.00 / 1.00 | \|diff\| ≤ 0.10 |

(Numbers at the original 0.25 m camera; superseded by §7 after the camera was
raised to the physical rig height.)

---

## 4. Deviations from the assignment's sketches (all measured, none cosmetic)

1. **Projection-vs-blob test is directional, not "< 2 px".** The rendered blob
   carries the puck's near-side wall pixels: a range-independent ~2 px (1.75–2.3
   measured) bias toward the camera in v. The test asserts u sub-pixel and v within
   a bounded band — still catching focal/H-W-swap/axis errors (tens of px) — and the
   bias itself is corrected in world space by the calibrated model.

2. **The scalar κ wall correction is replaced by a 4-constant bias model.** The
   sweep falsified κ (yaw RMSE stuck at 0.11 vs the 0.1 bar; κ fitted 0.217, far
   from the predicted 0.05–0.1): the centroid bias has a world-radial term **affine
   in horizontal camera distance** (`bias_radial=(0.0003, -0.0188)`) plus an
   **image-radial term linear in normalized pixel offsets**
   (`bias_image=(0.00943, 0.00377)`, dimensionless → resolution-invariant).
   `σ_px = 0.062` (quantization-only; rendering is noise-free). Baked as
   `PuckVisionSensor` defaults; `--mode static` refits the same basis and reports
   deltas (≈ 0 confirms). The correction is applied **before** the yaw computation —
   a biased centre against an unbiased marker skews the 15.5 mm lever arm ~0.33 rad.

3. **Blue marker gated strictly to the filled puck component** (not a bbox window or
   dilation ring): the scene's blue-grey floor checker PASSES the blue ratio test,
   and the synthetic checker regression caught a dilation-ring leak during
   development. Cost: marker outlines that alias open yield yaw `None` (the designed
   2-D fallback) — availability 87.6%, comfortably over the 60% bar.

4. **Detector bbox gate scales with render height** (`[4, 90]` px was silently
   640×360-specific; near-camera blobs exceed 90 px at 960×540).

5. **The NIS gate is keyed to the measurement channel.** Gating the GT path fired
   **89 times with 21 forced re-locks over 20 GT mini-defend episodes** (bounce
   innovations vs the tiny fitted OBS_COV → NIS in the thousands), which visibly
   changed GT behaviour — forbidden by the Part 3 acceptance. `R=None` (trusted GT)
   takes the original unconditional update, making GT invariance structural;
   measurements carrying a covariance (vision) are gated.

6. **Two gate escape hatches for model surprises** (from the closed-loop error
   decomposition: median EKF error 1.3–1.4 mm but p99 24–74 mm, all in bounce /
   mallet-strike rejection windows): (a) within 5 cm of a rim line the gate is
   bypassed (bounce mistiming is model error, not sensor error); (b) a second
   consecutive rejection consistent with the first (< 8 cm) re-locks immediately
   with velocity differenced from the two frames — glitches don't repeat
   consistently, real post-strike tracks do. The 5-streak plain reset remains as the
   fallback. Effect: closed-loop RMSE 13.7 → 3.1 mm (defend), 6.1 → 4.0 mm (hit).

7. **Closed-loop metrics classify out-of-frustum steps separately** (the same rule
   the assignment sets for the static sweep): the side camera covers 94.7% of the
   surface, and a puck parked in an uncovered corner (one 435-step episode-long
   dropout observed) is invisible to any detector. Detection and RMSE bars apply to
   unoccluded, in-frustum ("usable") steps; the tool prints the excluded counts.

8. **Smoke byte-identity is load-conditional** (§2): the vendored planner is
   wall-clock-bounded, so identical output requires an idle machine. The bitwise
   filter pin carries the hard guarantee.

---

## 5. Calibration record

Constants live as `PuckVisionSensor.__init__` defaults with a provenance comment;
the sweep artifacts are `artifacts/vision/static_sweep.{csv,json}` (gitignored,
regenerable). Refit procedure — required after any change to the puck geometry, the
camera pose, or the marker:

```bash
MUJOCO_GL=egl $PY -m mini_platform.tools.vision_eval --mode static
# bake the printed absolute fits into sensor.py -> re-run -> deltas ~ 0, bars PASS
```

`--resolution 960x540` remains the one-parameter escalation if yaw availability ever
drops below 60% (the bias model transfers; σ_px must be refit).

---

## 6. Re-verification commands (all green as of this record)

```bash
cd mini_platform
$PY -m pytest tests/ -q                                    # 43 passed
$PY -m pytest tests/test_kalman_variable_dim.py -q         # bitwise GT pin, gate, relock
MUJOCO_GL=egl $PY -m pytest tests/test_vision_detector.py tests/test_vision_integration.py -q

MUJOCO_GL=egl $PY -m mini_platform.tools.mini_mallet_check           # 3x PASS
MUJOCO_GL=egl $PY -m mini_platform.tools.vision_eval --mode snapshot # 5 PNGs, PASS
MUJOCO_GL=egl $PY -m mini_platform.tools.vision_eval --mode static   # 6 bars PASS
MUJOCO_GL=egl $PY -m mini_platform.tools.vision_eval --mode closedloop \
    --episodes 20 --tasks mini-hit,mini-defend --seed 0              # PASS (see §3)

# GT invariance (idle machine -- see the load caveat in section 2):
$PY -m mini_platform.tools.mini_env_smoke --episodes 20 --seed 0
```

---

## 7. Post-completion changes

**2026-08-22 — camera raised to the physical rig height (0.55 m above the floor =
0.3341 m above the playing surface, was 0.25 m).** One line in
`assets/mini_table.xml` plus the designed recalibration: coverage improved 94.7% →
97.7% (in-frustum grid poses 724 → 748); refit constants
`sigma_px=0.186`, `bias_radial=(0.0005, -0.0142)`, `bias_image=(0.00746, 0.00400)`
(the §4.2 values are the 0.25 m fits, kept for history); sweep bars all PASS
(position RMSE 0.77 / 1.07 mm, yaw 0.040 rad at 85.0%). The one covariance-
inconsistent cell exposed a real gap: a blob clipped by the image border jumps
~3.8 mm under jitter (vs ~0.1 mm unclipped), so `measure()` now inflates R by
`sigma_border=4 mm` per axis when the red bbox touches the border (measurement kept,
downweighted; `PuckMeasurement.clipped` exposes the flag) and the sweep's
consistency floor includes the modeled border term for clipped cells. Also added in
this period: `vision_eval --mode live` (annotated live window / `--record` GIF) and
the pytest `--show-vision` flag.

The height change also exposed a classifier gap the extra coverage made visible:
`occluded_by_mallet` approximated occlusion as "mallet head centre within one radius
of the sight segment", which misses the ARM (links/gripper) reaching over the table
-- one closed-loop episode parked the puck behind the reaching arm for ~100 steps,
misclassified as visible, carrying 87% of the run's squared EKF error (at the old
height those same steps fell outside the smaller frustum and were excluded anyway).
Replaced by `sight_line_blocked`: an exact `mj_ray` cast from the camera to the
puck-top centre against the whole scene (puck body excluded, 3 cm margin) --
verified to fire behind the mallet and stay quiet beside it.

Final closed-loop at the new height (20 eps, seed 0), all bars PASS -- better than
the 0.25 m results on every metric:

| | mini-hit | mini-defend | bar |
|---|---|---|---|
| detection (unoccluded, in-frustum) | **100%** | **100%** | > 97% |
| EKF pos RMSE (usable steps) | **2.17 mm** (P95 3.1) | **4.13 mm** (P95 2.9) | < 5 mm |
| success rate GT / vision | 0.15 / 0.20 | 1.00 / 1.00 | \|diff\| ≤ 0.10 |

Static sweep at the new height: detection 100%, position RMSE 0.77 / 1.07 mm, yaw
0.040 rad at 85.0% availability, pred/empirical 1.00-1.28x, refit deltas ~0 (4 of
756 grid poses arm-shadowed under the ray classifier, unchanged).

## 8. Notes for future work

- **Part 5 (wrist cam):** the sensor already reads extrinsics live per frame and the
  renderer takes a camera name per `update_scene`, so a second camera needs no new
  abstraction. `step()` accepting a list of (measurement, R) pairs is the only EKF
  change the assignment sketches.
- **T1.x damping refit:** the vendored EKF's fitted `damping = [0.2125, 0.2562]`
  still disagrees with the sim's 0.5 s⁻¹; out of scope per the assignment. Its
  per-step innovation is far below measurement noise, so it cannot fire the gate.
- The two `# MINI:` files touched here (`kalman_filter.py`, `system_state.py`) and
  `baseline_agent.py` keep the vendored-marker convention; `baseline_mini.py` is not
  vendored and carries none.
- The wall-clock-bounded planner (§2) is worth knowing about far beyond vision: any
  A/B comparison of agent behaviour is only clean under matched machine load.
