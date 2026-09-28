# Running the mini air-hockey rig on the physical arm

This package (`mini_platform.hw`) drives the **real** Metal arm (over `can0`) and a **real
fixed camera** with the platform's unmodified brain (agent + planner + FSM + EKF). The
low-level joint control is owned by the **vendor `metal_sdk`** — its own smooth NRT/RT
joint-position law, i.e. exactly the controller `core/sdk_control.py` reproduces in sim — so
nothing here re-implements torque control. Nothing from `manipulator_lib` is used.

```
real camera ─▶ RealPuckSensor.measure() ─▶ MiniBaselineAgent.draw_action(obs) ─▶ (2,6) q,qd
                                                                                     │
arm feedback (GetJointPosition/Velocity) ──────────────── obs ◀────────────────────┘
                                                                                     ▼
                                           cubic interpolate ─▶ metal_sdk SetFollowerArmJointPosition (RT)
```

## Bring-up order (at a glance)

Do these once, in order; each links to its section below. Later stages depend on the artefacts
the earlier ones write (`hw_calib.npz`, `puck_profile.json`, `base_frame_fixed.npz`,
`ee_height.npz`), so don't skip ahead. Keep the E-stop within reach from `set-origin` on.

| # | step | command (short) | writes / sets | §  |
|---|------|-----------------|---------------|----|
| 0 | runtime env + `can0` up | `conda activate metal_hw` | — | 0 |
| 1 | camera intrinsics + extrinsic | `calibrate_extrinsics …` | `hw_calib.npz` | 1 |
| 2 | puck colour | `tune_puck …` | `puck_profile.json` | 1b |
| 3 | play height (mallet on table) | `--stage table-height` | `ee_height.npz` | 1c |
| 4 | base_frame origin fix | `--stage set-origin --base-frame ../base_frame.npz --out ../base_frame_fixed.npz` | `base_frame_fixed.npz` | frame checks |
| 5 | frame + keep-out check | `--stage reach-probe --base-frame ../base_frame_fixed.npz` | pick `--min-goal-dist` | 1d / frame checks |
| 6 | staged bring-up | `--stage grav → home → reach → vision` | verifies each layer | 2 |
| 7 | **play** | `--stage play --slow … --base-frame ../base_frame_fixed.npz --ee-height-npz ee_height.npz --min-goal-dist <m>` | closed loop | 2 |

At step 5, confirm two things before trusting closed loop: **(a)** `dist-to-goal` *decreases*
as you push the mallet toward your own goal (chirality), and **(b)** it reads ~**0** with the
mallet *at* the goal line (origin). If either fails, fix the frame (toggle `--base-frame` /
re-run `set-origin`) before playing — the keep-out and defense both ride on it. If, once
playing, defense is skewed *across* the table, the frame's *rotation* is off → fresh
`calibrate_robot` (origin fix won't help that).

Every stage exits by handing the arm to **gravity compensation**; add `--mock` to rehearse any
stage with no hardware.

## 0. Runtime environment (once)

The vendor `.so` is **CPython 3.10**; the platform pins `mujoco==3.3.1` / `numpy<2`. One env
holds both:

```bash
conda create -n metal_hw python=3.10 -y && conda activate metal_hw
cd vision_sdk_platform && pip install -e .          # platform (mujoco, numpy<2, scipy, mushroom-rl)
pip install opencv-python                            # camera + calibration
cd ../metal_sdk && pip install .                     # vendor SDK (MetalSDKInterface)
```

Bring `can0` up (already on in your setup) with `metal_ros2/can_scripts/start_can0.sh`, and
make sure **no other MIT master** (a ROS node, another script) owns the bus.

## 1. Camera calibration (once, fixed camera)

Intrinsics + distortion, then the fixed camera→table extrinsic. The camera is a UVC webcam
at `/dev/videoN` (the "tty device"); find its index with `ls /dev/video*` /
`v4l2-ctl --list-devices`.

> **The ChArUco board is 5×5 squares, 41.5 mm square / 31 mm marker, `DICT_5X5_100`
> (~208×208 mm overall).** Those numbers live in *two* places — `cam_tests.py` and
> `calibrate_extrinsics.py` — and **must match each other and the print**. A detector built for
> the wrong grid still finds every ArUco marker and then interpolates zero charuco corners, so
> the only symptom is `board never detected`; nothing points at the geometry.
>
> From the table mount (0.62 m, fx = 341 px) a marker spans ~17 px — ~2.4 px per ArUco module
> against the ~3 px/module ArUco needs to decode, so the extrinsic is workable but marginal. If
> it will not detect from the mount, use `calibrate_robot` (gripped puck, 50 mm → ~26 px), or
> print ~60 mm squares.

```bash
# intrinsics (repo root tool) — calibrate at the SAME resolution you will run at
python cam_tests.py capture        # ~20 board views, push into the corners
python cam_tests.py calibrate      # -> calibration.npz (K, dist)

# extrinsic: place the ChArUco board flat on the table at a KNOWN spot in the world/table
# frame (origin = table centre, z up). Default assumes board origin at world origin, aligned.
python -m mini_platform.hw.calibrate_extrinsics \
    --intrinsics calibration.npz --camera-index 0 --resolution 640x480 \
    --board-x 0 --board-y 0 --board-z 0 --board-yaw 0 --out hw_calib.npz
```

`hw_calib.npz` bundles `K, dist, R_wc, C_world, resolution` — everything `RealPuckSensor`
needs. **Sanity-check it immediately** (no camera needed):

```bash
python -m mini_platform.hw.check_calib --calib hw_calib.npz
```

It must report the camera **above** the surface (height > 0) with rays pointing **down**. If
it says "camera at/below the surface" the extrinsic mirrored through the table plane and the
puck will be *detected but never measured* (`pixel_to_plane` returns None → the EKF stays
frozen → the arm holds but never moves). `calibrate_extrinsics` now auto-flips the board
normal to prevent this, but always run `check_calib` to confirm.

The world frame here **must** match the MJCF / `env_info['robot']['base_frame']`
(table-centred). If the real arm is not mounted exactly where `assets/arm1.xml` puts it,
measure the arm-base→table transform and pass `--base-frame T.npz` (key `T`, 4×4) at runtime.

## 1b. Puck colour profile (once, real puck)

The detector's default colour gate was tuned for the sim's dark-red puck and **leaks on the
real table** (skin, wood and the table surface are all reddish, so it grabs your hand instead
of the puck). Tune a profile for the real puck's colour with the interactive tuner, then pass
it to the vision/play stages with `--puck-profile`:

```bash
python -m mini_platform.hw.tune_puck --camera-index 4 --resolution 640x480 --out puck_profile.json
```

**Left-click** the puck (a few spots) and **right-click** the background (hand/table) to
sample; press **`a`** to auto-fit the hue/sat/value sliders (it prints the background-leakage
%), **`s`** to save, **`q`** to quit. The left panel shows the detector's *own* bbox/centroid
overlay; the right panel is the exact binary mask it segments — tune until the mask is the
puck and nothing else. Add `--marker` only if your real puck carries a blue yaw marker
(default off → yaw is unavailable, which the pipeline handles). Tune at the **same
resolution** you calibrated and run at.

## 1c. Play height — keep the mallet on the table (once, teleop)

The planner plays on a **flat plane** at `ee_desired_height` (robot-frame z of the grasp
frame). That constant is a *model* value that assumes the base sits exactly 0.2109 m below the
surface; on the real rig it is almost always off, so the gripped mallet **floats above the
table** and the arm looks like it has to swoop down to reach the puck. Measure the real
on-table height by teleop and pass it back with `--ee-height` — that drops the *whole*
controller (home, ready, strikes, defends) onto the surface, so the mallet stays in contact
and only ever slides.

```bash
# arm goes to gravity comp (back-drivable). Rest the gripped mallet FLAT on the table, hold
# it still, press ENTER to sample; do a few spread-out spots; 'q'+ENTER when done.
python -m mini_platform.hw.run_hw_match --stage table-height --ee-height-npz ee_height.npz
```

It reports the mean grasp-frame z (with the same FK the planner uses), flags the spread across
spots (table tilt / a loose grip), and writes `ee_height.npz`. Then pass **either** the printed
number as `--ee-height 0.xxxx` **or** the file as `--ee-height-npz ee_height.npz` to every
`home/reach/vision/play` run. Notes:

- `--contact-offset` (default 0 = light contact) lifts the mallet a hair off the table if it
  binds; `--init-lift` (default **0**) is the episode-start lift — 0 means the arm homes on the
  table with no start swoop (the sim agent uses 0.06).
- The table plane is flat, so a real table **tilt** can't be fully removed — the mean is a
  compromise plane. If the spread is large, level the table or re-measure.
- **Safety:** a wrong (too-low) height drives the mallet into the table. `set_play_height`
  warns if you ask for >50 mm below the model value; bring up with `--stage home --ee-height …`
  first and watch the arm settle before closing the loop.

## 1d. Own-goal keep-out — don't reach back and tip the table (once, teleop)

The arm stands **on the floor at the middle of its own short end, behind its own goal**. A plan
that routes the mallet toward its own goal drives it **head-on into the near rim** — and with the
mallet now resting *on* the table (§1c), that push shoves the whole table. (The vendored Bezier
near bound sits ~2.5 cm from the near wall — well past anything safe.)

> **After a re-mount, read `../../../recalibrate.md` first.** It is the ordered procedure for
> re-deriving everything that is referenced to the arm rather than to the table. The sections
> below are the per-tool reference it points at.

The keep-out is **on by default** on hardware and does two things: it *floors the planner's
back-reach* (Bezier near bound + hit/defend/prepare ranges + home) so no plan targets behind
the line, and it runs a *watchdog* that catches the arm under gravity comp if the **measured**
mallet still crosses a hard floor (bad puck estimate / IK overshoot). Set the limit by teleop:

```bash
# arm to gravity comp; move the mallet toward its OWN goal to the furthest SAFE spot (just
# short of pushing the near rim) and read 'dist-to-goal' — that mm value is --min-goal-dist.
python -m mini_platform.hw.run_hw_match --stage reach-probe
```

Then pass it to play (default 0.06 m ≈ the home rest spot):

```bash
python -m mini_platform.hw.run_hw_match --stage play --task mini-defend-sdk --view \
    --calib hw_calib.npz --camera-index 4 --puck-profile puck_profile.json \
    --ee-height-npz ee_height.npz --min-goal-dist 0.08
```

`--goal-keepout-hard-margin` (default 0.03) is how far past the plan floor the measured mallet
may drift before the watchdog catches; `--no-goal-keepout` disables it (**not recommended** —
it is what stops the table tipping). If the arm reaches too far back **toward its own goal**,
*raise* `--min-goal-dist`. Note: if a mis-calibrated puck (see the wrong-defend-spot /
x-inversion note) makes the arm defend the wrong end, the keep-out prevents the *crash* but you
still need the camera calibration fixed — watch `--stage play --view` (the `meas:`/`defend?`
lines) to tell them apart.

### Frame checks with reach-probe (do this before trusting defense or the keep-out)

`reach-probe` doubles as the base_frame sanity check, because the keep-out side, the defense
direction and this readout all live in the frame `--base-frame` defines:

1. **Chirality.** As you push the mallet **toward your own goal**, `dist-to-goal` must
   **decrease** (→ 0 at the line). If it *increases*, the frame is mirrored — toggle
   `--base-frame` on/off and keep whichever makes it decrease.
2. **Origin.** With the correct-chirality frame, the mallet **at the goal line** should read
   `dist ≈ 0`. A large constant offset (e.g. 357 mm) means the world origin is off — commonly
   because `calibrate_robot` couldn't reach the true table centre for its origin touch, so
   everything (defense, keep-out) is shifted by that much. Fix it with **one touch** at a
   reachable known point:

   ```bash
   # rest the mallet at the OWN-GOAL-LINE CENTRE, press ENTER; writes a corrected frame
   python -m mini_platform.hw.run_hw_match --stage set-origin \
       --base-frame base_frame.npz --out base_frame_fixed.npz
   ```

   Then use `--base-frame base_frame_fixed.npz` everywhere and re-run `reach-probe` — `dist`
   should now be ~0 at the goal. (`--ref-world-xy X Y` if you touch a different known point.)
   This corrects only the origin (translation); if defense is still skewed *across* the table
   the frame's rotation is off and needs a fresh `calibrate_robot`.

## 2. Staged bring-up (safety-gated — do them in order)

Every stage exits by handing the arm to **gravity compensation** so it holds, not collapses.
Add `--mock` to any stage to exercise the loop with no hardware (still needs mujoco for
`env_info`).

```bash
# a) arm holds itself — verifies the SDK link and the catch
python -m mini_platform.hw.run_hw_match --stage grav

# b) move to HOME and hold — verifies feedback + a commanded move + the gripper clamp
python -m mini_platform.hw.run_hw_match --stage home --gripper-mm 8

# c) slow reach HOME -> test -> HOME — verifies tracking
python -m mini_platform.hw.run_hw_match --stage reach --velocity-ratio 3

# d) vision open-loop — arm parked, prints live puck world-xy/yaw from the real camera
python -m mini_platform.hw.run_hw_match --stage vision \
    --calib hw_calib.npz --camera-index 4 --puck-profile puck_profile.json
#    add --view for an annotated live camera window (puck outline + world coords)
python -m mini_platform.hw.run_hw_match --stage vision --view \
    --calib hw_calib.npz --camera-index 4 --puck-profile puck_profile.json

# e) closed-loop play — start slow, watch the grip and the joint error codes.
#    --slow HARD-caps joint speed (safety for developing at home; see "Slow mode" below).
#    Keep it on until you trust the run, then drop it. --view = live puck/tactic/estimate.
python -m mini_platform.hw.run_hw_match --stage play --task mini-defend-sdk --view --slow \
    --calib hw_calib.npz --camera-index 4 --puck-profile puck_profile.json \
    --ee-height-npz ee_height.npz --stream-hz 250 --duration 30
python -m mini_platform.hw.run_hw_match --stage play --task mini-hit-sdk --view --slow \
    --calib hw_calib.npz --camera-index 4 --puck-profile puck_profile.json \
    --ee-height-npz ee_height.npz
```

**Slow mode (safety — developing at home).** `--slow` HARD-caps the RT-stream joint speed at
`0.4 rad/s` (~23 °/s) and forces the NRT `--velocity-ratio` to 1, so *every* stage moves slowly.
The cap is enforced in the streamer (each commanded step is rate-limited), so the arm cannot
swing fast **regardless of what the planner asks** — a hit/smash just crawls to the same place.
Use `--max-joint-vel <rad/s>` for an explicit cap (overrides `--slow`'s default; e.g. `0.25`
for extra-slow, raise it as you gain confidence). Without either flag the arm plays at full
planned speed — only drop the cap once you trust the setup and have clearance.

**Homing is always slow.** Before closed-loop play, `play` now does an explicit slow NRT move
to the hockey home pose and waits for it to settle, so the opening "go to home" sweep is gentle
even at full play speed (it used to jump straight into RT streaming from wherever the arm was).
Its speed is `--home-velocity-ratio` (1..10, **default 1 = slowest**); raise it once you trust
the homing move.

## 3. Things to calibrate / verify before trusting closed loop

- **Joint zero/sign** — `GetJointPosition()` must match the MJCF/URDF zero the FK/IK assume
  (the SDK reports the motor angle verbatim; the platform assumes identity). Pose to a known
  config in `--stage home` and compare `q-err`; if a joint is mirrored, that's a sign/offset
  map to add.
- **Base frame / reach** — confirm the play bands are reachable from the real mount
  (`--stage reach`, then `vision` while you roll the puck through the band).
- **Puck colour** — the vision/play stages need `--puck-profile puck_profile.json` (§1b).
  Without it they fall back to the **sim** red-puck gate (and warn), which mis-detects a real
  puck. Verify the profile in `--stage vision --view` before closing the loop.
- **Camera accuracy** — in `--stage vision`, a static puck's reported world-xy should match a
  tape measure to a few mm. If not, re-fit `sigma_px`/`bias_*` (template:
  `tools/vision_eval.py --mode static`) and bake into `RealPuckSensor` defaults.
- **Gripper stroke** — `--gripper-mm` must clamp your real mallet handle; calibrate the value.
- **Play height** — the mallet must rest ON the table, not float. Measure it once with
  `--stage table-height` (§1c) and pass `--ee-height`/`--ee-height-npz` to every run; without
  it the controller uses the model height and the mallet hovers (looks like it swoops to hit).
- **Own-goal keep-out** — the arm must not drive the mallet past its own goal line into the near
  rim (it shoves the table). On by default; set `--min-goal-dist` from `--stage reach-probe`
  (§1d). If it ever reaches too far back toward its own goal, raise the value. Leave
  `--base-keepout` OFF at this mount — the side-mount value 0.26 forbids the entire defensive
  band now that the base sits behind its own goal.
- **Aim bias** — run `--flat-mallet`. `AIM_BIAS_COMP` (6.5°) cancels a wrist lean *toward the
  base*, and the base direction changed at the 2026-09-10 re-mount, so it is unvalidated;
  `--flat-mallet` squares the contact normal and uses 0 instead.
- **Speed** — until you trust a run, play with `--slow` (hard joint-speed cap; §2 "Slow mode").
  Full planned speed is fast; only drop the cap once the setup is trusted and you have clearance.

## 4. Known hardware detail: RT follower vector length

`SetFollowerArmJointPosition` binds to the SDK's `vector<double>` overload, documented "all
joint position, **include gripper**". With a gripper attached (`arm_end_type=1`) that vector
**must be length 7** — the 7th element is the gripper stroke in mm (`example/one_slave.py`).
Streaming 6 trips a fatal C++ `CHECK` (`[y1]joint position size must be 7`) that **core-dumps
the process, bypassing the gravity-comp catch** — so if you hit it, re-catch the arm with
`--stage grav` (or the vendor `one_master` hold) and check it did not sag.

By default this is handled automatically: `--stream-include-gripper` is **auto = ON whenever a
gripper is attached** (`arm-end-type != 0`), so the streamed vector is `[q1..q6, gripper_mm]`.
Pass `--no-stream-include-gripper` to force a 6-vector for a no-gripper build. The
`--stage home/reach` NRT path uses the unambiguous `array<6>` overload and sidesteps this
entirely — use it to bring up first.

## Files

| file | role |
|---|---|
| `sdk_backend.py` | `MetalArm` / `MockMetalArm` over `metal_sdk` (modes, stream, feedback, health, catch) |
| `real_camera.py` | `V4L2Camera` capture + calibrated `RealPinholeCamera` + `CameraCalibration` (mujoco-free) |
| `real_sensor.py` | `RealPuckSensor` — subclasses `PuckVisionSensor`, real pixels + calibrated back-projection |
| `calibrate_extrinsics.py` | one-shot camera→world extrinsic from a board at a known table pose |
| `check_calib.py` | offline sanity check of `hw_calib.npz` (camera above table? rays hit plane?) |
| `tune_puck.py` | interactive real-puck colour tuner → `puck_profile.json` (`--puck-profile`) |
| `run_hw_match.py` | real-time loop + staged bring-up CLI |
```
