"""Vision-assignment evaluation tool: snapshots, the static accuracy map, closed-loop eval.

Modes:
  snapshot    teleport the puck to 5 poses, save PNGs and print red-pixel counts (Part 1)
  static      dense pose grid: detection/accuracy map, kappa and sigma_px fits (Part 2)
  closedloop  same-seed GT-vs-vision episodes with per-step logging (Part 4)
  live        watch the vision agent play, each camera frame annotated live: green
              outline + bbox around the detected puck, centroid cross, marker cross,
              red border on dropout frames (needs matplotlib and a display).
              With --record: no window, headless, writes a real-time-paced GIF to
              artifacts/vision/live_<task>.gif instead

Output lands under artifacts/vision/. Requires an offscreen GL context: on a headless
machine run with MUJOCO_GL=egl.

    MUJOCO_GL=egl python -m mini_platform.tools.vision_eval --mode snapshot
    MUJOCO_GL=egl python -m mini_platform.tools.vision_eval --mode static
    MUJOCO_GL=egl python -m mini_platform.tools.vision_eval --mode closedloop \
        --episodes 20 --tasks mini-hit,mini-defend --seed 0
    MUJOCO_GL=egl python -m mini_platform.tools.vision_eval --mode live \
        --task mini-defend --episodes 3

Offline tool: never import from the runtime path.
"""

import argparse
import json
import os

import mujoco
import numpy as np

import mini_platform
from mini_platform import ARTIFACT_DIR
from mini_platform.vision.camera import PUCK_TOP_Z

OUT_DIR = os.path.join(ARTIFACT_DIR, "vision")

#: Snapshot poses (x, y, yaw), world frame: table centre, the two half-table midfields,
#: and the far (-y, away from the camera) band where the puck is smallest.
SNAPSHOT_POSES = [
    (0.00, 0.00, 0.0),
    (-0.35, 0.10, np.pi / 4),
    (0.35, 0.10, np.pi / 2),
    (-0.30, -0.15, np.pi),
    (0.30, -0.15, -np.pi / 4),
]


def parse_resolution(text):
    """'640x360' -> (640, 360)."""
    w, h = text.lower().split("x")
    return int(w), int(h)


def teleport_puck(model, data, x, y, yaw):
    """Write the puck pose directly into qpos (the place_wrist_cam.measure() pattern)."""
    for name, val in (("puck_x", x), ("puck_y", y), ("puck_yaw", yaw)):
        data.qpos[model.joint(name).qposadr[0]] = val
    mujoco.mj_forward(model, data)


def red_pixel_count(rgb):
    """Chromaticity-ratio red mask (same gates the detector uses), summed."""
    im = rgb.astype(np.int16)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return int(((r > 60) & (r > 1.6 * g) & (r > 1.6 * b)).sum())


def first_render(sensor):
    """First render, catching GL-context failure with an actionable hint."""
    try:
        return sensor.render()
    except Exception as e:
        print(f"renderer creation failed: {type(e).__name__}: {e}")
        print("hint: set MUJOCO_GL=egl (headless) or MUJOCO_GL=glfw (desktop)")
        raise SystemExit(1)


def snapshot(args):
    from PIL import Image

    from mini_platform.vision.sensor import PuckVisionSensor

    env = mini_platform.make(args.task)
    env.reset()
    be = env.base_env
    sensor = PuckVisionSensor(be._model, be._data, resolution=args.resolution)
    os.makedirs(OUT_DIR, exist_ok=True)

    first_render(sensor)
    ok = True
    print(f"snapshot: {args.task}, camera table_side_left, "
          f"{args.resolution[0]}x{args.resolution[1]}")
    for i, (x, y, yaw) in enumerate(SNAPSHOT_POSES):
        teleport_puck(be._model, be._data, x, y, yaw)
        rgb = sensor.render()
        n = red_pixel_count(rgb)
        ok = ok and n > 0
        path = os.path.join(OUT_DIR, f"snapshot_{i}.png")
        Image.fromarray(rgb).save(path)
        print(f"  pose {i}: x={x:+.2f} y={y:+.2f} yaw={yaw:+.2f}  "
              f"red px {n:5d}  -> {os.path.relpath(path, ARTIFACT_DIR)}")
    sensor.close()
    print(f"snapshot: {'PASS' if ok else 'FAIL'} (red pixels at every pose: {ok})")


def sight_line_blocked(model, data, cam_pos, puck_xy):
    """True if any scene geometry physically blocks the camera's sight line to the
    puck-top centre. An exact mj_ray test: it catches the mallet head AND the arm
    links/gripper hovering over the table, which an earlier mallet-centre-vs-segment
    sphere approximation missed (measured: one closed-loop episode parked the puck
    behind the reaching arm for ~100 steps, misclassified as visible, and carried
    87% of the whole run's squared EKF error). The puck's own body is excluded so
    grazing self-hits on its top face don't count; the 3 cm margin forgives hits on
    the target's immediate surroundings.

    Shared with tests/test_vision_integration.py -- keep the signature stable."""
    a = np.asarray(cam_pos, dtype=float)
    b = np.array([puck_xy[0], puck_xy[1], PUCK_TOP_Z])
    vec = b - a
    dist = float(np.linalg.norm(vec))
    geomid = np.zeros(1, dtype=np.int32)
    hit = mujoco.mj_ray(model, data, a, vec / dist, None, 1,
                        model.body("puck").id, geomid)
    return 0 <= hit < dist - 0.03


#: Static-sweep grid (world frame) and the yaw slices measured at every cell.
GRID_X = np.linspace(-0.45, 0.45, 21)
GRID_Y = np.linspace(-0.20, 0.20, 9)
GRID_YAWS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

#: Deterministic pose jitters (mm) for the sigma_px calibration: rendering is
#: deterministic, so the "noise" a sub-GSD jitter excites is exactly the pixel
#: quantization error the EKF will see as measurement noise at runtime.
JITTERS_MM = ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2))


def static(args):
    import csv

    from mini_platform.vision.sensor import PuckVisionSensor

    env = mini_platform.make(args.task)
    env.reset()                       # arm + mallet parked at home; never stepped
    be = env.base_env
    model, data = be._model, be._data
    sensor = PuckVisionSensor(be._model, be._data, resolution=args.resolution)
    cam = sensor.camera
    w, h = args.resolution
    first_render(sensor)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"static sweep: {args.task}, {len(GRID_X)}x{len(GRID_Y)} grid x "
          f"{len(GRID_YAWS)} yaws, {w}x{h}, sigma_px={sensor.sigma_px}, "
          f"bias_radial={sensor.bias_radial}, bias_image={sensor.bias_image}")

    rows = []
    for x in GRID_X:
        for y in GRID_Y:
            for yaw in GRID_YAWS:
                teleport_puck(model, data, x, y, yaw)
                uv = cam.project(np.array([x, y, PUCK_TOP_Z]))
                in_frustum = (uv is not None and 0 <= uv[0] <= w - 1
                              and 0 <= uv[1] <= h - 1)
                occluded = sight_line_blocked(model, data, cam.pos, (x, y))
                m = sensor.measure()
                row = {"x": x, "y": y, "yaw": yaw, "in_frustum": int(in_frustum),
                       "occluded": int(occluded), "detected": int(m is not None),
                       "err_x": "", "err_y": "", "yaw_avail": 0, "yaw_err": "",
                       "pred_sqrt_trR": "", "n_pixels": 0}
                if m is not None:
                    err = m.pos_xy - np.array([x, y])
                    row.update(err_x=err[0], err_y=err[1], n_pixels=m.n_pixels,
                               pred_sqrt_trR=float(np.sqrt(np.trace(m.R_xy))))
                    if m.yaw is not None:
                        row.update(yaw_avail=1, yaw_err=float(
                            (m.yaw - yaw + np.pi) % (2 * np.pi) - np.pi))
                rows.append(row)

    # sigma_px calibration on the yaw-0 slice: empirical scatter over the jitter
    # cluster, divided by the cell's ground-sample distance sqrt(tr(J J^T)/2).
    cell_stats = []
    for x in GRID_X:
        for y in GRID_Y:
            residuals = []
            pred = None
            clipped = False
            for dx, dy in JITTERS_MM:
                gx, gy = x + dx / 1000.0, y + dy / 1000.0
                teleport_puck(model, data, gx, gy, 0.0)
                if sight_line_blocked(model, data, cam.pos, (gx, gy)):
                    residuals = None
                    break
                m = sensor.measure()
                if m is None:
                    residuals = None
                    break
                residuals.append(m.pos_xy - np.array([gx, gy]))
                pred = float(np.sqrt(np.trace(m.R_xy)))
                clipped = clipped or m.clipped
            if residuals is None:
                continue
            res = np.array(residuals)
            emp = float(np.sqrt(np.mean(np.sum((res - res.mean(0)) ** 2, axis=1))))
            J = cam.fd_jacobian(*cam.project(np.array([x, y, PUCK_TOP_Z])))
            gsd = float(np.sqrt(np.trace(J @ J.T) / 2))
            cell_stats.append({"x": x, "y": y, "emp_scatter": emp, "gsd": gsd,
                               "pred": pred, "clipped": clipped,
                               "sigma_px": emp / gsd})
    sensor.close()

    # -------------------------------------------------------------------- fits
    det_rows = [r for r in rows if r["detected"]]
    usable = [r for r in rows if r["in_frustum"] and not r["occluded"]]
    usable_det = [r for r in usable if r["detected"]]

    # Residual refit of the sensor's 4-constant centroid-bias model (see
    # PuckVisionSensor): e ~ u_hat*(da + db*hd) + J @ (f*[dcu*un, dcv*vn]). With
    # well-baked constants every delta is ~0; absolute fit = configured + delta.
    A = np.zeros((2 * len(usable_det), 4))
    evec = np.zeros(2 * len(usable_det))
    for i, r in enumerate(usable_det):
        gt = np.array([r["x"], r["y"]])
        uv = cam.project(np.array([r["x"], r["y"], PUCK_TOP_Z]))
        Jg = cam.fd_jacobian(*uv)
        u_hat = gt - cam.pos[:2]
        hd = float(np.linalg.norm(u_hat))
        u_hat = u_hat / hd
        un = (uv[0] - cam.cx) / cam.f
        vn = (uv[1] - cam.cy) / cam.f
        cols = (u_hat, u_hat * hd,
                Jg @ np.array([cam.f * un, 0.0]), Jg @ np.array([0.0, cam.f * vn]))
        for k, cvec in enumerate(cols):
            A[2 * i:2 * i + 2, k] = cvec
        evec[2 * i:2 * i + 2] = (r["err_x"], r["err_y"])
    delta, *_ = np.linalg.lstsq(A, evec, rcond=None)
    bias_radial_fit = (sensor.bias_radial[0] + delta[0], sensor.bias_radial[1] + delta[1])
    bias_image_fit = (sensor.bias_image[0] + delta[2], sensor.bias_image[1] + delta[3])
    sigma_px_fit = float(np.median([c["sigma_px"] for c in cell_stats]))

    # ---------------------------------------------------------------- DoD table
    def rmse(rs):
        e = np.array([[r["err_x"], r["err_y"]] for r in rs])
        return float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))) if len(rs) else np.nan

    center = [r for r in usable_det if abs(r["x"]) <= 0.3]
    corner = [r for r in usable_det if abs(r["x"]) > 0.3]
    det_rate = len(usable_det) / len(usable) if usable else np.nan
    yaw_avail = float(np.mean([r["yaw_avail"] for r in det_rows]))
    yaw_errs = np.array([r["yaw_err"] for r in det_rows if r["yaw_avail"]])
    yaw_rmse = float(np.sqrt(np.mean(yaw_errs ** 2)))

    # Predicted-vs-empirical consistency per jitter cell. The empirical scatter is
    # floored at the MODELED systematic terms before the ratio: sigma_floor (always)
    # and sigma_border (for cells whose blob clips the image edge) both model bias
    # bounds that deterministic jitters cannot excite, so cells with near-zero
    # quantization scatter would otherwise read as "overconfident" for carrying an
    # honest floor.
    ratios = np.array([
        c["pred"] / max(c["emp_scatter"],
                        np.sqrt(2) * np.sqrt(sensor.sigma_floor ** 2
                                             + (sensor.sigma_border ** 2
                                                if c["clipped"] else 0.0)))
        for c in cell_stats])
    ratio_ok = bool(np.all((ratios >= 0.5) & (ratios <= 2.0)))

    n_in = sum(r["in_frustum"] for r in rows)
    print(f"\n  poses: {len(rows)} total, {n_in} in-frustum, "
          f"{sum(r['occluded'] for r in rows)} occluded, {len(det_rows)} detected")
    print(f"  detection (unoccluded, in-frustum)  {det_rate * 100:6.2f}%   "
          f"[bar: 100%] {'PASS' if det_rate == 1.0 else 'FAIL'}")
    print(f"  position RMSE |x|<=0.3              {rmse(center) * 1000:6.2f} mm "
          f"[bar: <3 mm] {'PASS' if rmse(center) < 0.003 else 'FAIL'}")
    print(f"  position RMSE far corners           {rmse(corner) * 1000:6.2f} mm "
          f"[bar: <8 mm] {'PASS' if rmse(corner) < 0.008 else 'FAIL'}")
    print(f"  yaw availability (of detections)    {yaw_avail * 100:6.2f}%   "
          f"[bar: >=60%] {'PASS' if yaw_avail >= 0.60 else 'FAIL'}")
    print(f"  yaw RMSE where available            {yaw_rmse:6.4f} rad "
          f"[bar: <0.1] {'PASS' if yaw_rmse < 0.1 else 'FAIL'}")
    print(f"  pred/empirical sqrt(trR) ratio      {ratios.min():.2f}..{ratios.max():.2f} "
          f"[bar: 0.5..2 every cell] {'PASS' if ratio_ok else 'FAIL'}")
    print(f"  fitted bias_radial                  ({bias_radial_fit[0]:+.4f}, "
          f"{bias_radial_fit[1]:+.4f})  [delta {delta[0]:+.4f}, {delta[1]:+.4f}]")
    print(f"  fitted bias_image                   ({bias_image_fit[0]:+.5f}, "
          f"{bias_image_fit[1]:+.5f})  [delta {delta[2]:+.5f}, {delta[3]:+.5f}]")
    print(f"  fitted sigma_px                     {sigma_px_fit:.3f} px "
          f"(configured {sensor.sigma_px:.3f})")

    # ------------------------------------------------------------------ outputs
    csv_path = os.path.join(OUT_DIR, "static_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    meta = {
        "provenance": "measured in MuJoCo (static sweep; arm and mallet parked at reset home)",
        "method": f"{len(GRID_X)}x{len(GRID_Y)} grid x yaws {[round(v, 4) for v in GRID_YAWS]}; "
                  f"sigma_px from per-cell jitter scatter ({list(JITTERS_MM)} mm, yaw-0 slice) "
                  "over ground-sample distance sqrt(tr(J J^T)/2), median across cells; "
                  "4-constant centroid-bias model refit by least squares in world space "
                  "(radial affine in horizontal camera distance + normalized-image linear)",
        "task": args.task,
        "camera": "table_side_left",
        "resolution": list(args.resolution),
        "sigma_px_configured": sensor.sigma_px,
        "bias_radial_configured": list(sensor.bias_radial),
        "bias_image_configured": list(sensor.bias_image),
        "sigma_px_fit": sigma_px_fit,
        "bias_radial_fit": [float(v) for v in bias_radial_fit],
        "bias_image_fit": [float(v) for v in bias_image_fit],
        "detection_rate_unoccluded_in_frustum": det_rate,
        "rmse_mm_x_le_0.3": rmse(center) * 1000,
        "rmse_mm_far_corners": rmse(corner) * 1000,
        "yaw_availability": yaw_avail,
        "yaw_rmse_rad": yaw_rmse,
        "pred_over_empirical_ratio_min_max": [float(ratios.min()), float(ratios.max())],
    }
    json_path = os.path.join(OUT_DIR, "static_sweep.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  wrote {os.path.relpath(csv_path, ARTIFACT_DIR)} and "
          f"{os.path.relpath(json_path, ARTIFACT_DIR)}")


def _closedloop_run(task, source, args):
    """One agent config over the seeded episode set. Returns per-episode spawn poses
    (for the same-seed equality assertion), successes, and -- for the vision run --
    the per-step log rows."""
    from mini_platform.agents.baseline_mini import MiniBaselineAgent
    from mini_platform.envs.env_base import AirHockeyBase

    env = mini_platform.make(task)
    if source == "vision":
        agent = MiniBaselineAgent(env.env_info, agent_id=1, puck_obs_source="vision",
                                  vision_env=env.base_env)
        cam_pos = agent._sensor.camera.pos
        bf = env.env_info['robot']['base_frame'][0]
    else:
        agent = MiniBaselineAgent(env.env_info, agent_id=1)
    be = env.base_env
    model, data = be._model, be._data
    horizon = env.info.horizon

    spawns, successes, rows = [], [], []
    for ep in range(args.episodes):
        # Re-seed before EVERY reset: the env reset is the only consumer of the numpy
        # stream, so GT and vision runs draw identical spawns even though their
        # episodes end at different steps.
        np.random.seed(args.seed + ep)
        obs = env.reset()
        agent.episode_start()
        spawns.append([data.joint(n).qpos[0] for n in ("puck_x", "puck_y", "puck_yaw")]
                      + [data.joint(n).qvel[0] for n in ("puck_x", "puck_y", "puck_yaw")])
        success = 0
        for t in range(horizon):
            gt_pose = np.array([data.joint(n).qpos[0]
                                for n in ("puck_x", "puck_y", "puck_yaw")])
            action = agent.draw_action(obs)   # vision: renders THIS pre-step state
            if source == "vision":
                m = agent._last_measurement
                occ = sight_line_blocked(model, data, cam_pos, gt_pose[:2])
                cam = agent._sensor.camera
                uv = cam.project(np.array([gt_pose[0], gt_pose[1], PUCK_TOP_Z]))
                in_frustum = (uv is not None
                              and 0 <= uv[0] <= cam.width - 1
                              and 0 <= uv[1] <= cam.height - 1)
                gt_v = agent._r2v.pos2(
                    AirHockeyBase._puck_2d_in_robot_frame(gt_pose, bf))
                ekf = agent.state.puck_tracker.state
                err = ekf[:2] - gt_v
                rows.append({
                    "ep": ep, "t": t,
                    "gt_x": gt_pose[0], "gt_y": gt_pose[1], "gt_yaw": gt_pose[2],
                    "occluded": int(occ), "in_frustum": int(in_frustum),
                    "detected": int(m is not None),
                    "meas_x": m.pos_xy[0] if m is not None else "",
                    "meas_y": m.pos_xy[1] if m is not None else "",
                    "meas_yaw": "" if (m is None or m.yaw is None) else m.yaw,
                    "n_pixels": m.n_pixels if m is not None else 0,
                    "ekf_x_v": ekf[0], "ekf_y_v": ekf[1],
                    "err_x_v": err[0], "err_y_v": err[1],
                    "err_norm": float(np.linalg.norm(err)),
                })
            obs, reward, absorbing, info = env.step(action)
            success = max(success, info["success"])
            if absorbing:
                break
        successes.append(success)
    if source == "vision":
        agent._sensor.close()
    return np.array(spawns), np.array(successes, dtype=float), rows


GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)


def annotate_detection(rgb):
    """Copy of `rgb` with the detection drawn on it: green outline + bounding box
    around the picked-up component, a green centroid cross, a yellow cross on the
    resolved marker, and a red image border when nothing was detected.

    Runs the detector on the frame itself, so the highlight shows exactly what the
    sensor's own code picked up. Returns (annotated image, Detection | None)."""
    from scipy import ndimage

    from mini_platform.vision.detector import _red_mask, detect_puck

    im = rgb.copy()
    det = detect_puck(rgb)
    if det is None:
        im[:3, :] = im[-3:, :] = im[:, :3] = im[:, -3:] = RED
        return im, None

    # The component outline: rebuild the same mask/largest-component the detector
    # used and paint its 2 px dilation ring.
    lbl, n = ndimage.label(_red_mask(rgb.astype(np.int16)))
    comp = lbl == (int(np.argmax(np.bincount(lbl.ravel())[1:])) + 1)
    im[ndimage.binary_dilation(comp, iterations=2) & ~comp] = GREEN

    h, w = im.shape[:2]
    u0, v0, u1, v1 = det.bbox
    u0, v0 = max(u0 - 4, 0), max(v0 - 4, 0)
    u1, v1 = min(u1 + 4, w - 1), min(v1 + 4, h - 1)
    im[v0, u0:u1 + 1] = im[v1, u0:u1 + 1] = GREEN
    im[v0:v1 + 1, u0] = im[v0:v1 + 1, u1] = GREEN

    def cross(uv, color, r=4):
        cu, cv = int(round(uv[0])), int(round(uv[1]))
        im[np.clip(cv, 0, h - 1), max(cu - r, 0):min(cu + r, w - 1) + 1] = color
        im[max(cv - r, 0):min(cv + r, h - 1) + 1, np.clip(cu, 0, w - 1)] = color

    cross(det.centroid, GREEN)
    if det.marker_centroid is not None:
        cross(det.marker_centroid, YELLOW, r=3)
    return im, det


def burn_status(im, text):
    """Draw a status line into the frame's top-left corner (in place, via PIL)."""
    from PIL import Image, ImageDraw

    pil = Image.fromarray(im)
    ImageDraw.Draw(pil).text((6, 5), text, fill=YELLOW)
    im[:] = np.asarray(pil)


def live(args):
    """Play episodes with the VISION agent, camera feed annotated. Interactive window
    by default; --record instead writes artifacts/vision/live_<task>.gif headlessly
    (no matplotlib, real-time 50 fps playback timing -- the live window can never be
    smooth, because each frame waits for the planner's own wall-clock-bounded solve)."""
    from mini_platform.agents.baseline_mini import MiniBaselineAgent

    env = mini_platform.make(args.task)
    agent = MiniBaselineAgent(env.env_info, agent_id=1, puck_obs_source="vision",
                              vision_env=env.base_env)
    first_render(agent._sensor)

    plt = fig = frame = None
    frames = []
    encoder = None
    if args.record:
        # Prefer a streamed MP4 (constant memory, real compression): a long run --
        # 100 mini-hit episodes is ~35k frames -- is infeasible as an in-RAM GIF.
        import shutil
        import subprocess
        if shutil.which("ffmpeg"):
            os.makedirs(OUT_DIR, exist_ok=True)
            w, h = args.resolution
            record_path = os.path.join(OUT_DIR, f"live_{args.task}.mp4")
            encoder = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                 # 50 fps = the control rate, so playback runs in real time.
                 "-r", "50", "-i", "-",
                 "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                 record_path],
                stdin=subprocess.PIPE)
    if not args.record:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("live window needs matplotlib (pip install 'mini-platform[tools]');"
                  " or use --record for a headless GIF")
            raise SystemExit(1)
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 5.4))
        ax.set_axis_off()
        ax.set_title("vision live view -- close the window to stop", fontsize=10)
        fig.tight_layout()
        w, h = args.resolution
        frame = ax.imshow(np.zeros((h, w, 3), dtype=np.uint8))
        # Blit setup: only the image artist is redrawn per step, which is several
        # times faster than a full-figure draw.
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(ax.bbox)
        print(f"live: {args.task}, {args.episodes} episode(s) -- "
              "close the window to stop")
    else:
        print(f"recording: {args.task}, {args.episodes} episode(s), headless")

    try:
        for ep in range(args.episodes):
            np.random.seed(args.seed + ep)
            obs = env.reset()
            agent.episode_start()
            for t in range(env.info.horizon):
                action = agent.draw_action(obs)
                # Re-render the same (unchanged) scene state the agent just measured.
                im, det = annotate_detection(agent._sensor.render())
                m = agent._last_measurement
                if m is None:
                    status = "DROPOUT"
                else:
                    status = (f"puck ({m.pos_xy[0]:+.3f}, {m.pos_xy[1]:+.3f}) m  "
                              f"{m.n_pixels} px  "
                              + (f"yaw {m.yaw:+.2f}" if m.yaw is not None else "yaw --"))
                burn_status(im, f"{args.task}  ep {ep}  t {t:3d}   {status}")
                if args.record:
                    if encoder is not None:
                        encoder.stdin.write(im.tobytes())
                    else:
                        frames.append(im)
                else:
                    fig.canvas.restore_region(background)
                    frame.set_data(im)
                    ax.draw_artist(frame)
                    fig.canvas.blit(ax.bbox)
                    fig.canvas.flush_events()
                    if not plt.fignum_exists(fig.number):
                        return
                obs, reward, absorbing, info = env.step(action)
                if absorbing:
                    break
        if args.record:
            if encoder is not None:
                encoder.stdin.close()
                encoder.wait()
                print(f"wrote {os.path.relpath(record_path, ARTIFACT_DIR)}")
            else:
                from PIL import Image

                os.makedirs(OUT_DIR, exist_ok=True)
                path = os.path.join(OUT_DIR, f"live_{args.task}.gif")
                imgs = [Image.fromarray(f) for f in frames]
                # 20 ms per frame = the 50 Hz control rate: real-time playback.
                imgs[0].save(path, save_all=True, append_images=imgs[1:],
                             duration=20, loop=0)
                print(f"wrote {len(imgs)} frames -> "
                      f"{os.path.relpath(path, ARTIFACT_DIR)}")
        else:
            plt.ioff()
            print("done -- close the window to exit")
            plt.show()
    finally:
        agent._sensor.close()
        if encoder is not None and encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()
            encoder.wait()


def closedloop(args):
    import csv

    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    all_pass = True
    for task in tasks:
        print(f"\n=== closedloop {task}  ({args.episodes} episodes, seed {args.seed}) ===")
        gt_spawns, gt_succ, _ = _closedloop_run(task, "gt", args)
        vi_spawns, vi_succ, rows = _closedloop_run(task, "vision", args)

        # Same-seed comparability holds only if both runs actually saw the same
        # spawns -- assert it rather than assume it.
        assert np.array_equal(gt_spawns, vi_spawns), \
            f"{task}: spawn poses diverged between GT and vision runs"

        csv_path = os.path.join(OUT_DIR, f"closedloop_{task}.csv")
        with open(csv_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

        # The side camera covers 94.7% of the surface: out-of-frustum steps are
        # legitimately invisible to ANY detector and are classified separately, the
        # same rule the static sweep applies to its grid. RMSE is over every usable
        # (unoccluded, in-frustum) step, detected or not, so dropout coasting still
        # counts against the filter.
        usable = [r for r in rows if not r["occluded"] and r["in_frustum"]]
        det_rate = float(np.mean([r["detected"] for r in usable]))
        err = np.array([r["err_norm"] for r in usable])
        rmse = float(np.sqrt(np.mean(err ** 2)))
        p95 = float(np.percentile(err, 95))

        # Dropout run lengths: consecutive undetected steps within an episode.
        run_lengths, run = [], 0
        prev_ep = None
        for r in rows:
            if r["ep"] != prev_ep:
                if run:
                    run_lengths.append(run)
                run, prev_ep = 0, r["ep"]
            if r["detected"]:
                if run:
                    run_lengths.append(run)
                run = 0
            else:
                run += 1
        if run:
            run_lengths.append(run)
        hist = np.bincount(run_lengths) if run_lengths else np.array([0])

        sr_gt, sr_vi = float(gt_succ.mean()), float(vi_succ.mean())
        ok_det = det_rate > 0.97
        ok_rmse = rmse < 0.005
        ok_sr = abs(sr_gt - sr_vi) <= 0.10
        all_pass = all_pass and ok_det and ok_rmse and ok_sr

        n_occ = sum(r["occluded"] for r in rows)
        n_out = sum(not r["in_frustum"] for r in rows)
        print(f"  steps {len(rows)} ({n_occ} occluded, {n_out} out-of-frustum, "
              f"{len(usable)} usable)")
        print(f"  detection (unocc, in-frustum) {det_rate * 100:6.2f}%  "
              f"[bar: >97%] {'PASS' if ok_det else 'FAIL'}")
        print(f"  EKF pos RMSE (usable steps)   {rmse * 1000:6.2f} mm  "
              f"[bar: <5 mm] {'PASS' if ok_rmse else 'FAIL'}   P95 {p95 * 1000:.2f} mm")
        print(f"  dropout run lengths           "
              + (", ".join(f"{l}x{c}" for l, c in enumerate(hist) if l and c) or "none"))
        print(f"  success rate GT / vision      {sr_gt:.2f} / {sr_vi:.2f}  "
              f"[bar: |diff|<=0.10] {'PASS' if ok_sr else 'FAIL'}")
        print(f"  wrote {os.path.relpath(csv_path, ARTIFACT_DIR)}")
    print(f"\nclosedloop: {'PASS' if all_pass else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["snapshot", "static", "closedloop", "live"])
    ap.add_argument("--task", type=str, default="mini-defend",
                    help="task for snapshot/static modes")
    ap.add_argument("--tasks", type=str, default="mini-hit,mini-defend",
                    help="comma-separated task names for closedloop mode")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=str, default="640x360",
                    help="render resolution WxH; 960x540 is the yaw-availability escalation")
    ap.add_argument("--record", action="store_true",
                    help="live mode: no window, write artifacts/vision/live_<task>.gif "
                         "(headless, real-time playback)")
    args = ap.parse_args()
    args.resolution = parse_resolution(args.resolution)

    {"snapshot": snapshot, "static": static, "closedloop": closedloop,
     "live": live}[args.mode](args)


if __name__ == "__main__":
    main()
