"""Replay recorded puck rolls through the EKF and score its PREDICTIONS -- old vs new constants.

This is the acceptance test for the mini-table refit (``tools/fit_mini_table.py``), and it runs
entirely offline: no arm, no camera, no mujoco. Give it the same roll captures the fit was made
from (ideally *held-out* ones) and the fitted npz::

    python -m mini_platform.tools.ekf_replay rolls_bounce.npz --ekf-params mini_table_ekf.npz
    python -m mini_platform.tools.ekf_replay rolls_*.npz --ekf-params mini_table_ekf.npz \\
        --horizons 0.3 0.5 0.8 --plot ekf_replay.png

For every tick it asks the filter the same question the agent asks -- ``get_prediction(h)``,
"where will the puck be in h seconds" -- and compares the answer with where the puck ACTUALLY
was at that moment in the recording. That is the quantity the whole stack is built on: it is the
intercept every tactic aims at, and its error is what makes the arm late.

Why the replay is faithful to the runtime, not just to the maths
---------------------------------------------------------------
* the filter is stepped at the CONTROL rate (``env_info['dt']`` = 20 ms), not at the camera
  rate, and a measurement is fused only on the ticks where a genuinely new camera frame has
  arrived -- between frames ``step(None)`` coasts. That is exactly what ``ThreadedPuckSensor``
  does on hardware, and it matters: the coasting ticks are where a biased ``damping`` does its
  damage.
* measurements carry the same per-measurement covariance the live sensor computes
  (``sigma_px^2 J J^T + sigma_floor^2 I``, from the logged pixel Jacobian), so the NIS gate,
  the rim bypass and the re-lock logic all see what they would see live.
* positions are shifted world -> V by the constant offset (``estimation/mini_env``), which is
  the whole of the runtime's world -> robot -> V chain for agent 1.

The numbers to read
-------------------
Median / p90 prediction error per horizon, split into ticks whose horizon contains a **bounce**
and those that do not. The refit should cut the long-horizon error materially; the bounce rows
are where a wrong restitution shows up (the prediction reflects off the rail at the wrong angle
or the wrong speed, so the intercept lands somewhere else entirely).
"""

import argparse
import os

import numpy as np

from mini_platform.estimation.kalman_filter import DEFAULT_PARAMS, EkfParams, PuckTracker
from mini_platform.estimation.mini_env import DT, v_env_info, v_to_world, world_to_v
from mini_platform.hw.capture_puck_rolls import load_capture
from mini_platform.tools.fit_mini_table import (GAP_FRAMES, central_velocity,
                                                continuous_stretches, find_impacts,
                                                frame_period, smooth3)

#: Default prediction horizons (s) -- the handoff's 0.3 / 0.5 / 0.8, spanning "the puck is
#: nearly here" to "the arm must already be moving".
HORIZONS = (0.3, 0.5, 0.8)

#: The live sensor's noise model, mirrored so the replayed R matches the runtime's
#: (``hw/real_sensor.RealPuckSensor`` defaults; override sigma_px from the fitter's report).
SIGMA_PX = 0.186
SIGMA_FLOOR = 1e-3

#: Ticks are scored only once the filter has seen this many seconds of a roll -- it needs a few
#: frames to have a velocity at all, and scoring the acquisition transient measures the seeding,
#: not the model.
WARMUP_S = 0.30

#: ... and only where the puck ACTUALLY went somewhere over the horizon (metres, measured from
#: the log). Deliberately not a gate on the filter's own speed estimate: that estimate depends
#: on the constants under test, so it would score two candidate parameter sets on two different
#: sets of ticks and the comparison would be meaningless. This gate is the same for every
#: candidate, which is what makes the old-vs-new columns comparable -- and it keeps a puck
#: sitting still (predicted perfectly by anything) from diluting the median.
SCORE_MIN_TRAVEL = 0.03

#: Ground truth at t+h is interpolated between logged frames, and refused when the two
#: bracketing frames are further apart than this many frame periods -- i.e. across a dropout,
#: where we simply do not know where the puck was and must not pretend to score against it.
MAX_INTERP_FRAMES = 3


def measurement_R(J, sigma_px=SIGMA_PX, sigma_floor=SIGMA_FLOOR):
    """The 2x2 world-frame position covariance the live sensor would attach to this frame."""
    return sigma_px ** 2 * (J @ J.T) + sigma_floor ** 2 * np.eye(2)


def _interp(t_frames, xy_frames, t_query, max_gap=None):
    """Logged puck position at ``t_query``, or None if it is not bracketed by nearby frames."""
    max_gap = MAX_INTERP_FRAMES * frame_period(t_frames) if max_gap is None else max_gap
    if t_query < t_frames[0] or t_query > t_frames[-1]:
        return None
    j = int(np.searchsorted(t_frames, t_query))
    j = min(max(j, 1), len(t_frames) - 1)
    t0, t1 = t_frames[j - 1], t_frames[j]
    if t1 - t0 > max_gap:
        return None
    w = 0.0 if t1 == t0 else (t_query - t0) / (t1 - t0)
    return xy_frames[j - 1] * (1 - w) + xy_frames[j] * w


def replay_segment(t, xy, yaw, J, params, horizons, sigma_px, dt=DT):
    """Step one roll through a tracker and score every horizon at every tick.

    -> list of dicts (one per scored (tick, horizon) pair).
    """
    trk = PuckTracker(v_env_info(), agent_id=1, ekf_params=params)
    xy_v = world_to_v(xy)
    impact_times = [float(t[i0]) for i0, _, _ in
                    find_impacts(t, xy, smooth3(central_velocity(t, xy)))]

    out = []
    t_end = float(t[-1])
    tick = float(t[0])
    j = 0                       # next camera frame to be consumed
    seeded = False
    while tick <= t_end:
        # Fuse at most ONE new frame per tick, and only frames whose timestamp has passed --
        # the hardware sensor hands the control loop each frame exactly once.
        meas = R = None
        if j < len(t) and t[j] <= tick:
            while j + 1 < len(t) and t[j + 1] <= tick:
                j += 1          # the loop fell behind the camera: take the newest frame
            meas = xy_v[j]
            R = measurement_R(J[j], sigma_px)
            j += 1
        if not seeded:
            if meas is None:
                tick += dt
                continue
            trk.reset(np.append(meas, 0.0 if not np.isfinite(yaw[0]) else yaw[0]))
            t_seed = tick
            seeded = True
        else:
            if meas is not None and np.isfinite(yaw[min(j - 1, len(yaw) - 1)]):
                m3 = np.append(meas, yaw[j - 1])
                R3 = np.zeros((3, 3))
                R3[:2, :2], R3[2, 2] = R, DEFAULT_PARAMS.OBS_COV[2, 2]
                trk.step(m3, R3)
            else:
                trk.step(meas, R)

            if (tick - t_seed) >= WARMUP_S:
                here = _interp(t, xy, tick)
                for h in horizons:
                    actual = _interp(t, xy, tick + h)
                    if actual is None or here is None:
                        continue
                    if np.linalg.norm(actual - here) < SCORE_MIN_TRAVEL:
                        continue
                    # defend_line far below the table: we want the raw model prediction, not a
                    # tactic's truncation of it.
                    pred_v, _, _ = trk.get_prediction(h, defend_line=-1e6)
                    err = float(np.linalg.norm(v_to_world(pred_v[:2]) - actual))
                    out.append(dict(t=tick, h=h, err=err,
                                    bounce=any(tick < ti <= tick + h for ti in impact_times),
                                    speed=float(np.linalg.norm(trk.state[2:4]))))
        tick += dt
    return out


def replay(captures, params, horizons=HORIZONS, sigma_px=SIGMA_PX):
    """Replay every usable stretch of every capture. -> list of scored samples."""
    rows = []
    for cap in captures:
        idx = np.where(cap.good)[0]
        if len(idx) < 10:
            continue
        breaks = np.where(np.diff(cap.seq[idx]) > GAP_FRAMES + 1)[0]
        for run in np.split(idx, breaks + 1):
            if len(run) < 20:
                continue
            t, xy = cap.t[run], cap.xy[run]
            for lo, hi in continuous_stretches(t, xy):
                if hi - lo < 20:
                    continue
                sl = slice(lo, hi + 1)
                rows += replay_segment(t[sl], xy[sl], cap.yaw[run][sl], cap.J[run][sl],
                                       params, horizons, sigma_px)
    return rows


def summarize(rows, horizons):
    """{(horizon, subset): (n, median_mm, p90_mm, max_mm)} for subsets all/bounce/free."""
    out = {}
    for h in horizons:
        at_h = [r for r in rows if r["h"] == h]
        for name, sel in (("all", at_h),
                          ("bounce", [r for r in at_h if r["bounce"]]),
                          ("free", [r for r in at_h if not r["bounce"]])):
            e = np.array([r["err"] for r in sel], float) * 1000.0
            out[(h, name)] = ((len(e), float(np.median(e)), float(np.percentile(e, 90)),
                              float(e.max())) if len(e) else (0, np.nan, np.nan, np.nan))
    return out


def print_table(old, new, horizons, label_new):
    print("\n=============================== prediction-vs-actual ================================")
    print(f"   error (mm) between get_prediction(h) and where the puck really was, h seconds on")
    print(f"   {'horizon':>7}  {'subset':<7} {'n':>6}   {'KUKA median/p90':>18}   "
          f"{'NEW median/p90':>18}   change")
    for h in horizons:
        for name in ("all", "free", "bounce"):
            n_o, med_o, p90_o, _ = old[(h, name)]
            n_n, med_n, p90_n, _ = new[(h, name)]
            if not n_o:
                continue
            delta = (med_n - med_o) / med_o * 100 if med_o > 0 else np.nan
            print(f"   {h:6.2f}s  {name:<7} {n_o:6d}   {med_o:8.1f} / {p90_o:7.1f}   "
                  f"{med_n:8.1f} / {p90_n:7.1f}   {delta:+6.1f}%")
    print("-----------------------------------------------------------------------------------")
    print(f"   NEW = {label_new}")
    print("   'bounce' rows are ticks whose horizon contains a rim impact -- that is where a")
    print("   wrong restitution shows up; 'free' rows isolate the damping.")


def make_plot(path, rows_old, rows_new, horizons):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (--plot skipped: {type(e).__name__}: {e})")
        return
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 3.6), squeeze=False)
    for ax, h in zip(axes[0], horizons):
        for rows, name, color in ((rows_old, "KUKA", "tab:red"), (rows_new, "refit", "tab:blue")):
            e = np.array([r["err"] for r in rows if r["h"] == h]) * 1000
            if len(e):
                ax.hist(e, bins=40, range=(0, np.percentile(e, 99)), histtype="step",
                        color=color, label=f"{name} (med {np.median(e):.0f} mm)")
        ax.set(title=f"horizon {h:.2f} s", xlabel="prediction error [mm]", ylabel="ticks")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  histograms -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+", help="npz files from hw.capture_puck_rolls")
    ap.add_argument("--ekf-params", default=None,
                    help="fitted npz to evaluate (omitted: report the vendored constants alone)")
    ap.add_argument("--horizons", type=float, nargs="+", default=list(HORIZONS))
    ap.add_argument("--sigma-px", type=float, default=SIGMA_PX,
                    help="centroid noise the replayed measurement covariance is built from "
                         "(use the value fit_mini_table printed)")
    ap.add_argument("--plot", default=None, help="write error histograms to this PNG")
    args = ap.parse_args(argv)

    captures = [load_capture(p) for p in args.captures]
    for c in captures:
        print(f"{os.path.basename(c.path):28s} tag={c.tag:8s} {len(c):5d} frames "
              f"{100.0 * c.good.mean():5.1f}% detected  {c.fps:5.1f} fps")
    if any(c.tag == "static" for c in captures):
        print("  (static captures contribute nothing here -- a still puck's prediction is "
              "trivially right)")

    horizons = tuple(args.horizons)
    rows_old = replay(captures, DEFAULT_PARAMS, horizons, args.sigma_px)
    if not rows_old:
        raise SystemExit("no scorable ticks -- the captures contain no moving puck")
    if args.ekf_params is None:
        old = summarize(rows_old, horizons)
        print("\n   vendored KUKA constants only (pass --ekf-params to compare a refit)")
        for h in horizons:
            n, med, p90, mx = old[(h, "all")]
            print(f"   {h:6.2f}s  n={n:6d}  median {med:7.1f} mm  p90 {p90:7.1f} mm  "
                  f"max {mx:7.1f} mm")
        return rows_old, None

    params = EkfParams.load(args.ekf_params)
    print(f"\n{params.describe()}")
    rows_new = replay(captures, params, horizons, args.sigma_px)
    print_table(summarize(rows_old, horizons), summarize(rows_new, horizons), horizons,
                args.ekf_params)
    if args.plot:
        make_plot(args.plot, rows_old, rows_new, horizons)
    return rows_old, rows_new


if __name__ == "__main__":
    main()
