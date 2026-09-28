"""Offline analysis of ``run_hw_match --stage play --log`` telemetry (npz).

Computes the jerk / tracking-quality metrics used to accept the de-jerk changes, and can
PREVIEW the ``--intercept-max-jump`` / ``--intercept-ema-alpha`` rate-limit on any existing log
(a pure post-process on the logged intercept, matching ``MiniSystemState.update_prediction``) so
values can be chosen before touching hardware. Two logs can be compared with ``--compare``.

numpy-only; no sim / mujoco imports, so it runs anywhere the log does. ``--wrist-tilt`` and
``--controller`` are the two exceptions -- both turn logged JOINT angles into mallet geometry,
so they need the arm's kinematics and import the sim model lazily, saying so if it is missing.

``--controller`` adds the metrics the 2026-09-15 review judged the controller on, and that
CONTROLLER_FIX_PLAN.md states its pass criteria in. Give it the SAME geometry the run used:
the log stores joint angles, so the base frame and the play height are what turn those into a
mallet position, and a different calibration silently measures a different rig.

Usage:
    python -m mini_platform.tools.analyze_telem play_telem.npz
    python -m mini_platform.tools.analyze_telem after_recal.npz --intercept-max-jump 0.05
    python -m mini_platform.tools.analyze_telem after_recal.npz --intercept-max-jump 0.05 --intercept-ema-alpha 0.4
    python -m mini_platform.tools.analyze_telem before.npz --compare after.npz
    python -m mini_platform.tools.analyze_telem play_telem.npz --wrist-tilt
    python -m mini_platform.tools.analyze_telem vp_base.npz --controller \
        --base-frame base_frame_teleop.npz --ee-height-npz ee_height.npz --max-joint-vel 1.0

Telemetry columns (see ``run_hw_match._telemetry_row``):
    t, draw_ms, tactic, q_cmd(6), q_meas(6), qd_meas(6),
    meas_world(2), est_v(4)=[x, y, vx, vy] (V frame), intercept_v(2)=predicted_state[:2].
Added by step 0c, so logs recorded before it do not carry them. Everything that reads them
degrades to "not recorded" rather than refusing, and the six 2026-09-15 logs still analyse:
    q_streamed(6) -- what the RT streamer actually wrote, after its caps -- frame_seq,
    frame_age, gate_ok, vp_solved, vp_fallback, skipped; plus the run-constant scalars
    cap_joint_vel, cap_joint_accel, agent_tick, stream_hz, planner, goalie.

NB the intercept preview is open-loop: it rate-limits the *recorded* intercept sequence, which
is an accurate picture of the jump reduction but not of the full closed-loop response (the real
clamp also changes the plan, hence future measurements). Treat it as a lower bound on the
benefit; confirm on hardware with an A/B ``--log`` capture.
"""

import argparse

import numpy as np

# Acceptance targets from the de-jerk plan (baseline -> goal), printed alongside each metric.
TARGETS = {
    "intercept_jump_p99": ("< 60 mm", 60.0),
    "intercept_jump_max": ("< 150 mm", 150.0),
    "est_speed_p99": ("< 3 m/s", 3.0),
    "cmd_accel_max": ("< 80 rad/s^2", 80.0),
    "fresh_step": ("< 15 mm", 15.0),
}


def clamp_intercept(iv, max_jump=None, ema_alpha=None):
    """Replicate ``MiniSystemState``'s per-tick intercept rate-limit over a logged sequence.

    Anchor for tick i is the *clamped* value of tick i-1 (first tick unclamped, matching the
    runtime where the first acquisition is never clamped). EMA is applied before the hard clamp,
    same order as the runtime.
    """
    iv = np.asarray(iv, float)
    if max_jump is None and ema_alpha is None:
        return iv.copy()
    out = iv.copy()
    anchor = None
    for i in range(len(iv)):
        xy = iv[i].copy()
        if anchor is not None:
            if ema_alpha is not None:
                xy = anchor + ema_alpha * (xy - anchor)
            if max_jump is not None:
                d = xy - anchor
                n = float(np.linalg.norm(d))
                if n > max_jump:
                    xy = anchor + d * (max_jump / n)
        out[i] = xy
        anchor = xy
    return out


def _p(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def metrics(d, intercept=None):
    """Compute the acceptance metrics for one telemetry dict-like (npz)."""
    t = d["t"].astype(float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    hz = 1.0 / dt if dt > 0 else float("nan")

    q = d["q_cmd"].astype(float)                      # (N, 6)
    mw = d["meas_world"].astype(float)                # (N, 2)
    fresh = np.all(np.isfinite(mw), axis=1)           # a real measurement reached the loop
    ev = d["est_v"].astype(float)                     # (N, 4) = x, y, vx, vy
    iv = np.asarray(intercept, float) if intercept is not None else d["intercept_v"].astype(float)
    draw = d["draw_ms"].astype(float)

    # intercept jump (mm/tick) -- the planner-target discontinuity that drives jerk
    dij = np.linalg.norm(np.diff(iv, axis=0), axis=1) * 1000.0

    # estimated puck speed (m/s) and vx sign-flip rate (velocity instability)
    spd = np.linalg.norm(ev[:, 2:4], axis=1)
    vx = ev[:, 2]
    flips = int(np.sum(np.sign(vx[1:]) * np.sign(vx[:-1]) < 0))
    flip_rate = flips / max(len(vx) - 1, 1)

    # commanded joint acceleration proxy (rad/s^2): 2nd difference of q_cmd / dt^2.
    # NB this is the PLAN (agent q_cmd), UPSTREAM of the RT streamer's --max-joint-accel clamp;
    # it reflects planner jerk (fixed by a feasibility-aware planner), not the streamed command.
    acc = np.diff(q, axis=0, n=2) / (dt * dt)
    amax = np.max(np.abs(acc), axis=0)

    # measured joint acceleration (rad/s^2): 1st difference of the measured joint velocity.
    # This is what the arm ACTUALLY did -- the physical jerk the --max-joint-accel stream clamp
    # (and smoother plans) reduce. Noisier than the command, but the right A/B smoothness signal.
    qdm = d["qd_meas"].astype(float)                  # (N, 6)
    macc = np.diff(qdm, axis=0) / dt
    macc_abs = np.abs(macc)
    meas_amax = np.max(macc_abs, axis=0)
    meas_accel_p99 = _p(macc_abs.reshape(-1), 99)

    # measurement cadence: fraction of ticks with a fresh measurement + longest coast run
    fresh_rate = float(fresh.mean())
    maxrun = cur = 0
    for f in fresh:
        cur = 0 if f else cur + 1
        maxrun = max(maxrun, cur)

    # correction jump: est-position step on fresh vs coast ticks (fresh >> coast = snapping)
    dep = np.linalg.norm(np.diff(ev[:, 0:2], axis=0), axis=1) * 1000.0
    fn = fresh[1:]
    fresh_step = float(np.nanmean(dep[fn])) if fn.any() else float("nan")
    coast_step = float(np.nanmean(dep[~fn])) if (~fn).any() else float("nan")

    return dict(
        n=len(t), dur=float(t[-1] - t[0]) if len(t) > 1 else 0.0, hz=hz,
        intercept_jump_p50=_p(dij, 50), intercept_jump_p90=_p(dij, 90),
        intercept_jump_p99=_p(dij, 99), intercept_jump_max=float(np.nanmax(dij)) if dij.size else float("nan"),
        est_speed_p50=_p(spd, 50), est_speed_p99=_p(spd, 99),
        est_speed_max=float(np.nanmax(spd)) if spd.size else float("nan"),
        vx_flips=flips, vx_flip_rate=flip_rate,
        cmd_accel_per_joint=amax, cmd_accel_max=float(np.max(amax)),
        meas_accel_per_joint=meas_amax, meas_accel_max=float(np.max(meas_amax)),
        meas_accel_p99=meas_accel_p99,
        fresh_rate=fresh_rate, max_coast_run=maxrun,
        fresh_step=fresh_step, coast_step=coast_step,
        draw_p50=_p(draw, 50), draw_p99=_p(draw, 99),
        draw_max=float(np.nanmax(draw)) if draw.size else float("nan"),
        draw_over20=float(np.mean(draw > 20) * 100.0),
    )


def _tgt(key):
    return f"   [target {TARGETS[key][0]}]" if key in TARGETS else ""


def _flag(key, val):
    """'ok'/'!!' marker vs target, for at-a-glance scanning."""
    if key not in TARGETS:
        return "  "
    return "ok" if (np.isfinite(val) and val < TARGETS[key][1]) else "!!"


def print_report(name, m):
    print(f"== {name} ==  N={m['n']}  {m['dur']:.1f}s  {m['hz']:.1f} Hz")
    pj = m["cmd_accel_per_joint"]
    print(f"  intercept |jump| mm   p50={m['intercept_jump_p50']:6.1f} p90={m['intercept_jump_p90']:6.1f} "
          f"p99={m['intercept_jump_p99']:7.1f} max={m['intercept_jump_max']:7.1f}  "
          f"{_flag('intercept_jump_p99', m['intercept_jump_p99'])}{_tgt('intercept_jump_p99')}")
    print(f"  est speed m/s         p50={m['est_speed_p50']:5.2f} p99={m['est_speed_p99']:5.2f} "
          f"max={m['est_speed_max']:5.2f}  vx-sign-flips={m['vx_flips']} ({m['vx_flip_rate']*100:.1f}%)  "
          f"{_flag('est_speed_p99', m['est_speed_p99'])}{_tgt('est_speed_p99')}")
    print(f"  cmd accel rad/s^2     max/joint=[{', '.join(f'{x:.0f}' for x in pj)}]  "
          f"overall={m['cmd_accel_max']:.0f}  {_flag('cmd_accel_max', m['cmd_accel_max'])}{_tgt('cmd_accel_max')}"
          f"   (PLAN, pre-stream-clamp)")
    mj = m["meas_accel_per_joint"]
    print(f"  meas accel rad/s^2    max/joint=[{', '.join(f'{x:.0f}' for x in mj)}]  "
          f"p99={m['meas_accel_p99']:.0f} overall={m['meas_accel_max']:.0f}   (ARM actual -- what "
          f"--max-joint-accel smooths)")
    print(f"  fresh-rate            {m['fresh_rate']:.3f}  max coast-run={m['max_coast_run']} ticks   "
          f"[hold ~same after fixes]")
    print(f"  est-pos step mm       fresh={m['fresh_step']:5.1f}  coast={m['coast_step']:5.1f}  "
          f"{_flag('fresh_step', m['fresh_step'])}{_tgt('fresh_step')}")
    print(f"  draw_ms               p50={m['draw_p50']:5.2f} p99={m['draw_p99']:5.2f} "
          f"max={m['draw_max']:5.1f}  >20ms={m['draw_over20']:.1f}%")


def wrist_tilt_deg(d, which="q_meas"):
    """Angle (deg, per tick) between the mallet's pointing axis and straight down.

    Nothing in the vendored stack controls this -- the tracking QP is position-only -- so the
    wrist drifts tens of degrees off vertical and the mallet ends up playing on its rim. This is
    how to check, on REAL logs, whether ``--flat-mallet`` is doing what it does in sim: expect a
    ~55 deg median without it and a low single-digit median with it (higher, ~10 deg, while
    parked in front of the goal, where no vertical wrist exists).

    Lazily imports the sim model, which is the only thing here that does. Returns None if that
    import fails, so the rest of the report still runs on a machine without mujoco.
    """
    try:
        import mini_platform
        from mini_platform.core.kinematics import tool_tilt
    except Exception:
        return None
    env_info = mini_platform.make("mini-hit-sdk").env_info
    rm, rd = env_info["robot"]["robot_model"], env_info["robot"]["robot_data"]
    q = d[which].astype(float)
    return np.array([np.degrees(tool_tilt(rm, rd, row[:rm.nv])[0]) for row in q])


def print_wrist_tilt(name, d):
    tilt = wrist_tilt_deg(d)
    if tilt is None:
        print("  wrist tilt            (unavailable: needs the sim model for the arm kinematics)")
        return
    print(f"  wrist tilt deg        p50={np.median(tilt):5.1f} p90={_p(tilt, 90):5.1f} "
          f"max={tilt.max():5.1f}   [flat if p50 is single-digit; ~55 means the mallet is on "
          f"its rim]")


# ----------------------------------------------------------- controller metrics (--controller)
# Folded in from the 2026-09-15 review, which computed these in half a dozen throwaway scripts
# (analyze_logs / log_extras / joints / shots / dig). They are the numbers every fix in
# CONTROLLER_FIX_PLAN.md is judged on, so they belong in the tool the A/B runs already use.
#
# Unlike the rest of this file these need the arm's kinematics AND the rig geometry: a telemetry
# log records JOINT angles, not Cartesian mallet positions. So the sim model is imported lazily,
# exactly as --wrist-tilt does, and everything above stays numpy-only.

#: fsm.system_state.TACTICS, as logged (one int per tick).
TACTIC_NAMES = {0: "INIT", 1: "READY", 2: "PREPARE", 3: "DEFEND", 4: "REPEL", 5: "SMASH"}

#: The agent loop's NOMINAL tick (50 Hz). Slip is measured against this rather than against the
#: log's median interval, because the median is itself inflated by the overruns being measured.
AGENT_TICK = 0.02


def _meta(d, key, default=None):
    """A run-constant scalar written by ``run_hw_match._save_telemetry`` (step 0c), or ``default``.

    Logs recorded before 0c carry none of these, so every caller degrades to "not recorded"
    instead of refusing to analyse an older capture. A NaN cap means the run had no cap at all,
    which is also "nothing to report" rather than a number.
    """
    if key not in getattr(d, "files", []):
        return default
    v = np.asarray(d[key]).reshape(-1)[0]
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.floating):
        v = float(v)
        return default if not np.isfinite(v) else v
    return v


def _segments(mask):
    """(start, end_exclusive) index pairs of the True runs in ``mask``."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


class ControllerRig:
    """Arm kinematics + table geometry the controller metrics are measured against.

    Pass the SAME ``--base-frame`` and ``--ee-height-npz`` the run used: the log stores joint
    angles, so the mallet position, the play plane and the V-frame table bounds are all
    reconstructed here, and a different calibration silently describes a different rig.
    """

    def __init__(self, base_frame=None, ee_height_npz=None, ee_height=None,
                 goalie=False, min_goal_dist=0.02, defend_abort_dist=0.02):
        import mini_platform
        from mini_platform.agents.baseline_mini import MiniBaselineAgent, V_SETBACK
        from mini_platform.core.kinematics import forward_kinematics
        from mini_platform.envs.env_base import AirHockeyBase
        from mini_platform.hw.run_hw_match import load_base_frame, r2v_xform, set_play_height

        self._fk = forward_kinematics
        self._to_robot = AirHockeyBase._puck_2d_in_robot_frame
        ei = mini_platform.make("mini-hit-sdk").env_info
        if base_frame:
            load_base_frame(ei, base_frame)
        z = ee_height
        if z is None and ee_height_npz:
            z = float(np.load(ee_height_npz)["z"])
        if z is not None:
            set_play_height(ei, float(z))
        self.rm, self.rd = ei["robot"]["robot_model"], ei["robot"]["robot_data"]
        self.z_plane = float(ei["robot"]["ee_desired_height"])
        self.T_wr = np.asarray(ei["robot"]["base_frame"][0], float)
        self.r2v = r2v_xform(ei)
        # The jaw point FK reports is not where the mallet touches the table once the wrist
        # tilts -- ~6 cm away at the lean the arm takes near the goal line. A frame fitted by
        # hw.calibrate_teleop carries the jaw->contact length; without one, jaw == contact.
        self.tool_length = 0.0
        if base_frame:
            bf = np.load(base_frame)
            if "tool_length" in bf.files:
                self.tool_length = float(bf["tool_length"])
        agent = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=min_goal_dist,
                                  defend_abort_dist=defend_abort_dist if goalie else None)
        ap = agent.agent_params
        # The band should_defend() gates on: the abort line takes over the lower bound under
        # --goalie, the plant line is it otherwise (see MiniTacticMixin.should_defend).
        self.defend_lo = float(ap["defend_abort_line"] if goalie else ap["defend_range"][0])
        self.defend_hi = float(ap["defend_range"][1])
        self.hit_x_max = float(ap["hit_range"][1])
        self.goal_line = V_SETBACK - float(ei["table"]["length"]) / 2
        self.r_m, self.r_p = float(ei["mallet"]["radius"]), float(ei["puck"]["radius"])
        self.goal_half = float(ei["table"]["goal_width"]) / 2
        # Puck-CENTRE bounds (the rim inset by a puck radius), with the review's 10 mm tolerance
        # so ordinary calibration error is not counted as an off-table detection.
        self.x_lo = self.goal_line + self.r_p - 0.010
        self.x_hi = self.goal_line + float(ei["table"]["length"]) - self.r_p + 0.010
        self.y_hi = float(ei["table"]["width"]) / 2 - self.r_p + 0.010

    def fk(self, q):
        """(jaw xy in V, mallet-contact xy in V, jaw z, wrist tilt deg) for one joint row."""
        pos, rot = self._fk(self.rm, self.rd, np.asarray(q, float)[:6])
        tip = pos + self.tool_length * rot[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(-rot[2, 2], -1.0, 1.0))))
        return self.r2v.pos2(pos[:2]), self.r2v.pos2(tip[:2]), float(pos[2]), tilt

    def meas_to_v(self, meas_world):
        """Logged WORLD detections -> the V planning frame (NaN rows pass through)."""
        out = np.full((len(meas_world), 2), np.nan)
        for i, m in enumerate(meas_world):
            if np.all(np.isfinite(m)):
                out[i] = self.r2v.pos2(self._to_robot(np.array([m[0], m[1], 0.0]), self.T_wr))
        return out


def _shot_outcomes(t, tac, mal_xy, mv, rig, line=0.42):
    """Shot outcomes from a CLEANED puck track (off-table points and >4 m/s jumps dropped).

    A "shot" is the track crossing ``line`` toward our goal at more than 0.3 m/s, at most once
    per 0.6 s. The outcome is read from the following 1.2 s: reaching the mouth is a GOAL, a
    reversal within a mallet's reach is a block, any other reversal is rim/other.
    """
    pts = []
    last = None
    for i in range(len(t)):
        p = mv[i]
        if not np.all(np.isfinite(p)):
            continue
        if p[0] < rig.x_lo or p[0] > rig.x_hi or abs(p[1]) > rig.y_hi:
            continue
        if last is not None and t[i] - last[0] < 0.25 and \
                np.linalg.norm(p - last[1]) / (t[i] - last[0]) > 4.0:
            continue
        pts.append((t[i], p.copy(), i))
        last = (t[i], p.copy())

    vel = [None] * len(pts)
    for k in range(1, len(pts)):
        gap = pts[k][0] - pts[k - 1][0]
        if gap < 0.15:
            vel[k] = (pts[k][1] - pts[k - 1][1]) / gap

    shots, last_shot = [], -9.0
    for k in range(1, len(pts)):
        if vel[k] is None:
            continue
        if pts[k - 1][1][0] >= line > pts[k][1][0] and vel[k][0] < -0.3 \
                and pts[k][0] - last_shot > 0.6:
            shots.append(k)
            last_shot = pts[k][0]

    res = {"blocked": 0, "GOAL": 0, "rim/other": 0, "unknown": 0}
    lateral, speeds, latch = [], [], []
    for k in shots:
        ts = pts[k][0]
        speeds.append(-vel[k][0])
        outcome, dmin, lat = "unknown", np.inf, None
        for j in range(k, len(pts)):
            if pts[j][0] > ts + 1.2:
                break
            i = pts[j][2]
            dd = float(np.linalg.norm(pts[j][1] - mal_xy[i]))
            if dd < dmin:
                dmin, lat = dd, pts[j][1][1] - mal_xy[i][1]
            if pts[j][1][0] < rig.goal_line + 0.06 \
                    and abs(pts[j][1][1]) < rig.goal_half + rig.r_p:
                outcome = "GOAL"
                break
            if vel[j] is not None and vel[j][0] > 0.15 and j > k:
                prev = pts[j - 1]
                d_rev = float(np.linalg.norm(prev[1] - mal_xy[prev[2]]))
                outcome = "blocked" if (d_rev < 0.09 or dmin < 0.07) else "rim/other"
                break
        res[outcome] += 1
        dfd = np.where((tac == 3) & (t >= ts - 0.6) & (t <= ts + 1.0))[0]
        latch.append((t[dfd[0]] - ts) if len(dfd) else np.nan)
        if lat is not None:
            lateral.append(abs(lat))
    return res, np.array(speeds), np.array(lateral), np.array(latch)


def _cap_arg(text):
    """argparse type for ``--max-joint-vel``: one value, or one per joint.

    Mirrors ``run_hw_match.joint_cap_arg`` (not imported -- this tool stays numpy-only so it can
    read a log on a machine with no mujoco), so a run made with a per-joint cap can be analysed
    against the cap it actually used.
    """
    vals = [float(p) for p in str(text).replace(",", " ").split() if p]
    if not vals:
        raise argparse.ArgumentTypeError(f"expected a number or a list of them, got {text!r}")
    return float(vals[0]) if len(vals) == 1 else np.asarray(vals, float)


def _resolve_cap(d, override):
    """The joint-speed cap to analyse ``d`` against: an explicit override, else the log's own.

    Every log has recorded its own envelope since step 0c, and each log in a ``--compare`` pair
    carries its own -- the two runs being compared are usually NOT at the same cap, which is
    frequently the whole point of the comparison. Reading it per log is what stops the second
    log's over-cap rows being silently skipped (or, worse, computed against the first log's cap).

    Read here rather than through ``_meta``, which takes ``reshape(-1)[0]`` and so would flatten a
    PER-JOINT cap (fix 6d: ``[1.8, 1.4, 1.4, 1.4, 1.4, 1.4]``) to a uniform 1.8 -- silently too
    lenient for every joint but J1, and invisible because the report would print one tidy number.
    The missing-key and non-finite cases still degrade exactly as ``_meta`` does: a run with no
    cap recorded NaN, and that is "nothing to report", not a number.
    """
    if override is not None:
        return override
    if "cap_joint_vel" not in getattr(d, "files", []):
        return None
    v = np.asarray(d["cap_joint_vel"], float).reshape(-1)
    if not v.size or not np.all(np.isfinite(v)):
        return None
    return float(v[0]) if v.size == 1 else v


def controller_metrics(d, rig, cap=None):
    """The 2026-09-15 review's controller numbers for one telemetry log.

    ``cap`` is the run's ``--max-joint-vel`` (rad/s). The logs predating step 0c do not record
    it, so it has to be supplied; without it the over-cap rows are skipped rather than guessed.
    """
    t = d["t"].astype(float)
    n = len(t)
    dur = float(t[-1] - t[0]) if n > 1 else 0.0
    dt = float(np.median(np.diff(t))) if n > 1 else AGENT_TICK
    tac = d["tactic"].astype(int)
    qc, qm = d["q_cmd"].astype(float), d["q_meas"].astype(float)
    iv = d["intercept_v"].astype(float)
    est = d["est_v"].astype(float)

    fkc = [rig.fk(q) for q in qc]
    fkm = [rig.fk(q) for q in qm]
    plan_xy = np.array([f[0] for f in fkc])
    arm_xy = np.array([f[0] for f in fkm])
    zm = np.array([f[2] for f in fkm])
    tilt = np.array([f[3] for f in fkm])

    # --- plan feasibility: the joint speed the PLAN implies, against the stream's cap. A plan
    # over the cap is one the RT streamer throttles, which is how the arm leaves the planned path.
    dqp = np.max(np.abs(np.diff(qc, axis=0)), axis=1) / dt
    over, over_joint = {}, None
    # The cap is per JOINT since fix 6d (`--max-joint-vel` takes six values, to raise J1 alone), so
    # every joint is compared against its own entry and a tick counts as over if ANY joint is. A
    # scalar cap broadcasts, which is identical to the old max-over-joints test -- the pre-6d logs
    # read exactly as they did. The finite check also replaces `if cap:`, which was TRUE for the
    # NaN a no-cap run records, so those logs used to report a confident "0.0% over the cap".
    cap_v = None if cap is None else np.asarray(cap, float).reshape(-1)
    if cap_v is not None and cap_v.size and np.all(np.isfinite(cap_v)) and np.all(cap_v > 0):
        dqj = np.abs(np.diff(qc, axis=0)) / dt
        over_any = (dqj > cap_v * 1.02).any(axis=1)
        over["all"] = float(np.mean(over_any) * 100)
        for k in (3, 5, 2, 1):
            sel = tac[1:] == k
            if sel.any():
                over[TACTIC_NAMES[k]] = float(np.mean(over_any[sel]) * 100)
        act = np.isin(tac[1:], [3, 5])
        if act.any():
            over_joint = (dqj[act] > cap_v * 1.02).mean(axis=0) * 100

    # --- plan -> arm: how far the mallet actually is from where the plan put it.
    gap = np.linalg.norm(plan_xy - arm_xy, axis=1) * 1000
    gap_tac = {TACTIC_NAMES[k]: gap[tac == k] for k in (3, 5, 2, 1) if (tac == k).any()}

    # --- vision: detections and estimates that are not on the table at all.
    mv = rig.meas_to_v(d["meas_world"].astype(float))
    fresh = np.all(np.isfinite(mv), axis=1)
    p = mv[fresh]
    det_off = ((p[:, 0] < rig.x_lo) | (p[:, 0] > rig.x_hi) | (np.abs(p[:, 1]) > rig.y_hi)
               if len(p) else np.zeros(0, bool))
    # The estimate is NOT tested against x_lo: a puck entering our own goal mouth legitimately
    # runs past the near rim line, and with --ekf-own-goal the filter follows it there.
    est_off = (est[:, 0] > rig.x_hi) | (np.abs(est[:, 1]) > rig.y_hi)
    off_spots = None
    if len(p) and det_off.any():
        keys, counts = np.unique(np.round(p[det_off], 2), axis=0, return_counts=True)
        order = np.argsort(-counts)[:5]
        off_spots = [(tuple(np.round(keys[i], 2)), int(counts[i])) for i in order]

    # --- plan-clock slip: the agent advances its plan and its EKF by one fixed tick per loop
    # pass whatever the wall clock did, so every overrun stretches the plan against real time.
    d_t = np.diff(t)
    slip_all = float(np.sum(np.maximum(0.0, d_t - AGENT_TICK)))
    seg_slip = [float(np.sum(np.maximum(0.0, d_t[s:e - 1] - AGENT_TICK)))
                for k in (3, 5) for s, e in _segments(tac == k) if e - s > 1]

    # --- a latched DEFEND planning at an intercept its own gate would reject.
    dfd = tac == 3
    bad = dfd & ((iv[:, 0] < rig.defend_lo - 1e-3) | (iv[:, 0] > rig.defend_hi + 1e-3))

    # --- plan -> command: the gap the agent's plan opens on what the streamer could actually
    # write. Distinct from plan->arm above, which also contains the servo's own lag: this one is
    # purely the caps throttling, and it is what --resync-cmd acts on. Absent from logs recorded
    # before the column landed, so everything here stays optional.
    deficit = d["deficit"].astype(float) if "deficit" in d.files else None
    defc_age = defc_q = resync_n = None
    if deficit is not None:
        # Binned by how long the CURRENT engagement has been running. The deficit grows inside a
        # rally and collapses to zero at idle, so a whole-run average hides exactly the shape that
        # matters -- the error is largest at the moment of the strike.
        bins = {}
        for s, e in _segments(tac != 1):
            for k in range(e - s):
                bins.setdefault(min(k, 14), []).append(deficit[s + k])
        defc_age = [(k * AGENT_TICK * 1000, float(np.mean(v)))
                    for k, v in sorted(bins.items()) if len(v) > 5]
        eng = np.where(tac != 1)[0]
        if len(eng) >= 8:
            defc_q = [(float(t[g[0]]), float(np.mean(deficit[g])),
                       float(np.percentile(deficit[g], 95))) for g in np.array_split(eng, 4)]
        if "resync" in d.files and len(d["resync"]):
            resync_n = int(d["resync"][-1])

    # --- health. The one genuinely monotone physical degradation, and no log before this column
    # existed can speak to it at all.
    temp_rise = bad_codes = None
    if "rotor_temp" in d.files:
        rt = d["rotor_temp"].astype(float)
        if rt.size and np.isfinite(rt).any():
            h = max(1, len(rt) // 2)
            temp_rise = (np.nanmax(rt[:h], axis=0), np.nanmax(rt[h:], axis=0))
    if "error_code" in d.files and d["error_code"].size:
        ec = d["error_code"].astype(int)
        bad_codes = [(j, int(c)) for j, c in enumerate(np.max(ec, axis=0))
                     if c not in (0, 1, -1)] or None

    res, speeds, lateral, latch = _shot_outcomes(t, tac, arm_xy, mv, rig)

    return dict(
        n=n, dur=dur, cap=cap,
        over=over, over_joint=over_joint,
        plan_dq=dqp, gap=gap, gap_tac=gap_tac,
        dz=(zm - rig.z_plane) * 1000, tilt=tilt,
        det_off_pct=float(det_off.mean() * 100) if len(det_off) else float("nan"),
        det_n=int(fresh.sum()), est_off_pct=float(est_off.mean() * 100), off_spots=off_spots,
        slip_pct=float(slip_all / dur * 100) if dur else float("nan"),
        seg_slip=np.array(seg_slip), defend_n=int(dfd.sum()), defend_bad=int(bad.sum()),
        defend_bad_plan=plan_xy[bad, 0] if bad.any() else np.zeros(0),
        intercept_far_pct=float(np.mean((iv[:, 0] > rig.hit_x_max) & np.isin(tac, [3, 5])) * 100),
        shots=res, shot_speeds=speeds, shot_lateral=lateral, shot_latch=latch,
        defend_lo=rig.defend_lo, defend_hi=rig.defend_hi,
        deficit=deficit, defc_age=defc_age, defc_q=defc_q, resync_n=resync_n,
        temp_rise=temp_rise, bad_codes=bad_codes,
    )


def _q(a, qs=(50, 90, 99)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not a.size:
        return "n/a"
    return " ".join(f"p{x}={np.percentile(a, x):.3g}" for x in qs) + f" max={a.max():.3g}"


def print_controller_report(name, m):
    cap_txt = "  (no --max-joint-vel given)"
    if m["cap"] is not None:
        c = np.asarray(m["cap"], float).reshape(-1)
        if c.size and np.all(np.isfinite(c)) and np.all(c > 0):
            cap_txt = "  cap " + (f"{c[0]:g}" if np.allclose(c, c[0])
                                  else "[" + " ".join(f"{x:g}" for x in c) + "]") + " rad/s"
    print(f"\n== controller: {name} ==  N={m['n']}  {m['dur']:.1f}s" + cap_txt)
    print(f"  plan joint speed      {_q(m['plan_dq'])} rad/s")
    if m["over"]:
        row = "  ".join(f"{k} {v:.1f}%" for k, v in m["over"].items())
        print(f"  over the cap          {row}")
        if m["over_joint"] is not None:
            print(f"     per joint J1..J6   [{', '.join(f'{x:.0f}%' for x in m['over_joint'])}]"
                  f"   (DEFEND+SMASH ticks)")
    print(f"  plan->arm mallet gap  {_q(m['gap'])} mm")
    for k, v in m["gap_tac"].items():
        print(f"     {k:8s}           {_q(v)} mm")
    print(f"  jaw z off the plane   {_q(m['dz'], (1, 50, 99))} mm")
    print(f"  wrist tilt            {_q(m['tilt'])} deg")
    print(f"  detections off-table  {m['det_off_pct']:.1f}% of {m['det_n']}   "
          f"estimate off-table {m['est_off_pct']:.1f}% of ticks")
    if m["off_spots"]:
        print("     worst spots (V)    "
              + ", ".join(f"{s} x{c}" for s, c in m["off_spots"]))
    print(f"  plan-clock slip       {m['slip_pct']:.1f}% of wall time;  per DEFEND/SMASH segment "
          f"{_q(m['seg_slip'] * 1000, (50, 90))} ms")
    pct = 100.0 * m["defend_bad"] / max(1, m["defend_n"])
    print(f"  DEFEND out of zone    {m['defend_bad']}/{m['defend_n']} ticks ({pct:.0f}%) "
          f"outside [{m['defend_lo']:.3f}, {m['defend_hi']:.3f}]"
          + (f";  plan V-x there {_q(m['defend_bad_plan'], (50, 90))}" if m["defend_bad"] else ""))
    if m["deficit"] is None:
        print("  plan->cmd deficit     n/a (log predates the column)")
    else:
        print(f"  plan->cmd deficit     {_q(m['deficit'] * 1000)} mrad"
              + (f";  {m['resync_n']} resyncs" if m["resync_n"] is not None else ""))
        if m["defc_q"]:
            print("     by run quarter     " + "   ".join(
                f"{ts:.0f}s mean={mu * 1000:.0f} p95={p * 1000:.0f}"
                for ts, mu, p in m["defc_q"]))
        if m["defc_age"]:
            print("     ms into rally      " + "  ".join(
                f"{ms:.0f}:{mu * 1000:.0f}" for ms, mu in m["defc_age"][:8]) + "   (mrad)")
    if m["temp_rise"] is not None:
        a, b = m["temp_rise"]
        print(f"  rotor temp max        1st half [{', '.join(f'{x:.0f}' for x in a)}] -> "
              f"2nd [{', '.join(f'{x:.0f}' for x in b)}] C")
    if m["bad_codes"]:
        print(f"  joint error codes     {m['bad_codes']}  (>=2 is a fault)")
    print(f"  intercept beyond reach {m['intercept_far_pct']:.1f}% of DEFEND/SMASH ticks")
    print(f"  shots                 {m['shots']}   entry speed {_q(m['shot_speeds'], (50,))} m/s")
    print(f"     sideways miss      {_q(m['shot_lateral'] * 1000, (50, 90))} mm;  DEFEND latch "
          f"{_q(m['shot_latch'] * 1000, (50, 90))} ms rel. crossing")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="telemetry npz from run_hw_match --stage play --log")
    ap.add_argument("--compare", default=None, help="second telemetry npz to diff against")
    ap.add_argument("--intercept-max-jump", type=float, default=None,
                    help="preview the intercept rate-limit (m/tick) on the primary log")
    ap.add_argument("--intercept-ema-alpha", type=float, default=None,
                    help="preview the intercept EMA factor (0-1) on the primary log")
    ap.add_argument("--wrist-tilt", action="store_true",
                    help="also report how far off vertical the mallet was held (needs the sim "
                         "model for the arm kinematics). Use it to confirm --flat-mallet on "
                         "hardware.")
    ap.add_argument("--controller", action="store_true",
                    help="also report the 2026-09-15 review's controller metrics: plan joint "
                         "speed vs the stream cap per tactic, plan->arm mallet gap, jaw height "
                         "off the play plane and wrist tilt, off-table detections/estimates, "
                         "plan-clock slip, DEFEND ticks aimed outside their own gate, and shot "
                         "outcomes. Needs the arm kinematics (imports the sim model) plus the "
                         "run's geometry -- pass the SAME --base-frame / --ee-height-npz the "
                         "run used.")
    ap.add_argument("--base-frame", default=None,
                    help="(--controller) the run's base_frame npz (key 'T'). A 'tool_length' "
                         "key, which hw.calibrate_teleop writes, is used for the mallet-CONTACT "
                         "point; without one the jaw point is used instead.")
    ap.add_argument("--ee-height-npz", default=None,
                    help="(--controller) the run's play-height npz (key 'z')")
    ap.add_argument("--ee-height", type=float, default=None,
                    help="(--controller) the play height (m) directly, instead of the npz")
    ap.add_argument("--max-joint-vel", type=_cap_arg, default=None,
                    help="(--controller) the run's --max-joint-vel (rad/s), as one value or six "
                         "comma-separated (J1..J6) for a per-joint cap. Logs predating the "
                         "streamer caps in the telemetry do not record it, so the over-cap rows "
                         "are skipped rather than guessed unless you pass it. Every later log "
                         "carries its own cap, INCLUDING the --compare one, so you rarely need "
                         "this -- and passing it overrides BOTH logs, which is wrong whenever the "
                         "pair was run at different caps.")
    ap.add_argument("--goalie", action="store_true",
                    help="(--controller) the run used --goalie, so the DEFEND gate's lower "
                         "bound is the abort line rather than the plant line.")
    ap.add_argument("--min-goal-dist", type=float, default=0.02,
                    help="(--controller) the run's --min-goal-dist (m), which floors the ranges")
    args = ap.parse_args(argv)

    d = np.load(args.log, allow_pickle=True)
    print_report(args.log, metrics(d))
    if args.wrist_tilt:
        print_wrist_tilt(args.log, d)
    rig = None
    if args.controller:
        # Step 0c records the run's envelope in the log itself, so a log written since then does
        # not need its caps re-typed on the command line -- and, more to the point, cannot be
        # silently analysed against the wrong ones. An explicit flag still wins, which is what
        # the six pre-0c logs need.
        cap = _resolve_cap(d, args.max_joint_vel)
        goalie = args.goalie or bool(_meta(d, "goalie", False))
        rig = ControllerRig(base_frame=args.base_frame, ee_height_npz=args.ee_height_npz,
                            ee_height=args.ee_height, goalie=goalie,
                            min_goal_dist=args.min_goal_dist)
        print_controller_report(args.log, controller_metrics(d, rig, cap))

    imj = args.intercept_max_jump
    imj = imj if (imj is not None and imj > 0) else None
    if imj is not None or args.intercept_ema_alpha is not None:
        iv_clamped = clamp_intercept(d["intercept_v"].astype(float), imj, args.intercept_ema_alpha)
        m2 = metrics(d, intercept=iv_clamped)
        tag = f"intercept preview (max_jump={imj}, ema={args.intercept_ema_alpha})"
        print()
        print_report(tag, m2)
        raw = metrics(d)
        print(f"  -> intercept |jump| p99 {raw['intercept_jump_p99']:.0f}->{m2['intercept_jump_p99']:.0f} mm, "
              f"max {raw['intercept_jump_max']:.0f}->{m2['intercept_jump_max']:.0f} mm  (open-loop preview)")

    if args.compare:
        d2 = np.load(args.compare, allow_pickle=True)
        m_a, m_b = metrics(d), metrics(d2)
        print()
        print_report(args.compare, m_b)
        if args.wrist_tilt:
            print_wrist_tilt(args.compare, d2)
        if rig is not None:
            # Against the COMPARE log's own cap, not the primary's: an A/B that changes the cap
            # is the commonest reason to run this tool, and passing args.max_joint_vel (usually
            # None) silently dropped the second log's over-cap rows entirely.
            print_controller_report(args.compare,
                                    controller_metrics(d2, rig, _resolve_cap(d2, args.max_joint_vel)))
        print()
        print("== delta (compare - log) ==")
        for k in ("intercept_jump_p99", "intercept_jump_max", "est_speed_p99",
                  "cmd_accel_max", "meas_accel_p99", "meas_accel_max", "fresh_rate", "fresh_step"):
            print(f"  {k:22s} {m_a[k]:9.2f} -> {m_b[k]:9.2f}  ({m_b[k]-m_a[k]:+.2f})")


if __name__ == "__main__":
    main()
