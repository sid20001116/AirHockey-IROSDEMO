"""Synthetic puck rolls in the ``hw/capture_puck_rolls`` npz format (test data, not a test).

Ground truth for the Stage-2 EKF refit: a puck rolled under a KNOWN per-axis decay rate and a
KNOWN rim restitution, sampled at a camera-like frame rate with camera-like position noise,
timestamp jitter and dropouts. Both ``test_fit_mini_table`` (does the fitter recover the
constants?) and ``test_ekf_replay`` (do the recovered constants predict better?) build their
data here, so the two answer the same question about the same table.
"""

import numpy as np

from mini_platform.estimation.mini_env import rim_lines

FPS = 30.0
NOISE_M = 0.001           # per-axis position noise, m (~1 mm, the real detector's scale)
JITTER_S = 0.001          # timestamp jitter, s
PX_PER_M = 600.0          # rough world->pixel scale of the real rig (640 px over ~1 m)

#: The constants the synthetic table actually has. Every tolerance in the tests is a tolerance
#: on recovering THESE.
TRUTH = dict(d=0.55, e=0.62, tan=0.90, sigma=NOISE_M)


def roll(p0, v0, d, duration, dt=1.0 / FPS, e_normal=None, e_tangent=1.0):
    """Integrate a puck with exponential per-axis decay ``d``, reflecting off the rim lines.

    Exact within a free-flight step (the closed-form decay solution), so the samples carry no
    integration error -- what a fit sees is the model plus measurement noise, which is what
    makes a tolerance meaningful. ``e_normal=None`` means a GLIDE: it stops at the first rail.
    """
    lines = rim_lines()
    rims = [(0, lines["x-"], +1), (0, lines["x+"], -1), (1, lines["y-"], +1), (1, lines["y+"], -1)]
    p, v = np.asarray(p0, float).copy(), np.asarray(v0, float).copy()
    ts, ps = [], []
    t = 0.0
    while t < duration:
        ts.append(t)
        ps.append(p.copy())
        p = p + v * (1 - np.exp(-d * dt)) / d
        v = v * np.exp(-d * dt)
        for axis, line, inward in rims:
            if inward * (p[axis] - line) < 0:                      # crossed the rail
                p[axis] = line + inward * abs(p[axis] - line)      # reflect the overshoot
                if e_normal is None:
                    return np.array(ts), np.array(ps)
                v[axis] *= -e_normal
                v[1 - axis] *= e_tangent
        t += dt
    return np.array(ts), np.array(ps)


def concat_rolls(rolls, gap_frames=20, dt=1.0 / FPS, dropout=True):
    """Splice separate rolls into one capture the way a real session runs: the puck stops, sits
    where it landed, is picked up (``dropout=True`` -> the detector loses it for a few frames,
    which is what actually happens) and reappears at the next roll's start.

    With ``dropout=False`` the puck instead TELEPORTS between rolls with no gap -- the harsher
    case, which the tools have to survive on their own (``fit_mini_table.MAX_STEP_SPEED``)."""
    ts, ps = [], []
    t_end = 0.0
    for t, p in rolls:
        ts.append(t + t_end)
        ps.append(p)
        t_end = ts[-1][-1] + gap_frames * dt
        ts.append(ts[-1][-1] + dt * np.arange(1, gap_frames))
        sitting = np.tile(p[-1], (gap_frames - 1, 1))
        if dropout:
            sitting[-6:] = np.nan
        ps.append(sitting)
    return np.concatenate(ts), np.concatenate(ps)


def write_capture(path, t, xy, tag, rng, noise=NOISE_M):
    """Write a synthetic capture in the real npz format (noise + jitter applied here)."""
    t = t + rng.normal(0, JITTER_S, size=len(t))
    t = np.maximum.accumulate(t)                                   # time never runs backwards
    xy = xy + rng.normal(0, noise, size=xy.shape)
    uv = xy * PX_PER_M + np.array([320.0, 240.0])
    # a dropout frame carries no detection at all, exactly as the capture tool writes it
    uv[~np.isfinite(xy).all(axis=1)] = np.nan
    # seq counts CAMERA frames, so it advances through a gap in which nothing was detected --
    # deriving it from the timestamps is what makes a synthetic dropout look like a real one.
    seq = np.round((t - t[0]) * FPS).astype(int)
    np.savez(path, t=t, xy=xy, yaw=np.full(len(t), np.nan),
             area=np.full(len(t), 300, dtype=int), uv=uv,
             J=np.tile(np.eye(2) / PX_PER_M, (len(t), 1, 1)),
             seq=seq, tag=np.array(tag),
             calib_path=np.array("synthetic"), puck_profile_path=np.array(""),
             resolution=np.asarray([640, 480]), camera_index=np.array(-1),
             frame_rate=np.array(FPS), detect_rate=np.array(1.0))
    return str(path)


def glide_rolls(n=18, d=TRUTH["d"], dt=1.0 / FPS):
    """``n`` long single-direction glides at assorted speeds and headings, each starting
    UPSTREAM of its direction of travel so the whole roll fits on the table (the capture
    protocol's "long, clean, single-direction runs")."""
    out = []
    for i in range(n):
        speed = 0.5 + 0.12 * (i % 7)
        ang = [0.0, np.pi, 0.35, -0.35, np.pi - 0.3, np.pi + 0.3, 1.4, -1.4][i % 8]
        v0 = speed * np.array([np.cos(ang), np.sin(ang)])
        out.append(roll(-np.sign(v0) * np.array([0.36, 0.15]), v0, d, 1.6, dt))
    return out


def bounce_rolls(n=12, d=TRUTH["d"], dt=1.0 / FPS, e=TRUTH["e"], tan=TRUTH["tan"]):
    """``n`` rolls launched from the centre into the rails, mostly perpendicular plus oblique."""
    out = []
    for i in range(n):
        speed = 0.8 + 0.1 * (i % 5)
        ang = [-1.2, 1.2, 0.15, -0.15, 2.9, -2.9, 1.45, -1.45][i % 8]
        v0 = speed * np.array([np.cos(ang), np.sin(ang)])
        out.append(roll(np.array([0.0, 0.0]), v0, d, 3.0, dt, e_normal=e, e_tangent=tan))
    return out


def static_frames(spots=((0.0, 0.0), (-0.40, -0.20), (0.40, 0.20)), seconds=35.0,
                  dt=1.0 / FPS, move_seconds=2.0):
    """A still puck at each spot in turn (the measurement-noise capture).

    Between spots the puck is picked up and carried: the detector loses it, so those frames
    carry no detection at all -- which is also what stops anything downstream from reading the
    move as a slow roll."""
    ts, ps, t_end = [], [], 0.0
    for spot in spots:
        n = int(seconds * FPS)
        ts.append(t_end + dt * np.arange(n))
        ps.append(np.tile(spot, (n, 1)))
        t_end = ts[-1][-1] + dt
        m = int(move_seconds * FPS)
        ts.append(t_end + dt * np.arange(m))
        ps.append(np.full((m, 2), np.nan))
        t_end = ts[-1][-1] + dt
    return np.concatenate(ts), np.concatenate(ps)


def write_session(tmp_dir, seed=7, d=TRUTH["d"], e=TRUTH["e"], tan=TRUTH["tan"]):
    """A whole capture session -> (straight_path, bounce_path, static_path)."""
    rng = np.random.default_rng(seed)
    t, xy = concat_rolls(glide_rolls(d=d))
    straight = write_capture(f"{tmp_dir}/straight.npz", t, xy, "straight", rng)
    t, xy = concat_rolls(bounce_rolls(d=d, e=e, tan=tan))
    bounce = write_capture(f"{tmp_dir}/bounce.npz", t, xy, "bounce", rng)
    t, xy = static_frames()
    static = write_capture(f"{tmp_dir}/static.npz", t, xy, "static", rng)
    return straight, bounce, static
