"""Fit the EKF's puck-motion constants to the MINI table, from real camera rolls.

Input: one or more npz files from ``hw/capture_puck_rolls.py`` (tagged ``straight`` /
``bounce`` / ``static``). Output: ``mini_table_ekf.npz``, loaded by
``run_hw_match --ekf-params`` (see ``estimation/kalman_filter.EkfParams``)::

    python -m mini_platform.tools.fit_mini_table rolls_straight.npz rolls_bounce.npz \\
        rolls_static.npz --out mini_table_ekf.npz --plot fit_mini_table.png

Pure numpy/scipy: no camera, no arm, no mujoco. Everything is fitted in the capture's WORLD
frame, which shares its axes with the EKF's frame V (``estimation/mini_env``), so the
constants transfer verbatim.

What is fitted, and how
-----------------------
**damping** (``F_linear[2, 2]`` / ``[3, 3]``) -- from the straight glides. Each clean glide is
fitted to the closed-form solution of the filter's own decay law,

    x(t) = x0 + v0 * (1 - exp(-d*t)) / d,

i.e. **positions are fitted directly, never differentiated**. Finite-differencing a 1 mm-noise
position at 30 fps produces ~0.04 m/s of velocity noise, which is 20% of a slow roll -- fitting
the integral instead uses every sample and is limited by the model, not the noise. For a fixed
``d`` the model is linear in ``(x0, v0)``, so the fit is a 1-D bounded search over ``d`` with an
exact least-squares solve inside (no local minima, no initial guess). Per-axis estimates are
aggregated with a *weighted median* (weight = duration x |v0|), so one mis-segmented roll cannot
move the answer. The continuous ``d`` is then converted to the filter's discrete per-tick decay
exactly: ``(1 - exp(-d*dt)) / dt``.

**Bounce** (``b_params``/``n_params`` rows of ``_jac_local_collision``) -- from the wall
bounces. Impacts are found where the puck approaches a rim line and its normal velocity
reverses; the legs on either side are re-fitted with the same decay law and **extrapolated to
the impact instant** (the crossing of the rim line), which removes the "velocity right at
contact is the noisiest sample" problem. In the rim-local frame:

    n_params[1] = v_out_normal / v_in_normal     (= -restitution; vendored -0.79)
    b_params[0] = v_out_tangential / v_in_tangential   (vendored 0.8407)

NB the rim-local index convention, which is easy to invert: rim 0 is the -y long rail and its
``local_rim_transform`` is the identity, so local index 2 is the **tangential** velocity
(``b_params``' row) and index 3 the **normal** one (``n_params``' row). Restitution therefore
lives in ``n_params[1]``, not in ``b_params[0]``. ``theta_params`` (spin) is never fitted here:
it needs the puck's angular velocity, and the real puck profile runs with the yaw marker
disabled, so omega is unobserved. It keeps its vendored value.

**OBS_COV** -- from the static-puck spots: the per-spot variance of the measured world xy about
its own mean, averaged over spots (pooling *within* spots keeps per-spot calibration bias out
of what is meant to be a noise estimate). Yaw likewise when a marker is present.

**sigma_px** is reported too, from the same static frames in PIXEL space. That is the number
that actually matters on hardware: the vision sensor supplies its own per-measurement ``R``
(``sigma_px^2 J J^T + sigma_floor^2 I``), and ``PuckTracker.update`` uses ``OBS_COV`` only when
no ``R`` is passed -- i.e. on the sim/GT path. So pass the printed value to
``run_hw_match --sigma-px`` as well; OBS_COV alone will not change the hardware update.
"""

import argparse
import os

import numpy as np
from scipy.optimize import minimize_scalar

from mini_platform.estimation.kalman_filter import DEFAULT_PARAMS, EkfParams
from mini_platform.estimation.mini_env import (DT, GOAL_WIDTH, PUCK_RADIUS, TABLE_LENGTH,
                                               TABLE_WIDTH, rim_lines)
from mini_platform.hw.capture_puck_rolls import load_capture

# --------------------------------------------------------------------------------- gates
#: A dropout of more than this many frames splits a capture into separate runs -- the puck may
#: have been moved by hand in the gap, so samples either side are not one trajectory.
GAP_FRAMES = 2

#: Speed (m/s) above which the puck counts as rolling. Below it the position noise dominates
#: the motion and nothing is identifiable.
MIN_SPEED = 0.20

#: A glide must last this long (s) and travel this far (m) to be fitted: ``d`` is the CURVATURE
#: of the decay, so a short segment cannot distinguish it from a constant velocity.
MIN_SEG_TIME = 0.30
MIN_SEG_TRAVEL = 0.10

#: A frame-to-frame step implying more than this speed (m/s) is not a roll: it is the puck
#: being picked up and put down, or the detector jumping to another red object. It splits the
#: trajectory, and ``SPLIT_GUARD`` frames on each side of the split are discarded because the
#: 3-tap smoothing used for segmentation bleeds the discontinuity that far.
MAX_STEP_SPEED = 4.0
SPLIT_GUARD = 2

#: Moving stretches separated by fewer than this many sub-threshold frames are ONE roll. A
#: bounce is the reason this exists: a central difference that straddles the impact averages
#: the incoming and outgoing velocities, which for a head-on hit nearly cancel, so the measured
#: speed dips for 1-3 frames at exactly the moment the puck is fastest. Without the merge, every
#: perpendicular bounce splits its own roll in two and neither half contains a usable impact.
MERGE_FRAMES = 5

#: Minimum |v0| (m/s) on an axis for that axis to contribute its own damping estimate. A roll
#: that barely moves in y says nothing about damping[1].
AXIS_MIN_SPEED = 0.25

#: Reject a glide whose position-fit residual exceeds this (m) -- a mis-segmented roll, a
#: detection that jumped to the mallet, or an unnoticed contact.
MAX_FIT_RMSE = 0.006

#: Search bounds for the continuous decay rate (1/s). The vendored KUKA fits are ~0.21-0.26;
#: the mini table's MJCF models 0.5 (damping/mass, assets/mini_table.xml).
D_BOUNDS = (0.02, 6.0)

#: Impact detection: how close (m) the puck centre must come to a rim line, how much of the
#: trajectory on each side is used to fit the incoming/outgoing legs, and the minimum normal
#: speed (m/s) required on each side for the event to be a usable bounce.
#:
#: The leg window is a TIME, not a frame count, so it adapts to the camera rate (the captures
#: run 15-25 fps depending on the lighting the auto-exposure settles on). Three frames is the
#: floor: the decay-law leg fit has two free parameters per axis, so three samples leave one
#: degree of freedom -- fewer is an interpolation, not a fit.
CONTACT_MARGIN = 0.030
LEG_SECONDS = 0.30
MIN_LEG_FRAMES = 3
#: ... and the speed test looks at the frames ADJACENT to the impact, not at the median of the
#: whole leg. On a table where the puck decelerates at ~2.5 m/s^2 (measured), it has lost most
#: of its speed by the far end of a 0.3 s leg, so a median over the leg reads as "not moving"
#: for bounces that are perfectly clear at the moment they happen.
IMPACT_VN_FRAMES = 3
MIN_IMPACT_VN = 0.15
MIN_FIT_VN = 0.30          # ... and to be used for the RESTITUTION fit specifically
MIN_FIT_VT = 0.15          # ... for the tangential fit

#: Restitution / tangential ratios outside these bands are rejected as segmentation failures
#: rather than physics (a puck cannot leave a rail faster than it arrived).
E_BOUNDS = (0.10, 1.10)
T_BOUNDS = (0.10, 1.30)

#: How many usable impacts a rim row needs before it is written. A median of one event is not
#: a measurement -- and a single bad event would go straight into the filter the arm flies on.
#: Below this the row keeps its vendored value and the report says how many were short.
MIN_BOUNCE_EVENTS = 5

#: A frame-to-frame velocity jump is a contact, not free flight, when it exceeds BOTH this
#: floor (m/s) and :data:`CONTACT_DV_SIGMAS` times the capture's own frame-to-frame velocity
#: scatter. The second half matters: differencing position noise makes a velocity jump of
#: ~sqrt(2)*sigma/dt -- 0.05 m/s on a good capture but 0.2 m/s on a noisy one at a low frame
#: rate -- so a fixed threshold would either miss real strikes on one or shred free flight into
#: fragments on the other. A real contact reverses 1-2 m/s, well clear of either.
MIN_CONTACT_DV = 0.25
CONTACT_DV_SIGMAS = 5.0

#: A bounce leg must fit the free-flight decay law this well (m) to be believed. This is what
#: rejects the puck being CARRIED back by hand between rolls: a hand's path near a rail can
#: reverse like a bounce, but it does not glide like one. Set from the MEASURED noise of a
#: moving puck (~3 mm rms -- motion blur costs ~5x the 0.5 mm a static puck gives), so a
#: three-sample leg is allowed a couple of sigma before it is called "not a glide".
MAX_LEG_RMSE = 0.008

#: Static-spot detection. A spot is a stretch whose positions all stay within
#: ``MAX_STATIC_SPREAD`` of the stretch's first sample, lasting at least ``MIN_STATIC_TIME``.
#: It is deliberately NOT a speed threshold: per-frame position noise of 1 mm at 30 fps is
#: already ~0.03 m/s of apparent speed, so a speed gate chops one still puck into a dozen
#: fragments (and the fragments' variances are what we are trying to measure).
MIN_STATIC_TIME = 2.0
MAX_STATIC_SPREAD = 0.02

#: Rims, in the capture's world frame: (axis, line coordinate, inward sign).
#: ``inward * (xy[axis] - line) >= 0`` inside the table.
RIMS = {"x-": (0, rim_lines()["x-"], +1), "x+": (0, rim_lines()["x+"], -1),
        "y-": (1, rim_lines()["y-"], +1), "y+": (1, rim_lines()["y+"], -1)}


# ------------------------------------------------------------------------------- helpers
def weighted_median(values, weights):
    """Median of ``values`` under ``weights`` (the 50% point of the weight mass)."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def true_runs(mask):
    """[(i0, i1), ...] inclusive index ranges of the contiguous True stretches in ``mask``."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    d = np.diff(mask.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0])
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask) - 1)
    return list(zip(starts, ends))


def detected_runs(cap):
    """Index arrays of the contiguous stretches of DETECTED frames in a capture.

    A stretch is broken by a dropout longer than :data:`GAP_FRAMES` camera frames (measured on
    the camera's own ``seq`` counter, so a slow detector cannot masquerade as a gap).
    """
    idx = np.where(cap.good)[0]
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(cap.seq[idx]) > GAP_FRAMES + 1)[0]
    return [np.asarray(p) for p in np.split(idx, breaks + 1) if len(p) >= 3]


def frame_period(t):
    """The capture's median frame period (s) -- the yardstick for "is this a gap?"."""
    return float(np.median(np.diff(t))) if len(t) > 1 else DT


def continuous_stretches(t, xy):
    """(i0, i1) stretches over which the samples really are one continuous trajectory.

    Two ways a log stops being one trajectory, both of which would otherwise be fitted (or
    replayed) as motion that never happened:

    * a **position** step no puck can make -- between rolls the puck is carried back to the
      start, and if the detector kept seeing it (or jumped to another red object) the log
      contains a jump at ``MAX_STEP_SPEED`` or more;
    * a **time** gap -- frames missing because the puck was occluded or out of the gate. The
      samples either side are minutes-apart in puck terms even when they are adjacent rows,
      and the gap is measured in frame periods so this holds at any camera rate.
    """
    dt = np.diff(t)
    step_speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(dt, 1e-9)
    ok = np.ones(len(t), bool)
    bad = np.where((step_speed > MAX_STEP_SPEED) | (dt > (GAP_FRAMES + 1) * frame_period(t)))[0]
    for i in bad:
        ok[max(0, i - SPLIT_GUARD + 1):i + SPLIT_GUARD + 1] = False
    return true_runs(ok)


def moving_segments(t, xy, min_speed=None):
    """Index ranges (i0, i1) of the stretches where the puck is actually rolling.

    One segment = one roll. Splitting on motion is what keeps a fit from spanning the gap
    between two rolls -- and that gap is not empty: a real session leaves the puck sitting
    wherever it stopped, very often AGAINST a rail, so a "near a rim" stretch would otherwise
    run from one roll's last bounce straight through the pause into the next roll, and the
    bounce would be measured against a stationary puck (or against the next roll's first
    frames, which are somewhere else entirely).
    """
    min_speed = MIN_SPEED if min_speed is None else min_speed
    speed = np.linalg.norm(smooth3(central_velocity(t, xy)), axis=1)
    moving = speed > min_speed
    # Never let a segment span a discontinuity (see continuous_stretches): a naive "moving" run
    # swallows the jump, and then the fit sees a puck that instantaneously changed place --
    # which fits nothing, and loses the two perfectly good rolls either side of it.
    keep = np.zeros(len(t), bool)
    for a, b in continuous_stretches(t, xy):
        keep[a:b + 1] = True
    moving &= keep
    merged = []
    for a, b in true_runs(moving):
        if merged and a - merged[-1][1] <= MERGE_FRAMES:
            merged[-1] = (merged[-1][0], b)          # a bounce's velocity dip, not a stop
        else:
            merged.append((a, b))
    out = []
    for a, b in merged:
        if (b - a) < 5 or (t[b] - t[a]) < MIN_SEG_TIME:
            continue
        if float(np.linalg.norm(xy[b] - xy[a])) < MIN_SEG_TRAVEL:
            continue
        out.append((a, b))
    return out


def one_sided_velocity(t, xy):
    """``(v_back, v_fwd)``: raw backward and forward difference velocities, unsmoothed.

    The difference that matters at a contact. A central difference -- worse, a smoothed one --
    straddles the impact and averages the incoming and outgoing velocity, so the very spike it
    is being asked about is the thing it destroys: measured on real bounces, a genuine
    -1.5 m/s approach read as -0.07 m/s and the bounce was thrown away. Backward differences
    are valid strictly BEFORE an impact and forward differences strictly AFTER it, so each
    window is asked only about frames it can actually answer for.
    """
    dt = np.diff(t)[:, None]
    d = np.diff(xy, axis=0) / np.maximum(dt, 1e-9)
    v_back = np.vstack([d[:1], d])          # v_back[i] uses frames i-1 -> i
    v_fwd = np.vstack([d, d[-1:]])          # v_fwd[i]  uses frames i   -> i+1
    return v_back, v_fwd


def find_contacts(t, xy, min_dv=MIN_CONTACT_DV):
    """Frame indices where the velocity changes faster than free flight can explain.

    A puck in free flight loses at most a few cm/s per frame (measured: |a| well under
    1 m/s^2 with the air on, ~0.06 m/s per frame at 17 fps). A bigger jump is a CONTACT -- a
    rail, the mallet, or a hand batting the puck back mid-table, which is what a real capture
    session is full of. Free-flight fits must not span one, wherever it happened.

    The threshold adapts to the capture (see :data:`CONTACT_DV_SIGMAS`): the median jump IS
    this capture's noise level, because free flight is what the puck is doing almost all of
    the time.
    """
    v_back, v_fwd = one_sided_velocity(t, xy)
    dv = np.linalg.norm(v_fwd - v_back, axis=1)
    if len(dv) < 3:
        return np.array([], dtype=int)
    thr = max(min_dv, CONTACT_DV_SIGMAS * float(np.median(dv)))
    return np.where(dv > thr)[0]


def central_velocity(t, xy):
    """Central-difference velocity (endpoints one-sided). For SEGMENTATION decisions only --
    every fit works on positions."""
    v = np.empty_like(xy)
    v[1:-1] = (xy[2:] - xy[:-2]) / (t[2:] - t[:-2])[:, None]
    v[0] = (xy[1] - xy[0]) / (t[1] - t[0])
    v[-1] = (xy[-1] - xy[-2]) / (t[-1] - t[-2])
    return v


def smooth3(a):
    """3-tap moving average along axis 0, edges held. Segmentation only."""
    out = a.astype(float).copy()
    out[1:-1] = (a[:-2] + a[1:-1] + a[2:]) / 3.0
    return out


# ------------------------------------------------------------------- the decay-law fitters
def _decay_basis(t, d):
    """``(1 - exp(-d t)) / d`` -- the displacement of a velocity decaying at rate ``d``."""
    return np.expm1(-d * t) / (-d)


def fit_exp_decay(t, x, d_bounds=D_BOUNDS):
    """Fit ``x(t) = x0 + v0 * (1 - exp(-d(t-t0)))/d`` to one axis. -> (x0, v0, d, rmse).

    Linear in ``(x0, v0)`` for fixed ``d``, so this is a bounded 1-D search with an exact
    lstsq inside: no starting guess, no local minima.
    """
    t = np.asarray(t, float) - t[0]
    x = np.asarray(x, float)

    def solve(d):
        A = np.stack([np.ones_like(t), _decay_basis(t, d)], axis=1)
        coef = np.linalg.lstsq(A, x, rcond=None)[0]
        return coef, float(np.sqrt(np.mean((x - A @ coef) ** 2)))

    res = minimize_scalar(lambda d: solve(d)[1], bounds=d_bounds, method="bounded")
    coef, rmse = solve(res.x)
    return float(coef[0]), float(coef[1]), float(res.x), rmse


def fit_exp_decay_2d(t, xy, d_bounds=D_BOUNDS):
    """As :func:`fit_exp_decay` but with ONE ``d`` shared by both axes (the isotropic model).

    -> (p0 (2,), v0 (2,), d, rmse).
    """
    t = np.asarray(t, float) - t[0]
    xy = np.asarray(xy, float)

    def solve(d):
        A = np.stack([np.ones_like(t), _decay_basis(t, d)], axis=1)
        coef = np.linalg.lstsq(A, xy, rcond=None)[0]              # (2, 2): rows [p0; v0]
        return coef, float(np.sqrt(np.mean((xy - A @ coef) ** 2)))

    res = minimize_scalar(lambda d: solve(d)[1], bounds=d_bounds, method="bounded")
    coef, rmse = solve(res.x)
    return coef[0].copy(), coef[1].copy(), float(res.x), rmse


def fit_linear_decel(t, xy):
    """Constant-deceleration (Coulomb friction) alternative: |v| falls LINEARLY, not
    exponentially. Fitted only to report which model the table actually follows -- if this one
    is clearly better, the EKF's exponential form is the wrong shape and no value of ``damping``
    will make long-horizon predictions right. -> rmse."""
    t = np.asarray(t, float) - t[0]
    xy = np.asarray(xy, float)
    dirn = xy[-1] - xy[0]
    dirn = dirn / max(np.linalg.norm(dirn), 1e-9)
    s = (xy - xy[0]) @ dirn                                   # arc length along the roll
    A = np.stack([np.ones_like(t), t, -0.5 * t ** 2], axis=1)  # s = s0 + v0 t - a t^2 / 2
    coef = np.linalg.lstsq(A, s, rcond=None)[0]
    pred = xy[0] + np.outer(A @ coef, dirn)
    return float(np.sqrt(np.mean((xy - pred) ** 2)))


def continuous_to_discrete(d_c, dt=DT):
    """Continuous decay rate (1/s) -> the filter's per-tick constant, exactly.

    ``F_linear`` applies ``v <- (1 - dt*d)v`` once per tick, so matching ``exp(-d_c*dt)``
    over one tick gives ``d = (1 - exp(-d_c*dt))/dt``. At dt = 20 ms this is a ~0.5%
    correction -- small, but free."""
    return float((1.0 - np.exp(-d_c * dt)) / dt)


# ------------------------------------------------------------------------ impact detection
def leg_frames(t):
    """How many frames :data:`LEG_SECONDS` is, at this capture's rate (at least the floor)."""
    return max(MIN_LEG_FRAMES, int(round(LEG_SECONDS / frame_period(t))))


def find_impacts(t, xy, v=None):
    """Rim impacts inside one contiguous run. -> [(i_lo, i_hi, rim_key), ...] index bounds of
    the near-rim stretch. (``v`` is accepted and ignored -- the gate computes its own.)

    An impact is a stretch that comes within :data:`CONTACT_MARGIN` of a rim line, with the
    normal velocity clearly INTO the rail just before it and clearly OUT of the rail just
    after -- the reversal is what distinguishes a bounce from a puck sliding along the rail or
    dying against it. "Just" is :data:`IMPACT_VN_FRAMES` frames: on a fast-decaying table the
    puck is already slow at the far end of a leg, so a median over the whole leg misses real
    bounces (measured: it missed every one of them).
    """
    out = []
    n = len(t)
    lf = leg_frames(t)
    v_back, v_fwd = one_sided_velocity(t, xy)
    for key, (axis, line, inward) in RIMS.items():
        d = inward * (xy[:, axis] - line)
        # backward differences describe the APPROACH, forward differences the DEPARTURE;
        # neither window then contains a difference taken across the impact itself.
        vn_in = inward * v_back[:, axis]
        vn_out = inward * v_fwd[:, axis]
        for i0, i1 in true_runs(d < CONTACT_MARGIN):
            pre = slice(max(0, i0 - lf), i0)
            post = slice(i1 + 1, min(n, i1 + 1 + lf))
            if (pre.stop - pre.start) < MIN_LEG_FRAMES or (post.stop - post.start) < MIN_LEG_FRAMES:
                continue
            approaching = np.median(vn_in[pre][-IMPACT_VN_FRAMES:])
            receding = np.median(vn_out[post][:IMPACT_VN_FRAMES])
            if approaching < -MIN_IMPACT_VN and receding > MIN_IMPACT_VN:
                out.append((i0, i1, key))
    return sorted(out)


def _leg_bounds(i0, i1, impacts, k, lo, hi, lf):
    """Frame ranges for one impact's incoming and outgoing legs.

    Clipped to the enclosing moving segment AND to the neighbouring impacts (with a one-frame
    guard), so a fast puck rattling between the two long rails -- 0.46 m apart is under half a
    second, i.e. ~14 frames at 30 fps, less than two full legs -- never fits one bounce's leg
    across the next bounce.
    """
    prev_end = impacts[k - 1][1] + 2 if k > 0 else lo
    next_start = impacts[k + 1][0] - 1 if k + 1 < len(impacts) else hi + 1
    pre = slice(max(lo, prev_end, i0 - lf), i0)
    post = slice(i1 + 1, min(hi + 1, next_start, i1 + 1 + lf))
    return pre, post


def _leg_velocity_at(t, xy, sl, t_ref, d_axis):
    """Velocity at ``t_ref`` of the leg ``sl``, from the decay law with ``d`` held fixed.

    With ``d`` fixed the model ``x(t) = x_ref + v_ref*(1-exp(-d(t-t_ref)))/d`` is LINEAR in
    ``(x_ref, v_ref)``, so this is one lstsq per axis and the answer is the velocity AT THE
    IMPACT, extrapolated from the whole leg rather than read off the one or two frames next to
    the contact (which are the noisiest and, worse, may straddle it).
    """
    tt = np.asarray(t[sl], float) - t_ref
    pos = np.asarray(xy[sl], float)
    v_ref = np.empty(2)
    x_ref = np.empty(2)
    resid = np.empty(2)
    for a in range(2):
        A = np.stack([np.ones_like(tt), _decay_basis(tt, d_axis[a])], axis=1)
        coef = np.linalg.lstsq(A, pos[:, a], rcond=None)[0]
        x_ref[a], v_ref[a] = coef
        resid[a] = np.sqrt(np.mean((pos[:, a] - A @ coef) ** 2))
    return x_ref, v_ref, float(np.linalg.norm(resid))


def fit_one_bounce(t, xy, ev, d_axis, impacts=None, k=0, bounds=None):
    """Restitution + tangential ratio for one impact. -> dict, or None when it fails a gate.

    Impact time is defined as the moment the INCOMING leg's model crosses the rim line -- a
    physical definition (the puck touches the rail) that does not depend on which frame the
    camera happened to catch.
    """
    i0, i1, key = ev
    axis, line, inward = RIMS[key]
    n = len(t)
    impacts = [ev] if impacts is None else impacts
    lo, hi = (0, n - 1) if bounds is None else bounds
    pre, post = _leg_bounds(i0, i1, impacts, k, lo, hi, leg_frames(t))
    if (pre.stop - pre.start) < MIN_LEG_FRAMES or (post.stop - post.start) < MIN_LEG_FRAMES:
        return None

    # A bounce in the goal mouth is a bounce off the goal's own back/sides, not the rail the
    # model reflects on -- drop it.
    y_at_contact = float(np.mean(xy[i0:i1 + 1, 1]))
    if axis == 0 and abs(y_at_contact) < GOAL_WIDTH / 2 + PUCK_RADIUS:
        return None

    # 1st pass: velocities referenced to the nearest-frame time, to locate the crossing.
    d_near = inward * (xy[i0:i1 + 1, axis] - line)
    t_guess = float(t[i0 + int(np.argmin(d_near))])
    x_ref, v_ref, _ = _leg_velocity_at(t, xy, pre, t_guess, d_axis)
    vn_ref = inward * v_ref[axis]
    if vn_ref >= -1e-6:
        return None
    # crossing of the rim line by the incoming leg, from the same decay law:
    #   line = x_ref + v_ref*(1-exp(-d*dtau))/d  ->  solve for dtau (exactly, in closed form).
    d = max(d_axis[axis], 1e-6)
    arg = 1.0 - d * (line - x_ref[axis]) / v_ref[axis]
    t_imp = t_guess + (-np.log(arg) / d if arg > 1e-9 else 0.0)
    if not np.isfinite(t_imp) or not (t[max(0, i0 - 1)] <= t_imp <= t[min(n - 1, i1 + 1)]):
        t_imp = t_guess                       # extrapolation left the contact window: fall back

    # 2nd pass: both legs referenced to the impact instant.
    _, v_in, rms_in = _leg_velocity_at(t, xy, pre, t_imp, d_axis)
    _, v_out, rms_out = _leg_velocity_at(t, xy, post, t_imp, d_axis)
    if max(rms_in, rms_out) > MAX_LEG_RMSE:
        return None                           # not a free glide either side: not a rail bounce
    tangent = 1 - axis
    vn_in, vn_out = inward * v_in[axis], inward * v_out[axis]
    vt_in, vt_out = v_in[tangent], v_out[tangent]

    rec = dict(rim=key, t=t_imp, xy=xy[i0], vn_in=vn_in, vn_out=vn_out,
               vt_in=vt_in, vt_out=vt_out, e=np.nan, tan_ratio=np.nan)
    if -vn_in >= MIN_FIT_VN and vn_out > 0:
        e = vn_out / -vn_in
        if E_BOUNDS[0] <= e <= E_BOUNDS[1]:
            rec["e"] = e
    if abs(vt_in) >= MIN_FIT_VT:
        r = vt_out / vt_in
        if T_BOUNDS[0] <= r <= T_BOUNDS[1]:
            rec["tan_ratio"] = r
    return rec


# ------------------------------------------------------------------------ the three fits
def fit_damping(captures, verbose=True):
    """Per-axis + isotropic damping from every clean glide in the ``straight`` captures."""
    segs = []
    for cap in captures:
        for run in detected_runs(cap):
            t, xy = cap.t[run], cap.xy[run]
            for lo, hi in moving_segments(t, xy):
                # Free flight only. EVERY contact splits the roll, not just the ones at a rim:
                # a real session has the puck batted back by hand mid-table, and a glide fitted
                # across a strike measures the strike. A guard band on either side keeps the
                # contact frames themselves out. A bounce capture therefore also contributes
                # its (perfectly good) straight legs to the damping fit.
                ts_, ps_ = t[lo:hi + 1], xy[lo:hi + 1]
                keep = np.ones(hi + 1 - lo, bool)
                for i0, i1, _ in find_impacts(ts_, ps_):
                    keep[max(0, i0 - 2):i1 + 3] = False
                for i in find_contacts(ts_, ps_):
                    keep[max(0, i - 2):i + 3] = False
                for a0, b0 in true_runs(keep):
                    a, b = lo + a0, lo + b0
                    sl = slice(a, b + 1)
                    ts, ps = t[sl], xy[sl]
                    if len(ts) < 6 or ts[-1] - ts[0] < MIN_SEG_TIME:
                        continue
                    travel = float(np.linalg.norm(ps[-1] - ps[0]))
                    if travel < MIN_SEG_TRAVEL:
                        continue
                    p0, v0, d_iso, rmse_iso = fit_exp_decay_2d(ts, ps)
                    if rmse_iso > MAX_FIT_RMSE:
                        continue
                    per_axis = [fit_exp_decay(ts, ps[:, a_]) for a_ in (0, 1)]
                    # local image scale (m/px), for expressing the fit residual as a sigma_px
                    m_per_px = float(np.sqrt(np.abs(np.linalg.det(cap.J[run][a0:b0 + 1].mean(0)))))
                    segs.append(dict(src=os.path.basename(cap.path), t0=float(ts[0]),
                                     m_per_px=m_per_px,
                                     dur=float(ts[-1] - ts[0]), travel=travel, n=len(ts),
                                     v0=v0, d_iso=d_iso, rmse_iso=rmse_iso,
                                     rmse_lin=fit_linear_decel(ts, ps),
                                     d_axis=[per_axis[0][2], per_axis[1][2]],
                                     v0_axis=[per_axis[0][1], per_axis[1][1]],
                                     rmse_axis=[per_axis[0][3], per_axis[1][3]]))
    if verbose:
        print(f"  glide segments accepted: {len(segs)}")

    out = dict(segments=segs, d_axis=[np.nan, np.nan], d_iso=np.nan, n_axis=[0, 0])
    if not segs:
        return out
    w_iso = [s["dur"] * float(np.linalg.norm(s["v0"])) for s in segs]
    out["d_iso"] = weighted_median([s["d_iso"] for s in segs], w_iso)
    for a in (0, 1):
        use = [s for s in segs if abs(s["v0_axis"][a]) >= AXIS_MIN_SPEED
               and s["rmse_axis"][a] <= MAX_FIT_RMSE]
        out["n_axis"][a] = len(use)
        if use:
            out["d_axis"][a] = weighted_median(
                [s["d_axis"][a] for s in use],
                [s["dur"] * abs(s["v0_axis"][a]) for s in use])
    out["rmse_iso"] = float(np.median([s["rmse_iso"] for s in segs]))
    out["rmse_lin"] = float(np.median([s["rmse_lin"] for s in segs]))
    # The residual of a glide fit IS the measurement noise on a MOVING puck (the model has no
    # freedom left to absorb it), so express it in pixels the way the sensor's own noise model
    # is parameterised: sigma_px = residual / (metres per pixel).
    px = [s["rmse_iso"] / s["m_per_px"] for s in segs if s["m_per_px"] > 0]
    out["sigma_px_moving"] = float(np.median(px)) if px else np.nan
    return out


def fit_bounces(captures, d_axis, verbose=True):
    """Restitution + tangential retention from every usable impact in the ``bounce`` captures."""
    events = []
    for cap in captures:
        for run in detected_runs(cap):
            t, xy = cap.t[run], cap.xy[run]
            for lo, hi in moving_segments(t, xy):
                ts, ps = t[lo:hi + 1], xy[lo:hi + 1]
                impacts = find_impacts(ts, ps, smooth3(central_velocity(ts, ps)))
                for k, ev in enumerate(impacts):
                    rec = fit_one_bounce(ts, ps, ev, d_axis, impacts=impacts, k=k,
                                         bounds=(0, hi - lo))
                    if rec is not None:
                        rec["src"] = os.path.basename(cap.path)
                        events.append(rec)
    e = np.array([r["e"] for r in events], float)
    tr = np.array([r["tan_ratio"] for r in events], float)
    if verbose:
        print(f"  impacts found: {len(events)}   usable restitution: {int(np.isfinite(e).sum())}"
              f"   usable tangential: {int(np.isfinite(tr).sum())}")
    return dict(events=events,
                e=float(np.median(e[np.isfinite(e)])) if np.isfinite(e).any() else np.nan,
                e_mad=float(np.median(np.abs(e[np.isfinite(e)] - np.median(e[np.isfinite(e)]))))
                if np.isfinite(e).any() else np.nan,
                n_e=int(np.isfinite(e).sum()),
                tan=float(np.median(tr[np.isfinite(tr)])) if np.isfinite(tr).any() else np.nan,
                n_tan=int(np.isfinite(tr).sum()))


def _static_stretches(t, xy):
    """(i0, i1) ranges where the puck stayed inside a ``MAX_STATIC_SPREAD`` box for at least
    ``MIN_STATIC_TIME``. Greedy: anchor on a sample, extend while every later sample is still
    inside the box, close and re-anchor when one leaves."""
    out = []
    i = 0
    n = len(t)
    while i < n:
        j = i + 1
        while j < n and np.max(np.abs(xy[j] - xy[i])) <= MAX_STATIC_SPREAD / 2:
            j += 1
        if (t[j - 1] - t[i]) >= MIN_STATIC_TIME and (j - i) >= 20:
            out.append((i, j - 1))
            i = j
        else:
            i += 1                              # not a spot: slide the anchor forward
    return out


def fit_static(captures, verbose=True):
    """Measurement noise from the static-puck spots: world variance and pixel sigma."""
    spots = []
    for cap in captures:
        for run in detected_runs(cap):
            t, xy, uv = cap.t[run], cap.xy[run], cap.uv[run]
            yaw = cap.yaw[run]
            for a, b in _static_stretches(t, xy):
                sl = slice(a, b + 1)
                p = xy[sl]
                y = yaw[sl][np.isfinite(yaw[sl])]
                spots.append(dict(src=os.path.basename(cap.path), n=int(b - a + 1),
                                  dur=float(t[sl][-1] - t[sl][0]), mean=p.mean(axis=0),
                                  var=p.var(axis=0, ddof=1), var_px=uv[sl].var(axis=0, ddof=1),
                                  var_yaw=float(np.var(y, ddof=1)) if len(y) > 20 else np.nan,
                                  n_yaw=len(y)))
    if verbose:
        print(f"  static spots found: {len(spots)}")
    out = dict(spots=spots, var_xy=None, sigma_px=np.nan, var_yaw=np.nan)
    if not spots:
        return out
    out["var_xy"] = np.mean([s["var"] for s in spots], axis=0)
    out["sigma_px"] = float(np.sqrt(np.mean([s["var_px"].mean() for s in spots])))
    vy = [s["var_yaw"] for s in spots if np.isfinite(s["var_yaw"])]
    out["var_yaw"] = float(np.mean(vy)) if vy else np.nan
    return out


# ---------------------------------------------------------------------------- the report
def _mm(x):
    return f"{x * 1000:.2f} mm"


def report(damp, bounce, static, params, fitted, min_bounce_events=MIN_BOUNCE_EVENTS):
    old = DEFAULT_PARAMS
    print("\n================================ mini-table EKF fit ================================")
    print("-- damping (per-axis velocity decay, 1/s) ------------------------------------------")
    if damp["segments"]:
        print(f"   glides used            {len(damp['segments'])} "
              f"(x-axis {damp['n_axis'][0]}, y-axis {damp['n_axis'][1]})")
        print(f"   continuous fit         d_x={damp['d_axis'][0]:.4f}  d_y={damp['d_axis'][1]:.4f}"
              f"   isotropic d={damp['d_iso']:.4f}")
        print(f"   model check            exponential fit residual {_mm(damp['rmse_iso'])} vs "
              f"constant-decel {_mm(damp['rmse_lin'])} (median over glides)")
        if damp["rmse_lin"] < 0.6 * damp["rmse_iso"]:
            print("   !! constant deceleration fits MUCH better: the puck is friction-dominated, "
                  "not drag-dominated. The EKF's exponential form cannot match it at long "
                  "horizons -- treat `damping` as a best-effort compromise.")
        print(f"   -> damping             {np.array2string(old.damping, precision=4)} -> "
              f"{np.array2string(params.damping, precision=4)}  (per tick, dt={DT})")
    else:
        print("   no usable glides -- damping keeps the vendored KUKA values.")

    print("-- bounce (rim-local collision rows) -----------------------------------------------")
    if 0 < bounce["n_e"] < min_bounce_events:
        print(f"   only {bounce['n_e']} usable impact(s) (need {min_bounce_events}) -- n_params "
              f"KEEPS the vendored KUKA value.\n   The measured one(s) came to "
              f"restitution {bounce['e']:.3f} vs the vendored {-old.n_params[1]:.3f}; that is a "
              f"big enough gap\n   to be worth re-capturing bounces for. Lower the bar with "
              f"--min-bounce-events if the spread above is\n   tight and you accept the risk "
              f"of a systematic that only more rails would reveal.")
    if bounce["n_e"]:
        per_rim = {}
        for r in bounce["events"]:
            per_rim.setdefault(r["rim"], []).append(r["e"])
        for k, v in sorted(per_rim.items()):
            v = np.asarray(v, float)
            v = v[np.isfinite(v)]
            if len(v):
                print(f"   rim {k:3s}                n={len(v):2d}  restitution median "
                      f"{np.median(v):.3f}  spread {np.std(v):.3f}")
        print(f"   -> restitution         {bounce['e']:.3f} +/- {bounce['e_mad']:.3f} (MAD, "
              f"n={bounce['n_e']})   [vendored {-old.n_params[1]:.3f}]")
        print(f"   -> n_params            {np.array2string(old.n_params, precision=4)} -> "
              f"{np.array2string(params.n_params, precision=4)}")
    else:
        print("   no usable impacts -- n_params keeps the vendored KUKA values.")
    if np.isfinite(bounce["tan"]):
        print(f"   -> tangential          {bounce['tan']:.3f} (n={bounce['n_tan']})  "
              f"[vendored {old.b_params[0]:.3f}]")
        print(f"   -> b_params            {np.array2string(old.b_params, precision=4)} -> "
              f"{np.array2string(params.b_params, precision=4)}")
    print("   theta_params (spin)    NOT fitted -- needs the puck's angular velocity, which the "
          "real\n                          profile does not measure (yaw marker disabled).")

    print("-- measurement noise ---------------------------------------------------------------")
    if static["spots"]:
        for s in static["spots"]:
            print(f"   spot at ({s['mean'][0]:+.3f}, {s['mean'][1]:+.3f})  "
                  f"n={s['n']:5d} ({s['dur']:4.1f}s)  "
                  f"sd {_mm(np.sqrt(s['var'][0]))} / {_mm(np.sqrt(s['var'][1]))}  "
                  f"px sd {np.sqrt(s['var_px'][0]):.3f} / {np.sqrt(s['var_px'][1]):.3f}")
        print(f"   -> OBS_COV diag        {np.array2string(np.diag(old.OBS_COV), precision=3)} -> "
              f"{np.array2string(np.diag(params.OBS_COV), precision=3)}")
        print(f"      (i.e. position sd  {_mm(np.sqrt(params.OBS_COV[0, 0]))} x, "
              f"{_mm(np.sqrt(params.OBS_COV[1, 1]))} y)")
        print(f"   -> sigma_px (static)   {static['sigma_px']:.3f} px")
        if np.isfinite(damp.get("sigma_px_moving", np.nan)):
            print(f"   -> sigma_px (MOVING)   {damp['sigma_px_moving']:.2f} px  <-- PASS THIS to "
                  f"run_hw_match --sigma-px")
            print(f"      A moving puck is measured {damp['sigma_px_moving'] / static['sigma_px']:.1f}x "
                  f"worse than a still one (motion blur: the blob shrinks and its\n      centroid "
                  f"wanders). The filter's trust in a measurement should be set by the case that "
                  f"matters --\n      the puck in flight -- so use the moving figure. The static "
                  f"one is what OBS_COV encodes.")
        print("      NB OBS_COV only feeds the sim/GT update path; on hardware the sensor supplies "
              "its own R\n      from sigma_px, so refitting OBS_COV alone will NOT change the "
              "hardware filter.")
    else:
        print("   no static spots -- OBS_COV keeps the vendored KUKA values.")
    print("-----------------------------------------------------------------------------------")
    print(f"fitted from data: {', '.join(fitted) if fitted else 'NOTHING'}"
          f"   (everything else keeps the vendored KUKA value)")


def make_plot(path, damp, bounce, static):
    """Four diagnostic panels; skipped with a note if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (--plot skipped: {type(e).__name__}: {e})")
        return
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    a = ax[0, 0]
    for s in damp["segments"]:
        tt = np.linspace(0, s["dur"], 50)
        a.plot(tt, np.linalg.norm(s["v0"]) * np.exp(-s["d_iso"] * tt), lw=0.8, alpha=0.7)
    a.set(title=f"glide speed decay ({len(damp['segments'])} segments)",
          xlabel="t [s]", ylabel="|v| [m/s]")

    a = ax[0, 1]
    d = [s["d_iso"] for s in damp["segments"]]
    if d:
        a.hist(d, bins=max(6, len(d) // 3))
        a.axvline(damp["d_iso"], color="k", ls="--", label=f"median {damp['d_iso']:.3f}")
        a.axvline(DEFAULT_PARAMS.damping.mean(), color="r", ls=":",
                  label=f"KUKA {DEFAULT_PARAMS.damping.mean():.3f}")
        a.legend(fontsize=8)
    a.set(title="fitted decay rate d [1/s]", xlabel="d")

    a = ax[1, 0]
    ok = [r for r in bounce["events"] if np.isfinite(r["e"])]
    if ok:
        vin = [-r["vn_in"] for r in ok]
        vout = [r["vn_out"] for r in ok]
        a.scatter(vin, vout, s=18)
        lim = max(vin + vout + [0.1])
        a.plot([0, lim], [0, lim * bounce["e"]], "k--", label=f"e = {bounce['e']:.3f}")
        a.plot([0, lim], [0, lim * -DEFAULT_PARAMS.n_params[1]], "r:",
               label=f"KUKA e = {-DEFAULT_PARAMS.n_params[1]:.2f}")
        a.legend(fontsize=8)
    a.set(title="bounce: normal speed in vs out", xlabel="|v_in,n| [m/s]", ylabel="|v_out,n| [m/s]")

    a = ax[1, 1]
    for s in static["spots"]:
        a.scatter(*(np.sqrt(s["var"]) * 1000), s=40, label=f"{np.round(s['mean'], 2)}")
    if static["spots"]:
        a.legend(fontsize=7, title="spot xy [m]")
    a.set(title="static noise per spot", xlabel="sd x [mm]", ylabel="sd y [mm]")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  diagnostics plot -> {path}")


# ---------------------------------------------------------------------------------- main
def choose_damping(damp, per_axis="auto", min_axis_segments=4, damping_axis="both"):
    """The pair of continuous decay rates to write, or None when nothing was fitted.

    ``damping_axis`` in {"x", "y"} takes ONE axis's estimate and uses it for both entries.
    Drag is a property of the air film, not of a direction, so a single well-measured axis
    describes the table -- and a badly-measured one describes nothing. Use it when only one
    axis was cleanly exercised: cross-table rolls bouncing between the long rails give a clean
    y, while the along-table motion is the one an operator keeps interrupting by hand.
    """
    if not np.isfinite(damp["d_iso"]):
        return None
    if damping_axis in ("x", "y"):
        a = 0 if damping_axis == "x" else 1
        if not np.isfinite(damp["d_axis"][a]):
            raise SystemExit(f"--damping-axis {damping_axis}: no glide had enough speed along "
                             f"{damping_axis} to fit it (n={damp['n_axis'][a]})")
        print(f"  damping from the {damping_axis} axis only ({damp['n_axis'][a]} glides): "
              f"d={damp['d_axis'][a]:.4f} 1/s, used for both axes.")
        return [damp["d_axis"][a]] * 2

    # Per-axis only when there is enough data on BOTH axes AND the two estimates differ by
    # more than the scatter of the individual glide fits. On real captures that spread is
    # large (a short roll pins a decay rate loosely), so a few-percent gap between the axis
    # medians is noise, and baking it in would claim a table asymmetry that was not measured.
    spread = (float(np.median(np.abs(np.array([sg["d_iso"] for sg in damp["segments"]])
                                     - damp["d_iso"]))) if damp["segments"] else np.inf)
    damp["axis_spread"] = spread
    enough = (all(n >= min_axis_segments for n in damp["n_axis"])
              and all(np.isfinite(damp["d_axis"]))
              and abs(damp["d_axis"][0] - damp["d_axis"][1]) > spread)
    use_axis = enough if per_axis == "auto" else (per_axis == "yes")
    if use_axis and not all(np.isfinite(damp["d_axis"])):
        # --per-axis yes was asked for but an axis has no usable glide at all. Writing the NaN
        # would be refused by the load hook; falling back is what the user meant.
        print(f"  --per-axis yes: axis fits are {damp['d_axis']} -- one axis has no glide fast "
              f"enough along it, so the isotropic fit is used for both.")
        use_axis = False
    if use_axis:
        return list(damp["d_axis"])
    if per_axis == "auto" and all(np.isfinite(damp["d_axis"])):
        print(f"  per-axis damping needs >={min_axis_segments} glides on each axis "
              f"(have {damp['n_axis']}) AND an x-vs-y gap bigger than the per-glide scatter\n"
              f"  (gap {abs(damp['d_axis'][0] - damp['d_axis'][1]):.3f} vs scatter "
              f"{spread:.3f}) -- using the isotropic fit for both.")
    return [damp["d_iso"]] * 2


def fit(captures, per_axis="auto", min_axis_segments=4, damping_axis="both",
        min_bounce_events=MIN_BOUNCE_EVENTS):
    """Run the three fits and assemble an :class:`EkfParams`. -> (params, fitted, details)."""
    by_tag = {tag: [c for c in captures if c.tag == tag]
              for tag in ("straight", "bounce", "static")}
    for tag, caps in by_tag.items():
        print(f"{tag:9s}: {len(caps)} capture(s), "
              f"{sum(int(c.good.sum()) for c in caps)} detected frames")

    damp = fit_damping(by_tag["straight"] + by_tag["bounce"])   # bounce legs glide too

    # The bounce legs are extrapolated to the impact with the decay law, so they need a decay
    # rate: this run's per-axis fit, else its isotropic one, else the vendored value. (The
    # legs are ~0.25 s long, so the choice moves a restitution by well under a percent -- but
    # it must be SOME sane number even when the straight captures are missing.)
    def d_for(axis):
        for candidate in (damp["d_axis"][axis], damp["d_iso"]):
            if np.isfinite(candidate):
                return float(candidate)
        return float(DEFAULT_PARAMS.damping[axis])

    bounce = fit_bounces(by_tag["bounce"], d_axis=[d_for(0), d_for(1)])
    static = fit_static(by_tag["static"])

    fitted = []
    kw = {}
    d_c = choose_damping(damp, per_axis, min_axis_segments, damping_axis)
    if d_c is not None:
        kw["damping"] = np.array([continuous_to_discrete(d) for d in d_c])
        fitted.append("damping")
    if np.isfinite(bounce["e"]) and bounce["n_e"] >= min_bounce_events:
        n = DEFAULT_PARAMS.n_params.copy()
        n[1] = -bounce["e"]
        kw["n_params"] = n
        fitted.append("n_params")
    if np.isfinite(bounce["tan"]) and bounce["n_tan"] >= min_bounce_events:
        b = DEFAULT_PARAMS.b_params.copy()
        b[0] = bounce["tan"]
        kw["b_params"] = b
        fitted.append("b_params")
    if static["var_xy"] is not None:
        obs = DEFAULT_PARAMS.OBS_COV.copy()
        obs[0, 0], obs[1, 1] = static["var_xy"]
        if np.isfinite(static["var_yaw"]):
            obs[2, 2] = static["var_yaw"]
        kw["OBS_COV"] = obs
        fitted.append("OBS_COV")

    return EkfParams(**kw), fitted, dict(damping=damp, bounce=bounce, static=static)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+", help="npz files from hw.capture_puck_rolls")
    ap.add_argument("--out", default="mini_table_ekf.npz", help="EKF-params npz to write")
    ap.add_argument("--per-axis", choices=["auto", "yes", "no"], default="auto",
                    help="per-axis damping (auto: only when both axes have enough glides)")
    ap.add_argument("--min-axis-segments", type=int, default=4,
                    help="glides required per axis before 'auto' trusts a per-axis fit")
    ap.add_argument("--damping-axis", choices=["both", "x", "y"], default="both",
                    help="fit the decay from ONE axis and use it for both (drag is a property "
                         "of the air film, not of a direction). Use when only one axis was "
                         "cleanly exercised -- e.g. cross-table rolls off the long rails while "
                         "the along-table motion kept being interrupted by hand")
    ap.add_argument("--min-bounce-events", type=int, default=MIN_BOUNCE_EVENTS,
                    help="usable impacts required before a rim row is written at all "
                         "(a median of one or two events is not a measurement)")
    ap.add_argument("--plot", default=None, help="write diagnostic panels to this PNG")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)

    captures = [load_capture(p) for p in args.captures]
    for c in captures:
        print(f"{os.path.basename(c.path):28s} tag={c.tag:8s} {len(c):5d} frames "
              f"{100.0 * c.good.mean():5.1f}% detected  {c.fps:5.1f} fps")
    unknown = [c.path for c in captures if c.tag not in ("straight", "bounce", "static")]
    if unknown:
        raise SystemExit(f"unknown tag in {unknown}")

    params, fitted, details = fit(captures, args.per_axis, args.min_axis_segments,
                                  args.damping_axis, args.min_bounce_events)
    report(details["damping"], details["bounce"], details["static"], params, fitted,
           args.min_bounce_events)
    if args.plot:
        make_plot(args.plot, details["damping"], details["bounce"], details["static"])

    if not fitted:
        raise SystemExit(
            "\nnothing could be fitted, so there is nothing to write: the file would be the "
            "vendored\nKUKA constants under a new name, which is worse than no file at all. "
            "Check the report\nabove -- the usual causes are too few / too short rolls, a "
            "detection that never locked on\n(det % low in the capture), or captures saved "
            "under the wrong --tag.")
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return params, details
    # sigma_px recorded in the file is the MOVING-puck one: that is the case the filter's
    # trust actually has to be right for, and it is what the runtime's reminder quotes.
    sigma_px = details["damping"].get("sigma_px_moving", np.nan)
    if not np.isfinite(sigma_px):
        sigma_px = details["static"]["sigma_px"]
    params.save(args.out,
                fitted=np.array(fitted),
                sources=np.array([os.path.abspath(p) for p in args.captures]),
                sigma_px=np.array(sigma_px),
                sigma_px_static=np.array(details["static"]["sigma_px"]),
                n_glides=np.array(len(details["damping"]["segments"])),
                n_impacts=np.array(details["bounce"]["n_e"]),
                n_static_spots=np.array(len(details["static"]["spots"])),
                dt=np.array(DT))
    print(f"\nwrote {args.out}. Validate BEFORE trusting it on the arm:\n"
          f"    python -m mini_platform.tools.ekf_replay {' '.join(args.captures)} "
          f"--ekf-params {args.out}\n"
          f"then run with:  run_hw_match --stage play ... --ekf-params {args.out}"
          + (f" --sigma-px {sigma_px:.3f}" if np.isfinite(sigma_px) else ""))
    return params, details


if __name__ == "__main__":
    main()
