"""The EKF must acquire a struck puck's velocity when detections are sparser than the control tick.

THE BUG (found from hardware logs vp_calm.npz / vp_fast_cam.npz, 11-18 detections/s on a 50 Hz
loop): ``PuckTracker._step_impl`` cleared ``_last_rejected`` on every tick that carried no camera
frame. Since ~65% of ticks are coast ticks at a real detection rate, the first half of a two-point
re-lock was always wiped before the second measurement arrived, so the two-point re-lock -- the
ONLY path that seeds a velocity -- could never fire on the vision path. The filter's sole escape
was the 5-rejection plain reset, which zeroes velocity. Measured result: a puck crossing the table
at 2 m/s was estimated at v = 0.00 for the entire flight, which the arm executes as aiming at a
stationary puck (the "close miss, undershoot" report).

Two further per-STEP quantities had to scale with the real gap once the re-lock could fire:
``RELOCK_DIST`` (a travel bound) and the velocity divisor (was one control tick, not one frame).

The ground-truth path (R=None) never enters the gate, so sim behaviour is unchanged by all three.
"""
import numpy as np
import pytest

import mini_platform
from mini_platform.estimation.kalman_filter import PuckTracker


@pytest.fixture(scope="module")
def env_info():
    return mini_platform.make("mini-hit").env_info


def _settled_tracker(env_info):
    """A tracker whose P has converged on a puck at rest at the origin. Straight after reset()
    P = I, so S is huge and the gate accepts anything -- a rejection is only possible once the
    filter is actually confident."""
    t = PuckTracker(env_info, agent_id=1)
    t.reset(np.array([0.0, 0.0, 0.0]))
    R = np.eye(3) * 1e-8
    for _ in range(40):
        t.step(np.array([0.0, 0.0, 0.0]), R)
    return t


def _cross_table(env_info, v_true, det_hz, seconds=None, sigma=5e-4, seed=0):
    """Puck crossing at a constant ``v_true`` with detections at ``det_hz`` on the 50 Hz control
    loop (the vision thread coasts in between, exactly as ThreadedPuckSensor makes it). Returns
    the estimated-speed trace.

    The puck starts ON THE TABLE and the run is cut short of the far rim, because this file is
    about free-flight velocity acquisition: a rim is where ``_near_rim`` bypasses the NIS gate
    and the bounce model takes over, so a run that reaches one measures something else. The
    tracker's rim lines here are x = 0.138 and x = 1.104, |y| = 0.229, so 0.85 m of travel from
    (0.20, 0.05) stays in free flight at every speed tested.

    (It used to start at x = -0.45. That was on the table for the SIDE mount; since the
    2026-09-10 re-mount the tracker frame puts it 563 mm BEHIND the own goal line, off the
    table entirely, and the bounce model dominated the trace.)
    """
    rng = np.random.default_rng(seed)
    t = PuckTracker(env_info, agent_id=1)
    dt = t.system.dt
    p = np.array([0.20, 0.05])
    t.reset(np.array([p[0], p[1], 0.0]))
    R = np.eye(3) * sigma ** 2
    R[2, 2] = 1e-3
    every = max(1, round((1.0 / det_hz) / dt))
    if seconds is None:
        seconds = min(0.6, 0.85 / v_true)
    out = []
    for k in range(int(seconds / dt)):
        p = p + np.array([v_true, 0.0]) * dt
        if k % every == 0:
            t.step(np.array([p[0] + rng.normal(0, sigma), p[1] + rng.normal(0, sigma), 0.0]), R)
        else:
            t.step(None)
        out.append(np.linalg.norm(t.state[2:4]))
    return np.array(out)


@pytest.mark.parametrize("v_true", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("det_hz", [11, 18, 30])
def test_velocity_is_acquired_at_real_detection_rates(env_info, v_true, det_hz):
    """By the end of a 0.6 s crossing the filter must report the true speed, not zero.

    Fails on the pre-fix code with est = 0.00 for 1 and 2 m/s: the re-lock never fires and every
    plain reset re-zeroes the velocity.
    """
    est = _cross_table(env_info, v_true, det_hz)
    assert est[-1] == pytest.approx(v_true, rel=0.10), (
        f"{det_hz} Hz detections, true {v_true} m/s -> filter reported {est[-1]:.2f} m/s")


def test_locks_on_within_half_a_second(env_info):
    """A strike is over in a few tenths of a second, so acquisition has to be fast enough to aim."""
    est = _cross_table(env_info, 2.0, 11)
    dt = 0.02
    locked = np.flatnonzero(est > 0.5 * 2.0)
    assert locked.size, "never acquired the puck velocity at all"
    assert locked[0] * dt <= 0.5, f"took {locked[0] * dt:.2f}s to acquire (strike is already over)"


def test_last_rejected_survives_coast_ticks_but_ages_out(env_info):
    """A tick with no camera frame carries no evidence, so it must not discard the pending
    rejection -- but a stale one must not linger forever either."""
    t = _settled_tracker(env_info)
    R = np.eye(3) * 1e-8
    t.step(np.array([0.30, 0.0, 0.0]), R)              # far -> rejected, records _last_rejected
    assert t._last_rejected is not None, "fixture did not trigger a rejection"
    for _ in range(3):
        t.step(None)
    assert t._last_rejected is not None, "coast tick wiped the pending rejection (the bug)"
    assert t._last_rejected_age == 3
    for _ in range(t.RELOCK_MAX_AGE + 2):
        t.step(None)
    assert t._last_rejected is None, "a stale rejection must age out"


def test_relock_span_scales_with_the_gap(env_info):
    """RELOCK_DIST is a per-step travel bound; over N ticks the puck may legitimately travel N
    times as far, so the consistency window must scale or a fast puck can never re-lock."""
    t = _settled_tracker(env_info)
    R = np.eye(3) * 1e-8
    t.step(np.array([0.30, 0.0, 0.0]), R)
    for _ in range(3):
        t.step(None)
    # 0.12 m apart: beyond the single-tick RELOCK_DIST (0.08) but inside 4 ticks' worth.
    t.step(np.array([0.42, 0.0, 0.0]), R)
    assert np.linalg.norm(t.state[2:4]) > 1.0, "re-lock refused a legitimate one-frame displacement"
    assert np.linalg.norm(t.state[2:4]) == pytest.approx(0.12 / (4 * t.system.dt), rel=0.05)


def test_ground_truth_path_is_untouched(env_info):
    """R=None is the trusted GT channel: it must never enter the gate, so none of this applies."""
    t = PuckTracker(env_info, agent_id=1)
    t.reset(np.array([0.0, 0.0, 0.0]))
    for k in range(5):
        t.step(np.array([0.01 * k, 0.0, 0.0]))
    assert t._last_rejected is None
    assert t._last_rejected_age == 0
