"""Intercept rate-limit (``MiniSystemState``): the output-side clamp on ``predicted_state[:2]``
that tames the noisy-velocity intercept explosion (measured up to ~1.3 m/tick) driving hardware
joint jerk. Pins that the clamp bounds the per-tick jump when enabled, preserves direction, and
is an exact no-op (the vendored ``get_prediction`` result) when off -- the guard that keeps sim
behaviour unchanged.
"""
import numpy as np
import pytest

import mini_platform
from mini_platform.agents.baseline_mini import MiniBaselineAgent
from mini_platform.estimation.kalman_filter import PuckTracker


def _env_info(task="mini-hit-sdk"):
    return mini_platform.make(task).env_info


def _pin_static_puck(state, x, y):
    """Park the EKF at a static point so ``get_prediction`` returns ~that point (zero velocity
    => the intercept IS the position), giving a controllable raw intercept."""
    state.puck_tracker.state = np.array([x, y, 0., 0., 0., 0.])
    state.puck_tracker.P = np.eye(6) * 1e-4


def test_intercept_clamp_bounds_the_per_tick_jump():
    st = MiniBaselineAgent(_env_info(), agent_id=1, intercept_max_jump=0.05).state
    # tick 1 (first acquisition, never clamped): establishes the anchor. x=0.2 is inside the
    # reachable box (own-goal line ~0.113 .. hit reach ~0.511), so the workspace clamp is a no-op.
    st._intercept_anchor = None
    _pin_static_puck(st, 0.2, 0.0)
    st.update_prediction(1.0)
    i1 = st.predicted_state[:2].copy()
    assert i1 == pytest.approx([0.2, 0.0], abs=1e-3)
    # tick 2: a ~0.5 m raw jump is clamped to exactly max_jump from the frozen anchor
    st._intercept_anchor = st._intercept_last            # what update_observation freezes each tick
    _pin_static_puck(st, 0.5, 0.3)
    st.update_prediction(1.0)
    i2 = st.predicted_state[:2].copy()
    assert np.linalg.norm(i2 - i1) == pytest.approx(0.05, abs=1e-6)
    # clamped ALONG the raw step (direction preserved), not snapped elsewhere
    raw_dir = np.array([0.5, 0.3]) - i1
    assert np.allclose((i2 - i1) / 0.05, raw_dir / np.linalg.norm(raw_dir), atol=1e-6)


def test_intercept_clamp_off_is_raw_predicted_state():
    """Default (no clamp) -> predicted_state is the vendored get_prediction, jump unbounded."""
    st = MiniBaselineAgent(_env_info(), agent_id=1).state           # intercept_max_jump=None
    _pin_static_puck(st, 0.1, 0.0)
    st.update_prediction(1.0)
    i1 = st.predicted_state[:2].copy()
    _pin_static_puck(st, 0.5, 0.3)
    st.update_prediction(1.0)
    i2 = st.predicted_state[:2].copy()
    assert np.linalg.norm(i2 - i1) == pytest.approx(np.linalg.norm([0.4, 0.3]), abs=2e-3)


def test_intercept_ema_smooths_toward_anchor():
    st = MiniBaselineAgent(_env_info(), agent_id=1, intercept_ema_alpha=0.25).state
    st._intercept_anchor = None
    _pin_static_puck(st, 0.0, 0.0)
    st.update_prediction(1.0)
    i1 = st.predicted_state[:2].copy()                             # ~[0, 0]
    st._intercept_anchor = st._intercept_last
    _pin_static_puck(st, 0.4, 0.0)
    st.update_prediction(1.0)
    i2 = st.predicted_state[:2].copy()
    # EMA: new = anchor + alpha*(raw - anchor) = 0.25 * 0.4 ~ 0.1 from the ~0 anchor
    assert i2[0] == pytest.approx(i1[0] + 0.25 * (0.4 - i1[0]), abs=2e-3)


def _pin_moving_puck(state, x, y, vx, vy):
    """Park the EKF at a moving point so ``get_prediction`` forward-integrates a real velocity."""
    state.puck_tracker.state = np.array([x, y, vx, vy, 0., 0.])
    state.puck_tracker.P = np.eye(6) * 1e-4


def test_latency_lead_shifts_intercept_along_predicted_velocity():
    """--lead-time aims t_lead further along the predicted trajectory: the led intercept equals
    the raw predicted position + the predicted-point velocity * t_lead (workspace clamp a no-op
    here since the target stays inside the reachable box)."""
    # A slow puck that stays well inside the box over the short horizon (no defend-line early stop).
    seed = (0.30, 0.0, -0.20, 0.05)
    st0 = MiniBaselineAgent(_env_info(), agent_id=1).state          # lead_time=0 (raw reference)
    _pin_moving_puck(st0, *seed)
    st0.update_prediction(0.2)
    P = st0.predicted_state[:4].copy()                             # raw predicted [x, y, vx, vy]

    t_lead = 0.08
    stL = MiniBaselineAgent(_env_info(), agent_id=1, lead_time=t_lead).state
    _pin_moving_puck(stL, *seed)
    stL.update_prediction(0.2)
    assert stL.predicted_state[:2] == pytest.approx(P[:2] + P[2:4] * t_lead, abs=1e-6)
    # and it is a real forward shift (the puck is moving, so the lead is non-zero)
    assert np.linalg.norm(stL.predicted_state[:2] - P[:2]) > 1e-3


def test_latency_lead_zero_is_byte_identical():
    """Default lead_time=0 -> predicted_state is the exact vendored get_prediction (sim/tests
    unchanged), even for a moving puck."""
    seed = (0.30, 0.0, -0.20, 0.05)
    st = MiniBaselineAgent(_env_info(), agent_id=1, lead_time=0.0).state
    _pin_moving_puck(st, *seed)
    st.update_prediction(0.2)
    ref = MiniBaselineAgent(_env_info(), agent_id=1).state          # no lead arg at all
    _pin_moving_puck(ref, *seed)
    ref.update_prediction(0.2)
    assert st.predicted_state[:2] == pytest.approx(ref.predicted_state[:2], abs=1e-12)


def test_max_prediction_time_override():
    """--max-prediction-time overrides the vendored 1.0 s horizon cap in agent_params."""
    a = MiniBaselineAgent(_env_info(), agent_id=1, max_prediction_time=1.5)
    assert a.agent_params['max_prediction_time'] == pytest.approx(1.5)
    b = MiniBaselineAgent(_env_info(), agent_id=1)                  # default untouched
    assert b.agent_params['max_prediction_time'] == pytest.approx(1.0)


def test_max_puck_speed_caps_velocity():
    """The physical speed cap bounds the estimated velocity (kills detection-outlier spikes)."""
    pt = PuckTracker(_env_info(), agent_id=1, max_puck_speed=4.0)
    pt.reset(np.array([0.0, 0.0, 0.0]))
    pt.state[2:4] = [10.0, 0.0]                     # unphysical 10 m/s
    pt._clamp_speed()
    assert np.linalg.norm(pt.state[2:4]) == pytest.approx(4.0)


def test_max_puck_speed_unset_is_noop():
    """Default (None) leaves velocity untouched -> the GT/sim path stays bitwise-identical."""
    pt = PuckTracker(_env_info(), agent_id=1)       # max_puck_speed=None
    pt.reset(np.array([0.0, 0.0, 0.0]))
    pt.state[2:4] = [10.0, 0.0]
    pt._clamp_speed()
    assert np.linalg.norm(pt.state[2:4]) == pytest.approx(10.0)


def test_intercept_workspace_clamp_keeps_it_reachable():
    """A far off-table intercept (a velocity-spike extrapolation) is clamped into the arm's
    reachable box, so the arm can't slew out past its reach and over-extend into a joint-limit
    fault (the observed crash: intercept ran to V-x ~+1.0 and the arm ramped to its limits)."""
    from mini_platform.agents.baseline_mini import V_SETBACK
    a = MiniBaselineAgent(_env_info(), agent_id=1, intercept_max_jump=10.0)  # big jump -> only the
    st = a.state                                                             # workspace clamp acts
    st._intercept_anchor = None
    _pin_static_puck(st, V_SETBACK + 2.0, 1.0)      # way off-table toward the far end
    st.update_prediction(1.0)
    hit_x_max = a.agent_params['hit_range'][1]
    hw = 0.5 * a.env_info['table']['width']
    assert st.predicted_state[0] <= hit_x_max + 1e-9
    assert abs(st.predicted_state[1]) <= hw + 1e-9
