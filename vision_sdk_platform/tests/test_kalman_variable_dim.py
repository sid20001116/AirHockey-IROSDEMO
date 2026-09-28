"""Vision Part 3: variable-dimension EKF update, NIS gating, dropout (pure numpy).

The GT-regression case pins the default 3-D/R=None path BITWISE against a frozen
verbatim copy of the pre-change update math, so the vendored filter's behaviour on the
ground-truth path is provably untouched by the Part 3 edits.
"""

import numpy as np
import pytest

import mini_platform
from mini_platform.estimation.kalman_filter import PuckTracker


@pytest.fixture(scope="module")
def env_info():
    return mini_platform.make("mini-hit").env_info


def _frozen_original_update(tracker, measurement, predicted_state, P):
    """Verbatim copy of PuckTracker.update as of commit 1f82a86 (pre-Part 3),
    kalman_filter.py:194-202 -- the bitwise reference for the default GT path."""
    xy_innovation = measurement[:2] - predicted_state[:2]
    theta_innovation = (measurement[2] - predicted_state[4] + np.pi) % (np.pi * 2) - np.pi
    y = np.concatenate([xy_innovation, [theta_innovation]])
    S = tracker.H @ P @ tracker.H.T + tracker.R
    K = P @ tracker.H.T @ np.linalg.inv(S)
    state = predicted_state + K @ y
    P = (np.eye(6) - K @ tracker.H) @ P
    return state, P


def _smooth_sequence(n=200, dt=0.02):
    """Constant-velocity 3-D measurements that stay far off every rim of the tracker's
    table, so the bounce branch never fires: v = (-0.05, 0.02) m/s, omega = 0.1 rad/s."""
    t = np.arange(1, n + 1) * dt
    return np.stack([0.2 - 0.05 * t, 0.05 + 0.02 * t, 0.1 + 0.1 * t], axis=1)


def test_gt_regression_bitwise(env_info):
    """(a) 200 smooth 3-D steps with R=None: state AND P bitwise-identical to
    predict + the frozen original update at every step, and the gate never fired."""
    trk = PuckTracker(env_info)
    ref = PuckTracker(env_info)
    trk.reset(np.array([0.2, 0.05, 0.1]))
    ref.reset(np.array([0.2, 0.05, 0.1]))
    for m in _smooth_sequence():
        trk.step(m)
        pred, P = ref.predict(ref.state, ref.P)
        ref.state, ref.P = _frozen_original_update(ref, m, pred, P)
        assert np.array_equal(trk.state, ref.state)
        assert np.array_equal(trk.P, ref.P)
    assert trk._reject_count == 0  # proves the no-rejection path ran throughout


def test_vel_lpf_attenuates_a_velocity_jump(env_info):
    """The optional output-velocity LPF attenuates a sudden velocity change by the exact
    first-order factor alpha = dt/(tau+dt), tau = 1/(2*pi*fc). Deterministic (no EKF dynamics)."""
    fc = 5.0
    trk = PuckTracker(env_info, vel_lpf_hz=fc)
    trk.reset(np.array([0.2, 0.0, 0.0]))
    trk.state[2:4] = [0.0, 0.0]
    trk._apply_vel_lpf()                       # seeds the LPF state at [0, 0]
    trk.state[2:4] = [1.0, 0.0]                # impose a sudden 1 m/s velocity jump
    trk._apply_vel_lpf()                       # ... which the LPF attenuates
    alpha = trk.system.dt / (1.0 / (2.0 * np.pi * fc) + trk.system.dt)
    assert trk.state[2] == pytest.approx(alpha * 1.0, abs=1e-9)   # (1-alpha)*0 + alpha*1
    assert 0.0 < trk.state[2] < 1.0                               # strictly between old and new


def test_vel_lpf_default_is_byte_identical(env_info):
    """vel_lpf_hz=None (default) -> step() is bitwise the vendored filter (the LPF is a pure
    no-op), so sim / GT behaviour is unchanged."""
    a = PuckTracker(env_info)                        # default (no LPF)
    b = PuckTracker(env_info, vel_lpf_hz=None)        # explicit None
    a.reset(np.array([0.2, 0.05, 0.1]))
    b.reset(np.array([0.2, 0.05, 0.1]))
    for m in _smooth_sequence():
        a.step(m)
        b.step(m)
        assert np.array_equal(a.state, b.state)
        assert np.array_equal(a.P, b.P)


def test_2d_update_leaves_yaw_untouched(env_info):
    """(b) A 2-D update whose measurement equals the predicted xy: state bitwise
    unchanged, yaw variance untouched (xy and yaw blocks are decoupled off-bounce),
    but the xy covariance block did contract."""
    trk = PuckTracker(env_info)
    trk.reset(np.array([0.2, 0.05, 0.1]))
    pred, P = trk.predict(trk.state, trk.P)
    state, P_new = trk.update(pred[:2].copy(), pred, P)
    assert np.array_equal(state, pred)
    assert P_new[4, 4] == P[4, 4]
    assert not np.array_equal(P_new, P)


def test_dropout_predicts_ballistically(env_info):
    """(c) step(None) commits the prediction: position advances, tr(P) strictly grows."""
    trk = PuckTracker(env_info)
    trk.reset(np.array([0.2, 0.05, 0.1]))
    for m in _smooth_sequence(30):
        trk.step(m)
    state_before = trk.state.copy()
    pred, _ = trk.predict(trk.state, trk.P)
    tr_before = np.trace(trk.P)
    trk.step(None)
    assert np.array_equal(trk.state, pred)
    assert np.linalg.norm(trk.state[:2] - state_before[:2]) > 1e-4
    assert np.trace(trk.P) > tr_before


#: A typical vision-sensor covariance. The NIS gate applies only to measurements that
#: carry one -- R=None is the trusted GT channel and bypasses the gate entirely
#: (measured: gating GT fires ~89 times over 20 mini-defend episodes at bounces,
#: which would break the byte-identical-GT requirement).
R_VISION = np.diag([2e-6, 2e-6, 1.7e-3])


def test_gt_channel_bypasses_gate(env_info):
    """R=None must take the unconditional update even for a wild outlier -- the gate
    exists to defend against fallible sensors, and GT is not one."""
    trk = PuckTracker(env_info)
    trk.reset(np.array([0.2, 0.05, 0.1]))
    for _ in range(20):
        trk.step(np.array([0.2, 0.05, 0.1]))
    trk.step(np.array([1.2, 0.05, 0.1]))     # 1 m jump, no R: accepted
    assert trk._reject_count == 0
    assert abs(trk.state[0] - 0.2) > 0.1     # the update actually moved the state


def test_nis_gate_rejects_then_relocks(env_info):
    """(d) After ~20 consistent measurements (settled P -- straight after reset the
    diffuse eye(6) would let a 1 m outlier through with NIS ~ 1), a 1 m outlier is
    rejected predict-only; a SECOND consistent outlier is a real new track and
    re-locks immediately, with velocity differenced from the two frames."""
    trk = PuckTracker(env_info)
    trk.reset(np.array([0.2, 0.05, 0.1]))
    for _ in range(20):
        trk.step(np.array([0.2, 0.05, 0.1]), R_VISION)
    assert trk._reject_count == 0

    outlier = np.array([1.2, 0.05, 0.1])
    pred, P = trk.predict(trk.state, trk.P)
    trk.step(outlier, R_VISION)
    assert trk._reject_count == 1
    assert np.array_equal(trk.state, pred)
    assert np.array_equal(trk.P, P)

    trk.step(outlier, R_VISION)              # consistent repeat -> two-point re-lock
    assert trk._reject_count == 0
    assert np.array_equal(trk.state[[0, 1, 4]], outlier)
    assert np.array_equal(trk.state[2:4], np.zeros(2))   # differenced velocity: static
    assert np.array_equal(trk.P, np.eye(6))


def test_nis_gate_inconsistent_storm_falls_back(env_info):
    """(d') Scattered outliers (a genuine glitch storm) never two-point re-lock --
    glitches do not repeat consistently -- and the 5-streak plain reset remains the
    last-resort fallback."""
    trk = PuckTracker(env_info)
    trk.reset(np.array([0.2, 0.05, 0.1]))
    for _ in range(20):
        trk.step(np.array([0.2, 0.05, 0.1]), R_VISION)

    glitches = [np.array([1.2, 0.05, 0.1]), np.array([-0.8, -0.9, 0.1]),
                np.array([1.5, 0.8, 0.1]), np.array([-1.0, 0.7, 0.1])]
    for i, g in enumerate(glitches):
        trk.step(g, R_VISION)
        assert trk._reject_count == i + 1
    final = np.array([2.0, -0.6, 0.3])
    trk.step(final, R_VISION)                # 5th consecutive: plain re-lock
    assert trk._reject_count == 0
    assert np.array_equal(trk.state[[0, 1, 4]], final)
    assert np.array_equal(trk.P, np.eye(6))


def test_nis_gate_bypassed_near_rim(env_info):
    """A huge innovation NEXT TO A RIM is bounce-timing model error, not a sensor
    fault: the measurement must be accepted (near-unity gain snaps the state) rather
    than rejected into predict-only drift."""
    trk = PuckTracker(env_info)
    b = trk.system.table.boundary
    near_rim = np.array([0.3, b[1, 1, 1] - 0.02, 0.0])   # 20 mm off the +y rim line
    trk.reset(near_rim)
    for _ in range(20):
        trk.step(near_rim, R_VISION)
    # 25 mm jump: NIS ~ (25/1.5)^2 >> 16, and the measurement is still within
    # RIM_MARGIN of the rim line, as a really-bounced puck is one step after contact.
    bounced = near_rim + np.array([0.0, -0.025, 0.0])
    trk.step(bounced, R_VISION)
    assert trk._reject_count == 0
    # Steady-state gain closes ~60% per frame (process noise ~ R), so one frame
    # halves the innovation and two more make it millimetric -- vs 5 rejected frames
    # of ballistic drift without the bypass.
    assert np.linalg.norm(trk.state[:2] - bounced[:2]) < 0.012
    for _ in range(2):
        trk.step(bounced, R_VISION)
    assert np.linalg.norm(trk.state[:2] - bounced[:2]) < 0.003


def test_prior_reset_converges_fast(env_info):
    """(e) Reset from the own-half prior: the first real (vision) measurements pass
    the gate trivially (P = eye(6) is genuinely diffuse) and the near-unity gain
    snaps the state within 3 steps."""
    trk = PuckTracker(env_info)
    trk.reset(np.array([0.367, 0.0, 0.0]))
    target = np.array([0.2, -0.1, 0.5])
    for _ in range(3):
        trk.step(target, R_VISION)
    assert np.allclose(trk.state[[0, 1, 4]], target, atol=1e-3)
