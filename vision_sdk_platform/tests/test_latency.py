"""Latency (CONTROLLER_FIX_PLAN.md Fix 8).

The arm trails even a feasible plan by 60-180 ms, detections arrive 40-60 ms apart, and the
capture time of a measurement was never used at all. Three separable delays, each with its own
knob here:

  * **8c, measurement age** -- a camera measurement is a camera period + detection + thread
    hand-off old by the time the EKF sees it. Fusing it against the CURRENT prediction asserts
    the puck is there NOW: at 2 m/s, 50 ms of age is 100 mm of position error, which
    `get_prediction` then forward-integrates into the intercept the arm is sent to. The frame's
    own stamp now travels with the measurement, and `--meas-lag-comp` fuses against the state as
    of that stamp.
  * **8d, follower lag** -- the RT servo trails its setpoint. `--stream-lead-ms` commands the
    plan tau ahead of the clock, INSIDE the speed/acceleration caps, and `--stage track-test`
    measures the tau to use.
  * **8e, replan cadence** -- `--max-plan-steps` buys a fresher target for more replans.

Everything new is off by default: the sim and ground-truth paths must stay byte-identical, which
`test_kalman_variable_dim`'s bitwise pin enforces from the other side.
"""

import importlib
import time

import numpy as np
import pytest

import mini_platform
from mini_platform.estimation.kalman_filter import PuckTracker

run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")
_JointStreamer = run_hw_match._JointStreamer
N_ARM = run_hw_match.N_ARM

#: A typical vision-sensor covariance -- the NIS gate applies only to measurements carrying one.
R_VISION = np.diag([2e-6, 2e-6, 1.7e-3])


@pytest.fixture(scope="module")
def env_info():
    return mini_platform.make("mini-hit").env_info


# --------------------------------------------------------------- 8c: the stamp reaches the EKF
def _synthetic_calibration(resolution=(640, 480)):
    """A camera 1.2 m above the table centre looking straight down, no distortion."""
    from mini_platform.hw.real_camera import CameraCalibration

    w, h = resolution
    f = 500.0
    K = np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])
    R_wc = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return CameraCalibration(K=K, dist=np.zeros(5), R_wc=R_wc,
                             C_world=np.array([0.0, 0.0, 1.2]), resolution=resolution)


class _StampedCamera:
    """A capture that stamps its frames, like ``V4L2Camera``."""

    def __init__(self, frame, stamp):
        self._frame = frame
        self.stamp = stamp
        self.frame_seq = 1
        self.frame_time = stamp

    def read_rgb(self):
        return self._frame.copy()

    def read_rgb_stamped(self):
        return self._frame.copy(), self.frame_seq, self.stamp

    def close(self):
        pass


class _UnstampedCamera:
    """A capture predating the stamp -- read_rgb/close only, like every existing stand-in
    (`test_off_table._StillCamera`) and like the documented contract."""

    def __init__(self, frame):
        self._frame = frame
        self.frame_seq = 1
        self.frame_time = None

    def read_rgb(self):
        return self._frame.copy()

    def close(self):
        pass


def _frame_with_puck(calib):
    """A dark frame with one red disc on the table, and the pixel it lands at."""
    cv2 = pytest.importorskip("cv2")
    w, h = calib.resolution
    frame = np.full((h, w, 3), 30, np.uint8)
    from mini_platform.hw.real_sensor import RealPuckSensor
    probe = RealPuckSensor(_StampedCamera(frame, 0.0), calib)
    uv = probe.camera.project(np.array([0.20, 0.05, 0.007]))
    cv2.circle(frame, (int(uv[0]), int(uv[1])), 9, (200, 30, 30), -1)
    return frame


def test_a_measurement_carries_its_frames_capture_time(env_info):
    """The whole of 8c rests on this: without the stamp the age is unknowable downstream, and
    `frame_age` answers a different question (how old the NEWEST frame is right now)."""
    from mini_platform.hw.real_sensor import RealPuckSensor

    calib = _synthetic_calibration()
    frame = _frame_with_puck(calib)
    stamp = time.perf_counter() - 0.037
    sensor = RealPuckSensor(_StampedCamera(frame, stamp), calib)
    m = sensor.measure()
    assert m is not None, "the synthetic puck must be detected, or this pins nothing"
    assert m.stamp == pytest.approx(stamp)


def test_a_capture_without_stamps_still_works(env_info):
    """The documented capture contract is read_rgb()/close(). A stand-in that offers only that
    must keep working -- it simply cannot report an age."""
    from mini_platform.hw.real_sensor import RealPuckSensor

    calib = _synthetic_calibration()
    sensor = RealPuckSensor(_UnstampedCamera(_frame_with_puck(calib)), calib)
    m = sensor.measure()
    assert m is not None
    assert m.stamp is None


def test_the_sim_sensor_leaves_the_stamp_empty():
    """The sim renders on demand, so there is no separate capture instant to report."""
    from mini_platform.vision.sensor import PuckMeasurement

    m = PuckMeasurement(np.zeros(2), None, np.eye(2), None, 10)
    assert m.stamp is None, "the field must default empty, or every sim measurement claims a time"


# ------------------------------------------------------- 8c: fusing against the capture-time state
def _lagged_run(env_info, lag_ticks, comp, n=40, v=-1.2):
    """Fly a puck, handing the filter every measurement `lag_ticks` late.

    Two things this harness has to get right, or it measures itself instead of the fix:

      * the truth track is generated by the filter's OWN motion model, not by a straight
        constant-velocity line. The model damps the puck, so a straight-line truth would differ
        from it by centimetres over 40 ticks whatever the lag handling did, and that floor would
        sit on top of the effect under test;
      * the lag is consistent from the FIRST tick -- the filter starts at ``truth[lag]`` and is
        fed ``truth[k]`` -- rather than replaying ``truth[0]`` while the puck has already moved.
        Otherwise the opening ticks fuse a measurement no state in the history matches, and both
        configurations inherit a transient that has nothing to do with lag compensation;
      * the filter is seeded with the track's OWN state at that point, velocity included. The
        model damps the puck, so re-imposing the initial velocity would hand the filter an error
        it can never shed while coasting -- and a correctly compensated filter coasts, because
        its innovation is near zero. That artefact reads as the compensation failing when it is
        in fact the harness lying to it.

    Returns (mean position error over the settled last third, tracker).
    """
    t = PuckTracker(env_info, agent_id=1, meas_lag_comp=comp)
    s = np.zeros(6)
    s[:2] = [0.95, 0.02]
    s[2:4] = [v, 0.0]
    states = []
    for _ in range(n + lag_ticks + 1):
        states.append(s.copy())
        s = t.system.f(s)
    truth = [x[:2] for x in states]

    # "Now" starts lag ticks into the track; the camera is still showing tick 0.
    t.reset(np.array([truth[lag_ticks][0], truth[lag_ticks][1], 0.0]))
    t.state = states[lag_ticks].copy()
    errors = []
    for k in range(n):
        seen = truth[k + 1]                       # captured lag ticks before this tick's "now"
        t.step(np.array([seen[0], seen[1], 0.0]), R_VISION, age=lag_ticks * t.system.dt)
        errors.append(float(np.linalg.norm(t.state[:2] - truth[lag_ticks + k + 1])))
    return float(np.mean(errors[-(n // 3):])), t


def test_a_stale_frame_fused_as_current_parks_the_estimate_behind_the_puck(env_info):
    """THE defect 8c is about, pinned.

    Fusing a 3-tick-old frame as though it described the puck now doesn't scatter the estimate --
    it converges it, confidently, onto where the puck WAS. Instrumented, the filter tracks the
    lagged truth to within 0.1 mm and sits a steady ~72 mm (1.2 m/s x 60 ms) behind the real
    puck, and `get_prediction` then forward-integrates that offset into the intercept.
    """
    plain, _ = _lagged_run(env_info, lag_ticks=3, comp=False)
    assert 0.03 < plain < 0.09, f"expected roughly one lag of travel behind, got {plain}"


def test_retro_fusion_does_not_yet_beat_the_defect_it_targets(env_info):
    """Characterisation, not an aspiration: `--meas-lag-comp` measures WORSE than leaving it off.

    Two faults, both inherent to comparing against the filter's own past estimate:
      * it is circular -- that state is only the state "at capture time" if the filter already
        tracks NOW. From a cold start it first settles onto the lagged track, its history is then
        stale by 2*lag, and the innovation sits at a permanent +1 lag it never escapes;
      * a reference that straddles a rim bounce is on the wrong side of a discontinuity, so the
        innovation explodes, the NIS gate rejects, and it re-locks with a garbage velocity.

    Kept as a tripwire: when this assertion starts FAILING, retro-fusion has been improved and
    its default (and `--meas-lag-comp`'s help) should be reconsidered. Until then the shipped
    answer is `meas_age_lead`, below.
    """
    plain, _ = _lagged_run(env_info, lag_ticks=3, comp=False)
    comped, _ = _lagged_run(env_info, lag_ticks=3, comp=True)
    assert comped > plain, (
        f"retro-fusion now BEATS plain fusion ({comped} vs {plain}) -- good news: re-read the "
        f"docstring and promote it")


# ------------------------------------------------- 8c: leading the target by the measured age
def _fed_agent(env_info, age, **kw):
    """An agent that has seen a few stamped measurements of an approaching puck."""
    from mini_platform.agents.baseline_mini import MiniBaselineAgent
    from mini_platform.envs.env_base import INIT_SEED

    a = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.0, **kw)
    a.reset()
    q, dq = INIT_SEED.copy(), np.zeros(len(INIT_SEED))
    for x in (0.90, 0.86, 0.82, 0.78):
        a.state.update_observation(q, dq, np.array([x, 0.0]), R_VISION[:2, :2], age)
    return a


def test_the_measured_age_leads_the_intercept(env_info):
    """The shipped answer: leave the filter alone and aim where the puck will be.

    The estimate legitimately describes the capture instant, so the same extrapolation that
    already covers command latency (`--lead-time`) covers sensor latency -- except the arm
    measures this one instead of being told it.
    """
    age = 0.06
    plain = _fed_agent(env_info, age)
    led = _fed_agent(env_info, age, meas_age_lead=True)
    assert plain.state._age_ema == 0.0, "off by default, so every existing run is untouched"
    assert led.state._age_ema > 0.0

    t_plain = np.asarray(plain.state.intercept_at(0.2)[0][:2], float)
    t_led = np.asarray(led.state.intercept_at(0.2)[0][:2], float)
    v = np.asarray(led.state.puck_tracker.state[2:4], float)
    moved = t_led - t_plain
    assert float(np.dot(moved, v)) > 0, "the lead must aim ALONG the puck's motion"
    full = float(np.linalg.norm(v)) * age
    assert 0.15 * full < float(np.linalg.norm(moved)) < 1.2 * full


def test_the_age_lead_is_capped(env_info):
    """One stalled frame must not throw the mallet across the table."""
    led = _fed_agent(env_info, 5.0, meas_age_lead=True)
    assert led.state._age_ema <= led.state.MAX_AGE_LEAD


def test_the_age_lead_is_off_without_a_stamp(env_info):
    """No stamp -> no age -> the vendored target, so a camera that cannot stamp changes nothing."""
    led = _fed_agent(env_info, None, meas_age_lead=True)
    assert led.state._age_ema == 0.0
    assert led.state._effective_lead == 0.0


def test_an_age_changes_nothing_unless_the_compensation_is_on(env_info):
    """Passing an age must be inert on its own -- that is what keeps every existing caller, and
    the whole sim path, byte-identical while the stamp is plumbed through."""
    with_age, _ = _lagged_run(env_info, lag_ticks=3, comp=False)
    without, t = _lagged_run(env_info, lag_ticks=3, comp=False)
    assert with_age == without

    a = PuckTracker(env_info, agent_id=1)
    b = PuckTracker(env_info, agent_id=1)
    for t_ in (a, b):
        t_.reset(np.array([0.60, 0.0, 0.0]))
    for _ in range(10):
        a.step(np.array([0.60, 0.0, 0.0]), R_VISION)
        b.step(np.array([0.60, 0.0, 0.0]), R_VISION, age=0.06)
    assert np.array_equal(a.state, b.state)
    assert np.array_equal(a.P, b.P)


def test_a_sub_tick_age_is_not_rounded_into_a_lag(env_info):
    """Rounding a 5 ms age up to a whole tick would invent a lag rather than remove one."""
    t = PuckTracker(env_info, agent_id=1, meas_lag_comp=True)
    t.reset(np.array([0.60, 0.0, 0.0]))
    for _ in range(5):
        t.step(np.array([0.60, 0.0, 0.0]), R_VISION, age=0.005)
    predicted, _ = t.predict(t.state.copy(), t.P.copy())
    assert t._lag_reference(predicted, 0.005) is predicted


def test_no_compensation_until_the_history_is_deep_enough(env_info):
    """Just after a reset there is no state from `age` ago to compare against.

    Clamping to the oldest snapshot instead measures a stale frame against a much more recent
    state -- a big innovation pointing backwards, taken at the near-unity gain a fresh P gives.
    That corrupts the velocity, and a compensated filter coasts on it, because once it IS
    compensating its innovations are near zero and nothing pulls it back. Found by
    `test_lag_compensation_beats_fusing_a_stale_frame_as_current`, which it made 2.7x WORSE than
    no compensation at all.
    """
    t = PuckTracker(env_info, agent_id=1, meas_lag_comp=True)
    t.reset(np.array([0.60, 0.0, 0.0]))
    predicted, _ = t.predict(t.state.copy(), t.P.copy())
    assert t._lag_reference(predicted, 0.06) is predicted, "no history yet -> no compensation"
    for _ in range(4):
        t.step(np.array([0.60, 0.0, 0.0]), R_VISION, age=0.06)
    assert t._lag_reference(predicted, 0.06) is not predicted, "with history it compensates"
    assert t._lag_reference(predicted, 10.0) is predicted, "an age past the history is not faked"


def test_the_history_does_not_grow_without_bound(env_info):
    """A per-tick state copy kept forever is a leak in a process that runs for minutes."""
    t = PuckTracker(env_info, agent_id=1, meas_lag_comp=True)
    t.reset(np.array([0.60, 0.0, 0.0]))
    for _ in range(200):
        t.step(np.array([0.60, 0.0, 0.0]), R_VISION, age=0.04)
    assert len(t._history) <= t.MAX_LAG_TICKS


def test_a_relock_drops_the_history(env_info):
    """States from before a re-lock describe a different track; comparing a new measurement
    against them is worse than not compensating at all."""
    t = PuckTracker(env_info, agent_id=1, meas_lag_comp=True)
    t.reset(np.array([0.60, 0.0, 0.0]))
    for _ in range(5):
        t.step(np.array([0.60, 0.0, 0.0]), R_VISION, age=0.04)
    assert t._history
    t.reset(np.array([0.30, 0.0, 0.0]))
    assert t._history == []


# ------------------------------------------------------------------ 8d: the RT stream lead
class _RecordArm:
    def __init__(self):
        self.cmds = []

    def stream_rt(self, q):
        self.cmds.append(np.asarray(q, float).copy())


def _wide_limits():
    return np.array([[-10.0] * N_ARM, [10.0] * N_ARM])


def _stream_ramp(lead_ms, vmax=None, speed=1.0, seconds=0.2, stream_hz=200):
    """Stream a constant-velocity plan and return every raw setpoint written."""
    arm = _RecordArm()
    q0 = np.zeros(N_ARM)
    s = _JointStreamer(arm, stream_hz, 0.02, _wide_limits(), q0, max_joint_vel=vmax,
                       lead_s=lead_ms / 1000.0)
    s.start()
    wps = []
    for k in range(1, 6):
        q = np.zeros(N_ARM)
        qd = np.zeros(N_ARM)
        q[0] = speed * k * 0.02
        qd[0] = speed
        wps.append((q, qd))
    s.set_trajectory(wps)
    time.sleep(seconds)
    s.stop()
    return np.array(arm.cmds)


def test_the_stream_lead_commands_ahead_of_the_plan():
    """A position servo trails its setpoint, so the plan is commanded tau early. On a
    constant-velocity plan that is exactly v*tau of extra position."""
    speed, lead_ms = 1.0, 50.0
    plain = _stream_ramp(0.0, speed=speed)
    led = _stream_ramp(lead_ms, speed=speed)
    assert len(plain) > 10 and len(led) > 10
    expected = speed * lead_ms / 1000.0
    gap = float(led[:, 0].max() - plain[:, 0].max())
    # Bounded rather than exact: both runs are wall-clock paced, so a few ms of jitter is normal.
    assert 0.4 * expected < gap < 1.8 * expected, f"lead moved the command by {gap:.4f} rad"


def test_the_lead_cannot_smuggle_a_step_past_the_speed_cap():
    """The lead is applied INSIDE the limiters on purpose: a speed cap with a hole in it is not
    a speed cap, and this one is the at-home safety limit."""
    vmax, stream_hz = 0.3, 200
    cmds = _stream_ramp(80.0, vmax=vmax, speed=3.0, stream_hz=stream_hz)
    steps = np.diff(cmds, axis=0)
    worst = float(np.abs(steps).max())
    assert worst <= vmax / stream_hz + 1e-9, f"a step of {worst} rad exceeds the cap"


def test_zero_lead_is_the_untouched_path():
    """The default must be the behaviour every earlier log was recorded under."""
    s = _JointStreamer(_RecordArm(), 200, 0.02, _wide_limits(), np.zeros(N_ARM))
    assert s._lead == 0.0


# ------------------------------------------------------- 8d: reading the lag off a track test
FREQ, DT = 0.75, 0.02
W = 2 * np.pi * FREQ


def _sine(t, amp=0.08, tau=0.0):
    return amp * np.sin(W * (t - tau))


def test_follower_lag_is_recovered_below_one_sample():
    """`--stage track-test` is only as good as this estimator, and it cannot be checked on the
    rig without already knowing the answer -- so check it against a known shift.

    47 ms is deliberately NOT a multiple of the 20 ms sample grid: the original
    cross-correlation could only ever answer in whole samples, which against a real ~40 ms lag
    is a 50% quantisation error.
    """
    t = np.arange(0, 6.0, DT)
    lag, ratio, _ = run_hw_match.measure_follower_lag(t, _sine(t), _sine(t, tau=0.047), FREQ)
    assert lag == pytest.approx(0.047, abs=0.003)
    assert ratio == pytest.approx(1.0, abs=0.02)


def test_drift_does_not_masquerade_as_overshoot():
    """THE bug this estimator replaced, pinned.

    Gravity droop rides on top of the sine, and a peak-of-absolute amplitude picks it up: on the
    rig it turned an arm that was attenuating to 0.9x into a reported 1.3x "overshoot", which in
    turn drove advice to add a lead that made the next run worse. The fit carries an explicit
    drift term so the droop lands there instead of in the amplitude.

    The synthetic carries BOTH of the things the rig's own logs carried, because droop alone
    inflates a peak by only ~8% and would not reproduce the failure: 18 mrad of drift and 18 mrad
    RMS of non-sinusoidal content on an 80 mrad sine (`track_slow.npz`: drift 12.3, residual 18.0,
    peak inflation 1.43x). Seeded, so it is deterministic.
    """
    t = np.arange(0, 8.0, DT)
    cmd = _sine(t)
    rng = np.random.default_rng(0)
    meas = cmd + 0.018 * (t / t[-1]) + rng.normal(0.0, 0.018, size=t.shape)
    lag, ratio, diag = run_hw_match.measure_follower_lag(t, cmd, meas, FREQ)
    assert ratio == pytest.approx(1.0, abs=0.05), "droop and noise must not read as amplitude"
    assert diag["drift"] == pytest.approx(0.018, abs=0.006)
    assert diag["residual"] == pytest.approx(0.018, rel=0.25)
    naive = np.max(np.abs(meas - meas.mean())) / np.max(np.abs(cmd - cmd.mean()))
    assert naive > 1.25, "pins that the peak-based estimator really was fooled by this"


def test_an_attenuating_follower_is_not_called_an_overshoot():
    """A servo that trails attenuates; that is the NORMAL reading and must not look like a fault."""
    t = np.arange(0, 6.0, DT)
    lag, ratio, _ = run_hw_match.measure_follower_lag(
        t, _sine(t), _sine(t, amp=0.9 * 0.08, tau=0.04), FREQ)
    assert ratio == pytest.approx(0.9, abs=0.02)
    assert lag == pytest.approx(0.04, abs=0.003)


def test_an_overshooting_follower_still_reports_its_overshoot():
    """Too much lead overshoots every reversal, which on a mallet is a skid -- the number that
    says to back it off. Must survive the move away from peak amplitudes."""
    t = np.arange(0, 6.0, DT)
    _, ratio, _ = run_hw_match.measure_follower_lag(t, _sine(t), 1.3 * _sine(t), FREQ)
    assert ratio == pytest.approx(1.3, abs=0.05)


# --------------------------------------------------------------------- 8e: the replan cadence
def test_max_plan_steps_is_vendored_unless_asked(env_info):
    from mini_platform.agents.baseline_mini import MiniBaselineAgent

    assert MiniBaselineAgent(env_info, agent_id=1).agent_params["max_plan_steps"] == 5


def test_max_plan_steps_reaches_the_agent(env_info):
    """3 refreshes the target every 60 ms instead of 100; a 2 m/s puck moves 80 mm less between
    refreshes, at the cost of more replans on the CPU the vision thread wants."""
    from mini_platform.agents.baseline_mini import MiniBaselineAgent

    a = MiniBaselineAgent(env_info, agent_id=1, max_plan_steps=3)
    assert a.agent_params["max_plan_steps"] == 3


# ------------------------------------------------------------------------- 8b: the camera rate
def test_fourcc_round_trips():
    """A camera that silently fell back from MJPG to YUYV is the other way to lose half the
    frame rate, so `--stage cam-probe` prints the negotiated format."""
    from mini_platform.hw.real_camera import fourcc_str

    packed = sum(ord(c) << (8 * i) for i, c in enumerate("MJPG"))
    assert fourcc_str(packed) == "MJPG"
    assert fourcc_str(0) == ""
