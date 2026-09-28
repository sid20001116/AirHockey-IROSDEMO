"""Slow mode: a hard per-joint velocity cap on the RT stream so the arm cannot swing fast
while developing at home. The cap is enforced in _JointStreamer independently of the plan.
"""
import argparse
import importlib
import time

import numpy as np
import pytest

run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")
_JointStreamer = run_hw_match._JointStreamer
N_ARM = run_hw_match.N_ARM


class _RecordArm:
    """Minimal arm stand-in that records every raw setpoint streamed to it."""
    def __init__(self):
        self.cmds = []

    def stream_rt(self, q):
        self.cmds.append(np.asarray(q, float).copy())


def _wide_limits():
    return np.array([[-10.0] * N_ARM, [10.0] * N_ARM])


def _run_streamer(vmax, stream_hz=200, amax=None):
    arm = _RecordArm()
    q0 = np.zeros(N_ARM)
    s = _JointStreamer(arm, stream_hz, 0.02, _wide_limits(), q0,
                       max_joint_vel=vmax, max_joint_accel=amax)
    s.start()
    # a far target so an uncapped stream would jump hard
    s.set_trajectory([(np.full(N_ARM, 2.0), np.zeros(N_ARM))])
    time.sleep(0.2)
    s.stop()
    return arm.cmds, 1.0 / stream_hz


def _run_streamer_target(target, vmax, stream_hz=200, amax=None, limit_mode="coupled",
                         couple_spin=False):
    """Stream one far waypoint and return every raw setpoint written."""
    arm = _RecordArm()
    q0 = np.zeros(N_ARM)
    s = _JointStreamer(arm, stream_hz, 0.02, _wide_limits(), q0, max_joint_vel=vmax,
                       max_joint_accel=amax, limit_mode=limit_mode, couple_spin=couple_spin)
    s.start()
    s.set_trajectory([(np.asarray(target, float), np.zeros(N_ARM))])
    time.sleep(0.2)
    s.stop()
    return np.array(arm.cmds), 1.0 / stream_hz


def _moving_steps(cmds):
    steps = np.diff(cmds, axis=0)
    return steps[np.linalg.norm(steps, axis=1) > 1e-12]


def _three_to_one_target():
    """A move whose joints want 0.3 and 0.1 rad -- both far over any useful cap."""
    t = np.zeros(N_ARM)
    t[0], t[1] = 0.3, 0.1
    return t


# --------------------------------------------------------------- --resync-cmd (plan vs command)
_resync_target = run_hw_match._resync_target


def test_resync_leaves_a_tracking_command_alone():
    """Inside the threshold nothing is touched -- the resync must not fire on every capped tick."""
    q = np.full(N_ARM, 0.4)
    assert _resync_target(q, q + 0.01, 0.15) is None


def test_resync_pulls_the_plan_back_to_what_was_written():
    q_written = np.zeros(N_ARM)
    out = _resync_target(np.full(N_ARM, 1.0), q_written, 0.15)
    assert out is not None
    np.testing.assert_allclose(out, q_written)


def test_resync_is_idempotent_so_repeats_cannot_walk_the_command():
    """The droop guard.

    Re-seeding from the written COMMAND is a fixed point: once corrected, further resyncs are
    no-ops. Re-seeding from ``arm.q()`` instead would fold the standing ~11 mrad of gravity droop
    into the command on every pass, and the arm would sag further each time.
    """
    q_written = np.full(N_ARM, 0.25)
    q_cmd = np.zeros(N_ARM)
    for _ in range(50):
        out = _resync_target(q_cmd, q_written, 0.15)
        if out is not None:
            q_cmd = out
    np.testing.assert_allclose(q_cmd, q_written)
    assert _resync_target(q_cmd, q_written, 0.15) is None


def test_resync_ignores_a_non_finite_or_disabled_stream():
    q = np.zeros(N_ARM)
    assert _resync_target(q, np.full(N_ARM, np.nan), 0.15) is None   # diverged estimate / bad IK
    assert _resync_target(q, np.full(N_ARM, 5.0), None) is None      # --resync-cmd off
    assert _resync_target(q, np.full(N_ARM, 5.0), 0.0) is None


def test_a_capped_stream_opens_the_gap_the_resync_closes():
    """End to end on the real streamer: the cap throttles, so the plan runs away from the write."""
    target = np.full(N_ARM, 2.0)
    cmds, _ = _run_streamer_target(target, vmax=0.5, stream_hz=200)
    q_written = cmds[-1]
    gap = np.linalg.norm(target - q_written)
    assert gap > 0.15, f"expected the cap to open a gap, got {gap:.3f} rad"
    np.testing.assert_allclose(_resync_target(target, q_written, 0.15), q_written)


def test_coupled_limit_keeps_a_throttled_step_on_the_planned_direction():
    """A throttled move must slow down ALONG the plan, not bend off it.

    The Hermite from rest to rest moves every joint on the same scalar profile, so the raw step
    is parallel to (3, 1, 0, ...) by construction; scaling the whole vector by one factor keeps
    it that way however hard the cap bites.
    """
    cmds, _ = _run_streamer_target(_three_to_one_target(), vmax=0.5)
    moving = _moving_steps(cmds)
    assert len(moving) > 5
    for s in moving:
        assert s[0] * 1.0 == pytest.approx(s[1] * 3.0, abs=1e-9), "step left the 3:1 direction"
        assert np.abs(s[2:]).max() < 1e-12, "a joint that should not move, moved"


def test_per_joint_limit_bends_the_step_off_the_plan():
    """Pins the defect the coupled mode fixes: an independent clip shortens only the joints that
    are over their own cap, so a 3:1 move is streamed as 1:1 -- off the planned path, which is
    what lifts the mallet off the play plane."""
    cmds, _ = _run_streamer_target(_three_to_one_target(), vmax=0.5, limit_mode="per-joint")
    moving = _moving_steps(cmds)
    assert any(abs(s[0] - s[1] * 3.0) > 1e-6 for s in moving), \
        "per-joint clipping must actually distort the direction, or this pins nothing"


def test_coupled_limit_still_bounds_every_joint():
    """Direction preservation must not cost the safety bound the cap exists for."""
    vmax = 0.5
    cmds, period = _run_streamer_target(_three_to_one_target(), vmax=vmax)
    assert np.abs(np.diff(cmds, axis=0)).max() <= vmax * period + 1e-9
    assert cmds[-1][0] > cmds[0][0], "and it still made progress toward the target"


def test_cap_vector_accepts_a_scalar_or_one_value_per_joint():
    assert run_hw_match._cap_vector(None) is None
    assert run_hw_match._cap_vector(0.5) == pytest.approx(np.full(N_ARM, 0.5))
    v = np.arange(1, N_ARM + 1) * 0.1
    assert run_hw_match._cap_vector(v) == pytest.approx(v)
    for bad in ([0.1, 0.2], -1.0, 0.0, np.nan):
        with pytest.raises(ValueError):
            run_hw_match._cap_vector(bad)


def test_a_per_joint_cap_vector_bounds_each_joint_by_its_own_limit():
    """Fix 6d raises J1 alone, so the streamer has to honour a vector, not just a scalar."""
    caps = np.full(N_ARM, 1.0)
    caps[0] = 0.2
    cmds, period = _run_streamer_target(np.full(N_ARM, 0.5), vmax=caps)
    steps = np.abs(np.diff(cmds, axis=0))
    assert (steps <= caps * period + 1e-9).all()
    assert steps[:, 0].max() <= 0.2 * period + 1e-9


def _spin_and_base_target():
    """A move whose SPIN joint (J6) wants five times what the base joint does."""
    t = np.zeros(N_ARM)
    t[0], t[5] = 0.2, 1.0
    return t


def test_a_spin_joint_does_not_throttle_the_joints_that_carry_the_path():
    """J6 cannot move the mallet -- its position-Jacobian column is identically zero (measured max
    0.000000 m/rad over 2000 random configurations) -- so it must not set the coupled scale.

    Before this, a wrist chasing strike-aligned yaw dragged the base joint down with it: on
    goalie.npz / resync_base.npz J6 binds 52-72% of every over-cap tick, and on 10-13% of moving
    ticks it is the ONLY reason the vector is scaled at all.
    """
    vmax = 0.5
    free, _ = _run_streamer_target(_spin_and_base_target(), vmax=vmax)
    taxed, _ = _run_streamer_target(_spin_and_base_target(), vmax=vmax, couple_spin=True)
    j1_free = np.abs(np.diff(free, axis=0))[:, 0].max()
    j1_taxed = np.abs(np.diff(taxed, axis=0))[:, 0].max()
    assert j1_free > j1_taxed * 1.5, (
        f"the base joint should not be throttled by a spin it takes no part in "
        f"(free {j1_free:.5f} vs coupled {j1_taxed:.5f} rad per write)")


def test_a_spin_joint_is_still_bounded_by_its_own_cap():
    """Excluding J6 from the SCALE must not exempt it from the LIMIT -- it is clipped instead."""
    vmax = 0.5
    cmds, period = _run_streamer_target(_spin_and_base_target(), vmax=vmax)
    assert np.abs(np.diff(cmds, axis=0)).max() <= vmax * period + 1e-9


def test_the_path_joints_keep_their_direction_when_a_spin_joint_is_excluded():
    """The joints that DO carry the Cartesian path must still be scaled together, spin or no spin:
    that coupling is what keeps a throttled move on the play plane."""
    t = np.zeros(N_ARM)
    t[0], t[1], t[5] = 0.3, 0.1, 1.0
    cmds, _ = _run_streamer_target(t, vmax=0.5)
    moving = _moving_steps(cmds)
    assert len(moving) > 5
    for s in moving:
        assert s[0] == pytest.approx(s[1] * 3.0, abs=1e-9), "the 3:1 path direction was bent"


def test_joint_cap_arg_accepts_one_value_or_six():
    assert run_hw_match.joint_cap_arg("1.4") == pytest.approx(1.4)
    v = run_hw_match.joint_cap_arg("2.5,1.4,1.4,1.4,1.4,1.4")
    assert v == pytest.approx([2.5, 1.4, 1.4, 1.4, 1.4, 1.4])
    assert run_hw_match.joint_cap_arg("2.5 1.4 1.4 1.4 1.4 1.4") == pytest.approx(v)
    for bad in ("1.4,1.4", "abc", "0,1,1,1,1,1", "-1", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            run_hw_match.joint_cap_arg(bad)


class _FailingArm:
    """An arm whose every RT write raises -- the silent-failure case the counter makes visible."""
    def __init__(self):
        self.calls = 0

    def stream_rt(self, q):
        self.calls += 1
        raise RuntimeError("CAN write failed")


def test_a_stream_that_cannot_write_is_counted_not_silently_swallowed():
    """`_run` swallows write exceptions deliberately -- a transient SDK hiccup must not kill the
    RT thread -- but swallowing them SILENTLY made a stream failing every single write look
    identical to a healthy one, in the telemetry and in the banner alike."""
    arm = _FailingArm()
    s = _JointStreamer(arm, 200, 0.02, _wide_limits(), np.zeros(N_ARM), max_joint_vel=0.5)
    s.start()
    s.set_trajectory([(np.full(N_ARM, 0.4), np.zeros(N_ARM))])
    time.sleep(0.15)
    s.stop()
    assert arm.calls > 5, "the stream should have kept trying rather than died"
    assert s.write_errors == arm.calls
    summary = s.health_summary()
    assert summary is not None and "failed RT write" in summary


def test_a_healthy_stream_has_nothing_to_report():
    arm = _RecordArm()
    s = _JointStreamer(arm, 200, 0.02, _wide_limits(), np.zeros(N_ARM), max_joint_vel=0.5)
    s.start()
    s.set_trajectory([(np.full(N_ARM, 0.4), np.zeros(N_ARM))])
    time.sleep(0.1)
    s.stop()
    assert s.write_errors == 0
    assert s.health_summary() is None


def test_an_existing_log_is_flagged_before_the_run_not_after(tmp_path, capsys):
    """Overwriting a log destroys the baseline the next A/B compares against -- it has already
    cost `goalie.npz` twice. The warning has to arrive at startup, while there is still something
    to be done about it."""
    p = tmp_path / "goalie.npz"
    np.savez(p, t=np.arange(10, dtype=float))
    run_hw_match.warn_if_log_exists(str(p))
    out = capsys.readouterr().out
    assert "already exists" in out
    assert "10 ticks" in out
    assert "goalie_" in out, "it should suggest a dated name to use instead"


def test_a_fresh_log_name_is_not_flagged(tmp_path, capsys):
    run_hw_match.warn_if_log_exists(str(tmp_path / "brand_new.npz"))
    assert capsys.readouterr().out == ""
    run_hw_match.warn_if_log_exists(None)
    assert capsys.readouterr().out == ""


def test_the_cli_help_actually_renders():
    """argparse interpolates `help % params`, so ONE unescaped '%' in any help string turns
    `--help` into a TypeError -- and normal parsing never touches that path, so the rest of this
    suite runs green while the CLI is unusable. This caught exactly that: "44-58% of moving ticks"
    parses as '%' + space-flag + 'o', i.e. %o, and argparse demanded an integer.

    Covers every help string in the parser at once, not just the ones added today.
    """
    with pytest.raises(SystemExit) as e:
        run_hw_match.main(["--help"])
    assert e.value.code == 0


def test_a_per_joint_cap_survives_resolution_without_inflating_the_strike_envelope():
    """Fix 6d raises J1 alone. The Cartesian strike limits come off the SLOWEST joint, so freeing
    the base must not hand VPSTO a faster envelope than the rest of the arm can track."""
    A = type("A", (), {})
    a = A()
    a.max_joint_vel = np.array([2.5, 1.4, 1.4, 1.4, 1.4, 1.4])
    a.slow = False
    assert run_hw_match.resolve_max_joint_vel(a) == pytest.approx(a.max_joint_vel)
    a.vpsto_vel_lim = a.vpsto_acc_lim = None
    vel, _, derived = run_hw_match.resolve_vpsto_limits(a)
    assert "vel_lim" in derived
    assert vel == pytest.approx(round(run_hw_match.VPSTO_VEL_PER_JOINT_VEL * 1.4, 3))


def test_resolve_max_joint_vel():
    A = type("A", (), {})
    a = A(); a.max_joint_vel = 0.7; a.slow = False
    assert run_hw_match.resolve_max_joint_vel(a) == pytest.approx(0.7)
    b = A(); b.max_joint_vel = None; b.slow = True
    assert run_hw_match.resolve_max_joint_vel(b) == pytest.approx(run_hw_match.SLOW_MAX_JOINT_VEL)
    c = A(); c.max_joint_vel = None; c.slow = False
    assert run_hw_match.resolve_max_joint_vel(c) is None
    # explicit --max-joint-vel wins over --slow
    d = A(); d.max_joint_vel = 0.1; d.slow = True
    assert run_hw_match.resolve_max_joint_vel(d) == pytest.approx(0.1)


def test_resolve_max_joint_accel():
    A = type("A", (), {})
    a = A(); a.max_joint_accel = 55.0; a.slow = False
    assert run_hw_match.resolve_max_joint_accel(a) == pytest.approx(55.0)
    b = A(); b.max_joint_accel = None; b.slow = True
    assert run_hw_match.resolve_max_joint_accel(b) == pytest.approx(run_hw_match.SLOW_MAX_JOINT_ACCEL)
    c = A(); c.max_joint_accel = None; c.slow = False
    assert run_hw_match.resolve_max_joint_accel(c) is None
    # explicit --max-joint-accel wins over --slow
    d = A(); d.max_joint_accel = 20.0; d.slow = True
    assert run_hw_match.resolve_max_joint_accel(d) == pytest.approx(20.0)


def test_acceleration_cap_bounds_velocity_change():
    """With the accel cap, the commanded joint VELOCITY (per-write position delta / period) may
    change by no more than amax*period between consecutive writes on any joint -- i.e. the second
    difference of the streamed positions is bounded by amax*period^2. True by construction."""
    amax = 20.0
    cmds, period = _run_streamer(vmax=None, amax=amax)
    cmds = np.array(cmds)
    assert len(cmds) > 6
    second_diff = np.abs(np.diff(cmds, n=2, axis=0))       # |q_k - 2 q_{k-1} + q_{k-2}|
    assert second_diff.max() <= amax * period * period + 1e-9
    # and it still made real progress toward the far target (not frozen by the cap)
    assert cmds[-1].max() > cmds[0].max()


def test_acceleration_cap_composes_with_velocity_cap():
    """Both caps together: the velocity bound (step <= vmax*period) AND the accel bound
    (second-diff <= amax*period^2) hold simultaneously -- the accel clamp preserves the vmax
    bound rather than breaking it."""
    vmax, amax = 0.5, 10.0
    cmds, period = _run_streamer(vmax=vmax, amax=amax)
    cmds = np.array(cmds)
    assert len(cmds) > 6
    steps = np.abs(np.diff(cmds, axis=0))
    assert steps.max() <= vmax * period + 1e-9                     # velocity cap still holds
    second_diff = np.abs(np.diff(cmds, n=2, axis=0))
    assert second_diff.max() <= amax * period * period + 1e-9      # accel cap holds too


def test_uncapped_accel_changes_velocity_faster_than_the_cap():
    """Sanity: without the accel cap the same far target changes commanded velocity by more than
    amax*period in a single write -- i.e. the accel cap is doing real work."""
    amax = 10.0
    cmds, period = _run_streamer(vmax=None, amax=None)
    second_diff = np.abs(np.diff(np.array(cmds), n=2, axis=0))
    assert second_diff.max() > amax * period * period


def test_velocity_cap_bounds_every_step():
    """With the cap, no two consecutive streamed setpoints differ by more than vmax*period on
    any joint -- a hard speed ceiling, true by construction regardless of thread timing."""
    vmax = 0.5
    cmds, period = _run_streamer(vmax)
    assert len(cmds) > 5
    steps = np.abs(np.diff(np.array(cmds), axis=0))
    assert steps.max() <= vmax * period + 1e-9
    # and it actually moved toward the target (not stuck)
    assert np.array(cmds)[-1].max() > cmds[0].max()


def test_wait_until_config_arrives_and_times_out():
    """The play stage homes slowly and waits for arrival before RT play; the wait is bounded."""
    from mini_platform.hw.sdk_backend import MockMetalArm
    arm = MockMetalArm(q0=np.zeros(N_ARM))
    target = np.full(N_ARM, 0.2)
    arm.move_nrt(target)                       # mock reaches the target instantly
    assert run_hw_match.wait_until_config(arm, target, timeout=1.0)

    stuck = MockMetalArm(q0=np.zeros(N_ARM))   # never commanded to target -> stays at 0
    assert not run_hw_match.wait_until_config(stuck, target, timeout=0.3)


def test_uncapped_stream_jumps_more_than_the_cap():
    """Sanity: without the cap the same far target produces steps larger than the cap would
    allow -- i.e. the cap is doing real work."""
    vmax = 0.5
    cmds, period = _run_streamer(None)
    steps = np.abs(np.diff(np.array(cmds), axis=0))
    assert steps.max() > vmax * period


def test_non_finite_command_is_never_streamed():
    """A NaN/Inf plan must NEVER reach the arm -- clamp_pos(NaN)=NaN streams straight through and
    drives a wild extend + hard fault. The streamer drops the bad plan and holds the last good one.
    """
    arm = _RecordArm()
    q0 = np.zeros(N_ARM)
    s = _JointStreamer(arm, 200, 0.02, _wide_limits(), q0, max_joint_vel=None)
    s.start()
    try:
        s.set_trajectory([(np.full(N_ARM, 0.3), np.zeros(N_ARM))])    # good target
        time.sleep(0.1)
        s.set_trajectory([(np.full(N_ARM, np.nan), np.zeros(N_ARM))])  # non-finite -> must be dropped
        time.sleep(0.1)
    finally:
        s.stop()
    cmds = np.array(arm.cmds)
    assert len(cmds) > 5
    assert np.all(np.isfinite(cmds)), "a non-finite setpoint reached the arm"
    # the last good target (0.3) still governs -- the NaN plan was dropped, not latched
    assert cmds[-1].max() == pytest.approx(0.3, abs=1e-6)
