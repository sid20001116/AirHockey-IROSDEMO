"""Fix 1e: the agent advances by the ticks the WALL CLOCK took, not one per call.

The vendored `draw_action` pops exactly one waypoint and steps the EKF one fixed 20 ms tick per
call, however long the tick really took. Every overrun therefore stretches the plan against real
time -- measured 13-19% behind over a run, and a median 259-478 ms per VPSTO strike -- and biases
the filter's velocity by the same ratio exactly while stalls cluster, which is during strikes.

`skip=0` must stay the vendored path call for call, so the last test here pins that.
"""
import importlib

import numpy as np
import pytest

import mini_platform
from mini_platform.agents.baseline_agent import BaselineAgent
from mini_platform.envs.env_base import INIT_SEED, N_JOINTS

run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")


class _NoSensor:
    """A camera that never sees the puck -- every tick coasts, which keeps this test about the
    skip bookkeeping rather than about the filter."""

    def measure(self):
        return None


@pytest.fixture(scope="module")
def ei():
    return mini_platform.make("mini-hit-sdk").env_info


def _agent(env_info):
    a = run_hw_match.HwBaselineAgent(env_info, _NoSensor(), agent_id=1, min_goal_dist=0.02)
    a.reset()
    return a


def _obs(env_info, q):
    return run_hw_match.make_obs(env_info, q, np.zeros(len(q)))


def _buffer(q, n=5):
    """`n` distinguishable waypoints, so which one comes back is unambiguous."""
    buf = np.zeros((n, 2, N_JOINTS))
    for k in range(n):
        buf[k, 0] = q + (k + 1) * 0.01
    return buf


def _stub_pop(self, obs):
    """What the vendored `draw_action` does once its tactic loop has filled the buffer.

    Standing in for the tactic loop keeps these tests about the wall-clock bookkeeping: the real
    loop would replace `trajectory_buffer` with whatever it replanned, so the waypoints could not
    be identified on the way out.
    """
    puck, R = self.get_puck_measurement(obs)
    self.state.update_observation(self.get_joint_pos(obs), self.get_joint_vel(obs), puck, R)
    self.state.q_cmd, self.state.dq_cmd = self.state.trajectory_buffer[0]
    self.state.trajectory_buffer = self.state.trajectory_buffer[1:]
    self.state.x_cmd, self.state.v_cmd = self.state.update_ee_pos_vel(
        self.state.q_cmd, self.state.dq_cmd)
    return np.vstack([self.state.q_cmd, self.state.dq_cmd])


def _with_stub(fn):
    orig = BaselineAgent.draw_action
    BaselineAgent.draw_action = _stub_pop
    try:
        return fn()
    finally:
        BaselineAgent.draw_action = orig


def test_a_late_tick_hands_back_the_waypoint_due_now(ei):
    """A 60 ms tick at a 20 ms plan spacing is two ticks late: the command due now is the third
    waypoint, not the first."""
    a = _agent(ei)
    q = INIT_SEED.copy()
    buf = _buffer(q)

    def go():
        a.state.trajectory_buffer = buf.copy()
        return a.draw_action(_obs(ei, q), skip=2)

    action = _with_stub(go)
    assert a.last_skip == 2
    assert action[0] == pytest.approx(buf[2, 0]), "the arm must be sent where the plan is NOW"
    assert len(a.state.trajectory_buffer) == 2


def test_the_tracker_advances_one_step_per_wall_clock_tick(ei):
    """The filter's step is a fixed dt, so the skipped ticks have to be coasted or its velocity
    is biased by exactly the stall ratio."""
    a = _agent(ei)
    q = INIT_SEED.copy()
    seen = []
    real = a.state.puck_tracker.step

    def spy(measurement, R=None):
        seen.append(measurement)
        return real(measurement, R)

    a.state.puck_tracker.step = spy

    def go():
        a.state.trajectory_buffer = _buffer(q)
        return a.draw_action(_obs(ei, q), skip=3)

    try:
        _with_stub(go)
    finally:
        a.state.puck_tracker.step = real
    assert len(seen) == 4, "three coasted ticks plus this one"
    assert all(m is None for m in seen), "a coasted tick must not re-fuse a measurement"


def test_the_lookahead_is_never_emptied_by_a_skip(ei):
    """The streamer glides along the buffer through a replan freeze, so even an absurd skip must
    leave it something to glide along rather than strand the arm on a single point."""
    a = _agent(ei)
    q = INIT_SEED.copy()
    buf = _buffer(q)

    def go():
        a.state.trajectory_buffer = buf.copy()
        return a.draw_action(_obs(ei, q), skip=50)

    action = _with_stub(go)
    assert len(a.state.trajectory_buffer) == 1, "one waypoint of lookahead is always kept"
    assert action[0] == pytest.approx(buf[-2, 0]), "advanced as far as that rule allows"


def test_skip_zero_is_the_vendored_path(ei):
    """The contract: without --wall-clock, or on an on-time tick, nothing changes."""
    a, b = _agent(ei), _agent(ei)
    q = INIT_SEED.copy()
    buf = _buffer(q)

    def go():
        a.state.trajectory_buffer = buf.copy()
        b.state.trajectory_buffer = buf.copy()
        return a.draw_action(_obs(ei, q), skip=0), _stub_pop(b, _obs(ei, q))

    got, want = _with_stub(go)
    assert got == pytest.approx(want)
    assert a.last_skip == 0
    assert len(a.state.trajectory_buffer) == len(b.state.trajectory_buffer)
