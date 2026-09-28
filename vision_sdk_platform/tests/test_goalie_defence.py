"""Deep-defence ("goalie") fixes: the band in front of the arm's own goal.

The arm is side-mounted essentially ON its own goal line, so the region between where the mallet
stands and where the base sits is 138 mm of table that the vendored FSM cannot defend. Four
separate mechanisms conspire, and each has its own opt-in knob here:

  * the own goal is absent from the EKF bounce model, so a puck headed into it is predicted to
    rebound off the near rim and the arm swings FORWARD at a target that does not exist;
  * the intercept is the puck's position a fixed horizon (1.0 s) from now rather than the
    soonest reachable rendezvous;
  * `defend_range[0]` is at once the plant line and the give-up line, two jobs that want
    opposite values;
  * the block point is offset toward the centreline, leaving a third of the contact margin on
    the base side that it leaves on the other.

Every test here also pins the OFF state, because the contract for all of these is that an agent
built with no new arguments behaves exactly as the vendored one.
"""
import argparse
import importlib
import pathlib

import numpy as np
import pytest

import mini_platform
from mini_platform.agents.baseline_mini import (MiniAirHockeyTable, MiniBaselineAgent,
                                                MiniDefend, V_SETBACK)
from mini_platform.envs.env_base import INIT_SEED
from mini_platform.core.kinematics import inverse_kinematics
from mini_platform.estimation.kalman_filter import PuckTracker
from mini_platform.fsm.system_state import TACTICS

GOALIE = dict(ekf_own_goal=True, rendezvous_vel=0.35, defend_abort_dist=0.02,
              defend_block="centre", defend_centre_deadband=0.03,
              defend_give_up_ticks=10, defend_fresh_horizon=True, defend_gate_unled=True,
              defend_hold_target=True)


@pytest.fixture(scope="module")
def ei():
    return mini_platform.make("mini-hit-sdk").env_info


def _goal_line_v(env_info):
    return V_SETBACK - env_info["table"]["length"] / 2


def _tracker(env_info, own_goal=False):
    """A V-frame PuckTracker, optionally with the own goal opened."""
    v_robot = dict(env_info["robot"])
    v_robot["base_frame"] = [np.diag([1., 1., 1., 1.])]
    v_robot["base_frame"][0][0, 3] = -V_SETBACK
    tr = PuckTracker(dict(env_info, robot=v_robot), 1)
    if own_goal:
        sysm = tr.system
        sysm.table = MiniAirHockeyTable(
            env_info["table"]["length"], env_info["table"]["width"],
            env_info["table"]["goal_width"], env_info["puck"]["radius"],
            V_SETBACK, env_info["dt"], ekf_params=sysm.params)
    return tr


def _predict_from(tr, state4, horizon=1.0, stop_line=0.):
    n = 6
    tr.reset(np.array([state4[0], state4[1], 0.]))
    s = np.zeros(n)
    s[:4] = state4
    tr.state = s
    tr.P = np.eye(n) * 1e-4
    st, _, t = tr.get_prediction(horizon, stop_line)
    return np.asarray(st)[:4], t


# --------------------------------------------------------------------------- own goal in the EKF

def test_own_goal_default_predicts_a_phantom_rebound(ei):
    """Pin the bug: a shot on our own goal is predicted to bounce back up-table.

    This is the reason the arm swings forward instead of at the puck, so it is worth having a
    test that fails the day someone 'fixes' it in the vendored file instead.
    """
    tr = _tracker(ei, own_goal=False)
    st, _ = _predict_from(tr, [0.20, 0.00, -1.0, 0.0])
    assert st[2] > 0, "vendored model should reflect the puck off the phantom near rim"
    assert st[0] > 0.5, "and send it well up-table"


def test_own_goal_lets_a_shot_on_target_through(ei):
    tr = _tracker(ei, own_goal=True)
    st, _ = _predict_from(tr, [0.20, 0.00, -1.0, 0.0])
    assert st[2] < 0, "a puck inside the mouth must keep closing, not rebound"
    assert st[0] < 0.20


def test_own_goal_stops_at_a_stop_line_on_the_goal_line(ei):
    """With the mouth open the prediction is a real crossing, and it arrives soon."""
    tr = _tracker(ei, own_goal=True)
    gl = _goal_line_v(ei)
    st, t = _predict_from(tr, [0.20, 0.00, -1.0, 0.0], stop_line=gl)
    assert st[0] == pytest.approx(gl, abs=0.03)
    assert 0.0 < t < 0.20


@pytest.mark.parametrize("y", [0.10, 0.20, -0.15])
def test_own_goal_outside_the_mouth_still_bounces(ei, y):
    """The near rim is real everywhere except the mouth -- a corner puck must still rebound."""
    plain = _predict_from(_tracker(ei, own_goal=False), [0.20, y, -1.0, 0.0])[0]
    opened = _predict_from(_tracker(ei, own_goal=True), [0.20, y, -1.0, 0.0])[0]
    assert opened == pytest.approx(plain)
    assert opened[2] > 0


def test_own_goal_restores_the_boundary(ei):
    """`_check_collision_impl` slices `boundary`; `check_collision` reads it again afterwards."""
    tr = _tracker(ei, own_goal=True)
    table = tr.system.table
    before = table.boundary.copy()
    state = np.zeros(6)
    state[:4] = [0.14, 0.0, -1.0, 0.0]
    table.check_collision(state)
    assert table.boundary.shape == before.shape
    assert table.boundary == pytest.approx(before)


# ------------------------------------------------------------------------------- rendezvous

def test_rendezvous_off_matches_the_vendored_prediction(ei):
    """rendezvous_vel unset -> `update_prediction` is the vendored call, value for value."""
    off = MiniBaselineAgent(ei, agent_id=1)
    st = off.state
    st.puck_tracker.reset(np.array([0.42, 0.02, 0.]))
    s = np.zeros(6)
    s[:4] = [0.42, 0.02, -1.2, 0.0]
    st.puck_tracker.state = s
    stop = off.agent_params["defend_range"][0]
    st.update_prediction(1.0, stop)
    ref_st, _, ref_t = st.puck_tracker.get_prediction(1.0, stop)
    assert st.predicted_state == pytest.approx(np.asarray(ref_st))
    assert st.predicted_time == pytest.approx(ref_t)


def _intercept(env_info, **kw):
    """Intercept for a puck crossing the robot half toward the goal, from the parked mallet."""
    a = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.02, ekf_own_goal=True, **kw)
    st = a.state
    st.x_cmd = np.array([a.agent_params["x_home"][0], a.agent_params["x_home"][1], 0.])
    st.puck_tracker.reset(np.array([0.60, 0.0, 0.]))
    s = np.zeros(6)
    s[:4] = [0.60, 0.0, -0.80, 0.0]
    st.puck_tracker.state = s
    st.update_prediction(1.0, _goal_line_v(env_info))
    return float(st.predicted_state[0]), float(st.predicted_time)


def test_rendezvous_picks_a_nearer_intercept_than_the_fixed_horizon(ei):
    """The fixed horizon runs the puck all the way to the stop line; the rendezvous does not."""
    x_fixed, t_fixed = _intercept(ei)
    x_rv, t_rv = _intercept(ei, rendezvous_vel=0.35)
    assert t_rv < t_fixed
    assert x_rv > x_fixed, "the rendezvous intercept must be met earlier, further up-table"


def test_rendezvous_intercept_is_actually_reachable(ei):
    """The property that matters: the returned (point, time) satisfies the speed budget."""
    vel = 0.35
    a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, ekf_own_goal=True,
                          rendezvous_vel=vel)
    st = a.state
    mallet = np.array([a.agent_params["x_home"][0], a.agent_params["x_home"][1]])
    st.x_cmd = np.array([mallet[0], mallet[1], 0.])
    st.puck_tracker.reset(np.array([0.60, 0.0, 0.]))
    s = np.zeros(6)
    s[:4] = [0.60, 0.0, -0.80, 0.0]
    st.puck_tracker.state = s
    st.update_prediction(1.0, _goal_line_v(ei))
    reach = np.linalg.norm(st.predicted_state[:2] - mallet)
    assert reach <= vel * st.predicted_time + 1e-9


def _split_mallet_intercept(env_info, lead, from_measured):
    """`_intercept`, but with the COMMANDED mallet leading the MEASURED one by `lead` metres.

    That lead is what the RT streamer's cap produces on hardware -- the command runs ahead of the
    arm by a DEFEND median of 16.5 mm and a p99 of 138 mm -- and every fixture above hides it by
    putting both at the same point, which is exactly why the gap went unnoticed.
    Returns ``(state, measured_mallet_xy)``.
    """
    a = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.02, ekf_own_goal=True,
                          rendezvous_vel=0.35, rendezvous_from_measured=from_measured)
    st = a.state
    measured = np.asarray(a.agent_params["x_home"][:2], dtype=float)
    st.x_actual = np.array([measured[0], measured[1], 0.])
    st.x_cmd = np.array([measured[0] + lead, measured[1], 0.])
    st.puck_tracker.reset(np.array([0.60, 0.0, 0.]))
    s = np.zeros(6)
    s[:4] = [0.60, 0.0, -0.80, 0.0]
    st.puck_tracker.state = s
    st.update_prediction(1.0, _goal_line_v(env_info))
    return st, measured


def test_rendezvous_from_measured_is_off_by_default(ei):
    """Sim and GT stay byte-identical; only run_hw_match turns it on."""
    a = MiniBaselineAgent(ei, agent_id=1, rendezvous_vel=0.35)
    assert a.agent_params["rendezvous_from_measured"] is False


def test_rendezvous_measured_is_more_conservative_than_the_command(ei):
    """The command leads the arm, so judging reach from it accepts intercepts sooner and further
    up-table than the mallet can actually manage."""
    st_cmd, _ = _split_mallet_intercept(ei, lead=0.12, from_measured=False)
    st_meas, _ = _split_mallet_intercept(ei, lead=0.12, from_measured=True)
    assert st_meas.predicted_time > st_cmd.predicted_time
    assert st_meas.predicted_state[0] < st_cmd.predicted_state[0]


def test_the_command_based_rendezvous_promises_a_reach_the_mallet_does_not_have(ei):
    """Pins the defect. The returned (point, time) satisfies the speed budget measured from
    `x_cmd`, but NOT from where the mallet actually is -- and the real reach is the one that has
    to happen in the world."""
    vel = 0.35
    st_cmd, measured = _split_mallet_intercept(ei, lead=0.12, from_measured=False)
    reach_cmd = np.linalg.norm(st_cmd.predicted_state[:2] - st_cmd.x_cmd[:2])
    reach_real = np.linalg.norm(st_cmd.predicted_state[:2] - measured)
    assert reach_cmd <= vel * st_cmd.predicted_time + 1e-9
    assert reach_real > vel * st_cmd.predicted_time, \
        "the command lead is too small here to demonstrate the defect at all"

    st_meas, measured_on = _split_mallet_intercept(ei, lead=0.12, from_measured=True)
    reach_fixed = np.linalg.norm(st_meas.predicted_state[:2] - measured_on)
    assert reach_fixed <= vel * st_meas.predicted_time + 1e-9


def test_rendezvous_falls_back_to_the_command_before_any_measurement(ei):
    """`x_actual` is a literal zeros(3) until `update_observation` runs -- the V-frame ORIGIN, not
    a mallet position. Measuring reach from there would accept almost anything, so the flag has to
    fall back to the command until a measurement has actually arrived."""
    out = []
    for from_measured in (True, False):
        a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, ekf_own_goal=True,
                              rendezvous_vel=0.35, rendezvous_from_measured=from_measured)
        st = a.state
        assert not np.any(st.x_actual), "fixture precondition: no measurement has arrived yet"
        st.x_cmd = np.array([a.agent_params["x_home"][0], a.agent_params["x_home"][1], 0.])
        st.puck_tracker.reset(np.array([0.60, 0.0, 0.]))
        s = np.zeros(6)
        s[:4] = [0.60, 0.0, -0.80, 0.0]
        st.puck_tracker.state = s
        st.update_prediction(1.0, _goal_line_v(ei))
        out.append((float(st.predicted_state[0]), float(st.predicted_time)))
    assert out[0] == pytest.approx(out[1])


def test_rendezvous_never_returns_less_than_one_plan_block(ei):
    """`should_defend` refuses anything under max_plan_steps*dt; never hand it less."""
    a = MiniBaselineAgent(ei, agent_id=1, rendezvous_vel=0.35)
    st = a.state
    st.x_cmd = np.array([0.30, 0.0, 0.])
    st.puck_tracker.reset(np.array([0.30, 0.0, 0.]))
    s = np.zeros(6)
    s[:4] = [0.30, 0.0, -0.30, 0.0]
    st.puck_tracker.state = s
    st.update_prediction(1.0, 0.)
    t_min = a.agent_params["max_plan_steps"] * a.env_info["dt"]
    assert st.predicted_time >= t_min


# --------------------------------------------------------------------------- plant vs abort line

def test_abort_line_defaults_to_the_vendored_stop_line(ei):
    a = MiniBaselineAgent(ei, agent_id=1)
    d = a.tactics_processor[TACTICS.DEFEND.value]
    assert a.agent_params["defend_abort_line"] is None
    a.state.estimated_state = np.zeros(6)
    a.state.estimated_state[0] = 0.40                       # outside the plant line
    assert d._stop_line() == pytest.approx(a.agent_params["defend_range"][0])
    a.state.estimated_state[0] = 0.20                       # inside it
    assert d._stop_line() == 0.


def test_abort_line_takes_over_the_deep_stop_line(ei):
    a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, defend_abort_dist=0.02)
    d = a.tactics_processor[TACTICS.DEFEND.value]
    abort = a.agent_params["defend_abort_line"]
    assert abort == pytest.approx(_goal_line_v(ei) + 0.02)
    assert abort < a.agent_params["defend_range"][0], "the abort line must be DEEPER than the plant line"
    a.state.estimated_state = np.zeros(6)
    a.state.estimated_state[0] = 0.40
    assert d._stop_line() == pytest.approx(a.agent_params["defend_range"][0])
    a.state.estimated_state[0] = 0.20
    assert d._stop_line() == pytest.approx(abort)


def test_plant_line_is_unchanged_by_the_abort_line(ei):
    """defend_range keeps its value and its job -- test_goal_keepout depends on that."""
    base = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02)
    goalie = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, **GOALIE)
    assert goalie.agent_params["defend_range"] == pytest.approx(base.agent_params["defend_range"])
    assert goalie.agent_params["hit_range"] == pytest.approx(base.agent_params["hit_range"])


def test_is_puck_approaching_uses_the_abort_line(ei):
    """A puck already inside the plant line must still count as approaching."""
    def approaching(**kw):
        a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, **kw)
        a.reset()
        st = a.state
        q, dq = INIT_SEED.copy(), np.zeros(len(INIT_SEED))
        # deep (inside the plant line 0.2506) and closing
        st.update_observation(q, dq, np.array([0.20, 0.0, 0.]))
        for x in (0.18, 0.16, 0.14):
            st.update_observation(q, dq, np.array([x, 0.0, 0.]))
        return st.is_puck_approaching()

    assert not approaching(), "pins the bug: the vendored counter cannot latch inside the line"
    assert approaching(defend_abort_dist=0.02)


# ------------------------------------------------------------------------------- block target

def test_block_target_default_is_vendored(ei):
    a = MiniBaselineAgent(ei, agent_id=1)
    d = a.tactics_processor[TACTICS.DEFEND.value]
    puck = np.array([0.25, 0.08])
    pos, direction, vel = d._block_target(puck)
    r_m = ei["mallet"]["radius"]
    assert direction == pytest.approx(np.array([0., np.sign(puck[1] + 1e-6)]))
    assert pos == pytest.approx(puck - direction * r_m)
    assert vel == pytest.approx(direction * 0.05)


def test_block_target_centre_plants_on_the_crossing_point(ei):
    a = MiniBaselineAgent(ei, agent_id=1, defend_block="centre")
    d = a.tactics_processor[TACTICS.DEFEND.value]
    puck = np.array([0.25, 0.08])
    pos, _, _ = d._block_target(puck)
    assert pos == pytest.approx(puck)


def test_block_target_centre_doubles_the_base_side_margin(ei):
    """Contact needs |dy| <= r_m + r_p; measure the margin each mode leaves on each side."""
    r_m, r_p = ei["mallet"]["radius"], ei["puck"]["radius"]
    reach = r_m + r_p
    puck = np.array([0.25, 0.08])                    # crossing on the +y (base) side

    def margins(mode):
        a = MiniBaselineAgent(ei, agent_id=1, defend_block=mode)
        pos, _, _ = a.tactics_processor[TACTICS.DEFEND.value]._block_target(puck)
        outer = (pos[1] + reach) - puck[1]           # slack toward +y, i.e. toward the base
        inner = puck[1] - (pos[1] - reach)
        return outer, inner

    out_v, in_v = margins("mallet")
    out_c, in_c = margins("centre")
    out_k, _ = margins("contact")
    assert out_v == pytest.approx(r_p)                       # 25 mm
    assert out_c == pytest.approx(reach)                     # 50 mm
    assert out_c == pytest.approx(2 * out_v)
    assert out_c == pytest.approx(in_c), "centre is symmetric"
    assert out_k == pytest.approx(0.0), "the Prepare convention leaves NO base-side margin"


def test_block_target_deadband_latches_the_side_across_zero(ei):
    a = MiniBaselineAgent(ei, agent_id=1, defend_block="centre", defend_centre_deadband=0.03)
    d = a.tactics_processor[TACTICS.DEFEND.value]
    d._block_side = None
    sides = [float(d._block_target(np.array([0.25, y]))[1][1])
             for y in (0.005, -0.005, 0.004, -0.002)]
    assert len(set(sides)) == 1, "the tap direction must not flip for a shot down the middle"
    # leaving the deadband re-latches
    d._block_target(np.array([0.25, -0.10]))
    assert d._block_side == -1.0


def test_block_target_deadband_off_is_the_vendored_sign_flip(ei):
    a = MiniBaselineAgent(ei, agent_id=1, defend_block="centre")
    d = a.tactics_processor[TACTICS.DEFEND.value]
    sides = [float(d._block_target(np.array([0.25, y]))[1][1]) for y in (0.005, -0.005)]
    assert sides == [1.0, -1.0]


# ------------------------------------------------------------------------------- integration

def _home_q(env_info, agent):
    """Joints that put the mallet at the agent's park, ON the play plane.

    Starting from INIT_SEED instead puts the EE 250 mm below the plane, and every Cartesian
    plan the QP is handed then fails -- so the FSM thrashes and nothing is ever measured.
    """
    x_r = agent._v2r.pos3([agent.agent_params["x_home"][0],
                           agent.agent_params["x_home"][1],
                           env_info["robot"]["ee_desired_height"]])
    ok, q = inverse_kinematics(env_info["robot"]["robot_model"],
                               env_info["robot"]["robot_data"],
                               np.asarray(x_r, dtype=float), initial_q=INIT_SEED.copy())
    assert ok, "the park must be reachable, or the harness is measuring nothing"
    return np.asarray(q[:len(INIT_SEED)], dtype=float)


def _drive(env_info, puck0, n_ticks=60, min_goal_dist=0.02, **kw):
    """Run the real agent loop against a scripted glide, mirroring `BaselineAgent.draw_action`.

    Returns one row per tick:
    (tactic name, intercept V-x, true puck V-x, mallet xy in V, puck xy in V).
    """
    a = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=min_goal_dist, **kw)
    a.reset()
    st = a.state
    truth = _tracker(env_info, own_goal=True)
    truth.reset(np.array([puck0[0], puck0[1], 0.]))
    s = np.zeros(6)
    s[:4] = puck0
    truth.state = s
    q, dq = _home_q(env_info, a), np.zeros(len(INIT_SEED))
    rows = []
    for _ in range(n_ticks):
        gt = truth.state[:4].copy()
        st.update_observation(q, dq, np.array([gt[0], gt[1], 0.]))
        for _guard in range(20):                       # the vendored loop is `while True`
            a.tactics_processor[st.tactic_current.value].update_tactic()
            active = a.tactics_processor[st.tactic_current.value]
            if active.ready():
                active.apply()
            if len(st.trajectory_buffer) > 0:
                break
        if len(st.trajectory_buffer) > 0:
            st.q_cmd, st.dq_cmd = st.trajectory_buffer[0]
            st.trajectory_buffer = st.trajectory_buffer[1:]
            st.x_cmd, st.v_cmd = st.update_ee_pos_vel(st.q_cmd, st.dq_cmd)
            q, dq = np.asarray(st.q_cmd, float), np.asarray(st.dq_cmd, float)
        rows.append((st.tactic_current.name, float(st.predicted_state[0]), float(gt[0]),
                     np.asarray(st.x_cmd[:2], dtype=float).copy(), gt[:2].copy()))
        nxt, _ = truth.predict(truth.state, truth.P)
        truth.state = nxt
    return a, rows


def test_defend_persists_below_the_plant_line(ei):
    """The headline behaviour: keep defending a puck that is already deep.

    The puck is scripted for the END mount: it starts 50 mm outside the plant line (0.2506) and
    closes at 0.7 m/s -- fast enough to clear static_vel_threshold (0.202) and count as
    approaching, slow enough to take ~0.18 s to cross the 118 mm band between the plant line
    and the abort line (0.1329), and it is inside the plant line for the rest of the run.
    Measured on this script: the goalie holds DEFEND for 8 ticks while the puck is inside the
    plant line -- every one of them in that band, not after the puck is already through the
    goal -- and plans as deep as V-x 0.136.

    The old script (0.42 at 1.2 m/s) was derived at the side mount; at the end mount that puck
    crosses the deep band in ~0.1 s, so the goalie got 4 deep DEFEND ticks against the 5 asked
    for -- a geometry change reading as a behaviour regression.
    """
    plant = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02).agent_params["defend_range"][0]
    _, vendored = _drive(ei, [0.30, 0.02, -0.70, 0.0])
    _, goalie = _drive(ei, [0.30, 0.02, -0.70, 0.0], **GOALIE)

    deep_v = [r for r in vendored if r[0] == "DEFEND" and r[2] < plant]
    deep_g = [r for r in goalie if r[0] == "DEFEND" and r[2] < plant]
    assert not deep_v, "pins the bug: nothing defends inside the plant line today"
    assert len(deep_g) >= 5, "the goalie must keep defending after the puck crosses the line"

    intercepts = [r[1] for r in goalie if r[0] == "DEFEND"]
    assert min(intercepts) < plant, "and must be willing to plant deeper than the plant line"


def _worst_up_table_lead(rows):
    """How far UP-TABLE of the real puck the agent aimed, worst case, while acting on it."""
    live = [r[1] - r[2] for r in rows if r[0] in ("DEFEND", "SMASH", "PREPARE")]
    return max(live) if live else None


def test_goalie_does_not_aim_up_table_of_the_puck(ei):
    """"It swings, but forward, not at the puck."

    A puck settling deep in our own half is where this shows: the vendored `_update_prediction`
    hands `get_prediction` a stop line of 0. once the puck is inside the plant line, so the roll
    runs the full horizon, bounces off the phantom near rim, and comes back UP-TABLE. PREPARE
    then drives the mallet to that phantom -- past the puck, away from the goal.

    (The margin here is modest because the sim's vendored damping is ~2.4x the value fitted on
    the real table; with `mini_table_ekf.npz` the same puck predicts from V-x 0.20 to 0.64.)
    """
    puck = [0.20, 0.00, -0.15, 0.0]
    _, vendored = _drive(ei, puck, n_ticks=30)
    _, goalie = _drive(ei, puck, n_ticks=30, **GOALIE)

    assert _worst_up_table_lead(vendored) > 0.03, "pins the bug: the arm aims past the puck"
    assert _worst_up_table_lead(goalie) <= 0.0, "the goalie must never aim up-table of the puck"


def test_a_gentle_shot_on_goal_gets_defended_at_all(ei):
    """A slow shot that starts inside the defend band draws no response at all today.

    `should_defend` needs the crossing to be >= max_plan_steps*dt away AND the puck to be
    up-table of the plant line, so a shot that is already close simply never triggers.
    """
    puck = [0.22, 0.03, -0.40, 0.0]
    _, vendored = _drive(ei, puck, n_ticks=30)
    _, goalie = _drive(ei, puck, n_ticks=30, **GOALIE)

    assert not [r for r in vendored if r[0] == "DEFEND"], "pins the bug: no defend is attempted"
    assert len([r for r in goalie if r[0] == "DEFEND"]) >= 5


# ------------------------------------------------------------------- latched DEFEND's target

def _defend_plans(env_info, puck0, n_ticks=45, **kw):
    """Drive the agent and collect every target DEFEND actually planned a block to.

    `_drive` reports the INTERCEPT; what this section is about is the TARGET the tactic hands
    the planner, and the intercept is only that for as long as DEFEND's own gate still accepts
    it. The two can only diverge inside `apply`, so spy where they meet.
    """
    seen = []
    orig = MiniDefend._block_target

    def spy(self, puck_pos_2d):
        seen.append(np.asarray(puck_pos_2d[:2], dtype=float).copy())
        return orig(self, puck_pos_2d)

    MiniDefend._block_target = spy
    try:
        a, rows = _drive(env_info, puck0, n_ticks=n_ticks, **kw)
    finally:
        MiniDefend._block_target = orig
    return a, rows, seen


#: A shot at the near rim just OUTSIDE the goal mouth (|y| 0.18 against a 0.086 half-mouth):
#: DEFEND latches on the approach, then the puck rebounds off the rim -- which it really does
#: out there -- and heads back up-table. The intercept follows it out of the defend zone while
#: DEFEND is still latched, `defend_give_up_ticks` not yet spent. That is the `vp_base`
#: t = 0.45-1.11 s case, where the plan ran to V-x 0.78 and the arm chased it to 0.66.
_REBOUND_PUCK = [0.35, -0.18, -0.90, 0.0]


def test_latched_defend_chases_its_target_out_of_the_zone(ei):
    """Pin the bug: a latched DEFEND re-plans to the estimate without re-checking its gate."""
    a, _, plans = _defend_plans(ei, _REBOUND_PUCK, **{**GOALIE, "defend_hold_target": False})
    d = a.tactics_processor[TACTICS.DEFEND.value]
    assert plans, "the harness must actually plan some blocks, or it measures nothing"
    outside = [p for p in plans if not d._in_zone(p)]
    assert outside, "pins the bug: the target leaves the very zone DEFEND gated on"
    # Deliberately NO assertion on how far outside. Run on its own the scenario is
    # deterministic to the bit -- the worst target reaches V-x 0.46 against a 0.355 zone edge,
    # repeatably -- but run after the rest of the suite that worst case is 0.37. Something
    # earlier in the session changes this rebound and it is not yet understood, so a margin
    # here would pin that, not the defect.
    assert all(p[0] > a.agent_params["defend_range"][0] for p in outside), \
        "and they escape UP-TABLE, away from the goal -- the swing-forward failure"


def test_defend_hold_keeps_every_planned_target_inside_the_zone(ei):
    """The fix: keep blocking the last target that passed the gate."""
    a, rows, plans = _defend_plans(ei, _REBOUND_PUCK, **GOALIE)
    d = a.tactics_processor[TACTICS.DEFEND.value]
    assert plans
    assert all(d._in_zone(p) for p in plans), \
        f"held targets must stay in the zone: {[np.round(p, 3).tolist() for p in plans]}"
    assert len([r for r in rows if r[0] == "DEFEND"]) >= 5, \
        "and holding must not cost us the defend itself"


def test_defend_hold_still_tracks_a_target_that_stays_in_the_zone(ei):
    """The hold is a fallback, not a freeze: an in-zone intercept is followed as before."""
    puck = [0.30, 0.02, -0.70, 0.0]
    _, _, held = _defend_plans(ei, puck, **GOALIE)
    _, _, free = _defend_plans(ei, puck, **{**GOALIE, "defend_hold_target": False})
    assert held and len(held) == len(free)
    for h, f in zip(held, free):
        assert h == pytest.approx(f), "an in-zone defend must plan exactly as it does today"


def test_defaults_leave_every_goalie_knob_off(ei):
    """The contract: an agent built with no new arguments is the vendored one."""
    a = MiniBaselineAgent(ei, agent_id=1)
    p = a.agent_params
    assert p["ekf_own_goal"] is False
    assert p["rendezvous_vel"] is None
    assert p["defend_abort_line"] is None
    assert p["defend_block_offset"] == "mallet"
    assert p["defend_centre_deadband"] == 0.0
    assert p["defend_fresh_horizon"] is False
    assert p["defend_give_up_ticks"] == 4
    assert p["defend_gate_unled"] is False
    assert p["defend_hold_target"] is False
    assert a.state._workspace_clamp is False
    assert not isinstance(a.state.puck_tracker.system.table, MiniAirHockeyTable)


# ------------------------------------------------------------------------------- CLI wiring

def test_goalie_flag_expands_to_the_group():
    run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")
    args = argparse.Namespace(goalie=True, **{k: None for k in run_hw_match.GOALIE_DEFAULTS})
    args.ekf_own_goal = False                       # store_true flags arrive as False, not None
    run_hw_match.apply_goalie_defaults(args)
    for name, value in run_hw_match.GOALIE_DEFAULTS.items():
        assert getattr(args, name) == value


def test_goalie_flag_never_overrides_an_explicit_value():
    run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")
    args = argparse.Namespace(goalie=True, **{k: None for k in run_hw_match.GOALIE_DEFAULTS})
    args.ekf_own_goal = False
    args.rendezvous_vel = 0.5
    args.defend_block = "mallet"
    args.gate_unled = False
    run_hw_match.apply_goalie_defaults(args)
    assert args.rendezvous_vel == 0.5
    assert args.defend_block == "mallet"
    assert args.gate_unled is False                 # --no-gate-unled must survive --goalie
    assert args.defend_abort_dist == 0.02           # the rest still filled in


def test_no_goalie_flag_changes_nothing():
    run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")
    args = argparse.Namespace(goalie=False, **{k: None for k in run_hw_match.GOALIE_DEFAULTS})
    args.ekf_own_goal = False
    run_hw_match.apply_goalie_defaults(args)
    assert args.rendezvous_vel is None
    assert args.defend_abort_dist is None
    assert args.ekf_own_goal is False


# ------------------------------------------------------------------------------ base keep-out

def _real_rig_env():
    """env_info with the REAL side mount loaded, so the disc geometry is the rig's own."""
    run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")
    env_info = mini_platform.make("mini-hit-sdk").env_info
    path = pathlib.Path(__file__).resolve().parents[2] / "base_frame_fixed.npz"
    if not path.exists():
        pytest.skip("base_frame_fixed.npz not present")
    run_hw_match.load_base_frame(env_info, str(path))
    return env_info


def test_base_keepout_none_is_a_noop(ei):
    a = MiniBaselineAgent(ei, agent_id=1)
    assert a.agent_params["base_keepout_radius"] is None
    for xy in ([0.14, 0.22], [0.30, 0.0], [0.5, -0.2]):
        got = a.traj_generator.project_out_of_base(np.array(xy))
        assert got is not None and np.asarray(got) == pytest.approx(np.array(xy))


def test_base_keepout_projection_is_radial_in_the_robot_frame(ei):
    r = 0.26
    a = MiniBaselineAgent(ei, agent_id=1, base_keepout=r)
    inside = np.array([a.agent_params["x_home"][0], a.agent_params["x_home"][1]])
    # walk in toward the base until we are inside the disc, then project
    for _ in range(40):
        if np.linalg.norm(a._v2r.pos2(inside)) < r:
            break
        inside = np.asarray(a._r2v.pos2(a._v2r.pos2(inside) * 0.9))
    p_in = a._v2r.pos2(inside)
    assert np.linalg.norm(p_in) < r, "test setup: need a point inside the disc"
    out = a.traj_generator.project_out_of_base(inside)
    p_out = a._v2r.pos2(out)
    assert np.linalg.norm(p_out) == pytest.approx(r)
    # same bearing from the base, only the radius changed
    assert np.arctan2(*p_out[::-1]) == pytest.approx(np.arctan2(*p_in[::-1]))


def test_the_x_floor_permits_the_worst_fold_over_point_and_the_disc_does_not():
    """The whole argument for the disc, pinned on the real mount's numbers."""
    env_info = _real_rig_env()
    a = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.06)   # the x floor, ON
    reach = lambda xy: float(np.linalg.norm(a._v2r.pos2(np.array(xy))))

    worst = [a.agent_params["defend_range"][0],       # on the defend line...
             a.agent_params["defend_width"]]         # ...at the edge the defender will chase to
    mouth = [_goal_line_v(env_info), 0.0]  # dead centre of our own goal mouth
    assert reach(worst) == pytest.approx(0.198, abs=0.004)
    assert reach(mouth) == pytest.approx(0.364, abs=0.004)
    # the x floor allows the dangerous one and blocks the safe one
    assert worst[0] >= a.agent_params["defend_range"][0]
    assert mouth[0] < a.agent_params["defend_range"][0]
    # the disc does the opposite, which is the point
    r = 0.26
    assert reach(worst) < r
    assert reach(mouth) > r


def test_base_keepout_leaves_the_goal_mouth_reachable():
    """R = 0.26 must not clip the mouth: |y| up to the post plus a mallet radius."""
    pytest.skip(
        "the 0.26 disc is the SIDE-mount keep-out, and it is dead at the 2026-09-10 end mount. "
        "With the base standing behind its own goal on the table's own axis, that radius "
        "forbids the own goal line (0.113 m of reach), the ready-guard point (0.213) AND the "
        "defend plant line (0.251) -- the entire defensive band -- which is why "
        "project_out_of_base documents it as off here and --base-keepout defaults to None. "
        "Clipping the mouth is the EXPECTED behaviour of this radius at this mount, not a "
        "regression. Re-derive a radius with --stage reach-probe (expect well under 0.15) "
        "before re-enabling this.")
    env_info = _real_rig_env()
    a = MiniBaselineAgent(env_info, agent_id=1, base_keepout=0.26)
    gl = _goal_line_v(env_info)
    half_mouth = env_info["table"]["goal_width"] / 2 + env_info["mallet"]["radius"]
    for y in np.linspace(-half_mouth, half_mouth, 9):
        target = np.array([gl, y])
        assert a.traj_generator.project_out_of_base(target) == pytest.approx(target), \
            f"the keep-out must not fence off the goal mouth (y={y:+.3f})"


def test_base_keepout_pushes_trajectory_waypoints_onto_the_rim():
    """Projecting the TARGET is not enough -- the Bezier never clamps the curve itself."""
    env_info = _real_rig_env()
    r = 0.26
    a = MiniBaselineAgent(env_info, agent_id=1, base_keepout=r)
    gen = a.traj_generator
    # a straight sweep along the goal line, from the mouth out past the base-side rail
    n = 12
    traj = np.zeros((n, 9))
    traj[:, 0] = _goal_line_v(env_info)
    traj[:, 1] = np.linspace(0.0, 0.229, n)
    traj[:, 2] = env_info["robot"]["ee_desired_height"]
    traj[:, 3] = 0.0
    traj[:, 4] = 0.5                                     # sweeping toward the base side
    raw_r = traj[:, 0:2] @ gen._v2r.rot2.T + gen._v2r.t2
    assert (np.linalg.norm(raw_r, axis=1) < r).any(), "test setup: the sweep must enter the disc"

    seen = {}
    orig = type(gen).__mro__[1].optimize_trajectory

    def spy(self, cart_traj_r, *rest):
        seen["r"] = np.asarray(cart_traj_r).copy()
        return True, np.zeros((len(cart_traj_r), 2, env_info["robot"]["n_joints"]))

    type(gen).__mro__[1].optimize_trajectory = spy
    try:
        gen.optimize_trajectory(traj, np.zeros(6), np.zeros(6), INIT_SEED.copy())
    finally:
        type(gen).__mro__[1].optimize_trajectory = orig
    rad = np.linalg.norm(seen["r"][:, 0:2], axis=1)
    assert (rad >= r - 1e-9).all(), "every waypoint must be on or outside the keep-out rim"


def test_mallet_reach_r_matches_the_agent_park():
    """The watchdog and the planner must agree about what 'reach' means."""
    run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")
    env_info = _real_rig_env()
    a = MiniBaselineAgent(env_info, agent_id=1)
    q = _home_q(env_info, a)
    expected = float(np.linalg.norm(a._v2r.pos2(a.agent_params["x_home"][:2])))
    assert run_hw_match.mallet_reach_r(env_info, q) == pytest.approx(expected, abs=1e-3)


def test_unreachable_puck_aims_at_the_closest_approach_not_the_far_rail(ei):
    """When nothing on the predicted path is reachable, aim at the best attempt.

    A puck running down the base-side rail rebounds off the near rim and recedes faster than
    the mallet can close. The vendored fallback is "where the puck is at the END of the
    horizon", which after that rebound is up-table and then gets pinned to hit_range[1] by the
    workspace clamp -- so the arm lunges at the far rail while the puck is beside it.
    """
    def intercept(**kw):
        a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, ekf_own_goal=True, **kw)
        st = a.state
        st.x_cmd = np.array([a.agent_params["x_home"][0], a.agent_params["x_home"][1], 0.])
        st.puck_tracker.reset(np.array([0.24, 0.20, 0.]))
        s = np.zeros(6)
        s[:4] = [0.24, 0.20, -0.90, 0.0]
        st.puck_tracker.state = s
        st.update_prediction(1.0, a.agent_params["defend_abort_line"] or 0.)
        return np.asarray(st.predicted_state[:2], dtype=float), float(st.predicted_time)

    puck_x = 0.24
    far, t_far = intercept(defend_abort_dist=0.02)
    near, t_near = intercept(defend_abort_dist=0.02, rendezvous_vel=0.35)

    assert far[0] - puck_x > 0.3, "pins the bug: the fallback aims far up-table of the puck"
    assert t_far == pytest.approx(1.0), "and commits to the very end of the horizon"
    assert near[0] < far[0], "the fallback must aim nearer than the end of the horizon"
    assert t_near < t_far


def test_rendezvous_target_reaches_the_workspace_clamp(ei):
    """The rendezvous is an extrapolation, so the SAFETY clamp must not be skipped for it.

    The clamp sits behind an early return that fires when lead/jump/EMA are all unset -- which
    is the default hardware command line.
    """
    a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, ekf_own_goal=True,
                          defend_abort_dist=0.02, rendezvous_vel=0.35)
    st = a.state
    st.x_cmd = np.array([a.agent_params["x_home"][0], a.agent_params["x_home"][1], 0.])
    st.puck_tracker.reset(np.array([0.24, 0.20, 0.]))
    s = np.zeros(6)
    s[:4] = [0.24, 0.20, -3.0, 0.0]            # fast enough to extrapolate off the reach
    st.puck_tracker.state = s
    st.update_prediction(1.0, a.agent_params["defend_abort_line"])
    assert st.predicted_state[0] <= a.agent_params["hit_range"][1] + 1e-9
    assert abs(st.predicted_state[1]) <= ei["table"]["width"] / 2 + 1e-9


def test_workspace_clamp_applies_with_the_intercept_knobs_off(ei):
    """The clamp must not depend on whether lead / jump / EMA / rendezvous happen to be on.

    It sat behind their early return, so the PLAINEST hardware command line -- none of them set
    -- was the one running unclamped: this puck predicts to V-x 0.70 against a 0.51 m hit range,
    and the arm slews out to reach it, into its joint limits and an over-current fault.
    """
    def intercept(**kw):
        a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, ekf_own_goal=True,
                              defend_abort_dist=0.02, **kw)
        st = a.state
        st.x_cmd = np.array([a.agent_params["x_home"][0], a.agent_params["x_home"][1], 0.])
        st.puck_tracker.reset(np.array([0.24, 0.20, 0.]))
        s = np.zeros(6)
        s[:4] = [0.24, 0.20, -0.90, 0.0]
        st.puck_tracker.state = s
        st.update_prediction(1.0, a.agent_params["defend_abort_line"])
        assert st._knobs_off(), "the whole point of this test is the knobs-OFF path"
        return st, a.agent_params["hit_range"][1]

    loose, reach = intercept()
    clamped, _ = intercept(workspace_clamp=True)
    assert loose.predicted_state[0] > reach + 0.05, \
        "pins the bug: an unclamped intercept runs past the arm's reach"
    assert clamped.predicted_state[0] == pytest.approx(reach)
    assert abs(clamped.predicted_state[1]) <= ei["table"]["width"] / 2 + 1e-9
    assert clamped._intercept_last is None, \
        "the clamp must not switch on the rate-limit bookkeeping, which has its own knobs"


def test_keepout_clamps_waypoints_the_planner_would_otherwise_emit():
    """End-to-end: raw plans DO enter the disc, and the clamp is what keeps them out."""
    env_info = _real_rig_env()
    from mini_platform.agents.baseline_mini import MiniTrajectoryGenerator
    radius, raw = 0.26, []
    orig = MiniTrajectoryGenerator.optimize_trajectory

    def spy(self, cart_traj, *rest):
        t = np.asarray(cart_traj)
        raw.append(float(np.linalg.norm(
            t[:, 0:2] @ self._v2r.rot2.T + self._v2r.t2, axis=1).min()))
        return orig(self, cart_traj, *rest)

    MiniTrajectoryGenerator.optimize_trajectory = spy
    try:
        a, rows = _drive(env_info, [0.30, 0.20, -0.90, 0.0], n_ticks=25,
                         min_goal_dist=0.0, base_keepout=radius, **GOALIE)
    finally:
        MiniTrajectoryGenerator.optimize_trajectory = orig

    assert min(raw) < radius, "test setup: the planner must actually try to enter the disc"
    achieved = [float(np.linalg.norm(a._v2r.pos2(r[3]))) for r in rows]
    # the QP tracks a clamped waypoint with some lag, which is what the hard margin absorbs
    assert min(achieved) >= radius - 0.03, "commanded mallet must stay outside the hard catch"


# --------------------------------------------------------------------------------- ready guard

def _ready(env_info, **kw):
    a = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.02, **kw)
    return a, a.tactics_processor[TACTICS.READY.value]


def _set_puck(agent, xy):
    s = np.zeros(6)
    s[:2] = xy
    agent.state.estimated_state = s


def test_ready_guard_off_parks_at_x_home(ei):
    a, r = _ready(ei)
    assert a.agent_params["ready_guard"] is False
    _set_puck(a, [0.70, 0.20])
    assert r._guard_target() == pytest.approx(np.asarray(a.agent_params["x_home"][:2]))


def test_ready_t_stop_floor_defaults_to_the_vendored_second(ei):
    a, _ = _ready(ei)
    assert a.agent_params["ready_t_stop_min"] == 1.0


@pytest.mark.parametrize("puck", [[0.70, 0.20], [0.70, -0.20], [0.70, 0.0], [0.45, 0.15]])
def test_guard_point_sits_on_the_puck_to_goal_line(ei, puck):
    radius = 0.10
    a, r = _ready(ei, ready_guard=True, ready_guard_radius=radius)
    _set_puck(a, puck)
    goal = r._goal_centre_v()
    g = r._guard_target()
    d = np.asarray(puck, float) - goal
    d /= np.linalg.norm(d)
    along = float(np.dot(g - goal, d))
    assert float(np.linalg.norm((g - goal) - along * d)) == pytest.approx(0.0, abs=1e-9)
    assert along == pytest.approx(radius)


@pytest.mark.parametrize("puck", [[0.0, 0.0], [2.0, 0.0], [0.5, 0.9], [np.nan, 0.1]])
def test_guard_falls_back_to_the_park_when_the_puck_is_not_on_the_table(ei, puck):
    """A zeroed estimated_state before the first observation looks exactly like this."""
    a, r = _ready(ei, ready_guard=True)
    _set_puck(a, puck)
    assert r._guard_target() == pytest.approx(np.asarray(a.agent_params["x_home"][:2]))


def test_guard_never_waits_deeper_than_the_planner_can_route(ei):
    a, r = _ready(ei, ready_guard=True, ready_guard_radius=0.005, defend_abort_dist=0.02)
    _set_puck(a, [0.70, 0.0])
    floor = max(float(a.traj_generator.bound_points[:, 0].min()),
                a.agent_params["defend_abort_line"])
    assert r._guard_target()[0] >= floor - 1e-9


def test_guard_is_not_floored_at_the_plant_line(ei):
    """The plant line is where the defender MEETS the puck, not a limit on where we may wait.

    Flooring the guard there would push it back out of the band it exists to cover.
    """
    a, r = _ready(ei, ready_guard=True, ready_guard_radius=0.05, defend_abort_dist=0.02)
    _set_puck(a, [0.70, 0.0])
    assert r._guard_target()[0] < a.agent_params["defend_range"][0]


def test_guard_stays_inside_what_the_defender_chases(ei):
    a, r = _ready(ei, ready_guard=True, ready_guard_radius=0.40)   # deliberately way out
    _set_puck(a, [0.30, 0.25])
    assert abs(r._guard_target()[1]) <= a.agent_params["defend_width"] + 1e-9


def test_guard_respects_the_base_keepout():
    env_info = _real_rig_env()
    radius = 0.26
    a, r = _ready(env_info, ready_guard=True, ready_guard_radius=0.30, base_keepout=radius)
    for puck_y in (0.20, 0.25, 0.15):
        _set_puck(a, [0.30, puck_y])
        assert float(np.linalg.norm(a._v2r.pos2(r._guard_target()))) >= radius - 1e-9


def test_guard_does_not_re_aim_below_the_drift_threshold(ei):
    a, r = _ready(ei, ready_guard=True, ready_guard_drift=0.02)
    a.reset()
    a.state.is_new_tactic = True
    _set_puck(a, [0.70, 0.20])
    r.ready()
    first = r._guard_xy.copy()
    a.state.is_new_tactic = False
    a.state.trajectory_buffer = []
    _set_puck(a, [0.7005, 0.2005])          # EKF jitter
    r.ready()
    assert r._guard_xy == pytest.approx(first), "sub-threshold jitter must not re-aim the guard"
    _set_puck(a, [0.70, -0.20])             # a real move
    r.ready()
    assert not np.allclose(r._guard_xy, first)


def _settle(env_info, puck_xy, n=60, **kw):
    """Hold the puck still on the opponent half; return where the mallet chooses to wait."""
    a = MiniBaselineAgent(env_info, agent_id=1, **kw)
    a.reset()
    st = a.state
    q, dq = _home_q(env_info, a), np.zeros(len(INIT_SEED))
    for _ in range(n):
        st.update_observation(q, dq, np.array([puck_xy[0], puck_xy[1], 0.]))
        for _guard in range(20):
            a.tactics_processor[st.tactic_current.value].update_tactic()
            active = a.tactics_processor[st.tactic_current.value]
            if active.ready():
                active.apply()
            if len(st.trajectory_buffer) > 0:
                break
        if len(st.trajectory_buffer) > 0:
            st.q_cmd, st.dq_cmd = st.trajectory_buffer[0]
            st.trajectory_buffer = st.trajectory_buffer[1:]
            st.x_cmd, st.v_cmd = st.update_ee_pos_vel(st.q_cmd, st.dq_cmd)
            q, dq = np.asarray(st.q_cmd, float), np.asarray(st.dq_cmd, float)
    return np.asarray(st.x_cmd[:2], dtype=float)


@pytest.mark.parametrize("puck", [[0.70, 0.20], [0.70, -0.20], [0.70, 0.0]])
def test_guard_waits_on_the_shot_line_where_the_park_does_not(puck):
    """The guard's whole premise: be on the line BEFORE the shot, not sprint to it after."""
    env_info = _real_rig_env()
    goal = np.array([_goal_line_v(env_info), 0.0])
    d = goal - np.asarray(puck, float)
    d /= np.linalg.norm(d)

    def offset(mallet):
        v = mallet - np.asarray(puck, float)
        return float(np.linalg.norm(v - float(np.dot(v, d)) * d))

    parked = offset(_settle(env_info, puck, min_goal_dist=0.06))
    guarded = offset(_settle(env_info, puck, min_goal_dist=0.0, ready_guard=True,
                             ready_t_stop_min=0.25, base_keepout=0.26, **GOALIE))
    assert guarded == pytest.approx(0.0, abs=0.005)
    assert guarded < parked


# ------------------------------------------------------------------------------------- VPSTO

def _need_vpsto():
    pytest.importorskip("vpsto")
    pytest.importorskip("cma")


def test_vpsto_path_cursor_advances_by_the_waypoints_handed_out():
    """The cached VPSTO path used to replay at 1/max_plan_steps speed.

    `plan_strike_vpsto` runs once per apply(), i.e. once per max_plan_steps agent ticks, and
    hands out that many waypoints -- which are consumed one per tick. Advancing the sampling
    cursor by a single dt per call meant 100 ms of wall time bought 20 ms of path, with each
    buffer starting behind where the last one ended.
    """
    _need_vpsto()
    from mini_platform.agents.baseline_mini import MiniTrajectoryGenerator
    env_info = _real_rig_env()
    seen = []
    orig = MiniTrajectoryGenerator.plan_strike_vpsto

    def spy(self, *a, **k):
        before = self._vp_tick
        out = orig(self, *a, **k)
        if out[0]:
            seen.append((before, self._vp_tick, len(out[1])))
        return out

    MiniTrajectoryGenerator.plan_strike_vpsto = spy
    try:
        _drive(env_info, [0.42, 0.02, -1.20, 0.0], n_ticks=25, min_goal_dist=0.0,
               planner="vpsto", **GOALIE)
    finally:
        MiniTrajectoryGenerator.plan_strike_vpsto = orig

    assert seen, "the VPSTO strike path must actually run"
    for before, after, n in seen:
        assert n > 1, "one dt of path per apply IS the 1/5-speed replay bug"
        # a re-solve zeroes the cursor before sampling, so either it advanced from `before`
        # or it restarted at 0 and advanced from there
        assert after in (before + n, n), f"cursor {before} -> {after} for {n} waypoints"


def test_vpsto_rendezvous_uses_the_agent_pipeline_not_the_raw_tracker():
    """The gate and the executed target must be the same intercept.

    `_vp_rendezvous_solve` called `tracker.get_prediction` directly, skipping the latency lead,
    the intercept rate-limit and the workspace clamp -- so under `--planner vpsto` the tactics
    gated on one intercept while the arm was sent to another.
    """
    _need_vpsto()
    env_info = _real_rig_env()
    a = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.0, planner="vpsto",
                          lead_time=0.06, **GOALIE)
    a.reset()
    st = a.state
    q = _home_q(env_info, a)
    for x in (0.45, 0.42, 0.39, 0.36):
        st.update_observation(q, np.zeros(len(INIT_SEED)), np.array([x, 0.05, 0.]))

    calls = []
    real = st.intercept_at

    def spy(*args, **kw):
        calls.append(args)
        return real(*args, **kw)

    st.intercept_at = spy
    try:
        a.traj_generator._vp_rendezvous_solve(
            st.x_cmd[:2], st.v_cmd[:2], st, st.predicted_time,
            a.agent_params["defend_abort_line"], True, None, 0.05, q)
    finally:
        st.intercept_at = real
    assert calls, "the rendezvous must go through the agent's lead/clamp pipeline"


def test_intercept_at_does_not_mutate_the_committed_prediction(ei):
    """The rendezvous iterates candidate arrival times; that must not disturb the gates."""
    a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, lead_time=0.06, **GOALIE)
    st = a.state
    st.x_cmd = np.array([a.agent_params["x_home"][0], a.agent_params["x_home"][1], 0.])
    st.puck_tracker.reset(np.array([0.60, 0.02, 0.]))
    s = np.zeros(6)
    s[:4] = [0.60, 0.02, -0.80, 0.0]
    st.puck_tracker.state = s
    st.update_prediction(1.0, a.agent_params["defend_abort_line"])
    committed = (st.predicted_state.copy(), st.predicted_time,
                 st.predicted_gate_state.copy(), st._intercept_last)
    for t in (0.2, 0.4, 0.8):
        st.intercept_at(t, a.agent_params["defend_abort_line"])
    assert st.predicted_state == pytest.approx(committed[0])
    assert st.predicted_time == pytest.approx(committed[1])
    assert st.predicted_gate_state == pytest.approx(committed[2])
    assert st._intercept_last is committed[3]


def test_vpsto_rendezvous_honours_a_held_target():
    """The rendezvous re-queries `intercept_at`, so the DEFEND hold must reach inside it.

    Otherwise the bezier path holds its target and the VPSTO path quietly walks back onto the
    moving intercept -- the same bug, one layer down and only under `--planner vpsto`.
    """
    _need_vpsto()
    env_info = _real_rig_env()
    a = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.0, planner="vpsto", **GOALIE)
    a.reset()
    st = a.state
    q = _home_q(env_info, a)
    for x in (0.45, 0.42, 0.39, 0.36):
        st.update_observation(q, np.zeros(len(INIT_SEED)), np.array([x, 0.05, 0.]))

    calls = []
    real = st.intercept_at

    def spy(*args, **kw):
        calls.append(args)
        return real(*args, **kw)

    held = np.array([0.30, -0.04])
    st.intercept_at = spy
    try:
        _, _, P_v = a.traj_generator._vp_rendezvous_solve(
            st.x_cmd[:2], st.v_cmd[:2], st, st.predicted_time,
            a.agent_params["defend_abort_line"], True, None, 0.05, q, fixed_P_v=held)
    finally:
        st.intercept_at = real
    assert not calls, "a held target must not be re-derived from the tracker"
    assert P_v == pytest.approx(held)


# ------------------------------------------------------------------------- VPSTO bugs (fix 4)

def _vpsto_defend(env_info, puck_xs=(0.45, 0.42, 0.39, 0.36), **kw):
    """A latched DEFEND under `--planner vpsto`, ready to `apply()`.

    Built by hand rather than driven through `_drive`: these tests are about what ONE apply
    plans, and the FSM's route into DEFEND is covered by the integration tests above.
    """
    a = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.0, planner="vpsto",
                          **{**GOALIE, **kw})
    a.reset()
    st = a.state
    q = _home_q(env_info, a)
    st.q_cmd = np.asarray(q, dtype=float).copy()
    st.dq_cmd = np.zeros(len(INIT_SEED))
    st.x_cmd, st.v_cmd = st.update_ee_pos_vel(st.q_cmd, st.dq_cmd)
    for x in puck_xs:
        st.update_observation(q, np.zeros(len(INIT_SEED)), np.array([x, 0.05, 0.]))
    st.update_prediction(a.agent_params["max_prediction_time"],
                         a.agent_params["defend_abort_line"])
    d = a.tactics_processor[TACTICS.DEFEND.value]
    st.is_new_tactic = True
    d.ready()
    return a, st, d, q


def _strike_args(st, d):
    return (st.q_cmd, st.dq_cmd, st.x_cmd[:2], st.v_cmd[:2], st, st.predicted_time,
            d._stop_line())


def test_a_smash_never_replays_the_cached_defend_path():
    """4a: the VPSTO cache was keyed on the intercept POSITION, not on the tactic.

    A SMASH latching near where a DEFEND had been blocking found the drift under VP_DRIFT,
    skipped the solve and replayed the block -- ending at 0.05 m/s sideways (a plant) instead of
    max_hit_velocity toward the goal (a shot).
    """
    _need_vpsto()
    from mini_platform.agents.baseline_mini import MiniTrajectoryGenerator, TABLE_LENGTH
    env_info = _real_rig_env()
    a, st, d, q = _vpsto_defend(env_info)
    g = a.traj_generator
    P = np.asarray(st.predicted_state[:2], dtype=float)
    goal = np.array([TABLE_LENGTH / 2 + V_SETBACK, 0.0])

    # What the path solver was ASKED for, per solve. A re-solve is a second call, and the terminal
    # velocity is the whole difference between a shot and a plant. Reading it back off the
    # solution instead would run vpsto's own sampler, which is not what this test is about.
    plans = []
    orig_plan = g._vp.plan

    def spy(x0, v0, xT, vT):
        plans.append((np.asarray(xT, dtype=float).copy(), np.asarray(vT, dtype=float).copy()))
        return orig_plan(x0, v0, xT, vT)

    # Whether a vertical wrist reaches the SMASH's strike point is 4d's subject and has its own
    # test; at this mount it does not, so force it here and leave the cache as the only variable.
    orig_anchor = MiniTrajectoryGenerator.solve_anchor_pos_ik_null

    def vertical(self, hit_pos_2d, hit_dir_2d, q_0):
        out = orig_anchor(self, hit_pos_2d, hit_dir_2d, q_0)
        self.last_anchor_vertical = True
        return out

    g._vp.plan = spy
    MiniTrajectoryGenerator.solve_anchor_pos_ik_null = vertical
    try:
        ok, _, _ = g.plan_strike_vpsto(*_strike_args(st, d), is_defend=True, goal_pos_v=None,
                                       hit_vel_mag=0.05, P_now_v=P, q_seed=q)
        assert ok, "the block must solve, or this test proves nothing"
        assert len(plans) == 1
        # The SAME point 2 mm on -- well inside VP_DRIFT, so the tactic is the only thing that
        # has changed. That is what used to buy the smash a replay of the block.
        ok, _, _ = g.plan_strike_vpsto(*_strike_args(st, d), is_defend=False, goal_pos_v=goal,
                                       hit_vel_mag=float(a.agent_params["max_hit_velocity"]),
                                       P_now_v=P + 0.002, q_seed=q)
    finally:
        g._vp.plan = orig_plan
        MiniTrajectoryGenerator.solve_anchor_pos_ik_null = orig_anchor

    assert ok
    assert len(plans) == 2, "pins the bug: the smash replayed the block's cached path"
    (_, vT_block), (_, vT_smash) = plans
    assert float(np.linalg.norm(vT_block)) == pytest.approx(0.05, abs=1e-3), "a block plants"
    assert float(np.linalg.norm(vT_smash)) > 0.3, "a smash must end at speed, not at 0.05 m/s"
    assert vT_smash[0] > abs(vT_smash[1]), "and drive at the goal, not sideways like a block"


def test_the_vpsto_block_aims_where_defend_block_says():
    """4b: `_vp_aim` hard-coded the vendored offset, so --defend-block never reached the plan."""
    _need_vpsto()
    env_info = _real_rig_env()
    a, st, d, q = _vpsto_defend(env_info, defend_block="centre")
    g = a.traj_generator
    P = np.array([0.30, 0.06])

    vendored, _ = g._vp_aim(P, True, None)
    assert abs(float(vendored[1]) - float(P[1])) == pytest.approx(
        env_info["mallet"]["radius"]), "pins the bug: a mallet-radius offset whatever the flag says"

    d._vp_aim_latch = True
    centred, _ = d._vp_aim_block(P)
    assert centred == pytest.approx(P), "--defend-block centre plants ON the crossing"

    ok, _, _ = g.plan_strike_vpsto(*_strike_args(st, d), is_defend=True, goal_pos_v=None,
                                   hit_vel_mag=0.05, P_now_v=P, q_seed=q,
                                   fixed_P_v=P, aim_fn=d._vp_aim_block)
    assert ok
    assert g._vp_xT == pytest.approx(P, abs=1e-6), "and that is the point actually planned to"


def test_the_vpsto_block_latches_its_tap_side_within_one_apply():
    """The rendezvous aims once per candidate arrival time; a centreline puck must not flip the
    tap direction between iterations of a single solve, planning two opposite blocks."""
    _need_vpsto()
    env_info = _real_rig_env()
    a, st, d, q = _vpsto_defend(env_info)

    d._vp_aim_latch = True
    _, first = d._vp_aim_block(np.array([0.30, 0.06]))
    _, second = d._vp_aim_block(np.array([0.30, -0.06]))
    assert float(first[1]) == float(second[1]), "the side must hold across one apply's aims"

    d._vp_aim_latch = True                       # a NEW apply may re-decide, as it always could
    _, third = d._vp_aim_block(np.array([0.30, -0.06]))
    assert float(third[1]) == -float(first[1])


def test_a_bezier_fallback_after_a_vpsto_success_still_plans_the_block():
    """4c: after a VPSTO tick the bezier curve is stale and its t_final has run out.

    The fallback only re-aims a curve's TAIL, so with `plan_new_trajectory` left False the first
    fallback after a success re-aimed nothing, produced no trajectory, and the defend was dropped
    -- exactly when the VPSTO path had just failed and the block was needed most.
    """
    _need_vpsto()
    env_info = _real_rig_env()
    a, st, d, q = _vpsto_defend(env_info)

    d.apply()
    assert len(st.trajectory_buffer) > 0, "the VPSTO block must plan, or this proves nothing"
    assert d.plan_new_trajectory, "a VPSTO tick must leave the bezier curve marked stale"

    st.trajectory_buffer = st.trajectory_buffer[:0]
    st.update_prediction(a.agent_params["max_prediction_time"], d._stop_line())
    a.traj_generator.plan_strike_vpsto = lambda *A, **K: (False, None, 0.3)
    before = np.asarray(st.x_cmd[:2], dtype=float).copy()

    d.apply()
    assert len(st.trajectory_buffer) > 0, "pins the bug: the defend is dropped on the fallback"
    x0 = st.update_ee_pos_vel(st.trajectory_buffer[0][0], st.trajectory_buffer[0][1])[0][:2]
    assert np.asarray(x0) == pytest.approx(before, abs=0.002), \
        "the fallback must start from the live commanded state"


def test_a_smash_with_no_vertical_wrist_declines_instead_of_spraying():
    """4d: `q_anchor is None` could never fire, so the documented decline did not exist.

    The bezier SMASH hands back to READY when no vertical-wrist anchor reaches the strike point
    (a tilted mallet sprays, measured as bad as -143 deg off the goal ray). Under
    `--planner vpsto` that shot was planned and taken anyway. A BLOCK still proceeds: a mallet at
    the wrong angle still blocks.
    """
    _need_vpsto()
    from mini_platform.agents.baseline_mini import MiniTrajectoryGenerator, TABLE_LENGTH
    env_info = _real_rig_env()
    a, st, d, q = _vpsto_defend(env_info)
    g = a.traj_generator
    P = np.asarray(st.predicted_state[:2], dtype=float)
    goal = np.array([TABLE_LENGTH / 2 + V_SETBACK, 0.0])

    def no_vertical(self, hit_pos_2d, hit_dir_2d, q_0):
        self.last_anchor_vertical = False        # the flattest fallback anchor, not a vertical one
        return np.asarray(q_0, dtype=float).copy()

    orig = MiniTrajectoryGenerator.solve_anchor_pos_ik_null
    MiniTrajectoryGenerator.solve_anchor_pos_ik_null = no_vertical
    try:
        g.reset_vpsto()
        smash, _, _ = g.plan_strike_vpsto(*_strike_args(st, d), is_defend=False, goal_pos_v=goal,
                                          hit_vel_mag=float(a.agent_params["max_hit_velocity"]),
                                          P_now_v=P, q_seed=q)
        g.reset_vpsto()
        block, _, _ = g.plan_strike_vpsto(*_strike_args(st, d), is_defend=True, goal_pos_v=None,
                                          hit_vel_mag=0.05, P_now_v=P, q_seed=q)
    finally:
        MiniTrajectoryGenerator.solve_anchor_pos_ik_null = orig

    assert not smash, "a smash with no vertical wrist must hand back to the bezier path"
    assert block, "a block must still be planned"


def test_a_far_strike_point_is_clamped_into_the_workspace():
    """4e: the held-target path never consults `intercept_at`, so nothing clamped its xT."""
    _need_vpsto()
    env_info = _real_rig_env()
    a, st, d, q = _vpsto_defend(env_info)
    g = a.traj_generator

    g._vp_rendezvous_solve(st.x_cmd[:2], st.v_cmd[:2], st, st.predicted_time, d._stop_line(),
                           True, None, 0.05, q, fixed_P_v=np.array([0.95, 0.0]))
    assert float(g._vp_xT[0]) <= float(a.agent_params["hit_range"][1]) + 1e-9


def test_the_safety_gate_rejects_a_plan_that_overreaches():
    """4e: the ee_constr box has no FORWARD bound, so an overreaching plan met no gate at all."""
    _need_vpsto()
    from mini_platform.agents import baseline_mini as bm
    env_info = _real_rig_env()
    a, st, d, q = _vpsto_defend(env_info)
    g = a.traj_generator
    eeh = env_info["robot"]["ee_desired_height"]
    hit_ub = float(a.agent_params["hit_range"][1])

    def fk_at(x_v):
        """Every waypoint reads back at this V-x, on the centreline and on the play plane."""
        pos_r = np.asarray(a._v2r.pos2(np.array([x_v, 0.0])), dtype=float)
        return lambda *A, **K: (np.array([pos_r[0], pos_r[1], eeh]), None)

    P = np.asarray(st.predicted_state[:2], dtype=float)
    kw = dict(is_defend=True, goal_pos_v=None, hit_vel_mag=0.05, P_now_v=P, q_seed=q)
    orig_fk = bm.forward_kinematics
    try:
        bm.forward_kinematics = fk_at(hit_ub - 0.05)
        g.reset_vpsto()
        inside, _, _ = g.plan_strike_vpsto(*_strike_args(st, d), **kw)
        bm.forward_kinematics = fk_at(hit_ub + 0.05)
        g.reset_vpsto()
        outside, _, _ = g.plan_strike_vpsto(*_strike_args(st, d), **kw)
    finally:
        bm.forward_kinematics = orig_fk

    assert inside, "a plan inside the reach limit must still pass the gate"
    assert not outside, "a plan past hit_range[1] must fall back to bezier"


def test_the_anchor_is_reused_while_the_strike_point_barely_moves():
    """4g: the anchor IK search is the expensive half of a re-solve (up to ~35 ms of probing)."""
    _need_vpsto()
    env_info = _real_rig_env()
    a, st, d, q = _vpsto_defend(env_info)
    g = a.traj_generator
    P = np.asarray(st.predicted_state[:2], dtype=float)
    solves = []
    orig = g.solve_anchor_pos_ik_null

    def spy(hit_pos_2d, hit_dir_2d, q_0):
        solves.append(np.asarray(hit_pos_2d, dtype=float).copy())
        return orig(hit_pos_2d, hit_dir_2d, q_0)

    g.solve_anchor_pos_ik_null = spy
    common = dict(is_defend=True, goal_pos_v=None, hit_vel_mag=0.05, q_seed=q)
    g.plan_strike_vpsto(*_strike_args(st, d), P_now_v=P, fixed_P_v=P, **common)
    assert len(solves) == 1, "the first strike has nothing to reuse"

    # Drop the cached PATH without dropping the cached ANCHOR: that is the 4g case -- the path has
    # run out (or the tactic changed) while the strike point has barely moved. Going through
    # reset_vpsto() instead would clear the anchor too, which is what a genuinely new strike
    # wants; and leaving the path in place means no re-solve happens at all (drift < VP_DRIFT),
    # so the test would pass without ever reaching the code it is about.
    near = P + np.array([0.002, 0.0])            # 2 mm: inside VP_ANCHOR_REUSE
    g._vp_sol = None
    g.plan_strike_vpsto(*_strike_args(st, d), P_now_v=near, fixed_P_v=near, **common)
    assert len(solves) == 1, "a 2 mm move must not re-run the anchor IK"

    far = P + np.array([0.08, 0.0])              # 80 mm: a different reach entirely
    g._vp_sol = None
    g.plan_strike_vpsto(*_strike_args(st, d), P_now_v=far, fixed_P_v=far, **common)
    assert len(solves) == 2, "a real move must re-anchor"
