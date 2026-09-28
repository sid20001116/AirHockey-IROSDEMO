"""The clearing block: DEFEND returns an approaching puck instead of stopping it.

Four mechanisms stack to make a head-on puck come back slower than an angled one, and only the
last is in `planning/`:

1. `_block_target` aims `[0, +-1] * 0.05` -- a sideways tap with NO up-table component -- so
   DEFEND is structurally a block, never a return.
2. `can_smash` refuses any approaching puck, and `Repel` is a stub, so nothing else can take it.
3. The approach latch never clears for a head-on puck: once it recedes it matches neither reset
   arm (never "transversal" at |vx|/speed ~ 1, never "static" above the threshold, and vx > 0 so
   it never increments), so `can_smash` stays False for the rest of the run.
4. The Bezier collapses when travel is ~0, which is exactly the head-on case: measured 0.050 m/s
   of path speed with zero travel against 0.703 m/s over 0.149 m.

`--defend-clear` fixes 1 by standing the mallet off down-table and driving it THROUGH the
crossing along +x, and 3 outright (a successful clear MANUFACTURES the jamming state). 2 is
deliberately left: the clearing block IS the repel.

**A fifth cause, found on the rig (2026-09-22) and fixed by `--defend-early-frac`:** the plant
line the crossing is solved at (V-x 0.2506) sits 56 mm BEHIND the taught park (0.3064), so
`d_avail` is negative on a normal defend and the stroke is impossible by construction -- measured
p50 -0.010 m, only 31.8% of DEFEND ticks with any forward room, and the phase was PLANT at every
actual contact. Meeting a near-head-on puck further up-table fixes both halves: it puts the
crossing ahead of the mallet AND leaves time to get there. `--defend-clear-max-path` covers the
other half of that run, where the commit asked 3.58 m/s and the arm delivered 0.03.
"""

import argparse
import importlib

import numpy as np
import pytest

import mini_platform
from mini_platform.agents.baseline_mini import MiniBaselineAgent, MiniDefend, V_SETBACK
from mini_platform.envs.env_base import INIT_SEED
from mini_platform.fsm.system_state import TACTICS

#: The goalie harness, rather than a second copy of it: `_drive` runs the real agent loop
#: against a scripted glide and is the only thing here that measures end-to-end behaviour.
from tests.test_goalie_defence import GOALIE, _drive, _home_q


@pytest.fixture(scope="module")
def ei():
    return mini_platform.make("mini-hit-sdk").env_info


def _defender(env_info, x_cmd=(0.20, 0.0), predicted_time=0.5, **kw):
    """A DEFEND tactic with the mallet parked at `x_cmd` and `predicted_time` left to plan in."""
    a = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.02, **kw)
    a.reset()
    d = a.tactics_processor[TACTICS.DEFEND.value]
    d.state.x_cmd = np.array([x_cmd[0], x_cmd[1],
                              env_info["robot"]["ee_desired_height"]], dtype=float)
    d.state.predicted_time = float(predicted_time)
    d._clear_phase = None
    return a, d


def _floor(d):
    """The Bezier ROUTING floor -- already the max of the rim bound and the own-goal keep-out."""
    return float(d.generator.bound_points[:, 0].min())


# --------------------------------------------------------------------------------- off state

def test_defaults_leave_the_clearing_block_off(ei):
    """The contract: an agent built with no new arguments is the vendored one."""
    p = MiniBaselineAgent(ei, agent_id=1).agent_params
    assert p["defend_clear"] is False
    assert p["defend_clear_speed"] == 0.25
    assert p["defend_clear_standoff"] == 0.05
    assert p["defend_clear_margin"] == 0.02


@pytest.mark.parametrize("mode", ["mallet", "centre", "contact"])
def test_block_target_is_untouched_while_the_clearing_block_is_off(ei, mode):
    off = MiniBaselineAgent(ei, agent_id=1, defend_block=mode)
    d = off.tactics_processor[TACTICS.DEFEND.value]
    puck = np.array([0.30, 0.08])
    pos, direction, vel = d._block_target(puck)
    assert direction == pytest.approx(np.array([0., 1.]))
    assert vel == pytest.approx(direction * 0.05)
    r_m, r_p = ei["mallet"]["radius"], ei["puck"]["radius"]
    k = {"mallet": r_m, "centre": 0.0, "contact": r_m + r_p}[mode]
    assert pos == pytest.approx(puck - direction * k)


def test_the_goalie_switch_does_not_turn_the_clearing_block_on():
    """It moves the mallet TOWARD our own goal to cock, so it is opted into deliberately."""
    run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")
    for name in ("defend_clear", "defend_clear_speed", "defend_clear_standoff",
                 "defend_clear_margin"):
        assert name not in run_hw_match.GOALIE_DEFAULTS
    args = argparse.Namespace(goalie=True, **{k: None for k in run_hw_match.GOALIE_DEFAULTS})
    args.ekf_own_goal = False
    args.defend_clear = False
    run_hw_match.apply_goalie_defaults(args)
    assert args.defend_clear is False


# --------------------------------------------------------------------------------- geometry

def test_strike_plants_on_the_crossing_and_drives_up_table(ei):
    """The whole point: terminal velocity along +x, at the crossing itself."""
    a, d = _defender(ei, x_cmd=(0.20, 0.0), predicted_time=0.5, defend_clear=True)
    P = np.array([0.30, 0.02])                       # d_avail 0.10 -> v_need 0.40 >= 0.25
    pos, direction, vel = d._block_target(P)
    assert d._clear_phase == "strike"
    assert pos == pytest.approx(P), "the stroke lands ON the crossing -- no overshoot"
    assert direction == pytest.approx(np.array([1., 0.]))
    assert vel[0] == pytest.approx(0.25) and vel[1] == pytest.approx(0.0)


def test_cock_stands_off_down_table_on_the_crossings_own_line(ei):
    a, d = _defender(ei, x_cmd=(0.28, 0.0), predicted_time=0.5, defend_clear=True)
    P = np.array([0.30, 0.02])                       # d_avail 0.02 -> v_need 0.08 < 0.25
    pos, direction, vel = d._block_target(P)
    assert d._clear_phase == "cock"
    assert pos == pytest.approx(np.array([0.30 - 0.05, 0.02]))
    assert pos[1] == pytest.approx(P[1]), "the cock stays in the puck's path, just deeper"
    assert direction == pytest.approx(np.array([1., 0.]))
    assert vel == pytest.approx(np.zeros(2)), "arrive at rest, then stroke"


def test_the_standoff_is_bounded_by_the_routing_floor(ei):
    """The cock may never route behind the floor, whatever the standoff asks for."""
    a, d = _defender(ei, x_cmd=(0.28, 0.0), predicted_time=0.5,
                     defend_clear=True, defend_clear_standoff=0.40)
    P = np.array([0.30, 0.0])
    pos, _, _ = d._block_target(P)
    assert pos[0] == pytest.approx(_floor(d) + 0.02)
    assert pos[0] > _floor(d)


def test_the_floor_tracks_min_goal_dist(ei):
    """Raising the own-goal keep-out moves the floor, and the standoff with it."""
    deep = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, defend_clear=True)
    high = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.15, defend_clear=True)
    f_deep = float(deep.traj_generator.bound_points[:, 0].min())
    f_high = float(high.traj_generator.bound_points[:, 0].min())
    assert f_high > f_deep


def test_no_room_to_cock_falls_back_to_the_vendored_plant(ei):
    """Rung 1 of the ladder: a crossing with no space behind it is today's block, not a whiff."""
    a, d = _defender(ei, x_cmd=(0.28, 0.0), predicted_time=0.5, defend_clear=True,
                     defend_clear_margin=0.20, defend_block="centre")
    P = np.array([0.30, 0.02])
    pos, direction, vel = d._block_target(P)
    assert direction == pytest.approx(np.array([0., 1.])), "vendored lateral tap"
    assert vel == pytest.approx(direction * 0.05)
    assert pos == pytest.approx(P)


def test_a_mallet_already_up_table_never_strokes_backward(ei):
    """Rung 3: negative run-up would drive the mallet at our own goal. Plant instead."""
    a, d = _defender(ei, x_cmd=(0.35, 0.0), predicted_time=0.5, defend_clear=True)
    _, direction, vel = d._block_target(np.array([0.30, 0.02]))
    assert direction == pytest.approx(np.array([0., 1.]))
    assert vel == pytest.approx(direction * 0.05)


def test_no_run_up_falls_back_to_the_plant(ei):
    """Rung 2, the honest limit: a shot arriving with no room to stroke gets today's block."""
    a, d = _defender(ei, x_cmd=(0.2995, 0.0), predicted_time=0.5, defend_clear=True)
    _, direction, vel = d._block_target(np.array([0.30, 0.02]))
    assert direction == pytest.approx(np.array([0., 1.]))
    assert vel == pytest.approx(direction * 0.05)


# ------------------------------------------------------------------------------------- phase

def test_the_phase_is_latched_within_one_apply(ei):
    """The rendezvous aims more than once; both calls must plan the SAME block."""
    a, d = _defender(ei, x_cmd=(0.28, 0.0), predicted_time=0.5, defend_clear=True)
    P = np.array([0.30, 0.02])
    d._block_target(P, latch=True)
    first = d._clear_phase
    d.state.predicted_time = 0.05                    # would re-decide as 'strike' if re-latched
    d._block_target(P, latch=False)
    assert d._clear_phase == first


def test_a_new_defend_re_decides_the_phase(ei):
    a, d = _defender(ei, x_cmd=(0.28, 0.0), predicted_time=0.5, defend_clear=True)
    d._block_target(np.array([0.30, 0.02]))
    assert d._clear_phase is not None
    d.state.is_new_tactic = True
    d.ready()
    assert d._clear_phase is None


def test_the_cock_to_strike_flip_starts_a_fresh_curve(ei):
    """The live curve was shaped for a DOWN-table target; re-aiming only its tail strokes wrong."""
    z = ei["robot"]["ee_desired_height"]
    a, d = _defender(ei, x_cmd=(0.28, 0.0), predicted_time=0.5, defend_clear=True)
    d.state.q_cmd = _home_q(ei, a)
    d.state.dq_cmd = np.zeros(len(INIT_SEED))
    d.state.is_new_tactic = True
    d.ready()                                        # gives the tactic its anchor
    d.state.predicted_state = np.array([0.30, 0.02, -0.8, 0.0, 0., 0.])
    d.state.x_cmd = np.array([0.28, 0.0, z])
    d.state.predicted_time = 0.5
    d._block_target(np.array([0.30, 0.02]))
    assert d._clear_phase == "cock"

    # Now the mallet has run-up, so the next apply must commit -- with a FRESH curve.
    d.plan_new_trajectory = False
    d.replan_time = 0.06
    d.state.x_cmd = np.array([0.20, 0.0, z])
    fresh, tail = [], []
    bp = d.generator.bezier_planner
    o_cc, o_ub = bp.compute_control_point, bp.update_bezier_curve

    def cc(*args, **kw):
        fresh.append(1)
        return o_cc(*args, **kw)

    def ub(*args, **kw):
        tail.append(1)
        return o_ub(*args, **kw)

    bp.compute_control_point, bp.update_bezier_curve = cc, ub
    try:
        d.apply()
    finally:
        bp.compute_control_point, bp.update_bezier_curve = o_cc, o_ub
    assert d._clear_phase == "strike"
    assert fresh and not tail, "the flip must start a fresh curve, not re-aim a stale tail"


def test_the_vpsto_aim_uses_the_clearing_geometry(ei):
    """Both planners read one function, or the gate and the executed block diverge."""
    a, d = _defender(ei, x_cmd=(0.20, 0.0), predicted_time=0.5, defend_clear=True)
    d._vp_aim_latch = True
    P = np.array([0.30, 0.02])
    pos, direction = d._vp_aim_block(P)
    assert direction == pytest.approx(np.array([1., 0.]))
    assert pos == pytest.approx(P)


# ------------------------------------------------------------------- the own-goal floor clamp

def test_the_clearing_floor_clamps_waypoints_the_bezier_would_emit(ei):
    """The Bezier `boundary` shapes control points, it never clamps the CURVE."""
    tg = importlib.import_module("mini_platform.planning.trajectory_generator")
    a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, defend_clear=True)
    gen = a.traj_generator
    floor = float(gen.bound_points[:, 0].min())
    seen = {}
    orig = tg.TrajectoryGenerator.optimize_trajectory

    def spy(self, cart_traj, *args):
        seen["traj"] = np.asarray(cart_traj).copy()
        return True, []

    traj = np.zeros((3, 9))
    traj[:, 0] = [floor - 0.04, floor + 0.10, floor - 0.01]      # dips behind the floor
    traj[:, 3] = [-0.5, 0.20, -0.10]                             # and drives further in
    tg.TrajectoryGenerator.optimize_trajectory = spy
    try:
        gen.optimize_trajectory(traj, INIT_SEED.copy(), np.zeros(len(INIT_SEED)),
                                INIT_SEED.copy())
    finally:
        tg.TrajectoryGenerator.optimize_trajectory = orig

    back = np.array([gen._r2v.pos2(row[0:2]) for row in seen["traj"]])
    assert (back[:, 0] >= floor - 1e-6).all(), "no waypoint may sit behind the floor"
    vel_v = np.array([gen._r2v.vec2(row[3:5]) for row in seen["traj"]])
    assert vel_v[0, 0] >= -1e-9 and vel_v[2, 0] >= -1e-9, "inward velocity is stripped"


def test_the_floor_clamp_is_off_unless_the_clearing_block_is_on(ei):
    tg = importlib.import_module("mini_platform.planning.trajectory_generator")
    a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02)
    gen = a.traj_generator
    floor = float(gen.bound_points[:, 0].min())
    seen = {}
    orig = tg.TrajectoryGenerator.optimize_trajectory

    def spy(self, cart_traj, *args):
        seen["traj"] = np.asarray(cart_traj).copy()
        return True, []

    traj = np.zeros((2, 9))
    traj[:, 0] = [floor - 0.04, floor + 0.10]
    tg.TrajectoryGenerator.optimize_trajectory = spy
    try:
        gen.optimize_trajectory(traj, INIT_SEED.copy(), np.zeros(len(INIT_SEED)),
                                INIT_SEED.copy())
    finally:
        tg.TrajectoryGenerator.optimize_trajectory = orig
    back = np.array([gen._r2v.pos2(row[0:2]) for row in seen["traj"]])
    assert back[0, 0] < floor, "vendored behaviour is untouched with the flag off"


# ------------------------------------------------------------------------------- integration

def _block_commands(env_info, puck0, n_ticks=45, **kw):
    """Every (position, direction, velocity) DEFEND actually commanded, in order."""
    seen = []
    orig = MiniDefend._block_target

    def spy(self, puck_pos_2d, latch=True):
        out = orig(self, puck_pos_2d, latch=latch)
        seen.append(tuple(np.asarray(v, dtype=float).copy() for v in out))
        return out

    MiniDefend._block_target = spy
    try:
        a, rows = _drive(env_info, puck0, n_ticks=n_ticks, **kw)
    finally:
        MiniDefend._block_target = orig
    return a, rows, seen


#: A shot straight down the middle at the park's own line -- the case that never comes back.
_HEAD_ON = [0.70, 0.0, -1.0, 0.0]


def test_a_head_on_block_commands_no_up_table_speed_today(ei):
    """Pin the bug: with the flag off, nothing DEFEND commands pushes the puck back."""
    _, _, cmds = _block_commands(ei, _HEAD_ON, **GOALIE)
    assert cmds, "the harness must plan some blocks, or it measures nothing"
    assert max(abs(float(v[0])) for _, _, v in cmds) <= 0.01


def test_a_head_on_block_is_returned_up_table_with_the_clearing_block(ei):
    """The headline: the same shot now gets a stroke along +x at the commanded speed."""
    _, _, cmds = _block_commands(ei, _HEAD_ON, **{**GOALIE, "defend_clear": True})
    assert cmds, "the harness must plan some blocks, or it measures nothing"
    best = max(float(v[0]) for _, _, v in cmds)
    assert best >= 0.8 * 0.25, f"best up-table command was only {best:.3f} m/s"


def test_the_clearing_block_never_plans_behind_the_floor(ei):
    a, _, cmds = _block_commands(ei, _HEAD_ON, **{**GOALIE, "defend_clear": True})
    floor = float(a.traj_generator.bound_points[:, 0].min())
    assert cmds
    assert min(float(p[0]) for p, _, _ in cmds) >= floor - 1e-6


# ------------------------------------------------------------------------- the approach latch

def test_a_receding_head_on_puck_clears_the_approach_latch(ei):
    """Root cause 3. A clear MANUFACTURES this state, so the fix ships with the feature.

    Without the receding arm the counter matches neither reset (|vx|/speed ~ 1 is never
    'transversal', speed > static_vel_threshold is never 'static') and never increments either
    (vx > 0), so it keeps its last positive value and `can_smash` stays False for the whole run.
    """
    a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, defend_abort_dist=0.02)
    a.reset()
    st = a.state
    q, dq = INIT_SEED.copy(), np.zeros(len(INIT_SEED))
    for x in (0.44, 0.40, 0.36, 0.32, 0.28):                 # closing, dead head-on
        st.update_observation(q, dq, np.array([x, 0.0, 0.]))
    assert st.is_puck_approaching(), "must latch first, or the test measures nothing"
    for x in (0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60):   # returned, still head-on
        st.update_observation(q, dq, np.array([x, 0.0, 0.]))
    assert not st.is_puck_approaching()


def test_a_puck_still_closing_below_the_abort_line_keeps_its_latch(ei):
    """Refusing that one is `_in_zone`'s job, not this counter's.

    `ekf_own_goal` matters here: with our own goal CLOSED in the bounce model the filter turns a
    shot on target around at the goal line, and a puck the model believes has rebounded is one
    the new receding arm is right to un-latch. Opening the mouth is what --goalie does.
    """
    a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, defend_abort_dist=0.02,
                          ekf_own_goal=True)
    a.reset()
    st = a.state
    q, dq = INIT_SEED.copy(), np.zeros(len(INIT_SEED))
    for x in (0.30, 0.26, 0.22, 0.18, 0.14, 0.12):
        st.update_observation(q, dq, np.array([x, 0.0, 0.]))
    assert st.is_puck_approaching()


# -------------------------------------------------------------------------- folded-in defects

def test_a_zero_length_guard_curve_does_not_crash(ei):
    """`replan_time` is 0 on every fresh plan, so `t_final >= 0` accepted an empty sample grid."""
    a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02, ready_guard=True)
    a.reset()
    r = a.tactics_processor[TACTICS.READY.value]
    r.state.q_cmd = _home_q(ei, a)
    r.state.dq_cmd = np.zeros(len(INIT_SEED))
    r.state.x_cmd, r.state.v_cmd = r.state.update_ee_pos_vel(r.state.q_cmd, r.state.dq_cmd)
    r._guard_xy = np.asarray(r.state.x_cmd[:2], dtype=float)
    r.t_stop = 0.0
    r.plan_new_trajectory = True
    r.replan_time = 0
    r.apply()                                     # must not raise IndexError
    assert r.state.tactic_finish is True


def test_the_smash_retry_loop_backs_off_like_the_vendored_one(ei):
    """It had kept only the decay, so all ten retries re-planned the identical failing curve."""
    a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02)
    a.reset()
    s = a.tactics_processor[TACTICS.SMASH.value]
    s.state.q_cmd = _home_q(ei, a)
    s.state.dq_cmd = np.zeros(len(INIT_SEED))
    s.state.x_cmd, s.state.v_cmd = s.state.update_ee_pos_vel(s.state.q_cmd, s.state.dq_cmd)
    s.state.predicted_state = np.array([0.40, 0.05, 0., 0., 0., 0.])
    s.state.is_new_tactic = True
    s.ready()

    vels, times = [], []
    orig_cc = s.generator.bezier_planner.compute_control_point

    def spy_cc(p0, v0, p1, v1, t):
        vels.append(float(np.linalg.norm(v1)))
        times.append(float(t))
        return orig_cc(p0, v0, p1, v1, t)

    s.generator.bezier_planner.compute_control_point = spy_cc
    s.generator.solve_anchor_pos_ik_null = lambda p, d, q: q
    s.generator.last_anchor_vertical = True
    s.generator.optimize_trajectory = lambda *args, **kw: (False, [])
    s.apply()

    assert len(vels) >= 2, "the loop must actually retry"
    assert vels[1] < vels[0], "the commanded hit velocity must follow the decayed magnitude"
    assert times[1] > times[0], "and the plan must be given more time"
    assert s.state.tactic_finish is True, "ten failures must hand back a give-up signal"


# ------------------------------------------------------------------- the early defend line

def _early(env_info, puck, x_cmd=(0.3064, 0.0), **kw):
    """A DEFEND tactic with a scripted puck estimate and the mallet at the taught park.

    `static_vel_threshold` is forced to the RIG's 0.02: the sim default is 0.4*TABLE_SCALE =
    0.202, which would make every test puck under 0.2 m/s "static" and the head-on test dead.
    """
    a = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.02, **kw)
    a.reset()
    a.agent_params['static_vel_threshold'] = 0.02
    d = a.tactics_processor[TACTICS.DEFEND.value]
    d.state.estimated_state = np.array([puck[0], puck[1], puck[2], puck[3], 0., 0.])
    d.state.x_cmd = np.array([x_cmd[0], x_cmd[1], env_info["robot"]["ee_desired_height"]])
    return a, d


def test_the_early_line_is_off_by_default(ei):
    p = MiniBaselineAgent(ei, agent_id=1).agent_params
    assert p["defend_early_frac"] is None
    assert p["defend_early_cos"] == 0.90
    assert p["defend_clear_max_path"] == 0.25
    a, d = _early(ei, [0.75, 0.0, -0.5, 0.0])
    assert d._early_line() is None
    assert d._stop_line() == pytest.approx(d.agent_params["defend_range"][0])


def test_a_head_on_puck_is_met_up_table(ei):
    """The whole point: the crossing moves AHEAD of the park, so run-up exists at all."""
    a, d = _early(ei, [0.75, 0.0, -0.5, 0.0], defend_early_frac=0.6, rendezvous_vel=0.25)
    line = d._early_line()
    gl = V_SETBACK - ei["table"]["length"] / 2
    assert line == pytest.approx(gl + 0.6 * ei["table"]["length"] / 2)
    assert line > float(d.agent_params["x_home"][0]), "must be up-table of the park, or it is moot"
    assert d._stop_line() == pytest.approx(line)


def test_an_angled_puck_keeps_the_vendored_line(ei):
    a, d = _early(ei, [0.75, 0.0, -0.3, -0.5], defend_early_frac=0.6, rendezvous_vel=0.25)
    assert abs(-0.3) / np.hypot(0.3, 0.5) < 0.90, "the fixture must actually be angled"
    assert d._early_line() is None
    assert d._stop_line() == pytest.approx(d.agent_params["defend_range"][0])


def test_the_early_line_is_not_chased_when_it_cannot_be_reached(ei):
    """Without this it is the 'arm swings forward at a phantom' regression, back again."""
    a, d = _early(ei, [0.75, 0.0, -2.5, 0.0], defend_early_frac=0.6, rendezvous_vel=0.25)
    assert d._early_line() is None
    a2, d2 = _early(ei, [0.75, 0.0, -0.5, 0.0], defend_early_frac=0.6, rendezvous_vel=0.25)
    assert d2._early_line() is not None, "the slow twin must still qualify"


def test_the_early_line_is_capped_at_the_forward_reach(ei):
    a, d = _early(ei, [1.05, 0.0, -0.5, 0.0], defend_early_frac=2.0, rendezvous_vel=0.25)
    assert d._early_line() == pytest.approx(float(d.agent_params["hit_range"][1]))


def test_the_early_line_widens_the_defend_zone(ei):
    """A crossing up there has to pass the gate too, or the tactic refuses its own target."""
    ps = np.array([0.41, 0.0, -0.5, 0.0, 0., 0.])
    off_a, off_d = _early(ei, [0.75, 0.0, -0.5, 0.0], defend_abort_dist=0.02)
    assert not off_d._in_zone(ps), "0.41 is outside the vendored zone (dr[1] = 0.355)"
    on_a, on_d = _early(ei, [0.75, 0.0, -0.5, 0.0], defend_abort_dist=0.02,
                        defend_early_frac=0.6, rendezvous_vel=0.25)
    assert on_d._zone_hi() > float(on_d.agent_params["defend_range"][1])
    assert on_d._in_zone(ps)


def test_the_early_line_restores_run_up_for_the_clearing_block(ei):
    """The measured failure: plant line 0.251 sits BEHIND the park 0.306, so d_avail < 0."""
    park = 0.3064
    plant = float(MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.02)
                  .agent_params["defend_range"][0])
    assert plant < park, "this is the rig geometry the early line exists to fix"
    a, d = _early(ei, [0.75, 0.0, -0.5, 0.0], x_cmd=(park, 0.0), defend_clear=True,
                  defend_early_frac=0.6, rendezvous_vel=0.25)
    d.state.predicted_time = 1.2
    line = d._early_line()
    _, direction, vel = d._block_target(np.array([line, 0.0]))
    assert d._clear_phase in ("cock", "strike")
    assert direction == pytest.approx(np.array([1., 0.])), "a real up-table stroke, not a tap"
    # and the vendored crossing would have produced nothing but a plant
    _, dir_plant, vel_plant = d._block_target(np.array([plant, 0.0]))
    assert dir_plant == pytest.approx(np.array([0., 1.]))
    assert vel_plant == pytest.approx(dir_plant * 0.05)


# ------------------------------------------------------------------- stroke feasibility

def test_a_stroke_the_arm_cannot_walk_in_time_plants_instead(ei):
    """Pin the rig failure: it asked 3.58 m/s of mallet speed and delivered 0.03."""
    a, d = _defender(ei, x_cmd=(0.20, 0.0), predicted_time=0.20, defend_clear=True)
    _, direction, vel = d._block_target(np.array([0.30, 0.02]))   # 0.10 m in 0.20 s = 0.50 m/s
    assert direction == pytest.approx(np.array([0., 1.])), "must fall back to the vendored plant"
    assert vel == pytest.approx(direction * 0.05)


def test_no_time_left_plants_rather_than_swinging_wildly(ei):
    """The old `T <= 2*max_plan_steps*dt` escape hatch FORCED the least feasible case."""
    a, d = _defender(ei, x_cmd=(0.20, 0.0), predicted_time=0.06, defend_clear=True)
    _, direction, vel = d._block_target(np.array([0.30, 0.02]))
    assert direction == pytest.approx(np.array([0., 1.]))
    assert vel == pytest.approx(direction * 0.05)


def test_the_feasibility_budget_can_be_switched_off(ei):
    a, d = _defender(ei, x_cmd=(0.20, 0.0), predicted_time=0.20, defend_clear=True,
                     defend_clear_max_path=0.0)
    _, direction, _ = d._block_target(np.array([0.30, 0.02]))
    assert direction == pytest.approx(np.array([1., 0.]))
