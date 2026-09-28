"""Own-goal keep-out: the side-mounted arm must not reach back over its base into the near
rim (which tips the table). `min_goal_dist` floors the planner's back-reach; the hardware
watchdog (`mallet_x_v` vs the hard floor) catches the arm if the measured mallet crosses it.

These pin that the floor moves every back-reach knob, that the default (None) is a no-op, and
that the V-frame mallet position used by the watchdog matches the agent's own home target.
"""
import ast
import importlib
import pathlib

import numpy as np
import pytest

import mini_platform
from mini_platform.agents.baseline_mini import MiniBaselineAgent, V_SETBACK

run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")


def _env_info(task="mini-hit-sdk"):
    return mini_platform.make(task).env_info


def _goal_line_v(env_info):
    return V_SETBACK - env_info["table"]["length"] / 2


def test_min_goal_dist_none_is_unchanged():
    """No keep-out -> the vendored back-reach values, untouched (sim behaviour preserved)."""
    ei = _env_info()
    half_l = ei["table"]["length"] / 2
    r_m = ei["mallet"]["radius"]
    a = MiniBaselineAgent(ei, agent_id=1)                    # min_goal_dist=None
    gl = _goal_line_v(ei)
    assert a.agent_params["defend_range"][0] == pytest.approx(gl + 0.271 * half_l)
    assert a.agent_params["hit_range"][0] == pytest.approx(gl + 0.271 * half_l)
    near_x = a.traj_generator.bound_points[:, 0].min()
    assert near_x == pytest.approx(-(half_l - r_m) + V_SETBACK)


def test_min_goal_dist_floors_every_back_reach_knob():
    """A keep-out floors the Bezier near bound, the ranges and the home/init pose to >= floor."""
    ei = _env_info()
    d = 0.15                                                 # well in front of the defaults
    floor = _goal_line_v(ei) + d
    a = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=d)

    assert a.traj_generator.bound_points[:, 0].min() == pytest.approx(floor)
    assert a.agent_params["defend_range"][0] == pytest.approx(floor)
    assert a.agent_params["hit_range"][0] == pytest.approx(floor)
    assert a.agent_params["prepare_range"][0] == pytest.approx(floor)
    assert a.agent_params["x_home"][0] >= floor - 1e-9
    assert a.agent_params["x_init"][0] >= floor - 1e-9
    # the far bound and the range tops are NOT pulled in.
    assert a.traj_generator.bound_points[:, 0].max() > floor + 0.2
    assert a.agent_params["hit_range"][1] > floor + 0.2


def test_small_keepout_only_clamps_the_bezier_routing():
    """The reported failure mode: the Bezier near bound routed the mallet ~2.5 cm from the near
    wall while defend targets stayed further out. A keep-out at ~home clamps ONLY that routing,
    leaving the defend range (already further from the goal) alone."""
    ei = _env_info()
    a0 = MiniBaselineAgent(ei, agent_id=1)
    home_v_x = float(a0.agent_params["x_home"][0])
    d = home_v_x - _goal_line_v(ei)                          # keep-out exactly at the home spot

    ei2 = _env_info()
    a = MiniBaselineAgent(ei2, agent_id=1, min_goal_dist=d)
    # routing floor rises to home ...
    assert a.traj_generator.bound_points[:, 0].min() == pytest.approx(home_v_x)
    # ... but defend range (further from the goal than home) is unchanged.
    assert a.agent_params["defend_range"][0] == pytest.approx(a0.agent_params["defend_range"][0])


def test_mallet_x_v_matches_home_target():
    """The watchdog's V-frame mallet x (FK of the measured q, mapped R->V) matches the agent's
    own home V-x -- so comparing it to the keep-out floor is apples-to-apples."""
    ei = _env_info()
    r2v = run_hw_match.r2v_xform(ei)
    q_home = run_hw_match.home_config(ei)
    xv = run_hw_match.mallet_x_v(ei, r2v, q_home)

    a = MiniBaselineAgent(ei, agent_id=1)
    assert xv == pytest.approx(a.agent_params["x_home"][0], abs=2e-3)
    # and it is in FRONT of the own goal line (positive distance).
    assert xv > _goal_line_v(ei)


def test_shift_base_origin_maps_touched_point_to_ref():
    """shift_base_origin translates only xy so a touched point lands on the known reference,
    leaving rotation/tilt/z untouched. Any other point shifts by the same delta."""
    T = np.eye(4)
    T[:3, :3] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])   # +90 deg yaw
    T[:3, 3] = [-0.04, 0.38, -0.06]
    p_robot = np.array([0.3, 0.1, 0.27])
    world_now = (T[:3, :3] @ p_robot + T[:3, 3])[:2]
    ref = np.array([-0.508, 0.0])

    T2 = run_hw_match.shift_base_origin(T, world_now, ref)
    # the touched point now maps to ref
    assert np.allclose((T2[:3, :3] @ p_robot + T2[:3, 3])[:2], ref, atol=1e-9)
    # rotation and z untouched; xy shifted by exactly (ref - world_now)
    assert np.allclose(T2[:3, :3], T[:3, :3])
    assert T2[2, 3] == pytest.approx(T[2, 3])
    assert np.allclose(T2[:2, 3] - T[:2, 3], ref - world_now)


def test_origin_fix_zeroes_dist_at_the_reference():
    """After correcting the origin so a config maps to the own-goal-line centre, the reach-probe
    dist at that config is ~0 (the 357 mm-style offset is removed)."""
    ei = _env_info()
    half_l = ei["table"]["length"] / 2
    ref = np.array([-half_l, 0.0])
    T = ei["robot"]["base_frame"][0].astype(float).copy()
    # a config whose mallet is somewhere reachable; pretend the arm was touched to the goal here
    q = run_hw_match.home_config(ei)
    pos_r, _ = run_hw_match.forward_kinematics(ei["robot"]["robot_model"],
                                               ei["robot"]["robot_data"], q, link="ee")
    world_now = (T[:3, :3] @ pos_r + T[:3, 3])[:2]
    T_fixed = run_hw_match.shift_base_origin(T, world_now, ref)

    ei["robot"]["base_frame"][0] = T_fixed
    r2v = run_hw_match.r2v_xform(ei)
    dist = run_hw_match.mallet_x_v(ei, r2v, q) - _goal_line_v(ei)
    assert dist == pytest.approx(0.0, abs=1e-3)


def test_min_goal_dist_units_guard():
    """The metres/mm slip (e.g. --min-goal-dist 30 meaning 30 mm) is caught, not silently run."""
    half_l = 0.508
    assert run_hw_match.min_goal_dist_error(0.03, half_l) is None      # 30 mm, fine
    assert run_hw_match.min_goal_dist_error(0.06, half_l) is None      # default
    assert run_hw_match.min_goal_dist_error(0.0, half_l) is None
    # 30 (metres) -- way past the table: rejected with a mm hint
    err = run_hw_match.min_goal_dist_error(30.0, half_l)
    assert err is not None and "0.030" in err and "mm" in err
    # negative -> rejected
    assert run_hw_match.min_goal_dist_error(-0.01, half_l) is not None


def test_keepout_watchdog_condition():
    """A config at the home spot is safe; a floor set in front of it would trip the watchdog."""
    ei = _env_info()
    r2v = run_hw_match.r2v_xform(ei)
    xv_home = run_hw_match.mallet_x_v(ei, r2v, run_hw_match.home_config(ei))
    # hard floor BEHIND home -> safe (no trip)
    assert not (xv_home < xv_home - 0.02)
    # hard floor IN FRONT of home -> would trip (mallet is behind the floor toward the goal)
    assert xv_home < xv_home + 0.02


# --------------------------------------------------------------------------- where it PARKS
# `--home goal` moves the Ready/Init parking spot from the arm's own station to the centreline
# in front of the own goal. It is the one knob that puts the mallet where a centred shot has to
# be blocked, so what has to hold is: the default is untouched, the goal park is really on the
# centreline, and it can never be pushed behind the keep-out the rest of this file is about.
def _home_of(**kw):
    return MiniBaselineAgent(_env_info(), agent_id=1, **kw).agent_params["x_home"]


def test_home_base_is_the_unchanged_default():
    """The default park is HOME_PARK_XY (robot frame) mapped into V, floored, on the plane."""
    from mini_platform.agents.baseline_mini import HOME_POS, HOME_PARK_XY

    ei = _env_info()
    agent = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.06)
    ref = agent._r2v.pos3([HOME_PARK_XY[0], HOME_PARK_XY[1], HOME_POS[2]])
    want = np.array([max(ref[0], _goal_line_v(ei) + 0.06), ref[1],
                     ei["robot"]["ee_desired_height"]])
    assert np.array_equal(agent.agent_params["x_home"], want)
    assert agent.home_mode == "base"


def test_taught_park_moves_the_default_and_nothing_else():
    """Re-teaching HOME_PARK_XY moves where the mallet idles and leaves the geometry alone: the
    goal park, the strike ranges and the keep-out floor all stay where they were."""
    import mini_platform.agents.baseline_mini as bm

    ei = _env_info()
    before = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.06)
    taught = np.array([bm.HOME_PARK_XY[0] + 0.03, bm.HOME_PARK_XY[1] - 0.02])
    saved, bm.HOME_PARK_XY = bm.HOME_PARK_XY, taught
    try:
        after = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.06)
    finally:
        bm.HOME_PARK_XY = saved

    # the park followed the taught point, in the frame it was taught in
    assert after._v2r.pos2(after.agent_params["x_home"][:2]) == pytest.approx(taught, abs=1e-9)
    assert after.agent_params["x_home"][2] == ei["robot"]["ee_desired_height"]
    # ... and nothing else moved with it
    for key in ("hit_range", "defend_range", "prepare_range"):
        assert after.agent_params[key] == pytest.approx(before.agent_params[key])
    assert MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.06,
                             home="goal").agent_params["x_home"][1] == 0.0


def test_home_goal_parks_on_the_centreline_at_the_defend_line():
    """Default goal park = the defend line, y=0: a shot down the middle needs no travel."""
    ei = _env_info()
    agent = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.06, home="goal")
    x_home = agent.agent_params["x_home"]
    assert x_home[1] == 0.0                                   # dead centre of the goal mouth
    assert x_home[0] == pytest.approx(agent.agent_params["defend_range"][0])
    assert x_home[2] == ei["robot"]["ee_desired_height"]      # still on the play plane
    assert agent.home_mode == "goal"


def test_home_goal_dist_sets_the_distance_from_the_goal_line():
    ei = _env_info()
    gl = _goal_line_v(ei)
    x_home = _home_of(min_goal_dist=0.06, home="goal", home_goal_dist=0.12)
    assert x_home[0] == pytest.approx(gl + 0.12)
    assert x_home[1] == 0.0


def test_home_goal_can_never_be_pushed_behind_the_keepout():
    """Asking to park closer to the goal than min_goal_dist is floored, not obeyed -- the
    watchdog would otherwise fault on the parked pose itself."""
    ei = _env_info()
    gl = _goal_line_v(ei)
    x_home = _home_of(min_goal_dist=0.06, home="goal", home_goal_dist=0.0)
    assert x_home[0] == pytest.approx(gl + 0.06)
    # ... and it clears the watchdog's hard catch line (plan floor minus the hard margin)
    assert x_home[0] > gl + 0.06 - 0.03


def test_home_goal_is_inside_the_defend_box_the_agent_already_uses():
    """Reachability, argued from the agent's own geometry rather than an IK seed: the defend
    tactic already drives the mallet to this x on the centreline on every save."""
    ei = _env_info()
    agent = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.06, home="goal")
    x_home = agent.agent_params["x_home"]
    lo, hi = agent.agent_params["defend_range"]
    assert lo <= x_home[0] <= hi
    assert abs(x_home[1]) <= agent.agent_params["defend_width"]


def test_unknown_home_is_refused():
    with pytest.raises(ValueError, match="home"):
        MiniBaselineAgent(_env_info(), agent_id=1, home="goalie")


# ------------------------------------------------------------------- a park chosen by hand
# `--home-x` / `--home-y` move the park off its mode default, per axis, in the ROBOT BASE frame
# -- origin at the arm base, so hypot(x, y) is the reach and y=0 is straight out from it. The
# frames are ROTATED relative to each other, so the override cannot be substituted straight into
# x_home: the mode default goes into the robot frame, the named axis is replaced, and the pair
# comes back. These pin that round trip, and that the table-frame guards still bound the result.
def _robot_xy(agent):
    """The park the agent settled on, back in the robot base frame the flags are written in."""
    return np.asarray(agent._v2r.pos2(agent.agent_params["x_home"][:2]), float)


def _agent(**kw):
    return MiniBaselineAgent(_env_info(), agent_id=1, **kw)


def test_home_x_moves_only_x_and_home_y_moves_only_y_in_the_robot_frame():
    """Keep-out off, so this isolates the override itself: each flag moves its own robot-frame
    axis and leaves the other at whatever --home picked. (With the keep-out ON, a table-frame
    floor can still pull a robot-frame ask off both axes at once -- pinned separately below.)"""
    ref = _robot_xy(_agent())                                 # the base park, unoverridden

    only_x = _robot_xy(_agent(home_x=ref[0] + 0.03))
    assert only_x[0] == pytest.approx(ref[0] + 0.03)
    assert only_x[1] == pytest.approx(ref[1])                 # y kept the mode default

    only_y = _robot_xy(_agent(home_y=ref[1] - 0.03))
    assert only_y[0] == pytest.approx(ref[0])                 # x kept the mode default
    assert only_y[1] == pytest.approx(ref[1] - 0.03)


def test_the_default_base_park_is_the_taught_point_in_the_robot_frame(_sim_home_park):
    """The base park reads back as HOME_PARK_XY exactly -- the point capture_home printed -- so
    the constant reaches the agent unmangled and --home-x is a reach-out distance in the same
    units the capture reported.

    Opts OUT of the sim-station fixture by reading the SHIPPED constant from source: everywhere
    else the suite pins sim behaviour against HOME_POS (see conftest), but the one thing that
    must track the real rig is that the taught value is what the agent parks at."""
    import ast
    import mini_platform.agents.baseline_mini as bm

    src = ast.parse(pathlib.Path(bm.__file__).read_text())
    shipped = next(np.array(ast.literal_eval(n.value.args[0]))
                   for n in ast.walk(src)
                   if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "HOME_PARK_XY")

    saved, bm.HOME_PARK_XY = bm.HOME_PARK_XY, shipped
    try:
        assert _robot_xy(_agent()) == pytest.approx(shipped, abs=1e-9)
    finally:
        bm.HOME_PARK_XY = saved


def test_home_xy_overrides_the_goal_park_and_stays_on_the_plane():
    ei = _env_info()
    want = _robot_xy(_agent()) + np.array([0.03, -0.03])
    agent = _agent(min_goal_dist=0.06, home="goal", home_x=want[0], home_y=want[1])
    assert _robot_xy(agent) == pytest.approx(want, abs=1e-9)
    assert agent.agent_params["x_home"][2] == ei["robot"]["ee_desired_height"]


def test_home_x_beats_home_goal_dist():
    """Both name where the park sits; the explicit robot-frame coordinate is the one that wins."""
    want_x = _robot_xy(_agent())[0] + 0.03
    agent = _agent(min_goal_dist=0.06, home="goal", home_goal_dist=0.12, home_x=want_x)
    assert _robot_xy(agent)[0] == pytest.approx(want_x)


def test_a_robot_frame_park_can_never_be_pushed_behind_the_keepout():
    """The keep-out is a TABLE-frame floor and the frames are rotated, so flooring a robot-frame
    ask moves both of its coordinates -- what must hold is the floor itself, in V. It is a clamp,
    not an error: a park that only reaches back too far is rescued, never refused."""
    ei = _env_info()
    gl = _goal_line_v(ei)
    a = _agent()
    # Name the ask where the floor lives -- in V, 10 mm in front of the own goal line, well
    # behind the 60 mm floor -- and map it back into the robot frame the flags are written in.
    # Which robot axis reaches back down-table is a property of the mount (the frames were
    # rotated 90 deg at the side mount and are aligned at the end mount), so stepping along a
    # robot axis and guessing does not survive a re-mount: a +/-y step no longer moves V-x at
    # all, and the old probe ended up asking for a park that was never behind the floor.
    back_r = np.asarray(a._v2r.pos2(np.array([gl + 0.01, 0.0])), float)
    agent = _agent(min_goal_dist=0.06, home_x=back_r[0], home_y=back_r[1])
    assert agent.agent_params["x_home"][0] == pytest.approx(gl + 0.06)


def test_init_pose_follows_the_overridden_park():
    """x_init is the episode-start pose: it must land above the SAME spot, or the arm brings up
    at one park and the Init tactic immediately drags it to another."""
    agent = _agent(min_goal_dist=0.06, home_x=0.25, home_y=0.05, init_lift=0.06)
    x_home, x_init = agent.agent_params["x_home"], agent.agent_params["x_init"]
    assert np.array_equal(x_init[:2], x_home[:2])
    assert x_init[2] == pytest.approx(x_home[2] + 0.06)


def test_off_table_park_is_refused():
    """An off-table park would only surface as an IK failure at bring-up, or as the mallet driven
    into the rim -- refuse it where the numbers are still readable."""
    with pytest.raises(ValueError, match="off the table"):
        _agent(home_x=1.5)                                    # straight through the far rail
    with pytest.raises(ValueError, match="off the table"):
        _agent(home_y=1.5)                                    # straight through the side rail


def test_override_flags_the_park_as_custom_for_bringup():
    """run_hw_match reads this to bring up ON the park with the flat-wrist IK fallback rather
    than at the arm's station with the strict vertical solve."""
    assert _agent().home_xy_override is False
    assert _agent(home_x=_robot_xy(_agent())[0] + 0.03).home_xy_override is True
