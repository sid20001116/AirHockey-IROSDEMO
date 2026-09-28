"""Keeping the mallet FLAT on the table (`--flat-mallet`).

The tracking QP (`TrajectoryOptimizer._solve_aqp`) tracks POSITION only -- `jacobian(...)[:3]` --
and on this 6-DOF arm that leaves three spare joint dimensions, which the vendored code spends
pulling toward a fixed joint anchor. Nothing in that says which way the tool points, and it
shows: measured over a sim episode the wrist leaves homing at ~1 deg off vertical and drifts to
a median of ~55 deg within seconds, i.e. the mallet plays on its rim rather than its face.

`keep_mallet_flat` spends those same spare dimensions on the tool axis instead, in the whole
stack rather than only in the tracking loop:

  * the QP ranks levelling above the joint anchor INSIDE the position task's null space
    (`_flat_anchor_dq`), so the mallet stays on the commanded point;
  * the strike anchors are levelled wherever a strictly vertical wrist is unreachable, instead
    of falling back to a manipulability-only pose with the orientation left free;
  * homing solves for the flattest reachable pose (`core.kinematics.solve_flat_config`), which
    is what lets `--home goal` bring up on the centreline at all.

These tests check the tilt actually comes down, that the position task survives it, that the
planners' half agrees with the controller's, and that the default is exactly what it was.
"""

import numpy as np
import pytest
import scipy.linalg
from scipy.spatial.transform import Rotation as _Rot

import mini_platform
from mini_platform.agents.baseline_mini import MiniBaselineAgent
from mini_platform.core.kinematics import (flatten_config, forward_kinematics, inverse_kinematics,
                                           jacobian, solve_flat_config, tool_tilt)
from mini_platform.envs.env_base import INIT_SEED, N_JOINTS
from mini_platform.hw.run_hw_match import home_config, r2v_xform, v2r_xform
from mini_platform.planning.optimizer import TrajectoryOptimizer

EPISODE_STEPS = 200

#: Yaws probed when asking "is there a strictly vertical wrist here?". Denser than the six
#: `_vertical_hit_anchor` and `solve_flat_config`'s tier 1 actually try, so that "no vertical
#: wrist" is a statement about the ARM and not about which yaws happened to be probed.
_YAW_PROBES = np.arange(0.0, 2 * np.pi, np.pi / 8)


@pytest.fixture(scope="module")
def env_info():
    return mini_platform.make("mini-hit-sdk").env_info


def _vertical_wrist_exists(env_info, xy_r, budget=20e-3):
    """Is there a strictly mallet-down configuration at (``xy_r``, the play plane)?"""
    rm, rd = env_info["robot"]["robot_model"], env_info["robot"]["robot_data"]
    x_des = np.array([xy_r[0], xy_r[1], env_info["robot"]["ee_desired_height"]])
    for psi in _YAW_PROBES:
        rot = (_Rot.from_euler("z", psi) * _Rot.from_euler("x", np.pi)).as_matrix()
        if inverse_kinematics(rm, rd, x_des, rot, initial_q=INIT_SEED, max_time=budget)[0]:
            return True
    return False


@pytest.fixture(scope="module")
def no_vertical_xy(env_info):
    """ROBOT-frame xy of a point the arm can REACH but cannot hold the mallet flat at.

    Several tests below are about the GRADED fallback -- what the stack does where a strictly
    vertical wrist does not exist. WHICH points those are is a property of the MOUNT, not of the
    code: at the side mount the whole centreline qualified and these tests hard-coded points off
    it, but since the 2026-09-10 end mount a vertical wrist solves out to ~0.30 m of reach, so
    those same points began solving strictly and the tests were quietly measuring the strict
    branch instead of the fallback they were written for.

    So the point is SEARCHED for: walk out along the centreline until no yaw admits a
    mallet-down solve AND the graded solve still leaves >= 10 deg of tilt -- i.e. until the arm
    honestly cannot hold the mallet flat there. Finding none fails here, so the next re-mount
    fails on THIS precondition rather than on the behaviour the tests are really about.
    """
    rm, rd = env_info["robot"]["robot_model"], env_info["robot"]["robot_data"]
    z = env_info["robot"]["ee_desired_height"]
    v2r = v2r_xform(env_info)
    for x_v in np.arange(0.26, 0.76, 0.02):
        xy_r = np.asarray(v2r.pos2(np.array([x_v, 0.0])), float)
        x_des = np.array([xy_r[0], xy_r[1], z])
        if not inverse_kinematics(rm, rd, x_des, initial_q=INIT_SEED, max_time=20e-3)[0]:
            break                                  # past the arm's reach: stop walking out
        if _vertical_wrist_exists(env_info, xy_r):
            continue
        reached, _, tilt = solve_flat_config(rm, rd, x_des, INIT_SEED, n_joints=N_JOINTS,
                                             max_time=0.5, probe_time=20e-3)
        if reached and tilt >= np.radians(10.0):
            return xy_r
    pytest.fail("no reachable point on the centreline where a vertical wrist is out of reach -- "
                "the premise of the graded-fallback tests has moved with the mount")


def _tilt_deg(rot):
    """Angle between the tool's pointing axis (+Z) and straight down."""
    return float(np.degrees(np.arccos(np.clip(-(rot @ np.array([0., 0., 1.]))[2], -1, 1))))


def _run(keep_mallet_flat, home="base", steps=EPISODE_STEPS, seed=0, home_x=None, home_y=None):
    """One sim episode -> (tilts deg, max |z - play plane| m, min joint-limit margin rad).

    Seeded: the env randomises the puck on reset, so two runs compared against each other have
    to see the SAME puck or the comparison is between two different episodes (which is exactly
    what made an earlier version of this file flaky in a full-suite run and green alone).

    ``home_x``/``home_y`` park the mallet at an arbitrary ROBOT-frame spot, which is how the
    graded-fallback episode reaches a point with no vertical-wrist solution (see
    ``no_vertical_xy`` -- ``--home goal`` was that point at the side mount and is not at this one).
    """
    np.random.seed(seed)
    env = mini_platform.make("mini-hit-sdk")
    ei = env.env_info
    agent = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.06, home=home,
                              home_x=home_x, home_y=home_y,
                              keep_mallet_flat=keep_mallet_flat)
    obs = env.reset()
    agent.episode_start()
    lim = ei["robot"]["joint_pos_limit"]
    z_plane = ei["robot"]["ee_desired_height"]
    tilts, z_err, margin = [], [], []
    for _ in range(steps):
        obs, *_ = env.step(agent.draw_action(obs))
        q = agent.state.q_cmd[:6]
        _, rot = forward_kinematics(agent.state.robot_model, agent.state.robot_data, q)
        tilts.append(_tilt_deg(rot))
        z_err.append(abs(agent.state.x_cmd[2] - z_plane))
        margin.append(float(np.min(np.minimum(q - lim[0, :6], lim[1, :6] - q))))
    return np.array(tilts), max(z_err), min(margin)


# ------------------------------------------------------------------- the default is untouched
def test_default_optimizer_is_the_vendored_one(env_info):
    assert TrajectoryOptimizer(env_info).keep_mallet_flat is False
    agent = MiniBaselineAgent(env_info, agent_id=1)
    assert agent.traj_generator.optimizer.keep_mallet_flat is False


def test_the_optimizer_solves_standalone_on_this_arm(env_info):
    """The ctor used to hand a 6-DOF arm the 7-DOF iiwa's `anchor_weights`, so the FIRST
    `_solve_aqp` call raised `ValueError: ... size 7 is different from 6` -- `np.diag(w)` is 7x7
    against a 6-row null-space basis. It went unnoticed only because `MiniBaselineAgent`
    overwrites the attribute immediately after construction, which meant every path building a
    TrajectoryOptimizer or TrajectoryGenerator DIRECTLY was broken."""
    opt = TrajectoryOptimizer(env_info)
    n = env_info['robot']['n_joints']
    assert len(opt.anchor_weights) == n
    q0 = np.array([0.0014, -1.5243, 0.5668, -0.6133, 0.0, 0.0014])
    x0 = forward_kinematics(opt.robot_model, opt.robot_data, q0)[0]
    ok, dq = opt._solve_aqp(x0 + np.array([0.01, 0.01, 0.0]), q0.copy(), np.zeros(n))
    assert ok
    assert np.shape(dq) == (n,)


def test_default_trajectories_are_bitwise_unchanged(env_info):
    """Off => the optimiser emits exactly what it emitted before the option existed: the
    anchor branch, on the same numbers. This is what keeps sim and the suite green."""
    a = TrajectoryOptimizer(env_info)
    b = TrajectoryOptimizer(env_info, keep_mallet_flat=False)
    # Both already get ones(6) from the ctor now that it sizes the weights to the arm; kept
    # explicit so this comparison pins the weights it runs under rather than inheriting them.
    a.anchor_weights = b.anchor_weights = np.ones(6)
    rng = np.random.default_rng(0)
    q0 = np.array([0.0014, -1.5243, 0.5668, -0.6133, 0.0, 0.0014])
    x0 = forward_kinematics(a.robot_model, a.robot_data, q0)[0]
    cart = x0 + np.cumsum(rng.normal(0, 2e-4, size=(25, 3)), axis=0)
    ok_a, traj_a = a.optimize_trajectory(cart, q0.copy(), np.zeros(6), q0.copy())
    ok_b, traj_b = b.optimize_trajectory(cart, q0.copy(), np.zeros(6), q0.copy())
    assert ok_a and ok_b
    assert np.array_equal(traj_a, traj_b)


def test_the_drift_this_fixes_is_real(env_info):
    """Guards the premise: with the vendored null-space use, the wrist really does end up tens
    of degrees off vertical. If this ever stops being true, the option is no longer needed."""
    tilts, _, _ = _run(keep_mallet_flat=False)
    assert tilts[0] < 5.0, "the episode should start at the vertical-wrist home pose"
    assert np.median(tilts) > 30.0, "the vendored drift is the thing being corrected"


# ----------------------------------------------------------------------------- the correction
def test_flat_mallet_keeps_the_tool_axis_down(env_info):
    """Bounds are on the median and p90, not the max: the correction is a null-space PULL with
    a finite gain and a rate cap, so a fast transition (a strike commit, a direction reversal)
    tilts the tool transiently before it is brought back. What matters is where the mallet
    lives, which is flat -- 0.9 deg median, 10 deg p90 over a 250-step episode."""
    tilts, _, _ = _run(keep_mallet_flat=True)
    assert np.median(tilts) < 3.0
    assert np.percentile(tilts, 90) < 20.0


def test_flat_mallet_beats_the_default_by_a_lot(env_info):
    off, _, _ = _run(keep_mallet_flat=False)
    on, _, _ = _run(keep_mallet_flat=True)
    assert np.median(on) < 0.05 * np.median(off)


def test_position_task_is_not_disturbed(env_info):
    """The tool axis is held with the SPARE dimensions, so the mallet stays on the play plane."""
    _, z_err_off, _ = _run(keep_mallet_flat=False)
    _, z_err_on, _ = _run(keep_mallet_flat=True)
    assert z_err_on < 1e-3                       # sub-millimetre off the plane
    # ... and no worse than the vendored path on the same episode. This is the assertion that
    # the damping in _flat_anchor_dq exists to hold: an UNDAMPED exact levelling solve asks for
    # huge null-space velocities near a singularity of the tool-axis map, the QP's joint bounds
    # clip them, and the clipping lands on the position task -- 3.7 mm off the plane, 7x worse
    # than the vendored path it is supposed to leave alone.
    assert z_err_on <= z_err_off


def test_joints_stay_clear_of_their_limits(env_info):
    """The spare dimensions are being spent differently, so check the arm does not walk into a
    limit doing it."""
    _, _, margin = _run(keep_mallet_flat=True)
    assert margin > 0.1


def test_unreachable_vertical_degrades_instead_of_failing(env_info, no_vertical_xy):
    """Parked where NO vertical-wrist solution exists, the pull cannot null the tilt -- it has
    to get as close as the arm allows and keep tracking, which is the whole point of
    compensating with the wrist angle rather than demanding it.

    The honest-tilt bound is on the p90, not the median, and that is not a lowered bar: the
    episode spends only PART of its time on the park, and everywhere else on the table a
    vertical wrist does exist and the pull nulls the tilt to under a degree. So the median
    measures the MIX -- 2.5 to 11.4 deg across seeds, varying with nothing but how long that
    episode's puck kept the arm away from the park -- while the upper tail is the park itself
    and is stable: 7.7 to 11.4 deg over the same seeds, agreeing with the 11.5 deg
    `solve_flat_config` reports as the flattest pose that reaches it.
    """
    off, _, _ = _run(keep_mallet_flat=False,
                     home_x=no_vertical_xy[0], home_y=no_vertical_xy[1])
    on, z_err, _ = _run(keep_mallet_flat=True,
                        home_x=no_vertical_xy[0], home_y=no_vertical_xy[1])
    assert np.median(on) < 0.5 * np.median(off)   # much better
    assert np.percentile(on, 90) > 5.0            # ... but honestly not vertical, on the park
    assert z_err < 1e-3                           # and the position task still holds


def _null_space_at(opt, q):
    """The (N_J, b) the QP would hand `_flat_anchor_dq` for a HELD position (x_des == x_cur)."""
    jac = jacobian(opt.robot_model, opt.robot_data, q)[:3, :N_JOINTS]
    return scipy.linalg.null_space(jac), np.zeros(N_JOINTS)


def test_deadband_stops_pushing_a_mallet_that_is_already_flat(env_info):
    """No correction demanded inside the deadband, so the QP does not chatter on a flat tool."""
    opt = TrajectoryOptimizer(env_info, keep_mallet_flat=True)
    q_vertical = np.array([0.0014, -1.5243, 0.5668, -0.6133, 0.0, 0.0014])   # INIT_SEED
    assert np.degrees(tool_tilt(opt.robot_model, opt.robot_data, q_vertical)[0]) < 1.0
    N_J, b = _null_space_at(opt, q_vertical)
    assert np.allclose(opt._flat_anchor_dq(q_vertical, None, N_J, b), 0.0, atol=1e-9)


def test_correction_points_the_right_way(env_info):
    """A tilted tool must be pushed back toward vertical, not further over."""
    opt = TrajectoryOptimizer(env_info, keep_mallet_flat=True)
    q = np.array([0.0, -1.2, 0.9, -0.9, 0.3, 0.0])
    before = tool_tilt(opt.robot_model, opt.robot_data, q)[0]
    assert np.degrees(before) > 5.0                                # precondition: it is tilted
    N_J, b = _null_space_at(opt, q)
    dq = opt._flat_anchor_dq(q, None, N_J, b)
    after = tool_tilt(opt.robot_model, opt.robot_data, q + dq * 1e-3)[0]
    assert after < before


# ------------------------------------------------------- the planners agree with the controller
def test_flatten_config_levels_without_leaving_the_point(env_info):
    """The planners' levelling (`flatten_config`) is the same objective as the QP's, solved to
    convergence off-line. Position is the HARD task throughout: the pose it returns must still
    be at the point it was given, to well under a millimetre."""
    rm, rd = env_info["robot"]["robot_model"], env_info["robot"]["robot_data"]
    x_des = np.array([0.30, -0.06, env_info["robot"]["ee_desired_height"]])
    ok, q_seed = inverse_kinematics(rm, rd, x_des, initial_q=INIT_SEED)   # orientation FREE
    assert ok
    q_seed = np.asarray(q_seed, float)[:N_JOINTS]
    assert np.degrees(tool_tilt(rm, rd, q_seed)[0]) > 20.0, "a free-orientation IK pose is tilted"

    reached, q, tilt = flatten_config(rm, rd, q_seed, x_des, n_joints=N_JOINTS)
    assert reached
    assert tilt < np.radians(2.0)
    assert np.linalg.norm(forward_kinematics(rm, rd, q)[0] - x_des) < 1e-4


def test_solve_flat_config_finds_vertical_where_it_exists(env_info):
    rm, rd = env_info["robot"]["robot_model"], env_info["robot"]["robot_data"]
    x_des = np.array([0.30, -0.06, env_info["robot"]["ee_desired_height"]])
    reached, q, tilt = solve_flat_config(rm, rd, x_des, INIT_SEED, n_joints=N_JOINTS)
    assert reached and tilt < np.radians(1.0)
    assert np.linalg.norm(forward_kinematics(rm, rd, q)[0] - x_des) < 1e-3


def test_solve_flat_config_beats_free_orientation_ik_where_vertical_is_unreachable(
        env_info, no_vertical_xy):
    """Out at full stretch, where NO vertical wrist exists. The point of grading the objective
    is that this returns the flattest pose the arm can hold there instead of failing -- and that
    it is far better than what position-only IK happens to land on, which is what the arm used
    to park in. (At the side mount the goal-front park was such a point; at the end mount it is
    a comfortable 0.25 m straight out from the base, so the point is searched for.)"""
    rm, rd = env_info["robot"]["robot_model"], env_info["robot"]["robot_data"]
    x_des = np.array([no_vertical_xy[0], no_vertical_xy[1],
                      env_info["robot"]["ee_desired_height"]])

    ok, q_free = inverse_kinematics(rm, rd, x_des, initial_q=INIT_SEED)
    assert ok, "the point itself is reachable -- only a VERTICAL wrist there is not"
    free_tilt = tool_tilt(rm, rd, np.asarray(q_free, float)[:N_JOINTS])[0]

    reached, q, tilt = solve_flat_config(rm, rd, x_des, INIT_SEED, n_joints=N_JOINTS,
                                         max_time=0.5, probe_time=20e-3)
    assert reached
    assert tilt > np.radians(2.0), "vertical really is out of reach here"
    assert tilt < 0.5 * free_tilt, f"{np.degrees(tilt):.1f} vs {np.degrees(free_tilt):.1f} deg"
    assert np.linalg.norm(forward_kinematics(rm, rd, q)[0] - x_des) < 1e-3


def test_the_secondary_anchor_pull_cannot_rotate_the_tool(env_info):
    """The QP's null-space target ranks levelling first and the joint anchor second. The ranking
    has to survive the solver, so the anchor's contribution must be confined to the freedom
    levelling did not need -- adding it may not change the tool's angular velocity."""
    opt = TrajectoryOptimizer(env_info, keep_mallet_flat=True)
    opt.anchor_weights = np.ones(N_JOINTS)
    q = np.array([0.0, -1.2, 0.9, -0.9, 0.3, 0.0])
    jac = jacobian(opt.robot_model, opt.robot_data, q)[:, :N_JOINTS]
    N_J = scipy.linalg.null_space(jac[:3])
    b = np.linalg.lstsq(jac[:3], np.zeros(3), rcond=None)[0]

    without = opt._flat_anchor_dq(q, None, N_J, b)
    with_anchor = opt._flat_anchor_dq(q, INIT_SEED - q, N_J, b)
    assert not np.allclose(without, with_anchor), "the anchor should still do something"
    # ... but only about the tool's own axis, which a disc mallet does not care about
    z_tool = forward_kinematics(opt.robot_model, opt.robot_data, q)[1] @ np.array([0., 0., 1.])
    d_omega = jac[3:6] @ (with_anchor - without)
    sideways = np.linalg.norm(d_omega - np.dot(d_omega, z_tool) * z_tool)
    assert sideways < 0.02 * np.linalg.norm(d_omega)   # >98% of it is spin about the tool axis


def test_hit_anchors_are_levelled_only_when_asked(env_info, no_vertical_xy):
    """The strike anchor, where no strictly vertical wrist reaches the hit point.

    What the vendored fallback leaves behind is the thing being fixed: it maximises velocity
    manipulability with the orientation entirely free, so the anchor the QP spends its
    redundancy chasing is tens of degrees off vertical -- and a rim contact rotates the shot
    (~-1.5 deg of aim per deg of tilt). Under flat mode the anchor is levelled in place, and
    because levelling holds position as the hard task it stays on the hit point while doing it.

    At the SIDE mount the fallback was worse still: `solve_hit_config_ik_null` did not converge
    at all and returned INIT_SEED, a pose a third of a metre from the point being struck. At the
    end mount it does reach the point, so what is left to pin is the POSTURE -- which is what
    the option was always really about.
    """
    off = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.06)
    on = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.06, keep_mallet_flat=True)
    hit_pos_v = np.asarray(r2v_xform(env_info).pos2(no_vertical_xy), float)
    hit_dir_v = np.array([1.0, 0.0])

    q_off = off.traj_generator.solve_anchor_pos_ik_null(hit_pos_v, hit_dir_v, INIT_SEED.copy())
    q_on = on.traj_generator.solve_anchor_pos_ik_null(hit_pos_v, hit_dir_v, INIT_SEED.copy())
    assert not off.traj_generator.last_anchor_vertical      # precondition: the fallback ran
    assert not on.traj_generator.last_anchor_vertical

    rm, rd = env_info["robot"]["robot_model"], env_info["robot"]["robot_data"]
    x_des = np.array([no_vertical_xy[0], no_vertical_xy[1],
                      env_info["robot"]["ee_desired_height"]])
    err = lambda q: np.linalg.norm(
        forward_kinematics(rm, rd, np.asarray(q, float)[:N_JOINTS])[0] - x_des)
    tilt = lambda q: tool_tilt(rm, rd, np.asarray(q, float)[:N_JOINTS])[0]

    assert err(q_on) < 1e-3, "the levelled anchor is AT the hit point"
    assert tilt(q_off) > np.radians(20.0), "the vendored anchor really is way off vertical"
    assert tilt(q_on) < 0.5 * tilt(q_off)      # measured 11.6 vs 32.7 deg at the end mount
    assert tilt(q_on) < np.radians(20.0)       # ... and flat in absolute terms, not just better


def test_hit_anchors_are_untouched_where_a_vertical_wrist_exists(env_info):
    """Off the centreline the strict probe succeeds, so both paths return the SAME vertical
    anchor -- the graded fallback is a fallback, not a replacement."""
    off = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.06)
    on = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.06, keep_mallet_flat=True)
    hit_pos_v, hit_dir_v = np.array([0.28, -0.10]), np.array([1.0, 0.0])

    q_off = off.traj_generator.solve_anchor_pos_ik_null(hit_pos_v, hit_dir_v, INIT_SEED.copy())
    q_on = on.traj_generator.solve_anchor_pos_ik_null(hit_pos_v, hit_dir_v, INIT_SEED.copy())
    assert off.traj_generator.last_anchor_vertical and on.traj_generator.last_anchor_vertical
    assert np.array_equal(q_off, q_on)


# --------------------------------------------------------------------------------- bring-up
def test_default_homing_is_unchanged(env_info):
    """`--home base` still homes with a strictly vertical wrist, by the same constrained IK."""
    q = home_config(env_info)
    assert q.shape == (N_JOINTS,)
    assert home_config.last_tilt == 0.0
    assert tool_tilt(env_info["robot"]["robot_model"], env_info["robot"]["robot_data"],
                     q)[0] < np.radians(1.0)


def test_homing_refuses_an_impossible_vertical_unless_flatness_is_graded(env_info,
                                                                        no_vertical_xy):
    """A park with no vertical-wrist solution. Without `flat=True` that is still a hard failure
    (nothing silently changes for the default path); with it, the arm homes to the flattest pose
    that reaches the park -- which is what lets a park out at stretch bring up there at all
    instead of homing to its own station and gliding across under the play loop."""
    xy_r = no_vertical_xy

    with pytest.raises(RuntimeError):
        home_config(env_info, target_xy=xy_r)

    q = home_config(env_info, target_xy=xy_r, flat=True)
    rm, rd = env_info["robot"]["robot_model"], env_info["robot"]["robot_data"]
    x = forward_kinematics(rm, rd, q)[0]
    assert np.linalg.norm(x[:2] - xy_r) < 1e-3                      # it parks where asked
    assert abs(x[2] - env_info["robot"]["ee_desired_height"]) < 1e-3  # ... on the play plane
    assert 0 < home_config.last_tilt < np.radians(25.0)             # tilted, but honestly so


# --------------------------------------------------------------------------------- aim
def test_the_aim_bias_compensation_follows_the_wrist(env_info):
    """`AIM_BIAS_COMP` pre-rotates every smash to cancel the shot rotation a TILTED mallet head
    causes. Holding the mallet flat removes the tilt -- and so removes what there was to cancel,
    leaving the pre-rotation as pure error (measured: +7.9 deg of outgoing bias with it, +0.4
    without). So the compensation has to be a property of how the wrist is being held, not a
    constant."""
    from mini_platform.agents.baseline_mini import AIM_BIAS_COMP, AIM_BIAS_FLAT

    off = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.06)
    on = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.06, keep_mallet_flat=True)
    assert off.traj_generator.aim_bias == AIM_BIAS_COMP      # default: unchanged
    assert on.traj_generator.aim_bias == AIM_BIAS_FLAT == 0.0

    # and it reaches the shot: the VPSTO aim path rotates the goal ray by exactly that much
    P_v, goal_v = np.array([0.30, 0.0]), np.array([1.20, 0.0])
    ang = lambda g: np.arctan2(*g._vp_aim(P_v, False, goal_v)[1][::-1])
    assert ang(off.traj_generator) == pytest.approx(AIM_BIAS_COMP, abs=1e-9)
    assert ang(on.traj_generator) == pytest.approx(0.0, abs=1e-9)
