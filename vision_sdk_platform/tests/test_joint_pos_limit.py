"""A hard POSITION band for the base joint (``--j1-limit``), clipped in the PLANNER.

J1's MJCF stops are +-2.79 rad (+-160 deg), far wider than the sweep the arm can use at the end
mount, so the planner is free to route the base somewhere the rig cannot follow. Clipping only at
the stream would turn that into a plan the command then has to throw away -- the plan-vs-command
deficit this controller already pays for elsewhere. So the band goes in where the motion is
CHOSEN, which means TWO arrays, because the planner and the IK read different ones:

  * ``env_info['robot']['joint_pos_limit']`` -- the tracking QP's position bound, the nlopt
    hit-config bounds, ``JointPosConstraint``, and the streamer's final ``clamp_pos``;
  * ``robot_model.jnt_range`` -- the box every IK path keeps inside, i.e. the home pose and every
    hit anchor.

These tests pin that both move, that the band only ever narrows, that a band which actually BITES
is respected by the plan (not just by the clamp), that paying for it comes out of the arm's
redundancy rather than out of the play plane, and that the default is exactly what it was.
"""

import argparse
import importlib
import time

import numpy as np
import pytest

import mini_platform
from mini_platform.agents.baseline_mini import MiniBaselineAgent
from mini_platform.core.kinematics import inverse_kinematics
from mini_platform.envs.env_base import INIT_SEED

run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")
narrow_joint_pos_limit = run_hw_match.narrow_joint_pos_limit
home_config = run_hw_match.home_config
J_BASE = run_hw_match.J_BASE
N_ARM = run_hw_match.N_ARM

#: J1's stops as the MJCF declares them (assets/metal_arm_only.xml joint1 range).
MJCF_J1 = (-2.79, 2.79)


@pytest.fixture
def env_info():
    """A FRESH env_info per test: these tests mutate the joint box, so a module-scoped one would
    leak a narrowed band into whatever ran next."""
    return mini_platform.make("mini-hit-sdk").env_info


def _j1(env_info):
    return tuple(np.asarray(env_info["robot"]["joint_pos_limit"], float)[:, J_BASE])


# ------------------------------------------------------------------- the default is untouched
def test_default_band_is_the_mjcf_stops(env_info):
    """No flag => nothing moved. Every existing log and A/B stays comparable."""
    assert _j1(env_info) == pytest.approx(MJCF_J1)
    assert tuple(env_info["robot"]["robot_model"].jnt_range[J_BASE]) == pytest.approx(MJCF_J1)


# --------------------------------------------------------------------------- what it narrows
def test_it_narrows_both_the_qp_bound_and_the_ik_box(env_info):
    """The whole point: one call, and the planner AND the IK anchors are bound. Narrowing only
    env_info would leave the anchors free to aim at a pose the QP then refuses to track."""
    narrow_joint_pos_limit(env_info, J_BASE, -0.60, 0.55)
    assert _j1(env_info) == pytest.approx((-0.60, 0.55))
    assert tuple(env_info["robot"]["robot_model"].jnt_range[J_BASE]) == pytest.approx((-0.60, 0.55))


def test_it_touches_no_other_joint(env_info):
    before = np.asarray(env_info["robot"]["joint_pos_limit"], float).copy()
    narrow_joint_pos_limit(env_info, J_BASE, -0.60, 0.55)
    after = np.asarray(env_info["robot"]["joint_pos_limit"], float)
    others = [j for j in range(N_ARM) if j != J_BASE]
    assert after[:, others] == pytest.approx(before[:, others])


def test_it_mutates_in_place_so_every_holder_of_the_array_sees_it(env_info):
    """``stage_play`` binds ``pos_limit = env_info[...]`` and hands that object to the streamer.
    A helper that REBOUND the key instead of writing through it would silently leave the stream
    on the old band."""
    held = env_info["robot"]["joint_pos_limit"]
    narrow_joint_pos_limit(env_info, J_BASE, -0.60, 0.55)
    assert held is env_info["robot"]["joint_pos_limit"]
    assert (float(held[0, J_BASE]), float(held[1, J_BASE])) == pytest.approx((-0.60, 0.55))


def test_it_only_ever_narrows(env_info):
    """A flag that could WIDEN the box would be a way to drive the arm past a stop the model
    declares. Intersected with what is already there instead. (Wider still -- past +-2pi -- is
    refused outright by the degrees guard below, which is a different complaint about the same
    mistake.)"""
    narrow_joint_pos_limit(env_info, J_BASE, -5.0, 5.0)
    assert _j1(env_info) == pytest.approx(MJCF_J1)


def test_two_calls_compose_monotonically(env_info):
    narrow_joint_pos_limit(env_info, J_BASE, -0.60, 0.55)
    narrow_joint_pos_limit(env_info, J_BASE, -0.80, 0.30)     # wider below, tighter above
    assert _j1(env_info) == pytest.approx((-0.60, 0.30))


# ----------------------------------------------------------------------------- what it refuses
@pytest.mark.parametrize("lo,hi", [(0.5, -0.5), (0.2, 0.2), (np.nan, 0.5), (-0.5, np.inf)])
def test_a_malformed_band_is_refused(env_info, lo, hi):
    with pytest.raises(SystemExit):
        narrow_joint_pos_limit(env_info, J_BASE, lo, hi)


def test_degrees_are_caught_rather_than_silently_widening(env_info):
    """The band is in RADIANS but the readout an operator eyeballs at the rig invites degrees,
    and degrees would sail through as a band wider than the arm -- i.e. as a no-op. Same guard as
    ``min_goal_dist_error``, and it has to name the radian value."""
    with pytest.raises(SystemExit) as e:
        narrow_joint_pos_limit(env_info, J_BASE, -35.0, 40.0)
    msg = str(e.value)
    assert "RADIANS" in msg and "0.6109" in msg          # radians(-35)


def test_a_band_that_misses_the_stops_entirely_is_refused(env_info):
    """An empty intersection would otherwise leave lo > hi in the box, which every consumer reads
    as "no configuration exists" only once it has already failed."""
    with pytest.raises(SystemExit):
        narrow_joint_pos_limit(env_info, J_BASE, 3.0, 4.0)


# ------------------------------------------------------------------ the IK half is really bound
def test_the_ik_keeps_inside_the_narrowed_band(env_info):
    """``inverse_kinematics`` clamps to ``robot_model.jnt_range``, NOT to env_info -- which is
    exactly why the helper has to write both. Solve for a point that needs base rotation and
    check the solution respects a band that excludes the free answer."""
    rm, rd = env_info["robot"]["robot_model"], env_info["robot"]["robot_data"]
    z = env_info["robot"]["ee_desired_height"]
    target = np.array([0.24, 0.16, z])                 # off the centreline: J1 has to turn
    ok_free, q_free = inverse_kinematics(rm, rd, target, initial_q=INIT_SEED)
    assert ok_free, "pick a reachable probe point or this pins nothing"
    assert abs(q_free[J_BASE]) > 0.1, "probe point must actually need the base joint"

    band = (-0.05, 0.05)
    narrow_joint_pos_limit(env_info, J_BASE, *band)
    _, q_cl = inverse_kinematics(rm, rd, target, initial_q=INIT_SEED)
    assert band[0] - 1e-9 <= q_cl[J_BASE] <= band[1] + 1e-9, \
        "the IK left the band, so the hit anchors and the home pose are not bound"


def test_the_home_pose_stays_inside_a_sane_band(env_info):
    """Home is the arm's own station, J1 ~ 0, so a band taught from the usable sweep contains it.
    If it ever does not, the home IK fails loudly at startup -- before the arm moves."""
    narrow_joint_pos_limit(env_info, J_BASE, -0.60, 0.55)
    q = home_config(env_info)
    assert -0.60 <= q[J_BASE] <= 0.55


# --------------------------------------------------------- the PLAN half is really bound
def _roll(band, steps=250, seed=3):
    """One sim episode -> (planned J1 per tick, max |commanded z - play plane|).

    ``state.q_cmd`` is what the optimiser CHOSE, upstream of the RT stream's ``clamp_pos``, so an
    assertion on it is an assertion about the planner rather than about the backstop. Seeded: the
    env randomises the puck on reset, so two rollouts compared with each other have to see the
    same puck.
    """
    np.random.seed(seed)
    env = mini_platform.make("mini-hit-sdk")
    ei = env.env_info
    if band is not None:
        narrow_joint_pos_limit(ei, J_BASE, *band)
    agent = MiniBaselineAgent(ei, agent_id=1, min_goal_dist=0.06, keep_mallet_flat=True)
    obs = env.reset()
    agent.episode_start()
    z_plane = ei["robot"]["ee_desired_height"]
    j1, z_err = [], []
    for _ in range(steps):
        obs, *_ = env.step(agent.draw_action(obs))
        j1.append(float(agent.state.q_cmd[J_BASE]))
        z_err.append(abs(float(agent.state.x_cmd[2]) - z_plane))
    return np.array(j1), max(z_err)


#: A band that the seed-3 episode actually pushes against (its free J1 reaches ~+0.30 rad), so
#: these tests measure the clip biting rather than a band the plan never approached.
BITING_BAND = (-0.35, 0.15)


def test_the_planner_stays_inside_a_band_that_bites():
    free, _ = _roll(None)
    assert free.max() > BITING_BAND[1], \
        "the band no longer bites on this episode -- retune it or this pins nothing"
    clipped, _ = _roll(BITING_BAND)
    assert clipped.min() >= BITING_BAND[0] - 1e-9
    assert clipped.max() <= BITING_BAND[1] + 1e-9


def test_the_clip_comes_out_of_the_redundancy_not_the_play_plane():
    """A position limit paid for by the POSITION task would lift the mallet off the table -- the
    same defect per-joint velocity clipping had (jaw-z p99 57 mm vs 9 mm coupled). Spent out of
    the arm's spare dimensions instead, the commanded point stays on the plane.

    Bounded, not ordered: measured, a biting band costs ~0.1 mm of plane error against the free
    plan (0.49 vs 0.39 mm on this episode) and the sign of that 0.1 mm is not an invariant. What
    IS one is that both stay far inside the band the mallet can be off the surface by and still
    be playing on its face.
    """
    _, z_free = _roll(None)
    _, z_clipped = _roll(BITING_BAND)
    assert z_clipped < 1e-3, f"commanded point left the play plane by {z_clipped * 1000:.2f} mm"
    assert z_clipped < 4 * max(z_free, 1e-4), "the clip is landing on the position task"


def test_a_band_the_plan_never_reaches_changes_nothing():
    """Byte-identical when it does not bind: a band taught from the real sweep must not quietly
    re-shape ordinary play."""
    free, _ = _roll(None)
    wide, _ = _roll((-2.0, 2.0))
    assert wide == pytest.approx(free)


# ------------------------------------------------------- the stream backstop cannot wind up
class _RecordArm:
    def __init__(self):
        self.cmds = []

    def stream_rt(self, q):
        self.cmds.append(np.asarray(q, float).copy())


def test_the_stream_anchor_cannot_wind_up_outside_the_band():
    """``clamp_pos`` is applied to the ANCHOR, not only to the outgoing command. Clamping just the
    command would let ``_q_written`` march outside a binding band while the arm sat pinned at it,
    so the next re-anchor would continue from a pose the arm was never in -- which is precisely
    what the velocity cap's own comment claims does not happen.
    """
    band = (-0.20, 0.20)
    limits = np.array([[band[0]] + [-10.0] * (N_ARM - 1),
                       [band[1]] + [10.0] * (N_ARM - 1)])
    arm = _RecordArm()
    s = run_hw_match._JointStreamer(arm, 200, 0.02, limits, np.zeros(N_ARM))
    s.start()
    s.set_trajectory([(np.full(N_ARM, 2.0), np.zeros(N_ARM))])   # far outside the band on J1
    time.sleep(0.2)
    s.stop()
    cmds = np.array(arm.cmds)
    assert len(cmds) > 5
    assert cmds[:, J_BASE].max() <= band[1] + 1e-9, "a command left the band"
    assert s._q_written[J_BASE] <= band[1] + 1e-9, "the anchor wound up outside the band"
    assert s._q_last[J_BASE] <= band[1] + 1e-9, "the re-anchor point is outside the band"


def test_the_backstop_is_a_no_op_while_the_band_is_the_stops():
    """Every existing run keeps its byte-identical stream: the MJCF stops are nowhere near."""
    wide = np.array([[-10.0] * N_ARM, [10.0] * N_ARM])
    arm = _RecordArm()
    s = run_hw_match._JointStreamer(arm, 200, 0.02, wide, np.zeros(N_ARM))
    s.start()
    s.set_trajectory([(np.full(N_ARM, 1.0), np.zeros(N_ARM))])
    time.sleep(0.15)
    s.stop()
    cmds = np.array(arm.cmds)
    assert len(cmds) > 5
    assert np.all(np.abs(cmds) < 10.0)
    assert s._q_written == pytest.approx(s._q_last)


# --------------------------------------------------------------------------------- telemetry
def test_the_band_is_recorded_as_applied(env_info):
    """A run taken inside a 60 deg sweep is not comparable with one taken inside the model's 320,
    and nothing else in the log would say which it was. Recorded post-intersection, so the log
    says what the run actually ran under rather than what was typed."""
    narrow_joint_pos_limit(env_info, J_BASE, -5.0, 0.30)      # asks wider below than the stops
    pos_limit = env_info["robot"]["joint_pos_limit"]
    assert np.asarray(pos_limit[:, J_BASE], float) == pytest.approx([MJCF_J1[0], 0.30])
