"""The --planner vpsto CLI must construct the striker it actually ships.

Regression test for a real hardware crash: the CLI still passed the JOINT-space kwargs
(``vel_frac``/``accel_cap``) after the striker was pivoted to a CARTESIAN formulation
(``vel_lim``/``acc_lim``), so every ``--planner vpsto`` run died with

    TypeError: VpstoStriker.__init__() got an unexpected keyword argument 'vel_frac'

at agent construction. It went unnoticed because ``tools/vpsto_bench`` builds the agent with
NO vpsto_kwargs (defaults), so the sim A/B exercised a path the CLI never took.
"""
import inspect

import pytest

from mini_platform.hw.run_hw_match import vpsto_kwargs_from_args


def test_cli_kwargs_match_the_striker_signature():
    """Every key the CLI passes must be a real VpstoStriker parameter."""
    vpsto_striker = pytest.importorskip(
        "mini_platform.planning.vpsto_striker",
        reason="needs the vp-sto package (pip install -e vp-sto/); the default bezier path "
               "never imports it, so the suite stays dependency-free without it")

    class _Args:
        vpsto_vel_lim = 1.5
        vpsto_acc_lim = 8.0
        vpsto_smooth = 0.0
        vpsto_max_iter = 30

    kwargs = vpsto_kwargs_from_args(_Args())
    params = inspect.signature(vpsto_striker.VpstoStriker.__init__).parameters
    unknown = set(kwargs) - set(params)
    assert not unknown, (
        f"--planner vpsto would crash at agent construction: {sorted(unknown)} "
        f"is not a VpstoStriker parameter (it takes {sorted(set(params) - {'self'})})")


def test_agent_constructs_with_the_cli_kwargs():
    """End-to-end: the kwargs the CLI builds actually construct the agent."""
    pytest.importorskip("vpsto", reason="needs the vp-sto package")
    import mini_platform
    from mini_platform.agents.baseline_mini import MiniBaselineAgent

    class _Args:
        vpsto_vel_lim = 1.5
        vpsto_acc_lim = 8.0
        vpsto_smooth = 0.0
        vpsto_max_iter = 30

    env = mini_platform.make("mini-hit")
    agent = MiniBaselineAgent(env.env_info, agent_id=1, planner="vpsto",
                              vpsto_kwargs=vpsto_kwargs_from_args(_Args()))
    striker = agent.traj_generator._vp
    assert striker is not None
    assert striker.vel_lim.tolist() == [1.5, 1.5]
    assert striker.acc_lim.tolist() == [8.0, 8.0]


def test_max_iter_zero_disables_cma():
    """--vpsto-max-iter 0 must select the deterministic mean path (no cma dependency)."""
    class _Args:
        vpsto_vel_lim = 1.5
        vpsto_acc_lim = 8.0
        vpsto_smooth = 0.0
        vpsto_max_iter = 0

    assert vpsto_kwargs_from_args(_Args())["use_cma"] is False


def test_smooth_zero_is_pure_minimum_time():
    """smooth=0 must leave the objective byte-identical to the original `lambda c: c['T']`."""
    vs = pytest.importorskip("mini_platform.planning.vpsto_striker")
    import numpy as np
    s = vs.VpstoStriker(None, smooth=0.0)
    c = {"T": np.array([0.4, 0.7]),
         "acc": np.random.default_rng(0).normal(size=(2, 20, 2)) * 5.0}
    assert np.array_equal(s._loss(c), c["T"])


def test_smooth_penalises_acceleration_effort():
    """A positive weight must rank a high-acceleration candidate worse than a gentle one of the
    same duration -- that is the whole point of trading time for control."""
    vs = pytest.importorskip("mini_platform.planning.vpsto_striker")
    import numpy as np
    s = vs.VpstoStriker(None, acc_lim=8.0, smooth=0.5)
    gentle = np.full((1, 20, 2), 0.5)
    harsh = np.full((1, 20, 2), 8.0)
    T = np.array([0.5])
    c_gentle = s._loss({"T": T, "acc": gentle})
    c_harsh = s._loss({"T": T, "acc": harsh})
    assert c_harsh[0] > c_gentle[0]
    assert c_gentle[0] >= T[0]          # never cheaper than the pure-time cost


def test_smooth_reaches_the_striker_from_the_cli():
    vs = pytest.importorskip("mini_platform.planning.vpsto_striker")

    class _Args:
        vpsto_vel_lim = 1.5
        vpsto_acc_lim = 8.0
        vpsto_smooth = 0.6
        vpsto_max_iter = 30

    assert vpsto_kwargs_from_args(_Args())["smooth"] == 0.6
    assert vs.VpstoStriker(None, **vpsto_kwargs_from_args(_Args())).smooth == 0.6


# ------------------------------------------------------- limits derived from the joint cap
#
# The shipped 1.5 m/s / 8.0 m/s^2 are 4-8x what this arm manages (0.19-0.25 m/s sideways at
# --max-joint-vel 1.0), so VPSTO planned sprints the RT streamer then throttled: 81-96% of DEFEND
# ticks over the joint cap, replans stalling at ~60 ms p99. A feasibility-aware planner handed an
# infeasible envelope is just a slower Bezier, so absent an explicit flag the limits now come from
# the cap the stream is going to enforce anyway.

def _limits(**over):
    from mini_platform.hw.run_hw_match import resolve_vpsto_limits

    class _Args:
        vpsto_vel_lim = None
        vpsto_acc_lim = None
        max_joint_vel = None
        slow = False

    for k, v in over.items():
        setattr(_Args, k, v)
    return resolve_vpsto_limits(_Args())


def test_limits_come_from_the_joint_cap_when_the_flags_are_absent():
    vel, acc, derived = _limits(max_joint_vel=1.0)
    assert vel == pytest.approx(0.3), "0.3 m/s per rad/s is the measured worst-direction speed"
    assert acc == pytest.approx(2.5)
    assert set(derived) == {"vel_lim", "acc_lim"}


def test_an_explicit_limit_wins_and_the_other_is_still_derived():
    vel, acc, derived = _limits(vpsto_vel_lim=0.9, max_joint_vel=1.0)
    assert vel == pytest.approx(0.9)
    assert acc == pytest.approx(2.5)
    assert derived == ("acc_lim",), "only the flag that was left out may be derived"


def test_slow_mode_drives_the_derived_limits_too():
    """--slow is a joint cap like any other; the strike path must plan inside it."""
    from mini_platform.hw.run_hw_match import SLOW_MAX_JOINT_VEL
    vel, _, _ = _limits(slow=True)
    assert vel == pytest.approx(round(0.3 * SLOW_MAX_JOINT_VEL, 3))


def test_with_no_joint_cap_at_all_the_shipped_defaults_stand():
    """Nothing to derive FROM -- deriving 0 m/s from "no cap" would ground the planner."""
    vel, acc, _ = _limits()
    assert (vel, acc) == (1.5, 8.0)


def test_the_derived_limits_reach_the_striker():
    """The whole point: what the CLI derives is what the solver plans inside."""
    vs = pytest.importorskip("mini_platform.planning.vpsto_striker")

    class _Args:
        vpsto_vel_lim = None
        vpsto_acc_lim = None
        vpsto_smooth = 0.0
        vpsto_max_iter = 10
        max_joint_vel = 1.0
        slow = False

    s = vs.VpstoStriker(None, **vpsto_kwargs_from_args(_Args()))
    assert s.vel_lim.tolist() == [0.3, 0.3]
    assert s.acc_lim.tolist() == [2.5, 2.5]
