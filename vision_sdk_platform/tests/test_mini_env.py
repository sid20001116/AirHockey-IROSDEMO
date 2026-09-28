"""`estimation/mini_env.py` is a copy of six numbers -- this is the guard that it stays true.

The offline EKF tools (roll capture, fit, replay) must run on a box with a camera and nothing
else, so they cannot build ``env_info`` through mujoco + mushroom_rl just to learn how big the
table is. `mini_env` restates those numbers; if the real ones ever move, everything fitted
against the copy silently describes a different table.

Two independent checks, so neither environment can hide a drift:

* the **literals** in ``envs/env_base.py`` / ``agents/baseline_mini.py``, read with `ast` --
  no imports, so this runs anywhere;
* the **live** ``env_info`` and the agent's own virtual frame -- skipped without mujoco.
"""

import ast
import pathlib

import numpy as np
import pytest

from mini_platform.estimation import mini_env

_SRC = pathlib.Path(mini_env.__file__).resolve().parents[1]


def _literals(relpath, names):
    """Top-level ``NAME = <number>`` assignments in a module's SOURCE (no import)."""
    tree = ast.parse((_SRC / relpath).read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in names and isinstance(node.value, ast.Constant):
            out[node.targets[0].id] = node.value.value
    missing = set(names) - set(out)
    assert not missing, f"{relpath} no longer defines {missing} as a plain literal"
    return out


def test_table_constants_match_env_base_source():
    want = _literals("envs/env_base.py",
                     {"TABLE_LENGTH", "TABLE_WIDTH", "GOAL_WIDTH", "PUCK_RADIUS",
                      "MALLET_RADIUS"})
    for k, v in want.items():
        assert getattr(mini_env, k) == v, f"mini_env.{k} drifted from envs/env_base.py"


def test_v_setback_matches_baseline_mini_source():
    want = _literals("agents/baseline_mini.py", {"V_SETBACK"})
    assert mini_env.V_SETBACK == want["V_SETBACK"]


def test_dt_matches_the_env_step_definition():
    """dt = MuJoCo timestep x n_intermediate_steps x n_substeps, from env_base's own defaults."""
    tree = ast.parse((_SRC / "envs/env_base.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "__init__"
              and any(a.arg == "n_intermediate_steps" for a in n.args.args))
    kw = {a.arg: d for a, d in zip(fn.args.args[-len(fn.args.defaults):], fn.args.defaults)}
    # the defaults are arithmetic (``timestep=1 / 1000.``), so evaluate the expression itself
    # -- in an empty namespace, since it is source we are inspecting rather than trusting.
    def val(name):
        return eval(compile(ast.Expression(kw[name]), "<default>", "eval"), {"__builtins__": {}})
    assert mini_env.DT == pytest.approx(val("timestep") * val("n_intermediate_steps")
                                        * val("n_substeps"))


# ------------------------------------------------------------------------- world <-> V frame
def test_world_to_v_round_trips():
    xy = np.array([[0.0, 0.0], [-0.5, 0.25], [0.508, -0.254]])
    assert np.allclose(mini_env.v_to_world(mini_env.world_to_v(xy)), xy)


def test_world_to_v_is_a_pure_shift_so_velocities_are_shared():
    """The claim the whole fit rests on: only positions move between the frames."""
    a, b = np.array([0.1, -0.2]), np.array([0.4, 0.15])
    assert np.allclose(mini_env.world_to_v(b) - mini_env.world_to_v(a), b - a)


def test_rim_lines_are_the_table_inset_by_a_puck_radius():
    lines = mini_env.rim_lines()
    assert lines["x+"] == pytest.approx(mini_env.TABLE_LENGTH / 2 - mini_env.PUCK_RADIUS)
    assert lines["y-"] == pytest.approx(-mini_env.TABLE_WIDTH / 2 + mini_env.PUCK_RADIUS)


def test_agent_two_is_refused():
    with pytest.raises(ValueError, match="agent 1"):
        mini_env.v_env_info(agent_id=2)


# --------------------------------------------------------------------------- the live check
def test_matches_the_live_env_info():
    pytest.importorskip("mujoco")
    import mini_platform

    ei = mini_platform.make("mini-hit-sdk").env_info
    assert mini_env.TABLE_LENGTH == ei["table"]["length"]
    assert mini_env.TABLE_WIDTH == ei["table"]["width"]
    assert mini_env.GOAL_WIDTH == ei["table"]["goal_width"]
    assert mini_env.PUCK_RADIUS == ei["puck"]["radius"]
    assert mini_env.MALLET_RADIUS == ei["mallet"]["radius"]
    assert mini_env.DT == pytest.approx(ei["dt"])


def test_offline_tracker_sees_the_same_table_as_the_agents():
    """The offline env_info must place the rims where the AGENT's tracker places them --
    otherwise every fitted bounce is measured against the wrong wall."""
    pytest.importorskip("mujoco")
    import mini_platform
    from mini_platform.agents.baseline_mini import MiniBaselineAgent
    from mini_platform.estimation.kalman_filter import PuckTracker

    ei = mini_platform.make("mini-hit-sdk").env_info
    live = MiniBaselineAgent(ei, agent_id=1).state.puck_tracker
    offline = PuckTracker(mini_env.v_env_info(), agent_id=1)
    assert np.allclose(live.system.table.boundary, offline.system.table.boundary)
    assert live.system.dt == pytest.approx(offline.system.dt)

    # ... and a world point maps onto that boundary exactly as the runtime maps it
    world = np.array([mini_env.rim_lines()["x+"], 0.0])
    assert mini_env.world_to_v(world)[0] == pytest.approx(live.system.table.boundary[0, 1, 0])
