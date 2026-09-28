"""The two-player hit variants: both arms run the baseline agent.

`mini-hit` follows the organizers' `IiwaPositionHit` -- the opponent traces a lissajous and
never plays the puck. `mini-hit-2p` puts a real `MiniBaselineAgent(agent_id=2)` on the other
side instead. These tests pin the wiring, not the outcome.
"""

import numpy as np
import pytest

import mini_platform
from mini_platform.agents.baseline_mini import MiniBaselineAgent
from mini_platform.envs.position_control import _lissajous_opponent_gen


@pytest.fixture(scope="module")
def envs():
    return {t: mini_platform.make(t).base_env
            for t in ("mini-hit", "mini-hit-2p", "mini-hit-sdk-2p")}


def test_registered():
    for t in ("mini-hit-2p", "mini-hit-sdk-2p"):
        assert t in mini_platform.TASKS
        from mini_platform.envs import ENV_DICT
        assert t in ENV_DICT


def test_one_player_opponent_is_the_scripted_wander(envs):
    b = envs["mini-hit"]
    assert b._opponent_policy is None, "stock mini-hit must keep the lissajous wander"
    assert b._opponent_agent_gen is not None


def test_two_player_opponent_is_a_real_agent(envs):
    for t in ("mini-hit-2p", "mini-hit-sdk-2p"):
        b = envs[t]
        assert isinstance(b._opponent_policy, MiniBaselineAgent)
        # agent_id 2 -- it must read the puck in ROBOT 2's frame, not robot 1's.
        assert b._opponent_policy.agent_id == 2


def test_two_player_opponent_gets_cubic_interpolation(envs):
    """A real agent emits a proper (pos, vel) trajectory, so it wants the same cubic agent 1
    gets. Order -1 is only right for the wander, whose velocity is a finite difference."""
    assert envs["mini-hit"].interp_order == (3, -1)
    assert envs["mini-hit-2p"].interp_order == (3, 3)


def test_opponent_episode_start_is_called_on_reset(envs, monkeypatch):
    b = envs["mini-hit-2p"]
    calls = []
    monkeypatch.setattr(type(b._opponent_policy), "episode_start",
                        lambda self, *a, **k: calls.append(1))
    b.reset()
    assert calls, "a real opponent needs episode_start() per episode, like agent 1"


def test_explicit_agent_object_still_works():
    """The original `opponent_agent=<object>` path must be untouched."""
    env = mini_platform.make("mini-hit")
    policy = MiniBaselineAgent(env.env_info, agent_id=2)
    env2 = mini_platform.make("mini-hit", opponent_agent=policy)
    assert env2.base_env._opponent_policy is policy


def test_unknown_opponent_string_is_rejected():
    with pytest.raises(ValueError, match="unknown opponent_agent"):
        mini_platform.make("mini-hit", opponent_agent="nope")


def test_two_player_opponent_actually_contests_the_puck():
    """Behavioural, not cosmetic: the second arm must engage, not park.

    Measured over 25 episodes the baseline opponent touches the puck 6x as often as the wander
    and cuts arrivals at its own goal from 14/25 to 3/25. Here we assert the cheap, robust half
    of that -- the opponent's FSM leaves READY and enters a puck-driven tactic.
    """
    env = mini_platform.make("mini-hit-2p")
    b = env.base_env
    agent = MiniBaselineAgent(env.env_info, agent_id=1)
    seen = set()
    # Full episodes, several seeds: the opponent only leaves READY once the puck actually
    # reaches its side, and the side-mounted baseline declines unstrikeable pucks rather
    # than spraying them, so 250-step windows are no longer enough for every seed.
    for ep in range(6):
        np.random.seed(ep)
        obs = env.reset()
        agent.episode_start()
        for _ in range(env.info.horizon):
            obs, r, absorbing, _ = env.step(agent.draw_action(obs))
            seen.add(b._opponent_policy.state.tactic_current.name)
            if absorbing:
                break
        if seen - {"INIT", "READY"}:
            break
    assert seen - {"INIT", "READY"}, f"opponent never left READY/INIT: {seen}"
