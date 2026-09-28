"""The slow-puck smash: offence against a puck that is slow but not dead still.

The vendored FSM claims a puck through exactly three gates, and every one of them keys off
`is_puck_static()` -- speed under `static_vel_threshold` for more than three ticks. Hardware runs
that threshold at 0.02 m/s (`run_hw_match --static-vel-threshold`, deliberately 10x more sensitive
than the sim's table-scaled 0.202 so DEFEND latches on gentle at-home pushes). A puck creeping at
0.05-0.15 m/s in front of the arm is therefore claimed by NOTHING:

  * `can_smash`      -- refuses it, not static;
  * `should_defend`  -- refuses it unless it is closing on our own goal;
  * `is_puck_stuck`  -- refuses it too, also not static.

The FSM sits in READY and watches it. `smash_slow` opens exactly that gap.

Every test here also pins the OFF state, because the contract is that an agent built with no new
arguments behaves exactly as the vendored one.
"""
import importlib

import numpy as np
import pytest

import mini_platform
from mini_platform.agents.baseline_mini import MiniBaselineAgent
from mini_platform.envs.env_base import INIT_SEED
from mini_platform.fsm.system_state import TACTICS

#: The hardware `static_vel_threshold`. The sim default (0.4 * TABLE_SCALE = 0.202) sits ABOVE
#: `smash_slow_vel`, which would make every "slow" puck already static and the new branch dead
#: code -- so these tests run the rig's value, exactly as `run_hw_match` sets it post-construction.
HW_STATIC_VEL = 0.02

#: Slow, settled, and drifting SIDEWAYS: fast enough not to be static (0.05 > 0.02), slow enough
#: to be claimed (0.05 < 0.15), and transversal, so it is not approaching our goal and its x does
#: not move under the prediction -- which keeps the band assertions about the band.
SLOW_PUCK = [0.30, -0.03, 0.0, 0.05]
#: The same speed, aimed at our own goal. DEFEND's puck, not the smash's.
APPROACHING_PUCK = [0.30, 0.0, -0.08, 0.0]


@pytest.fixture(scope="module")
def ei():
    return mini_platform.make("mini-hit-sdk").env_info


def _agent(env_info, **kw):
    """An agent with the RIG's static threshold, mirroring run_hw_match's post-build override."""
    a = MiniBaselineAgent(env_info, agent_id=1, min_goal_dist=0.02, **kw)
    a.agent_params["static_vel_threshold"] = HW_STATIC_VEL
    return a


def _glide(a, state4, n):
    """Feed `n` consistent measurements of a puck gliding at `state4`; return its SystemState.

    The tracker is SEEDED with the true velocity first. An unseeded filter re-locks from two
    points and reports a wildly wrong velocity for the first few ticks, which is the difference
    between measuring this gate and measuring the seeding artefact.
    """
    st = a.state
    x, y, vx, vy = (float(v) for v in state4)
    dt = float(a.env_info["dt"])
    tr = st.puck_tracker
    tr.reset(np.array([x, y, 0.]))
    s = np.zeros(len(tr.state))
    s[:4] = [x, y, vx, vy]
    tr.state = s
    st.restart = False
    q, dq = INIT_SEED.copy(), np.zeros(len(INIT_SEED))
    for _ in range(n):
        x += vx * dt
        y += vy * dt
        st.update_observation(q, dq, np.array([x, y, 0.]))
    st.update_prediction(a.agent_params["max_prediction_time"],
                         a.agent_params["defend_range"][0])
    return st


def _ready_gate(a):
    """`can_smash` as the FSM asks it, through the READY tactic that owns the dispatcher."""
    return a.tactics_processor[TACTICS.READY.value].can_smash()


# ------------------------------------------------------------------------------------- OFF state

def test_defaults_leave_the_slow_smash_off(ei):
    """The contract: an agent built with no new arguments is the vendored one."""
    a = MiniBaselineAgent(ei, agent_id=1)
    p = a.agent_params
    assert p["smash_slow"] is False
    assert p["smash_slow_vel"] == 0.3
    assert p["smash_slow_ticks"] == 5
    assert a.state.is_puck_slow() is False, "the dwell must never latch while the knob is off"


def test_off_a_slow_puck_is_claimed_by_nothing(ei):
    """Pin the gap this feature exists to close, so it fails the day the FSM grows a claim."""
    a = _agent(ei)
    st = _glide(a, SLOW_PUCK, 40)
    ready = a.tactics_processor[TACTICS.READY.value]
    assert not st.is_puck_static(), "test setup: the puck must be MOVING at the rig threshold"
    assert not ready.can_smash()
    assert not ready.should_defend()
    assert not ready.is_puck_stuck()
    assert st._slow_count == 0, "the dwell counter must not even run while the knob is off"


def test_off_the_dwell_counter_never_runs(ei):
    a = _agent(ei)
    _glide(a, SLOW_PUCK, 40)
    assert a.state._slow_count == 0
    assert a.state.is_puck_slow() is False


# -------------------------------------------------------------------------------------- ON state

def test_a_slow_settled_puck_becomes_smashable(ei):
    """The headline behaviour."""
    a = _agent(ei, smash_slow=True)
    st = _glide(a, SLOW_PUCK, 40)
    assert not st.is_puck_static(), "test setup: still not static -- this is the NEW branch"
    assert st.is_puck_slow()
    assert _ready_gate(a), "a slow, settled, in-band puck must be smashable"


def test_the_dwell_must_elapse_first(ei):
    """Slow for a moment is not settled: it is also what a mid-bounce puck looks like, and what
    the EKF reports for ~300 ms after a re-lock."""
    a = _agent(ei, smash_slow=True, smash_slow_ticks=20)
    _glide(a, SLOW_PUCK, 5)
    assert not a.state.is_puck_slow()
    assert not _ready_gate(a), "the smash must wait for the dwell"


def test_the_dwell_resets_when_the_puck_speeds_up(ei):
    a = _agent(ei, smash_slow=True)
    _glide(a, SLOW_PUCK, 40)
    assert a.state.is_puck_slow()
    _glide(a, [0.30, 0.0, 0.0, 0.9], 6)          # a real shove
    assert a.state._slow_count == 0
    assert not _ready_gate(a)


def test_an_approaching_puck_is_left_to_defend(ei):
    """DEFEND keeps priority: the vendored dispatcher asks can_smash FIRST, so the gate itself
    has to hand an approaching puck back."""
    a = _agent(ei, smash_slow=True)
    st = _glide(a, APPROACHING_PUCK, 40)
    assert st.is_puck_approaching(), "test setup: this puck must read as approaching"
    assert st.is_puck_slow(), "and it is slow -- only the approach may disqualify it"
    assert not _ready_gate(a), "an approaching puck belongs to DEFEND, never the smash"


# ----------------------------------------------------------------------------------------- band

def test_the_band_default_reaches_just_past_the_plant_line(ei):
    """0.4 buys ~45 mm past the defend plant line without leaving the vendored hit range."""
    a = _agent(ei, smash_slow=True)
    hi = a.agent_params["smash_slow_max_x"]
    assert hi == pytest.approx(0.4)
    assert hi > a.agent_params["defend_range"][1], \
        "the default deliberately chases a little past the post"
    assert hi < a.agent_params["hit_range"][1], \
        "but stays inside the band the vendored smash would already have used"


def test_a_puck_past_the_band_is_not_smashed(ei):
    a = _agent(ei, smash_slow=True)
    far = [0.45, -0.03, 0.0, 0.05]               # inside hit_range, past the defend line
    st = _glide(a, far, 40)
    assert st.is_puck_slow()
    assert not _ready_gate(a)


def test_the_band_can_be_opened_up_table(ei):
    """Same puck, band widened on the flag alone -- no code change."""
    a = _agent(ei, smash_slow=True, smash_slow_max_x=0.50)
    _glide(a, [0.45, -0.03, 0.0, 0.05], 40)
    assert _ready_gate(a)


def test_a_deep_puck_is_left_to_prepare(ei):
    """Below hit_range[0] is PREPARE's case; the new branch must not steal it."""
    a = _agent(ei, smash_slow=True)
    deep = [0.20, -0.03, 0.0, 0.05]
    st = _glide(a, deep, 40)
    assert st.is_puck_slow()
    assert not _ready_gate(a)


def test_a_wide_puck_is_not_smashed(ei):
    """The lateral bound is the vendored one, so PREPARE's wall case is untouched."""
    a = _agent(ei, smash_slow=True)
    wide = [0.30, 0.20, 0.0, 0.03]
    _glide(a, wide, 40)
    assert not _ready_gate(a)


# ------------------------------------------------------------------- the three-call-site contract

def test_a_latched_smash_is_not_dropped_five_ticks_later(ei):
    """The regression that proves the gate reached all THREE call sites.

    `can_smash` is asked by `Ready._update_tactic_impl` (enter), `Smash.ready` (keep planning) and
    `Smash._update_tactic_impl` (give up after 5 failing evaluations). Widen only the dispatcher
    and SMASH is entered and then immediately dropped -- so this drives the give-up path directly.
    """
    a = _agent(ei, smash_slow=True)
    st = _glide(a, SLOW_PUCK, 40)
    ready = a.tactics_processor[TACTICS.READY.value]
    ready._update_tactic_impl()
    assert st.tactic_current is TACTICS.SMASH, "the dispatcher must enter SMASH"

    sm = a.tactics_processor[TACTICS.SMASH.value]
    st.tactic_finish = False
    st.trajectory_buffer = []
    sm.switch_count = 0
    for _ in range(8):                            # well past the vendored give-up count of 4
        sm._update_tactic_impl()
    assert st.tactic_current is TACTICS.SMASH, "SMASH must hold while the puck is still smashable"


def test_off_the_same_puck_never_enters_smash(ei):
    """The OFF half of the test above."""
    a = _agent(ei)
    st = _glide(a, SLOW_PUCK, 40)
    a.tactics_processor[TACTICS.READY.value]._update_tactic_impl()
    assert st.tactic_current is TACTICS.READY


# ------------------------------------------------------------------------------------ CLI wiring

def test_the_goalie_switch_does_not_turn_the_smash_on():
    """It changes where the arm GOES, so it is opted into deliberately -- like --ready-guard."""
    run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")
    assert "smash_slow" not in run_hw_match.GOALIE_DEFAULTS
    for name in ("smash_slow_vel", "smash_slow_ticks", "smash_slow_max_x"):
        assert name not in run_hw_match.GOALIE_DEFAULTS


def test_the_overlay_reason_is_silent_while_the_feature_is_off(ei):
    run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")
    a = _agent(ei)
    _glide(a, SLOW_PUCK, 40)
    assert run_hw_match._smash_reason(a) is None


def test_the_overlay_reason_names_the_blocking_condition(ei):
    run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")
    a = _agent(ei, smash_slow=True)
    _glide(a, SLOW_PUCK, 3)
    assert run_hw_match._smash_reason(a).startswith("DWELL")
    _glide(a, SLOW_PUCK, 40)
    assert run_hw_match._smash_reason(a) == "smash ok"
    _glide(a, [0.30, 0.0, 0.0, 0.9], 6)
    assert run_hw_match._smash_reason(a).startswith("NOT SLOW")
