"""Unit tests for the ported METAL SDK control law (`core/sdk_control.py`).

These do not need the planner, the agent or a display. Each is sub-second apart from the two
env fixtures, which build a MuJoCo model once per module.

Run with::

    cd mini_platform && pytest tests/test_sdk_control.py -q
"""

import mujoco
import numpy as np
import pytest

import mini_platform
from mini_platform.core import sdk_params
from mini_platform.core.sdk_control import SdkJointControl, _quantize


@pytest.fixture(scope="module")
def defend():
    return mini_platform.make("mini-defend-sdk").base_env


@pytest.fixture(scope="module")
def hit():
    """`mini-hit` has n_agents == 2. It MUST be in the suite: at n_agents == 1 the legacy
    `list * n_agents` gain-tiling idiom is silently correct, so only this fixture can catch a
    regression to it."""
    return mini_platform.make("mini-hit-sdk").base_env


# --------------------------------------------------------------------------------- T1
def test_hold_mask_is_400hz_phase_locked():
    mask = SdkJointControl._hold_mask(20, 1e-3, 400.0)
    idx = np.flatnonzero(mask)

    assert idx.tolist() == [2, 4, 7, 9, 12, 14, 17, 19]
    assert mask.sum() == 8, "8 latches per 20 ms env step == 400 Hz exactly"
    assert np.diff(idx).tolist() == [2, 3, 2, 3, 2, 3, 2]
    # The last latch must land on the final substep, i.e. exactly on t = dt, the segment
    # endpoint the planner actually asked for.
    assert idx[-1] == 19

    # None -> re-latch every substep (the 1 kHz upper bound, not the hardware).
    assert SdkJointControl._hold_mask(20, 1e-3, None).all()

    # A rate that does not divide the env step would alias across steps; refuse it.
    with pytest.raises(AssertionError):
        SdkJointControl._hold_mask(20, 1e-3, 333.0)


# --------------------------------------------------------------------------------- T2
def test_rigid_body_terms_match_mj_forward_exactly(defend):
    """The G/C split and the inertia term must be bit-identical to the reference path.

    `mj_rne` reads `d.cvel`, not `d.qvel`, so a missing `mj_comVel` after writing qvel would
    silently return G twice -- which this catches as a large C error, not a small one.
    """
    m = defend.robot_model
    ref = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]

    for _ in range(200):
        q = rng.uniform(lo, hi)
        qd = rng.uniform(-3, 3, m.nv)
        qdd = rng.uniform(-50, 50, m.nv)

        G, C, Mqdd = defend._rigid_body_terms(q, qd, qdd)

        ref.qpos[:] = q
        ref.qvel[:] = 0.0
        mujoco.mj_forward(m, ref)
        assert np.array_equal(G, ref.qfrc_bias), "G(q) != qfrc_bias at qvel = 0"

        ref.qvel[:] = qd
        mujoco.mj_forward(m, ref)
        # The raw pipeline output is bit-identical to mj_forward's. This is the claim that
        # matters: it proves mj_kinematics/comPos/comVel/rne is a drop-in for mj_forward here.
        assert np.array_equal(defend._Bias, ref.qfrc_bias), "bias != qfrc_bias(q, qd)"
        # G + C only round-trips to floating-point precision, because C is *defined* as
        # Bias - G and (G + (Bias - G)) != Bias exactly in IEEE 754.
        assert np.allclose(G + C, ref.qfrc_bias, atol=1e-12, rtol=0.0)

        want = np.zeros(m.nv)
        mujoco.mj_mulM(m, ref, want, qdd)
        assert np.allclose(Mqdd, want, atol=0.0, rtol=0.0), "M(q) @ qdd mismatch"


# --------------------------------------------------------------------------------- T3
def test_host_tick_formula(defend):
    """Hand-compute the feedforward and compare, on a velocity that straddles every branch."""
    n = sdk_params.N
    cfg = defend.sdk
    q = defend.init_state.copy()
    # J5 (index 4) sits inside the 0.05 deadband; J3/J4 straddle the +/-0.8 / +/-1.2 clamps.
    qd = np.array([0.3, -0.7, 0.9, 1.5, 0.02, -2.0])
    q_ref, qd_ref, qdd_ref = q + 0.01, np.zeros(n), np.zeros(n)

    defend._host_tick(q_ref, qd_ref, qdd_ref, q, qd)
    got = defend._cmd_tau.copy()

    qd_ff = np.where(np.abs(qd) < cfg.vel_deadband, 0.0, qd)
    assert qd_ff[4] == 0.0, "the deadband must zero J5"
    assert np.count_nonzero(qd_ff == 0.0) == 1, "the deadband must zero ONLY J5"

    G, C, Mqdd = defend._rigid_body_terms(q, qd_ff, qdd_ref)
    want = cfg.k_gravity * G + cfg.k_coriolis * C
    if cfg.inertia_ff:
        want = want + Mqdd

    visc = cfg.b_viscous * np.clip(qd_ff, cfg.visc_vel_min, cfg.visc_vel_max)
    # The clamp is on the VELOCITY, so J3 saturates at 0.8*0.52 and J4 at 1.2*0.42.
    assert visc[2] == pytest.approx(0.8 * 0.52)
    assert visc[3] == pytest.approx(1.2 * 0.42)
    # +b*qd CANCELS drag. If anyone "fixes" this into a damping term the sign flips and this
    # assertion is the only thing standing between them and a silently doubled drag model.
    want = want + visc

    want = np.clip(want, -cfg.tmax, cfg.tmax)
    if cfg.quantize:
        want = _quantize(want, -cfg.tmax, cfg.tmax, 12)
    assert np.allclose(got, want, atol=1e-12)

    assert np.allclose(defend._cmd_q, np.clip(_quantize(
        np.clip(q_ref, defend._pos_min, defend._pos_max),
        -defend._pmax, defend._pmax, 16), defend._pos_min, defend._pos_max), atol=1e-3)


# --------------------------------------------------------------------------------- T4
def test_quantisation_matches_the_can_frame():
    # 12-bit over [0, 500]: exact on multiples of the step, truncating otherwise.
    assert _quantize(300.0, 0.0, 500.0, 12) == pytest.approx(300.0, abs=0.13)
    assert _quantize(25.0, 0.0, 500.0, 12) == pytest.approx(24.908425, abs=1e-5)
    assert _quantize(20.0, 0.0, 500.0, 12) == pytest.approx(19.902320, abs=1e-5)
    # 12-bit over [0, 5]
    assert _quantize(0.2, 0.0, 5.0, 12) == pytest.approx(0.199023, abs=1e-5)

    x = np.linspace(-30.0, 30.0, 997)
    q = _quantize(x, -30.0, 30.0, 12)
    assert np.all(np.diff(q) >= 0), "quantisation must be monotone"
    assert q.min() >= -30.0 and q.max() <= 30.0, "must never leave the encodable range"
    # DEVIATION 2: we clamp before truncating, the SDK wraps. Out-of-range must saturate.
    assert _quantize(1e6, -30.0, 30.0, 12) <= 30.0
    assert _quantize(-1e6, -30.0, 30.0, 12) >= -30.0


# --------------------------------------------------------------------------------- T5
def test_gain_arrays_are_tiled_not_list_repeated(hit):
    assert hit.n_agents == 2, "this test is meaningless at n_agents == 1"
    n = hit._num_env_joints
    assert n == 12

    for name, base in (("_kp", hit.sdk.kp), ("_kd", hit.sdk.kd),
                       ("_b", hit.sdk.b_viscous), ("_tmax_ff", hit.sdk.tmax)):
        got = getattr(hit, name)
        assert got.shape == (n,), f"{name} is {got.shape}, expected {(n,)}"

    # kp/kd go through quantisation, so compare the tiling structure rather than raw values:
    # the second agent's block must equal the first's.
    for name in ("_kp", "_kd", "_b", "_tmax_ff", "_tmax_out", "_pos_min", "_pos_max"):
        a = getattr(hit, name)
        assert np.array_equal(a[:6], a[6:]), f"{name} is not tiled per agent"


# --------------------------------------------------------------------------------- T6
def test_generator_and_latch_pairing(defend, monkeypatch):
    """One env step must consume exactly 20 interpolator samples and latch exactly 8 times.

    Gating `next(self.traj)` on the hold mask would desynchronise this pair, truncating every
    20 ms segment to its first 40 % -- silently, with plausible-looking numbers.
    """
    counts = {"next": 0, "latch": 0}
    cls = type(defend)
    orig_step_init, orig_host = cls._step_init, cls._host_tick

    class Counting:
        def __init__(self, it):
            self.it = it

        def __iter__(self):
            return self

        def __next__(self):
            counts["next"] += 1
            return next(self.it)

    def step_init(self, obs, action):
        orig_step_init(self, obs, action)
        self.traj = Counting(self.traj)

    def host_tick(self, *a):
        counts["latch"] += 1
        return orig_host(self, *a)

    monkeypatch.setattr(cls, "_step_init", step_init)
    monkeypatch.setattr(cls, "_host_tick", host_tick)

    defend.reset()
    counts["next"] = counts["latch"] = 0          # reset() seeds one deliberate host tick
    defend.step(np.vstack([defend.init_state, np.zeros(6)]))

    assert counts["next"] == 20
    assert counts["latch"] == 8


# --------------------------------------------------------------------------------- T7
def test_grip_is_preserved(defend):
    """`AirHockeyBase._simulation_pre_step` re-commands the 3 N clamp every substep and does
    NOT call super(). Shadowing it drops the mallet -- a free body held only by friction --
    with no exception at all."""
    assert "_simulation_pre_step" not in vars(SdkJointControl)

    defend.reset()
    defend.step(np.vstack([defend.init_state, np.zeros(6)]))

    assert np.allclose(defend._data.ctrl[defend._grip_ctrl_ids], 3.0)
    slip, tilt, jaw = defend.grip_state(1)
    assert slip < 1e-3, f"mallet slipped {slip * 1000:.2f} mm holding still"
    assert 4.5e-3 < jaw < 5.5e-3, f"jaw at {jaw * 1000:.2f} mm, expected ~5 mm"


# --------------------------------------------------------------------------------- T8
def test_position_clamp_is_the_mjcf_range(defend):
    """MEASURED: the MJCF joint range is narrower than the SDK soft limit on all six joints,
    so the intersection is exactly the array the planner and JointPositionConstraint use."""
    want = np.tile(defend.env_info['robot']['joint_pos_limit'], (1, defend.n_agents))
    assert np.allclose(defend._pos_min, want[0])
    assert np.allclose(defend._pos_max, want[1])


def test_legacy_controller_is_unreachable(defend):
    with pytest.raises(NotImplementedError):
        defend._controller(None, None, None, None, None)
