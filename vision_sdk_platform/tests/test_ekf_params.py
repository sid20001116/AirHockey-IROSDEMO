"""The EKF refit load hook (`estimation/kalman_filter.EkfParams`).

Two things have to be true for the mini-table refit to be safe to ship:

* **off by default, byte-identically.** Every run without ``--ekf-params`` must execute the
  vendored KUKA constants, on the same array objects, so sim, the GT regression and the 105-test
  suite cannot move. That is what makes the change reversible by dropping a flag.
* **on when asked, everywhere.** A loaded file has to reach all five places the constants are
  consumed -- the two ``F_linear`` damping entries, the three rows of the rim-collision
  Jacobian, the measurement covariance, and the two process-noise matrices -- or the refit is
  silently partial.

Runs without mujoco: `PuckTracker` needs only the handful of numbers in
`estimation/mini_env.v_env_info()`. The one test that needs the whole agent skips instead.
"""

import numpy as np
import pytest

from mini_platform.estimation import kalman_filter as K
from mini_platform.estimation.kalman_filter import DEFAULT_PARAMS, EkfParams, PuckTracker
from mini_platform.estimation.mini_env import v_env_info


@pytest.fixture(scope="module")
def env_info():
    return v_env_info()


def _sequence(n=150, dt=0.02):
    """Constant-velocity 3-D measurements well clear of every rim (no bounce branch)."""
    t = np.arange(1, n + 1) * dt
    return np.stack([0.62 - 0.15 * t, 0.05 + 0.02 * t, 0.1 + 0.1 * t], axis=1)


# --------------------------------------------------------------------- default = untouched
def test_defaults_are_the_module_constants_themselves():
    """Not copies: the same objects, so the default path's arithmetic is unchanged."""
    assert DEFAULT_PARAMS.damping is K.damping
    assert DEFAULT_PARAMS.b_params is K.b_params
    assert DEFAULT_PARAMS.n_params is K.n_params
    assert DEFAULT_PARAMS.theta_params is K.theta_params
    assert DEFAULT_PARAMS.OBS_COV is K.OBS_COV
    assert DEFAULT_PARAMS.LIN_COV is K.LIN_COV
    assert DEFAULT_PARAMS.COL_COV is K.COL_COV


def test_tracker_without_params_is_bitwise_the_tracker_with_none(env_info):
    a, b, c = (PuckTracker(env_info), PuckTracker(env_info, ekf_params=None),
               PuckTracker(env_info, ekf_params=DEFAULT_PARAMS))
    for trk in (a, b, c):
        trk.reset(np.array([0.62, 0.05, 0.1]))
    for m in _sequence():
        a.step(m)
        b.step(m)
        c.step(m)
        assert np.array_equal(a.state, b.state) and np.array_equal(a.P, b.P)
        assert np.array_equal(a.state, c.state) and np.array_equal(a.P, c.P)


def test_default_tracker_still_binds_the_module_arrays(env_info):
    trk = PuckTracker(env_info)
    assert trk.R is K.OBS_COV
    assert trk.Q is K.LIN_COV
    assert trk.system.table.col_cov is K.COL_COV
    assert trk.system.F_linear[2, 2] == 1 - env_info["dt"] * K.damping[0]
    assert trk.system.F_linear[3, 3] == 1 - env_info["dt"] * K.damping[1]


# ------------------------------------------------------------------ loaded params take hold
def test_loaded_params_reach_every_consumer(env_info):
    p = EkfParams(damping=np.array([0.9, 1.1]),
                  b_params=np.array([0.5, 0.1, 1e-3]),
                  n_params=np.array([0.02, -0.55, 0.03]),
                  theta_params=np.array([-1.0, 2.0, 0.5]),
                  OBS_COV=np.diag([1e-6, 2e-6, 3e-3]),
                  LIN_COV=np.eye(6) * 7.0, COL_COV=np.eye(6) * 5.0)
    trk = PuckTracker(env_info, ekf_params=p)
    dt = env_info["dt"]
    assert trk.system.F_linear[2, 2] == pytest.approx(1 - dt * 0.9)
    assert trk.system.F_linear[3, 3] == pytest.approx(1 - dt * 1.1)
    jac = trk.system.table._jac_local_collision
    assert np.array_equal(jac[2, [2, 3, 5]], p.b_params)
    assert np.array_equal(jac[3, [2, 3, 5]], p.n_params)
    assert np.array_equal(jac[5, [2, 3, 5]], p.theta_params)
    assert trk.R is p.OBS_COV
    assert trk.Q is p.LIN_COV
    assert trk.system.table.col_cov is p.COL_COV


def test_new_damping_actually_changes_the_prediction(env_info):
    """The point of the whole exercise: a different decay must move the forward integration,
    and move it MORE the further ahead it looks."""
    slow = PuckTracker(env_info)                                       # KUKA ~0.21 1/s
    fast = PuckTracker(env_info, ekf_params=EkfParams(damping=np.array([1.2, 1.2])))
    err = []
    for h in (0.2, 0.5, 0.8):
        for trk in (slow, fast):
            trk.reset(np.array([0.62, 0.0, 0.0]))
            trk.state[2:4] = [-1.5, 0.0]
            trk.P = np.eye(6) * 1e-6
        a = slow.get_prediction(h, defend_line=-1e6)[0][:2]
        b = fast.get_prediction(h, defend_line=-1e6)[0][:2]
        err.append(float(np.linalg.norm(a - b)))
    assert err[0] > 0.01                       # already visible at 0.2 s
    assert err[2] > 2 * err[0]                 # and compounds with the horizon


# ------------------------------------------------------------------------- the file format
def test_partial_file_keeps_the_vendored_rest(tmp_path):
    """The fitter writes only what the rolls could measure; the rest must fall back."""
    path = tmp_path / "partial.npz"
    np.savez(path, damping=np.array([0.5, 0.6]))
    p = EkfParams.load(path)
    assert np.array_equal(p.damping, [0.5, 0.6])
    for k in ("b_params", "n_params", "theta_params", "OBS_COV", "LIN_COV", "COL_COV"):
        assert np.array_equal(getattr(p, k), getattr(DEFAULT_PARAMS, k))
    assert p.diff_from_default() == ["damping"]


def test_round_trip_and_metadata(tmp_path):
    path = tmp_path / "ekf.npz"
    src = EkfParams(damping=np.array([0.51, 0.52]), n_params=np.array([0., -0.6, 0.]))
    src.save(path, fitted=np.array(["damping", "n_params"]), sigma_px=np.array(0.42))
    back = EkfParams.load(path)
    for k in EkfParams.KEYS:
        assert np.array_equal(getattr(back, k), getattr(src, k))
    assert back.sigma_px == pytest.approx(0.42)      # metadata, not a filter constant
    assert back.source == str(path)


def test_load_rejects_a_wrong_shape(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, damping=np.array([0.5, 0.6, 0.7]))
    with pytest.raises(ValueError, match="damping"):
        EkfParams.load(path)


def test_load_rejects_non_finite_values(tmp_path):
    """A NaN would propagate into F_linear and poison the filter silently -- on the arm."""
    path = tmp_path / "nan.npz"
    np.savez(path, damping=np.array([0.5, np.nan]))
    with pytest.raises(ValueError, match="non-finite"):
        EkfParams.load(path)


def test_load_rejects_a_file_with_nothing_it_recognises(tmp_path):
    path = tmp_path / "not_ekf.npz"
    np.savez(path, q_cmd=np.zeros(6))                 # e.g. a telemetry log by mistake
    with pytest.raises(ValueError, match="not an EKF-params file"):
        EkfParams.load(path)


def test_describe_names_only_what_changed():
    assert "nothing overridden" in DEFAULT_PARAMS.describe()
    p = EkfParams(damping=np.array([0.5, 0.6]), source="x.npz")
    text = p.describe()
    assert "damping" in text and "n_params" not in text and "x.npz" in text


# ------------------------------------------------------------------------- runtime wiring
def test_agent_threads_ekf_params_to_its_tracker():
    """`run_hw_match --ekf-params` -> MiniBaselineAgent -> MiniSystemState -> PuckTracker."""
    pytest.importorskip("mujoco")
    import mini_platform
    from mini_platform.agents.baseline_mini import MiniBaselineAgent

    ei = mini_platform.make("mini-hit-sdk").env_info
    p = EkfParams(damping=np.array([0.77, 0.88]))
    agent = MiniBaselineAgent(ei, agent_id=1, ekf_params=p)
    assert agent.state.puck_tracker.params is p
    assert agent.state.puck_tracker.system.F_linear[2, 2] == pytest.approx(1 - ei["dt"] * 0.77)
    # ... and the default agent is untouched
    assert MiniBaselineAgent(ei, agent_id=1).state.puck_tracker.params is DEFAULT_PARAMS
