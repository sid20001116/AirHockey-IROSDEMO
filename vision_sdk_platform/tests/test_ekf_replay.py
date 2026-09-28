"""The offline prediction replay (`tools/ekf_replay.py`) -- the Stage-2 acceptance harness.

It has to be worth trusting before it is used to accept a refit on hardware, so what is pinned
here is that it (a) reproduces the runtime's measurement cadence rather than an idealised one,
(b) refuses to score against ground truth it does not have, and (c) actually shows the
improvement when the constants are right -- on data whose true constants are known.

Pure numpy/scipy: no mujoco, no camera.
"""

import numpy as np
import pytest

from mini_platform.estimation.kalman_filter import DEFAULT_PARAMS, EkfParams
from mini_platform.hw.capture_puck_rolls import load_capture
from mini_platform.tools import ekf_replay as R
from mini_platform.tools import fit_mini_table as F
from tests.synthetic_rolls import FPS, TRUTH, write_session

HORIZONS = (0.3, 0.5, 0.8)


@pytest.fixture(scope="module")
def captures(tmp_path_factory):
    return [load_capture(p) for p in write_session(tmp_path_factory.mktemp("replay"))]


@pytest.fixture(scope="module")
def rolling(captures):
    """Only the moving captures -- a still puck's prediction is trivially right."""
    return [c for c in captures if c.tag in ("straight", "bounce")]


def _truth_params():
    """The synthetic table's real constants, in the filter's own parameterisation."""
    n = DEFAULT_PARAMS.n_params.copy()
    n[1] = -TRUTH["e"]
    b = DEFAULT_PARAMS.b_params.copy()
    b[0] = TRUTH["tan"]
    d = F.continuous_to_discrete(TRUTH["d"])
    return EkfParams(damping=np.array([d, d]), n_params=n, b_params=b, source="truth")


# ------------------------------------------------------------------ the headline acceptance
def test_true_constants_beat_the_kuka_ones_at_every_horizon(rolling):
    old = R.summarize(R.replay(rolling, DEFAULT_PARAMS, HORIZONS), HORIZONS)
    new = R.summarize(R.replay(rolling, _truth_params(), HORIZONS), HORIZONS)
    for h in HORIZONS:
        assert new[(h, "all")][1] < old[(h, "all")][1], f"no improvement at {h}s"
    # ... and the gain grows with the horizon, which is the mechanism the refit is fixing:
    # a wrong decay rate is a small position error after 0.3 s and a large one after 0.8 s.
    gain = [old[(h, "all")][1] - new[(h, "all")][1] for h in HORIZONS]
    assert gain[2] > gain[0]
    # Thresholds are on the RATIO, and are set where they are because the score gate admits
    # every tick over which the puck actually travelled 3 cm -- including slow ones, where even
    # badly wrong constants predict well. The long horizon is where the claim lives.
    assert new[(0.5, "all")][1] < 0.7 * old[(0.5, "all")][1]
    assert new[(0.8, "all")][1] < 0.5 * old[(0.8, "all")][1]


def test_bounce_horizons_improve_the_most(rolling):
    """A wrong restitution puts the reflected intercept in the wrong place; the ticks whose
    horizon contains an impact are where that shows, so they should gain at least as much."""
    old = R.summarize(R.replay(rolling, DEFAULT_PARAMS, HORIZONS), HORIZONS)
    new = R.summarize(R.replay(rolling, _truth_params(), HORIZONS), HORIZONS)
    assert old[(0.8, "bounce")][0] > 20, "the bounce capture produced no bounce-horizon ticks"
    rel_bounce = new[(0.8, "bounce")][1] / old[(0.8, "bounce")][1]
    rel_free = new[(0.8, "free")][1] / old[(0.8, "free")][1]
    assert rel_bounce <= rel_free + 0.05


def test_the_whole_pipeline_improves_prediction(tmp_path, captures, rolling):
    """capture -> fit -> npz -> load hook -> replay, with nothing hand-fed in between."""
    params, fitted, details = F.fit(captures)
    out = tmp_path / "mini_table_ekf.npz"
    params.save(out, sigma_px=np.array(details["static"]["sigma_px"]))

    loaded = EkfParams.load(out)
    old = R.summarize(R.replay(rolling, DEFAULT_PARAMS, HORIZONS), HORIZONS)
    new = R.summarize(R.replay(rolling, loaded, HORIZONS), HORIZONS)
    assert new[(0.5, "all")][1] < 0.7 * old[(0.5, "all")][1]
    assert new[(0.8, "all")][1] < 0.5 * old[(0.8, "all")][1]
    assert set(fitted) == {"damping", "n_params", "b_params", "OBS_COV"}


# ------------------------------------------------------------------- faithfulness to the rig
def test_measurements_are_fused_once_per_camera_frame(rolling, monkeypatch):
    """The replay steps at the CONTROL rate and coasts between camera frames, exactly as
    `ThreadedPuckSensor` makes the runtime do -- not once per frame at the camera rate, which
    would hide the coasting error the refit exists to reduce."""
    from mini_platform.estimation import kalman_filter

    counts = {"ticks": 0, "fused": 0}
    real_step = kalman_filter.PuckTracker.step

    def counting_step(self, measurement, R_=None):
        counts["ticks"] += 1
        counts["fused"] += measurement is not None
        return real_step(self, measurement, R_)

    monkeypatch.setattr(kalman_filter.PuckTracker, "step", counting_step)
    cap = rolling[0]
    R.replay([cap], DEFAULT_PARAMS, (0.5,))
    # 30 fps camera, 50 Hz filter -> markedly more ticks than fusions, and never more fusions
    # than there were detections in the log.
    assert counts["fused"] <= int(cap.good.sum())
    assert counts["ticks"] > 1.4 * counts["fused"]


def test_ground_truth_is_refused_across_a_dropout():
    t = np.array([0.0, 0.033, 0.067, 0.5, 0.533])          # a 0.43 s hole
    xy = np.array([[0., 0.], [0.1, 0.], [0.2, 0.], [1.0, 0.], [1.1, 0.]])
    assert R._interp(t, xy, 0.05) is not None               # inside a normal frame gap
    assert R._interp(t, xy, 0.30) is None                   # inside the hole
    assert R._interp(t, xy, 0.9) is None                    # past the end
    assert R._interp(t, xy, -0.1) is None                   # before the start


def test_measurement_R_matches_the_live_sensor_model():
    """The replayed covariance is the one `PuckVisionSensor.measure` would have attached."""
    J = np.array([[1.7e-3, 1e-4], [-2e-4, 1.6e-3]])
    got = R.measurement_R(J, sigma_px=0.42, sigma_floor=1e-3)
    want = 0.42 ** 2 * (J @ J.T) + 1e-6 * np.eye(2)
    assert np.allclose(got, want)


def test_replay_reports_nothing_for_a_still_puck(captures):
    """A static capture has no scorable tick: the puck never travels far enough over a horizon
    to be worth scoring (and it would be predicted perfectly by any constants, so it would only
    dilute the medians)."""
    static = [c for c in captures if c.tag == "static"]
    assert R.replay(static, DEFAULT_PARAMS, HORIZONS) == []
