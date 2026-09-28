"""The mini-table EKF refit (Stage 2): does the fitter recover constants it was given?

Ground truth is synthetic (``tests/synthetic_rolls.py``): a puck rolled under a KNOWN decay
rate and a KNOWN rim restitution, sampled at a camera-like frame rate with camera-like position
noise, timestamp jitter and dropouts, written in exactly the ``hw/capture_puck_rolls`` npz
format. The fits must come back within tolerances tight enough to matter for a 0.5 s prediction
-- which is the whole point of the refit -- from a realistic amount of data (18 glides, ~30
bounces, 3 static spots).

Pure numpy/scipy: no mujoco, no camera, so this runs anywhere the tools do.
"""

import numpy as np
import pytest

from mini_platform.estimation.kalman_filter import DEFAULT_PARAMS
from mini_platform.estimation.mini_env import rim_lines
from mini_platform.hw.capture_puck_rolls import load_capture
from mini_platform.tools import fit_mini_table as F
from tests.synthetic_rolls import (FPS, NOISE_M, PX_PER_M, TRUTH, concat_rolls, glide_rolls,
                                   roll, write_capture, write_session)


@pytest.fixture(scope="module")
def truth():
    return TRUTH


@pytest.fixture(scope="module")
def captures(tmp_path_factory):
    """A realistic session: 18 straight glides, ~30 bounces, 3 static spots."""
    return [load_capture(p) for p in write_session(tmp_path_factory.mktemp("rolls"))]


# ------------------------------------------------------------------------------ the fits
def test_damping_recovered(captures, truth):
    """The fitted per-tick damping matches the truth converted to the filter's discrete form."""
    params, fitted, det = F.fit(captures)
    assert "damping" in fitted
    expected = F.continuous_to_discrete(truth["d"])
    assert det["damping"]["d_iso"] == pytest.approx(truth["d"], rel=0.06)
    assert params.damping == pytest.approx([expected, expected], rel=0.08)
    # ... and it actually moved off the KUKA values it is replacing
    assert not np.allclose(params.damping, DEFAULT_PARAMS.damping)


def test_restitution_recovered(captures, truth):
    """n_params[1] comes back as -restitution (the NORMAL row -- not b_params)."""
    params, fitted, det = F.fit(captures)
    assert "n_params" in fitted
    assert det["bounce"]["n_e"] >= 10
    assert det["bounce"]["e"] == pytest.approx(truth["e"], abs=0.05)
    assert params.n_params[1] == pytest.approx(-truth["e"], abs=0.05)
    # the untouched entries of the row keep their vendored values
    assert params.n_params[0] == DEFAULT_PARAMS.n_params[0]
    assert params.n_params[2] == DEFAULT_PARAMS.n_params[2]


def test_tangential_recovered(captures, truth):
    params, fitted, det = F.fit(captures)
    assert "b_params" in fitted
    assert params.b_params[0] == pytest.approx(truth["tan"], abs=0.08)


def test_obs_cov_and_sigma_px_recovered(captures, truth):
    """Static spots give back the measurement noise they were generated with, in both
    world units (OBS_COV) and pixels (sigma_px)."""
    params, fitted, det = F.fit(captures)
    assert "OBS_COV" in fitted
    assert len(det["static"]["spots"]) == 3
    sd = np.sqrt(np.diag(params.OBS_COV)[:2])
    assert sd == pytest.approx([truth["sigma"], truth["sigma"]], rel=0.15)
    assert det["static"]["sigma_px"] == pytest.approx(truth["sigma"] * PX_PER_M, rel=0.15)
    # yaw is unmeasurable without the marker -> the vendored variance is kept
    assert params.OBS_COV[2, 2] == DEFAULT_PARAMS.OBS_COV[2, 2]


def test_spin_row_is_never_invented(captures):
    """theta_params needs omega, which the marker-less real profile cannot supply."""
    params, fitted, _ = F.fit(captures)
    assert "theta_params" not in fitted
    assert np.array_equal(params.theta_params, DEFAULT_PARAMS.theta_params)


def test_segmentation_finds_every_roll(captures):
    """Each of the 18 synthetic glides is recovered as at least one clean segment, and none
    of them straddles a rim contact."""
    det = F.fit_damping([captures[0]], verbose=False)
    assert 18 <= len(det["segments"]) <= 22
    assert all(s["rmse_iso"] < F.MAX_FIT_RMSE for s in det["segments"])


def test_bounce_events_are_attributed_to_the_right_rim(captures):
    """Every detected impact sits ON the rim line it was attributed to."""
    lines = rim_lines()
    det = F.fit_bounces([captures[1]], d_axis=[0.55, 0.55], verbose=False)
    assert det["n_e"] >= 10
    for r in det["events"]:
        axis = 0 if r["rim"].startswith("x") else 1
        assert abs(r["xy"][axis] - lines[r["rim"]]) < F.CONTACT_MARGIN


def test_goal_mouth_bounces_are_skipped():
    """A 'bounce' inside the goal mouth is the goal, not the modelled rail -- it must not
    contribute a restitution sample."""
    dt = 1.0 / FPS
    t, xy = roll(np.array([0.30, 0.0]), np.array([1.2, 0.0]), 0.55, 2.0, dt,
                 e_normal=0.62, e_tangent=0.9)
    impacts = F.find_impacts(t, xy)
    assert impacts, "the head-on roll should reach the far rail"
    assert all(F.fit_one_bounce(t, xy, ev, [0.55, 0.55]) is None
               for ev in impacts if ev[2].startswith("x"))


def test_noisier_camera_still_lands_in_tolerance(tmp_path, truth):
    """3x the position noise (a poorly-lit table) must not break the damping fit -- the
    positions-not-differences design is what buys this."""
    rng = np.random.default_rng(11)
    dt = 1.0 / FPS
    glides = [roll(np.array([-0.35, 0.10 * ((i % 5) - 2)]),
                   (0.5 + 0.1 * (i % 6)) * np.array([1.0, 0.0]), truth["d"], 1.6, dt)
              for i in range(12)]
    t, xy = concat_rolls(glides)
    cap = load_capture(write_capture(tmp_path / "noisy.npz", t, xy, "straight", rng,
                                     noise=3 * NOISE_M))
    det = F.fit_damping([cap], verbose=False)
    assert det["d_iso"] == pytest.approx(truth["d"], rel=0.12)


def test_written_npz_round_trips_through_the_load_hook(captures, tmp_path):
    """What the fitter writes is what ``EkfParams.load`` (and so ``--ekf-params``) reads."""
    from mini_platform.estimation.kalman_filter import EkfParams
    params, fitted, det = F.fit(captures)
    out = tmp_path / "mini_table_ekf.npz"
    params.save(out, fitted=np.array(fitted), sigma_px=np.array(det["static"]["sigma_px"]))
    back = EkfParams.load(out)
    for k in EkfParams.KEYS:
        assert np.array_equal(getattr(back, k), getattr(params, k))
    assert set(back.diff_from_default()) == set(fitted)
