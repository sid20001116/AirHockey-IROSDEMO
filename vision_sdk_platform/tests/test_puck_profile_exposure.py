"""A puck colour profile is only valid at the exposure it was fitted at.

`sat_min` / `val_min` are FIXED thresholds, so a gate tuned under one exposure mis-segments under
another -- and it fails silently, as lost detections rather than an error. Below ~18 detections/s
this EKF cannot track a struck puck at all, so a silent loss here is expensive.

Until 2026-09-20 `tune_puck` could not set the camera controls at all: it inherited whatever the
device happened to hold (V4L2 exposure settings persist between processes), so matching the
calibration to the runtime was luck.
"""
import importlib
import json

import pytest

from mini_platform.vision.detector import PuckColorProfile

run_hw_match = importlib.import_module("mini_platform.hw.run_hw_match")


class _Args:
    puck_profile = None
    cam_exposure = None
    camera_index = 0


def _write(tmp_path, **kw):
    p = tmp_path / "profile.json"
    PuckColorProfile(mode="hsv", **kw).save(p)
    return str(p)


def test_the_exposure_survives_a_save_load_round_trip(tmp_path):
    path = _write(tmp_path, tuned_exposure=50, tuned_exposure_auto=1)
    back = PuckColorProfile.load(path)
    assert back.tuned_exposure == 50
    assert back.tuned_exposure_auto == 1


def test_a_profile_tuned_before_stamping_existed_still_loads(tmp_path):
    """Old profiles have no such key. `from_dict` filters to the dataclass fields, so they must
    load with the gates intact and the provenance simply absent."""
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"mode": "hsv", "hue_lo": 340.0, "sat_min": 0.5}))
    prof = PuckColorProfile.load(str(p))
    assert prof.hue_lo == 340.0
    assert prof.sat_min == 0.5
    assert prof.tuned_exposure is None


def test_a_mismatched_exposure_is_called_out(tmp_path, capsys):
    a = _Args()
    a.puck_profile = _write(tmp_path, tuned_exposure=157, tuned_exposure_auto=1)
    a.cam_exposure = 50
    run_hw_match.load_puck_profile(a)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "157" in out and "50" in out
    assert "tune_puck" in out, "the warning should say how to fix it"


def test_a_matching_exposure_is_not_called_out(tmp_path, capsys):
    a = _Args()
    a.puck_profile = _write(tmp_path, tuned_exposure=50, tuned_exposure_auto=1)
    a.cam_exposure = 50
    run_hw_match.load_puck_profile(a)
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "tuned at exposure 50" in out


def test_an_unpinned_run_is_flagged_even_when_the_profile_knows_its_exposure(tmp_path, capsys):
    """The driver is free to move an unpinned exposure, so a stamped profile plus no
    --cam-exposure is still a mismatch waiting to happen."""
    a = _Args()
    a.puck_profile = _write(tmp_path, tuned_exposure=50, tuned_exposure_auto=1)
    a.cam_exposure = None
    run_hw_match.load_puck_profile(a)
    out = capsys.readouterr().out
    assert "pins no" in out and "--cam-exposure 50" in out


def test_an_unstamped_profile_says_it_cannot_be_checked(tmp_path, capsys):
    a = _Args()
    a.puck_profile = _write(tmp_path)
    a.cam_exposure = 50
    run_hw_match.load_puck_profile(a)
    out = capsys.readouterr().out
    assert "predates exposure stamping" in out


def test_no_profile_at_all_still_warns_and_falls_back(capsys):
    a = _Args()
    assert run_hw_match.load_puck_profile(a) is None
    assert "no --puck-profile" in capsys.readouterr().out


def test_the_tuner_accepts_the_same_camera_flags_as_the_play_stage():
    """Parity with run_hw_match: the calibration step has to be able to REACH the exposure the
    runtime pins, or the two can only ever match by accident -- which is what it did before.

    Rendering --help also guards this module against an unescaped '%' in any help string, which
    breaks the CLI only when help is rendered and so leaves every other test green.
    """
    import contextlib
    import io

    tune_puck = importlib.import_module("mini_platform.hw.tune_puck")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as e:
        tune_puck.main(["--help"])
    assert e.value.code == 0
    help_text = buf.getvalue()
    for flag in ("--cam-fps", "--cam-exposure", "--cam-auto-exposure", "--cam-auto-priority"):
        assert flag in help_text, f"{flag} missing from tune_puck"
