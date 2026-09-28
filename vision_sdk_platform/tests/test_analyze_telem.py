"""The offline log analyser's cap resolution -- which every A/B comparison depends on.

`--compare` is the tool's main job, and the compare log used to be analysed against
`args.max_joint_vel` (normally None) instead of its own recorded cap, so the second log's
over-cap rows were silently dropped from exactly the comparison they existed for.
"""
import argparse
import importlib

import numpy as np
import pytest

at = importlib.import_module("mini_platform.tools.analyze_telem")


def _log(tmp_path, name, cap):
    p = tmp_path / name
    np.savez(p, t=np.arange(5, dtype=float), cap_joint_vel=np.asarray(cap, float))
    return np.load(p, allow_pickle=True)


def test_cap_arg_takes_one_value_or_a_per_joint_vector():
    assert at._cap_arg("1.4") == pytest.approx(1.4)
    assert at._cap_arg("2.5,1.4,1.4,1.4,1.4,1.4") == pytest.approx([2.5, 1.4, 1.4, 1.4, 1.4, 1.4])
    assert at._cap_arg("2.5 1.4") == pytest.approx([2.5, 1.4])
    with pytest.raises(argparse.ArgumentTypeError):
        at._cap_arg("")


def test_each_log_is_resolved_against_its_own_recorded_cap(tmp_path):
    """The two runs in an A/B are usually NOT at the same cap -- that is frequently the point of
    the comparison -- so each log has to be read against the envelope it was actually taken under."""
    uniform = _log(tmp_path, "a.npz", 1.4)
    per_joint = _log(tmp_path, "b.npz", [1.8, 1.4, 1.4, 1.4, 1.4, 1.4])
    assert at._resolve_cap(uniform, None) == pytest.approx(1.4)
    assert at._resolve_cap(per_joint, None) == pytest.approx([1.8, 1.4, 1.4, 1.4, 1.4, 1.4])


def test_an_explicit_cap_still_wins(tmp_path):
    """The six pre-0c logs record no cap at all, so the override has to remain available."""
    d = _log(tmp_path, "a.npz", 1.4)
    assert at._resolve_cap(d, 2.0) == pytest.approx(2.0)


def test_a_log_with_no_recorded_cap_resolves_to_nothing(tmp_path):
    p = tmp_path / "old.npz"
    np.savez(p, t=np.arange(5, dtype=float))
    d = np.load(p, allow_pickle=True)
    assert at._resolve_cap(d, None) is None


def test_a_run_with_no_cap_resolves_to_nothing_not_to_nan(tmp_path):
    """A run with no cap records NaN. `if cap:` is TRUE for NaN, so such logs used to report a
    confident '0.0% over the cap' -- the cap has to come back as None so the row is skipped."""
    p = tmp_path / "nocap.npz"
    np.savez(p, t=np.arange(5, dtype=float), cap_joint_vel=np.float64(np.nan))
    assert at._resolve_cap(np.load(p, allow_pickle=True), None) is None


def test_a_per_joint_cap_is_not_flattened_to_its_first_entry(tmp_path):
    """`_meta` returns `reshape(-1)[0]`, so reading the cap through it turned
    [1.8, 1.4, 1.4, 1.4, 1.4, 1.4] into a uniform 1.8 -- too lenient for J2..J6, and invisible
    because the report printed one tidy number. This is the regression pin for that."""
    p = tmp_path / "perjoint.npz"
    caps = np.array([1.8, 1.4, 1.4, 1.4, 1.4, 1.4])
    np.savez(p, t=np.arange(5, dtype=float), cap_joint_vel=caps)
    got = at._resolve_cap(np.load(p, allow_pickle=True), None)
    assert np.shape(got) == (6,)
    assert got == pytest.approx(caps)
