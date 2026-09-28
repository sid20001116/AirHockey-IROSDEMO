"""The roll capture (`hw/capture_puck_rolls.py`): its measurement math and its file format.

The capture is the one link in the Stage-2 chain that cannot be checked by the fit tests -- they
start from an npz it is assumed to have written correctly. So this renders a red puck into a
synthetic camera frame at a KNOWN world position, runs the tool's own detect -> back-project
path over it, and requires the world xy back within a millimetre. If that link is wrong, every
constant fitted downstream is wrong in the same way and nothing else would notice.

cv2 only (`RealPinholeCamera` undistorts through it); no mujoco, no camera, no arm.
"""

import numpy as np
import pytest

from mini_platform.estimation.mini_env import PUCK_RADIUS, TABLE_LENGTH, TABLE_WIDTH
from mini_platform.hw import capture_puck_rolls as C
from mini_platform.vision.detector import PuckColorProfile, detect_puck

cv2 = pytest.importorskip("cv2")

RES = (640, 480)


@pytest.fixture(scope="module")
def camera():
    """A calibrated pinhole above the table looking straight down (no distortion).

    Deliberately synthetic rather than the rig's ``hw_calib.npz``: the point is to pin the
    tool's geometry, and a test that needs a calibration artefact to exist is a test that
    silently stops running.
    """
    from mini_platform.hw.real_camera import RealPinholeCamera

    K = np.array([[600.0, 0.0, RES[0] / 2], [0.0, 600.0, RES[1] / 2], [0.0, 0.0, 1.0]])
    # world +x -> camera +x, world +y -> camera -y, world -z -> camera +z (looking down)
    R_wc = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return RealPinholeCamera(K, np.zeros(5), R_wc, np.array([0.0, 0.0, 0.9]), RES)


@pytest.fixture(scope="module")
def profile():
    """The real rig's gate shape: a red hue band, yaw marker off."""
    return PuckColorProfile(mode="hsv", hue_lo=348.0, hue_hi=6.0, sat_min=0.5, val_min=0.4,
                            min_red_area=20, bbox_major_range=(3, 400), marker_enabled=False)


def _frame_with_puck(camera, world_xy):
    """A grey frame with a red disc drawn where the puck at ``world_xy`` would appear."""
    frame = np.full((RES[1], RES[0], 3), 90, dtype=np.uint8)
    centre = camera.project(np.array([world_xy[0], world_xy[1], camera.puck_top_z]))
    edge = camera.project(np.array([world_xy[0] + PUCK_RADIUS, world_xy[1], camera.puck_top_z]))
    assert centre is not None and edge is not None, "the test point is not in view"
    radius = int(round(abs(edge[0] - centre[0])))
    cv2.circle(frame, (int(round(centre[0])), int(round(centre[1]))), radius, (220, 20, 20), -1)
    return frame


# ------------------------------------------------------------------------ measurement math
@pytest.mark.parametrize("world_xy", [(0.0, 0.0), (0.30, 0.15), (-0.42, -0.20), (0.45, 0.0)])
def test_detected_world_xy_matches_where_the_puck_was_drawn(camera, profile, world_xy):
    frame = _frame_with_puck(camera, world_xy)
    det = detect_puck(frame, profile, backend="numpy")
    assert det is not None, "the puck was drawn but not detected"
    xy, yaw, J = C._measure(camera, det)
    assert xy is not None and J is not None
    assert np.allclose(xy, world_xy, atol=1e-3)      # within a millimetre
    assert yaw is None                               # no marker in the frame, so no yaw
    # the Jacobian is the local metres-per-pixel scale, ~1.5 mm/px for this camera height
    assert 5e-4 < abs(np.linalg.det(J)) ** 0.5 < 5e-3


def test_measure_gives_up_rather_than_guessing_off_plane(camera, profile):
    """A detection whose back-projection leaves the plane yields no measurement at all (the
    capture logs a dropout) instead of an invented world point."""
    class _Det:
        centroid = (5.0, 5.0)
        marker_centroid = None
    broken = type("cam", (), {"pixel_to_plane": lambda *_a, **_k: None,
                              "fd_jacobian": lambda *_a, **_k: None})()
    assert C._measure(broken, _Det()) == (None, None, None)


def test_off_table_guard_flags_a_stale_extrinsic():
    assert not C._off_table(np.array([0.0, 0.0]))
    assert not C._off_table(np.array([TABLE_LENGTH / 2 - 0.01, TABLE_WIDTH / 2 - 0.01]))
    assert C._off_table(np.array([TABLE_LENGTH, 0.0]))
    assert C._off_table(np.array([0.0, TABLE_WIDTH]))


# ---------------------------------------------------------------------------- live counter
def test_roll_watcher_counts_a_roll_and_measures_its_peak():
    w = C.RollWatcher()
    dt, done = 1.0 / 30, None
    for i in range(40):                     # 1.3 s at 0.9 m/s in +x
        got = w.update(i * dt, np.array([0.9 * i * dt, 0.0]))
        done = got if got is not None else done
    assert w.speed == pytest.approx(0.9, rel=1e-6)
    for i in range(40, 60):                 # ... then it stops
        got = w.update(i * dt, np.array([0.9 * 39 * dt, 0.0]))
        done = got if got is not None else done
    assert len(w.rolls) == 1 and done is not None
    _, dur, peak, path = done
    assert peak == pytest.approx(0.9, rel=0.01)
    assert path == pytest.approx(0.9 * 39 * dt, rel=0.05)
    assert dur > C.ROLL_MIN_DURATION


def test_roll_watcher_ignores_a_nudge():
    w = C.RollWatcher()
    dt = 1.0 / 30
    for i in range(3):
        w.update(i * dt, np.array([0.9 * i * dt, 0.0]))
    for i in range(3, 30):
        w.update(i * dt, np.array([0.9 * 2 * dt, 0.0]))
    assert w.rolls == []


def test_a_dropout_does_not_fabricate_a_speed():
    """A gap must break the difference, not produce a huge apparent step across it."""
    w = C.RollWatcher()
    w.update(0.0, np.array([0.0, 0.0]))
    w.update(0.033, np.array([0.03, 0.0]))
    fast = w.speed
    w.update(0.067, None)                    # dropout
    w.update(2.0, np.array([0.40, 0.0]))     # reappears far away, ~2 s later
    assert w.speed == fast                   # unchanged: no difference was taken across the gap


# ------------------------------------------------------------------------- the file format
def test_save_load_round_trip(tmp_path, camera, profile):
    """What `save` writes is exactly what `load_capture` (and so the fitter) reads back."""
    from mini_platform.hw.real_camera import CameraCalibration

    n = 5
    rows = dict(t=[i * 0.033 for i in range(n)], x=[0.1 * i for i in range(n)],
                y=[0.0] * n, yaw=[np.nan] * n, area=[300] * n,
                u=[300.0 + i for i in range(n)], v=[240.0] * n,
                J=[np.full(4, 1e-3) for _ in range(n)], seq=list(range(n)))
    rows["x"][2] = rows["y"][2] = np.nan                      # one dropout frame
    args = type("a", (), dict(tag="straight", calib="hw_calib.npz", camera_index=4))()
    calib = CameraCalibration(camera.K, camera.dist, camera.R_wc, camera.pos, RES)
    out = tmp_path / "rolls.npz"
    C.save(out, rows, args, calib, "puck_profile.json", n_frames=n, n_det=n - 1)

    cap = C.load_capture(out)
    assert len(cap) == n and cap.tag == "straight"
    assert cap.resolution == RES
    assert cap.good.tolist() == [True, True, False, True, True]
    assert cap.J.shape == (n, 2, 2)
    assert cap.fps == pytest.approx(1 / 0.033, rel=1e-3)


def test_load_capture_rejects_a_foreign_npz(tmp_path):
    path = tmp_path / "telemetry.npz"
    np.savez(path, q_cmd=np.zeros((3, 6)), t=np.arange(3.0))
    with pytest.raises(ValueError, match="not a capture_puck_rolls npz"):
        C.load_capture(path)
