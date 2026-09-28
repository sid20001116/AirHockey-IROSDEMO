"""Off-table puck detections (CONTROLLER_FIX_PLAN.md Fix 2, parts 2a and 2c).

`detect_puck` returns the LARGEST blob passing the colour gate, so any red object bigger than
the visible puck replaces the puck for that frame. Measured across the six 2026-09-15 logs, one
such object at V (1.14, 0.11) -- world (0.52, +0.11), just past the far end -- won 158 frames,
and the EKF estimate sat off the table on 4-15% of ticks.

Two things ship here:

  * **2a, the pixel mask** (`hw/real_sensor.RealPuckSensor`): the play area and its two goal
    mouths are projected into the image once and everything else is zeroed BEFORE detection, so
    the object never competes for the blob at all. This is the primary defence.
  * **2c, the one-sided rim bypass** (`PuckTracker._near_rim`): the bypass measured ABSOLUTE
    distance to a rim line, so a point 36 mm PAST the far rim counted as "near" it and skipped
    the NIS gate entirely -- one frame moved the estimate 306 mm and added 0.9 m/s. A point
    outside the box no longer gets that free pass.

2b -- a tracker-level gate that DROPPED off-table measurements outright -- was deliberately not
taken: it cannot coexist with the NIS/re-lock contract that `test_kalman_variable_dim` pins.
What that leaves exposed is pinned below in `test_a_repeating_phantom_can_still_relock`, and is
the reason the mask is the primary defence rather than a convenience.

The ground-truth path (R=None) never enters any of this, so sim stays byte-identical.
"""

import numpy as np
import pytest

import mini_platform
from mini_platform.estimation.kalman_filter import PuckTracker

#: A typical vision-sensor covariance -- the gate applies only to measurements carrying one.
R_VISION = np.diag([2e-6, 2e-6, 1.7e-3])


@pytest.fixture(scope="module")
def env_info():
    return mini_platform.make("mini-hit").env_info


def _settled(env_info, at=(0.60, 0.0)):
    """A tracker that is CONFIDENT about a puck at ``at``.

    Straight after reset P = I, so S is huge and the gate accepts anything; a rejection is only
    possible once the filter actually believes something.
    """
    t = PuckTracker(env_info, agent_id=1)
    t.reset(np.array([at[0], at[1], 0.0]))
    for _ in range(20):
        t.step(np.array([at[0], at[1], 0.0]), R_VISION)
    assert t._reject_count == 0
    return t


def test_the_geometry_these_tests_assume(env_info):
    """Pin the numbers the rest of the file is written against, so a table or mount change
    fails HERE rather than as a puzzling assertion further down."""
    t = PuckTracker(env_info, agent_id=1)
    x_lo, x_hi, y_lo, y_hi = t._box()
    assert (x_lo, x_hi) == pytest.approx((0.13794, 1.10394), abs=1e-4)
    assert (y_lo, y_hi) == pytest.approx((-0.229, 0.229), abs=1e-4)


# ------------------------------------------------------------- 2c: the bypass is one-sided
def test_the_rim_bypass_is_one_sided(env_info):
    """The bypass is for a mistimed BOUNCE, which happens on the table.

    Checked at a y inside the far goal mouth as well, because the logged phantom sits within a
    couple of millimetres of the mouth's edge: a mouth-aware test here would hand it back
    exactly the free pass this removes.
    """
    t = PuckTracker(env_info, agent_id=1)
    _, x_hi, _, y_hi = t._box()
    assert t._near_rim(np.array([x_hi - 0.02, 0.10])), "just INSIDE the rim: bypass applies"
    assert t._near_rim(np.array([x_hi + 0.010, 0.10])), "within the 15 mm tolerance: applies"
    assert not t._near_rim(np.array([x_hi + 0.036, 0.10])), "36 mm PAST it: no bypass"
    assert not t._near_rim(np.array([x_hi + 0.036, 0.113])), "... nor outside the mouth"
    assert not t._near_rim(np.array([0.60, y_hi + 0.04])), "past a side rail: no bypass"


def test_the_phantom_no_longer_skips_the_nis_gate(env_info):
    """The logged phantom, V (1.14, 0.113): 36 mm past the far inset rim.

    It used to be waved through as "near the rim" and moved the estimate ~306 mm in a single
    frame. Now it reaches the NIS gate like any other implausible measurement, and a settled
    filter rejects it into a predict-only coast.
    """
    t = _settled(env_info)
    predicted, _ = t.predict(t.state.copy(), t.P.copy())
    before = t.state.copy()
    phantom = np.array([1.14, 0.113, 0.0])
    assert not t._near_rim(phantom[:2]), "precondition: the bypass must not apply"
    t.step(phantom, R_VISION)
    assert t._reject_count == 1, "it must be REJECTED, not fused"
    assert np.array_equal(t.state, predicted), "a rejected measurement only coasts"
    assert np.linalg.norm(t.state[:2] - before[:2]) < 0.01      # (was ~0.306 m)
    assert np.linalg.norm(t.state[2:4]) < 0.05                  # (was ~0.9 m/s)


def test_a_mistimed_bounce_still_gets_its_bypass(env_info):
    """The behaviour 2c must NOT break: a real puck one step after an unmodelled bounce sits
    beside a rim with a huge innovation, and rejecting it into multi-step drift is what the
    bypass exists to prevent. (`test_kalman_variable_dim.test_nis_gate_bypassed_near_rim` pins
    the fusion itself; this pins that the predicate still admits it.)"""
    t = PuckTracker(env_info, agent_id=1)
    _, _, _, y_hi = t._box()
    assert t._near_rim(np.array([0.30, y_hi - 0.02]))
    assert t._near_rim(np.array([0.30, y_hi - 0.045]))


def test_a_repeating_phantom_can_still_relock_onto_itself(env_info):
    """The residual exposure left by NOT taking Fix 2b, pinned so it is not a surprise.

    The first phantom frame is rejected and recorded; a second, identical one is "consistent"
    with it, which is precisely what the two-point re-lock treats as a real new track -- so the
    filter re-locks ONTO the phantom. A tracker-level off-table drop would stop this, but it
    cannot coexist with the NIS/re-lock contract `test_kalman_variable_dim` pins.

    This is why the pixel MASK is the primary defence and not a convenience: upstream of the
    filter, the object never becomes a measurement in the first place.
    """
    t = _settled(env_info)
    phantom = np.array([1.14, 0.113, 0.0])
    t.step(phantom, R_VISION)
    t.step(phantom, R_VISION)
    assert t.state[0] == pytest.approx(1.14, abs=1e-6), \
        "documents the gap the mask closes: the filter does re-lock onto a repeated phantom"


def test_ground_truth_path_is_untouched(env_info):
    """R=None is the trusted channel: it never enters the gate, so sim stays byte-identical."""
    t = _settled(env_info)
    before = t.state[0]
    t.step(np.array([1.14, 0.113, 0.0]))            # no R -> unconditional update
    assert t._reject_count == 0
    assert t.state[0] - before > 0.4, "the GT update must still have been applied"


# --------------------------------------------------------------------- 2a: the pixel mask
def _synthetic_calibration(resolution=(640, 480)):
    """A camera 1.2 m above the table centre looking straight down, no distortion.

    Enough for the mask geometry: the outline projects to a rectangle in the middle of the
    frame, so "inside the table" and "outside it" are unambiguous in pixels.
    """
    from mini_platform.hw.real_camera import CameraCalibration

    w, h = resolution
    f = 500.0
    K = np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])
    # World +x -> image +x, world +y -> image -y (camera looking down -z).
    R_wc = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return CameraCalibration(K=K, dist=np.zeros(5), R_wc=R_wc,
                             C_world=np.array([0.0, 0.0, 1.2]), resolution=resolution)


class _StillCamera:
    """A capture that always returns the same frame."""

    def __init__(self, frame):
        self._frame = frame
        self.frame_seq = 1
        self.frame_time = None

    def read_rgb(self):
        return self._frame.copy()

    def close(self):
        pass


def test_the_mask_keeps_the_table_and_drops_the_object(env_info):
    """The headline: a BIG red disc off the table and a SMALLER one on it.

    Unmasked, the detector picks the bigger blob -- which is exactly how the object won 158
    frames. Masked, only the on-table puck survives to be detected at all.
    """
    cv2 = pytest.importorskip("cv2")
    from mini_platform.hw.real_sensor import RealPuckSensor
    from mini_platform.vision.detector import detect_puck

    calib = _synthetic_calibration()
    w, h = calib.resolution
    table = dict(length=env_info["table"]["length"], width=env_info["table"]["width"],
                 goal_width=env_info["table"]["goal_width"],
                 radius=env_info["puck"]["radius"])

    frame = np.full((h, w, 3), 30, np.uint8)
    probe = RealPuckSensor(_StillCamera(frame), calib, table=table)
    on_uv = probe.camera.project(np.array([0.20, 0.05, 0.007]))      # on the table
    off_uv = probe.camera.project(np.array([0.62, 0.11, 0.007]))     # past the far end
    assert on_uv is not None and off_uv is not None
    cv2.circle(frame, (int(on_uv[0]), int(on_uv[1])), 8, (200, 30, 30), -1)
    cv2.circle(frame, (int(off_uv[0]), int(off_uv[1])), 20, (200, 30, 30), -1)   # BIGGER

    unmasked = detect_puck(frame, backend="numpy")
    assert unmasked is not None
    assert np.linalg.norm(np.asarray(unmasked.centroid) - off_uv) < 3.0, \
        "pins the bug: unmasked, the bigger OFF-TABLE blob is what gets detected"

    sensor = RealPuckSensor(_StillCamera(frame), calib, table=table)
    masked = detect_puck(sensor.render(), backend="numpy")
    assert sensor.mask_keep_frac is not None, "the mask must have been built"
    assert masked is not None, "the on-table puck must still be found"
    assert np.linalg.norm(np.asarray(masked.centroid) - on_uv) < 3.0
    assert np.array_equal(sensor._last_raw_frame, frame), "--view needs the unmasked frame"


def test_the_mask_keeps_the_goal_mouths(env_info):
    """A shot on target carries the puck past the end line, and with --ekf-own-goal the model
    follows it there -- so the mask must not clip the mouths."""
    pytest.importorskip("cv2")
    from mini_platform.hw.real_sensor import RealPuckSensor

    calib = _synthetic_calibration()
    w, h = calib.resolution
    table = dict(length=env_info["table"]["length"], width=env_info["table"]["width"],
                 goal_width=env_info["table"]["goal_width"],
                 radius=env_info["puck"]["radius"])
    sensor = RealPuckSensor(_StillCamera(np.full((h, w, 3), 30, np.uint8)), calib, table=table)
    sensor.render()
    assert sensor._mask is not None
    half_l = env_info["table"]["length"] / 2
    for x in (-half_l - 0.03, half_l + 0.03):                   # 30 mm into each mouth
        uv = sensor.camera.project(np.array([x, 0.0, 0.007]))
        assert sensor._mask[int(round(uv[1])), int(round(uv[0]))] > 0, \
            f"the mask clipped the goal mouth at world x={x:+.3f}"


def test_no_table_geometry_leaves_the_frame_untouched(env_info):
    """Fail-open by construction: no geometry -> exactly the pre-mask behaviour."""
    from mini_platform.hw.real_sensor import RealPuckSensor

    calib = _synthetic_calibration()
    w, h = calib.resolution
    frame = np.full((h, w, 3), 30, np.uint8)
    frame[10:20, 10:20] = (200, 30, 30)                # outside any plausible table region
    sensor = RealPuckSensor(_StillCamera(frame), calib, table=None)
    assert np.array_equal(sensor.render(), frame)
    assert sensor.mask_keep_frac is None
