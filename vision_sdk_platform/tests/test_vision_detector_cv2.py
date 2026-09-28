"""The OpenCV detector backend must be a drop-in equivalent of the numpy reference path.

The hardware sensor runs ``detect_puck(..., backend="cv2")`` for a ~10x speedup; these pin
that it returns the SAME ``Detection`` the sim's numpy path does -- byte-for-byte on the
``rgb_ratio`` gate (the two full-frame ops it replaces, hole-filling and connected-component
labelling, are exact equivalents of ``scipy.ndimage``), and within OpenCV's hue quantization
on the ``hsv`` gate. If these drift, the hardware detector no longer matches the model the
bias/sigma constants were calibrated on.
"""
import importlib

import numpy as np
import pytest

from mini_platform.vision.detector import (
    PuckColorProfile, detect_puck, puck_mask, _fill_holes_cv2,
)

cv2 = pytest.importorskip("cv2")
ndimage = pytest.importorskip("scipy.ndimage")


# The synthetic image builder mirrors tests/test_vision_detector.py so the two suites probe
# the same shapes (red disc + blue marker, optional specular holes, blue-grey checker bg).
def _synthetic(marker_at=(8, 0), holes=(), checker=False):
    im = np.full((240, 320, 3), 120, np.uint8)
    if checker:
        yy, xx = np.mgrid[:240, :320]
        cells = ((xx // 20) + (yy // 20)) % 2
        im[cells == 0] = (51, 77, 102)
        im[cells == 1] = (26, 51, 77)
    yy, xx = np.mgrid[:240, :320]
    disc = (xx - 160) ** 2 + (yy - 120) ** 2 <= 15 ** 2
    im[disc] = (200, 30, 30)
    for hx, hy, hr in holes:
        hole = (xx - 160 - hx) ** 2 + (yy - 120 - hy) ** 2 <= hr ** 2
        im[hole & disc] = (255, 255, 255)
    mx, my = 160 + marker_at[0], 120 + marker_at[1]
    dot = (xx - mx) ** 2 + (yy - my) ** 2 <= 4 ** 2
    im[dot] = (30, 30, 200)
    return im


def _assert_same_detection(a, b, centroid_atol, area_atol=0):
    assert (a is None) == (b is None), (a, b)
    if a is None:
        return
    np.testing.assert_allclose(a.centroid, b.centroid, atol=centroid_atol)
    assert abs(a.area - b.area) <= area_atol, (a.area, b.area)
    if centroid_atol == 0:
        assert a.bbox == b.bbox, (a.bbox, b.bbox)
    assert (a.marker_centroid is None) == (b.marker_centroid is None)
    if a.marker_centroid is not None:
        np.testing.assert_allclose(a.marker_centroid, b.marker_centroid,
                                   atol=max(centroid_atol, 0.5))


# ------------------------------------------------------------------- rgb_ratio: byte-exact

@pytest.mark.parametrize("kwargs", [
    {}, {"holes": ((-6, 0, 6),)}, {"checker": True}, {"marker_at": (0, 8)},
    {"holes": ((5, -5, 4), (-7, 2, 3))},
])
def test_cv2_matches_numpy_rgb_ratio_exactly(kwargs):
    """The sim gate: cv2 must reproduce the numpy Detection to floating-point precision."""
    im = _synthetic(**kwargs)
    _assert_same_detection(detect_puck(im, backend="numpy"),
                           detect_puck(im, backend="cv2"), centroid_atol=0)


def test_cv2_matches_numpy_none_cases():
    flat = np.full((240, 320, 3), 120, np.uint8)
    assert detect_puck(flat, backend="numpy") is None
    assert detect_puck(flat, backend="cv2") is None
    # blue-grey checker with no puck: both reject
    yy, xx = np.mgrid[:240, :320]
    checker = np.full((240, 320, 3), 120, np.uint8)
    cells = ((xx // 20) + (yy // 20)) % 2
    checker[cells == 0] = (51, 77, 102)
    assert detect_puck(checker, backend="numpy") is None
    assert detect_puck(checker, backend="cv2") is None


def test_cv2_picks_same_largest_component():
    """Two red discs: both backends must select the larger one (same centroid/bbox)."""
    im = np.full((240, 320, 3), 120, np.uint8)
    yy, xx = np.mgrid[:240, :320]
    im[(xx - 90) ** 2 + (yy - 120) ** 2 <= 18 ** 2] = (200, 30, 30)   # big
    im[(xx - 240) ** 2 + (yy - 120) ** 2 <= 9 ** 2] = (200, 30, 30)   # small
    _assert_same_detection(detect_puck(im, backend="numpy"),
                           detect_puck(im, backend="cv2"), centroid_atol=0)


# --------------------------------------------------------------------- hsv: within quant

def test_cv2_matches_numpy_hsv_close():
    """Real-puck HSV gate: OpenCV's 2-deg hue quantization shifts nothing meaningful."""
    hp = PuckColorProfile(mode="hsv", hue_lo=345, hue_hi=15, sat_min=0.45, val_min=0.25,
                          marker_enabled=True, bbox_major_range=(4, 90))
    for kwargs in ({}, {"checker": True}, {"holes": ((-6, 0, 6),)}):
        im = _synthetic(**kwargs)
        _assert_same_detection(detect_puck(im, hp, backend="numpy"),
                               detect_puck(im, hp, backend="cv2"),
                               centroid_atol=0.5, area_atol=6)


def test_hsv_wrapped_band_matches_unwrapped_logic():
    """A hue band that wraps through 0 (lo>hi, the red case) selects the same red disc as
    the numpy _hue_band OR-branch -- the two-inRange path is not silently dropping the wrap."""
    hp = PuckColorProfile(mode="hsv", hue_lo=350, hue_hi=10, sat_min=0.4, val_min=0.2,
                          marker_enabled=False, bbox_major_range=(4, 90))
    a = detect_puck(_synthetic(), hp, backend="numpy")
    b = detect_puck(_synthetic(), hp, backend="cv2")
    assert a is not None and b is not None
    np.testing.assert_allclose(a.centroid, b.centroid, atol=0.5)


# ------------------------------------------------------------- primitive equivalence pins

def test_fill_holes_matches_scipy_exactly():
    """The cv2 floodFill hole-fill is pixel-identical to ndimage.binary_fill_holes, including
    a blob touching the image corner (where the naive (0,0)-seed trick would fail)."""
    yy, xx = np.mgrid[:120, :160]
    m = (xx - 80) ** 2 + (yy - 60) ** 2 <= 30 ** 2
    m &= ~((xx - 80) ** 2 + (yy - 60) ** 2 <= 8 ** 2)       # centred hole
    m &= ~((xx - 92) ** 2 + (yy - 60) ** 2 <= 4 ** 2)       # off-centre hole
    m |= (xx - 0) ** 2 + (yy - 0) ** 2 <= 6 ** 2            # blob at the very corner
    sci = ndimage.binary_fill_holes(m)
    cvb = _fill_holes_cv2(cv2, m.astype(np.uint8) * 255) > 0
    assert np.array_equal(sci, cvb)


def test_puck_mask_backends_agree_rgb_ratio():
    im = _synthetic(checker=True)
    assert np.array_equal(puck_mask(im, backend="numpy"), puck_mask(im, backend="cv2"))


# ------------------------------------------------------------------- backend dispatch

def test_auto_backend_uses_cv2_and_matches():
    im = _synthetic()
    _assert_same_detection(detect_puck(im, backend="numpy"),
                           detect_puck(im, backend="auto"), centroid_atol=0)


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown detect backend"):
        detect_puck(_synthetic(), backend="opencl")
    with pytest.raises(ValueError, match="unknown detect backend"):
        puck_mask(_synthetic(), backend="opencl")


# ------------------------------------------------------------ GL-gated: real rendered frame

def test_cv2_matches_numpy_on_rendered_frame(require_gl):
    """The decisive real-pixel pin: on an actual MuJoCo-rendered puck frame (not synthetic),
    the cv2 rgb_ratio path reproduces the numpy Detection exactly."""
    import mujoco

    import mini_platform
    from mini_platform.vision.sensor import PuckVisionSensor

    env = mini_platform.make("mini-defend")
    env.reset()
    be = env.base_env
    sensor = PuckVisionSensor(be._model, be._data, resolution=(640, 360))
    try:
        matched = 0
        for x, y, yaw in [(0.0, 0.0, 0.3), (0.3, 0.1, 1.2), (-0.3, -0.05, -0.8),
                          (0.15, 0.18, 2.9)]:
            for name, val in (("puck_x", x), ("puck_y", y), ("puck_yaw", yaw)):
                be._data.qpos[be._model.joint(name).qposadr[0]] = val
            mujoco.mj_forward(be._model, be._data)
            frame = sensor.render()
            _assert_same_detection(detect_puck(frame, backend="numpy"),
                                   detect_puck(frame, backend="cv2"), centroid_atol=0)
            matched += 1
        assert matched == 4
    finally:
        sensor.close()
