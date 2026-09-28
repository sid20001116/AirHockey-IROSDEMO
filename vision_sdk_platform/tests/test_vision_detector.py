"""Vision Parts 1-2: the pinhole camera against its own math (GL-free) and against
rendered pixels (GL-gated via the conftest `require_gl` fixture)."""

import mujoco
import numpy as np
import pytest

import mini_platform
from mini_platform.vision.camera import PUCK_TOP_Z, PinholeCamera

RESOLUTION = (640, 360)


@pytest.fixture(scope="module")
def env():
    e = mini_platform.make("mini-defend")
    e.reset()
    return e.base_env


@pytest.fixture(scope="module")
def cam(env):
    return PinholeCamera(env._model, env._data, "table_side_left", RESOLUTION)


@pytest.fixture(scope="module")
def sensor(env, require_gl):
    from mini_platform.vision.sensor import PuckVisionSensor
    s = PuckVisionSensor(env._model, env._data, resolution=RESOLUTION)
    yield s
    s.close()


def teleport(env, x, y, yaw=0.0):
    for name, val in (("puck_x", x), ("puck_y", y), ("puck_yaw", yaw)):
        env._data.qpos[env._model.joint(name).qposadr[0]] = val
    mujoco.mj_forward(env._model, env._data)


# ---------------------------------------------------------------------------- GL-free

def test_pixel_round_trip(cam):
    """project(pixel_to_plane(u, v)) must return (u, v) to 1e-6 wherever the ray lands.

    At this 45-deg-down, fovy-80 camera the top rows and upper corners point above the
    horizon and pixel_to_plane correctly returns None there -- those are skipped, and the
    assertion on `tested` proves the loop exercised real geometry rather than vacuously
    passing."""
    w, h = RESOLUTION
    tested = 0
    for u in np.linspace(5.0, w - 5.0, 7):
        for v in np.linspace(5.0, h - 5.0, 7):
            X = cam.pixel_to_plane(u, v)
            if X is None:
                continue
            uv = cam.project(X)
            assert uv is not None
            np.testing.assert_allclose(uv, (u, v), atol=1e-6)
            tested += 1
    assert tested >= 4


def test_fd_jacobian_consistency(cam):
    """A +1 px horizontal probe must move the plane point by ~J[:, 0] (20% tolerance:
    the FD stencil is +-0.5 px, the probe a full pixel, so curvature costs a few %)."""
    u, v = 320.0, 300.0
    J = cam.fd_jacobian(u, v)
    assert J is not None
    X0 = cam.pixel_to_plane(u, v)
    X1 = cam.pixel_to_plane(u + 1.0, v)
    np.testing.assert_allclose(X1[:2] - X0[:2], J[:, 0], rtol=0.2)


# --------------------------------------------------------------------------- GL-gated

def test_projection_matches_rendered_blob(env, cam, sensor, vision_viewer):
    """The camera model and the renderer must agree on where the puck is.

    Projecting the GT puck-top centre is compared against the rendered red blob's
    centroid. The blob includes the puck's near-side WALL pixels (the far wall hides
    under the top face), which drag the centroid ~2 px toward the camera (larger v) --
    a range-independent wall/top area ratio, measured 1.75-2.3 px over the table at
    640x360. That bias is real signal (Part 2's kappa corrects it in world space), so
    the assertion is directional: u agrees sub-pixel at the camera's own x, and the
    projection sits ABOVE the centroid by the known bounded amount. An intrinsics error
    (3x focal from the declared 1080p), a Renderer(H, W) swap, or an axis-sign flip all
    miss by tens of pixels and fail both clauses."""
    from scipy import ndimage

    x_gt, y_gt = 0.0, 0.0
    teleport(env, x_gt, y_gt, 0.0)
    rgb = sensor.render()
    vision_viewer(rgb, "projection vs rendered blob (table centre)")
    assert rgb.shape == (RESOLUTION[1], RESOLUTION[0], 3)

    im = rgb.astype(np.int16)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    red = (r > 60) & (r > 1.6 * g) & (r > 1.6 * b)
    # Fill the blue-marker and specular holes so the centroid is the disc's, not the
    # rim's, and keep only the largest component (stray 1-2 px specks exist).
    red = ndimage.binary_fill_holes(red)
    lbl, n = ndimage.label(red)
    assert n >= 1
    largest = np.argmax(ndimage.sum_labels(np.ones_like(lbl), lbl, range(1, n + 1))) + 1
    ys, xs = np.nonzero(lbl == largest)
    assert xs.size >= 20
    centroid = np.array([xs.mean(), ys.mean()])

    uv = cam.project(np.array([x_gt, y_gt, PUCK_TOP_Z]))
    assert uv is not None
    du, dv = np.asarray(uv) - centroid
    assert abs(du) < 1.0, (uv, centroid)
    assert -3.5 < dv < 0.5, (uv, centroid)


# ----------------------------------------------------- Part 2, GL-free (synthetic)

def _synthetic(marker_at=(8, 0), holes=(), checker=False):
    """320x240 image: red disc (r=15) at (160, 120), blue dot (r=4) at disc+marker_at,
    optional white 'specular' holes (offsets from disc centre), optional blue-grey
    checker background in the scene's own colours (which PASS the blue ratio test)."""
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


def test_detector_basic():
    from mini_platform.vision.detector import detect_puck
    det = detect_puck(_synthetic())
    assert det is not None
    np.testing.assert_allclose(det.centroid, (160, 120), atol=0.5)
    assert det.marker_centroid is not None
    np.testing.assert_allclose(det.marker_centroid, (168, 120), atol=0.5)
    assert det.area > 500


def test_detector_specular_hole_neutralized():
    """A white off-centre highlight punches a hole in the ratio mask; fill_holes must
    keep the centroid on the disc centre (unfilled, this hole shifts it ~1 px)."""
    from mini_platform.vision.detector import detect_puck
    det = detect_puck(_synthetic(holes=((-6, 0, 6),)))
    assert det is not None
    np.testing.assert_allclose(det.centroid, (160, 120), atol=0.3)


def test_detector_blue_background_does_not_leak():
    """The scene's blue-grey checker PASSES the blue ratio test; the marker search and
    the mask union must stay confined to the puck's neighbourhood."""
    from mini_platform.vision.detector import detect_puck
    det = detect_puck(_synthetic(checker=True))
    assert det is not None
    np.testing.assert_allclose(det.centroid, (160, 120), atol=0.5)
    np.testing.assert_allclose(det.marker_centroid, (168, 120), atol=0.5)


def test_detector_none_without_puck():
    from mini_platform.vision.detector import detect_puck
    im = np.full((240, 320, 3), 120, np.uint8)
    assert detect_puck(im) is None
    yy, xx = np.mgrid[:240, :320]
    cells = ((xx // 20) + (yy // 20)) % 2
    im[cells == 0] = (51, 77, 102)
    assert detect_puck(im) is None


# ---------------------------------------------------- Part 2, GL-gated (rendered)

#: Canonical measurement poses: spread over both halves, clear of the parked mallet
#: (which occludes the sight line around (-0.45, -0.15)) and of the far corners the
#: frustum misses.
CANONICAL_POSES = [(0.0, 0.0), (0.3, 0.1), (-0.3, 0.1), (0.3, -0.1),
                   (-0.3, -0.05), (0.4, 0.0), (-0.2, -0.15), (0.15, 0.18)]


def test_measure_position_accuracy(env, sensor, vision_viewer):
    for x, y in CANONICAL_POSES:
        teleport(env, x, y, 0.3)
        m = sensor.measure()
        vision_viewer(sensor.render(), f"measure() at ({x:+.2f}, {y:+.2f})")
        assert m is not None, (x, y)
        err = np.linalg.norm(m.pos_xy - np.array([x, y]))
        assert err < 0.005, (x, y, err, m.pos_xy)


def test_measure_yaw_world_frame(env, sensor, vision_viewer):
    """Yaw against the puck_yaw qpos directly: pins the WORLD-frame sign convention
    before any robot-frame conversion exists (the marker sits on the puck's local +x,
    so atan2 of the back-projected offset must equal the joint angle). Yaws where the
    marker outline aliases open onto the silhouette return None by design (the 2-D
    fallback; grid availability is the static sweep's metric) -- but every yaw that IS
    reported must be accurate, and two accurate yaws at well-separated angles already
    rule out a sign flip (a flipped sign errs by 2*yaw, 0.8-2.4 rad here)."""
    available = 0
    for yaw in (0.4, 1.2, -0.8, 2.9):
        teleport(env, 0.1, 0.0, yaw)
        m = sensor.measure()
        vision_viewer(sensor.render(), f"yaw pin: qpos yaw {yaw:+.2f}")
        assert m is not None, yaw
        if m.yaw is None:
            continue
        available += 1
        err = (m.yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        assert abs(err) < 0.15, (yaw, m.yaw, err)
        assert m.R_yaw is not None
    assert available >= 2


def test_measure_none_off_table(env, sensor, vision_viewer):
    teleport(env, 3.0, 3.0, 0.0)
    m = sensor.measure()
    vision_viewer(sensor.render(), "puck off-table: expect DROPOUT (red border)")
    assert m is None
