import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--show-vision", action="store_true", default=False,
        help="pop a window during GL vision tests showing each camera frame with the "
             "detection annotated (green outline/box, centroid, marker); needs "
             "matplotlib and a display. Off by default so CI/headless runs are "
             "untouched.")


@pytest.fixture(scope="session")
def vision_viewer(request):
    """show(rgb, title='', pause=0.4): display an annotated camera frame.

    A no-op lambda unless pytest ran with --show-vision, so tests can call it
    unconditionally at zero cost. With the flag, frames are annotated through the
    same `annotate_detection` the live tool uses -- what you see is what the
    detector picked up."""
    if not request.config.getoption("--show-vision"):
        yield lambda rgb, title="", pause=0.4: None
        return

    import matplotlib.pyplot as plt

    from mini_platform.tools.vision_eval import annotate_detection

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.set_axis_off()
    fig.tight_layout()
    state = {"handle": None}

    def show(rgb, title="", pause=0.4):
        im, _ = annotate_detection(rgb)
        if state["handle"] is None:
            state["handle"] = ax.imshow(im)
        else:
            state["handle"].set_data(im)
        ax.set_title(title, fontsize=9)
        fig.canvas.draw_idle()
        plt.pause(pause)

    yield show
    plt.close(fig)


@pytest.fixture(scope="session")
def require_gl():
    """Skip GL-dependent tests on machines without an offscreen rendering context.

    The physics/maths half of the suite runs anywhere; only tests that render pixels
    request this fixture. On headless machines set MUJOCO_GL=egl (or glfw on a desktop).
    """
    import mujoco
    try:
        r = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>"), 16, 16)
        r.close()
    except Exception as e:
        pytest.skip(f"no offscreen GL context ({type(e).__name__}); try MUJOCO_GL=egl")


@pytest.fixture(autouse=True)
def _sim_home_park():
    """Pin the park to the vendored station for the SIM suite.

    ``baseline_mini.HOME_PARK_XY`` is a per-rig calibration -- a pose teleoped on the real table
    and read back through the arm's own FK (``hw/capture_home.py``), so it is measured against
    the REAL table height. The sim runs at the model grasp height, ~140 mm above it, where that
    same xy is a much longer stretch with no vertical-wrist solution: left in place it changes
    what every sim episode measures, and a rig recalibration would silently move sim assertions.
    So the sim tests run against ``HOME_POS`` and the taught value is asserted on its own, in
    ``test_goal_keepout.test_the_default_base_park_is_the_taught_point_in_the_robot_frame``.
    """
    import numpy as np
    import mini_platform.agents.baseline_mini as bm

    saved = bm.HOME_PARK_XY
    bm.HOME_PARK_XY = np.array([bm.HOME_POS[0], bm.HOME_POS[1]])
    try:
        yield
    finally:
        bm.HOME_PARK_XY = saved
