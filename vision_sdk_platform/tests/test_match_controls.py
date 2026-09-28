"""The on-screen viewer controls (`tools/match_controls.py`).

GL-free by construction: `MatchControls` never touches the renderer outside `draw`, and the
click path is pure arithmetic once the two glfw queries are stubbed. What is worth pinning is
the y-axis flip -- glfw reports the cursor from the window's TOP-left, MuJoCo's `MjrRect` is
anchored at the BOTTOM-left, so an unflipped hit test silently targets the mirror image of
every button and only shows up as "the buttons do nothing".
"""

import numpy as np
import pytest

import mini_platform
from mini_platform.tools import match_controls
from mini_platform.tools.match_controls import BTN_H, BTN_W, GAP, MARGIN, MatchControls

WIN_H = 720


class FakeWindow:
    """Stands in for a glfw window handle; `cursor` is in glfw's top-left convention."""

    def __init__(self, cursor):
        self.cursor = cursor


@pytest.fixture
def controls(monkeypatch):
    monkeypatch.setattr(match_controls.glfw, "get_cursor_pos", lambda w: w.cursor)
    monkeypatch.setattr(match_controls.glfw, "get_window_size", lambda w: (1280, WIN_H))
    env = mini_platform.make("mini-hit-sdk-2p")
    ctl = MatchControls(env, nudge_speed=1.0)
    # draw() is what populates the hit-test rects; call the geometry half directly rather
    # than standing up a GL context.
    ctl._rects = [(match_controls.mujoco.MjrRect(MARGIN + i * (BTN_W + GAP), MARGIN,
                                                 BTN_W, BTN_H), b)
                  for i, b in enumerate(ctl.buttons)]
    return ctl


def button_centre(index):
    """Cursor position (glfw top-left origin) over the middle of button `index`."""
    x = MARGIN + index * (BTN_W + GAP) + BTN_W / 2
    return x, WIN_H - (MARGIN + BTN_H / 2)


def test_registered_buttons():
    labels = [b.label.split()[0] for b in MatchControls(mini_platform.make("mini-hit")).buttons]
    assert labels == ["PLAY", "PAUSE", "RESET", "HIT"]


def test_click_dispatches_to_the_button_under_the_cursor(controls):
    assert controls.click(FakeWindow(button_centre(3)))     # HIT PUCK
    assert controls.nudges == 1
    assert controls.click(FakeWindow(button_centre(2)))     # RESET
    assert controls.take_reset()


def test_click_outside_the_strip_is_not_swallowed(controls):
    """A miss must return False so the viewer still gets the event and orbits the camera."""
    assert not controls.click(FakeWindow((640, 300)))
    assert controls.nudges == 0


def test_hit_test_is_not_vertically_mirrored(controls):
    """The exact bug the flip prevents: the button's mirror-image row must be a miss."""
    x, y = button_centre(3)
    assert controls.click(FakeWindow((x, y)))
    assert not controls.click(FakeWindow((x, WIN_H - y)))
    assert controls.nudges == 1


def test_take_reset_fires_once(controls):
    assert not controls.take_reset()
    controls._reset()
    assert controls.take_reset()
    assert not controls.take_reset()


def test_nudge_gives_a_stalled_puck_the_requested_speed(controls):
    """The whole point of the button: whatever direction the policy picks, the puck must
    leave with `nudge_speed`, from rest and from a slow drift alike."""
    data = controls._env.base_env._data
    controls._env.reset()
    for start in ([0.0, 0.0], [0.02, -0.01]):
        data.qvel[controls._puck_dofs] = start
        controls._nudge()
        assert np.linalg.norm(data.qvel[controls._puck_dofs]) == pytest.approx(1.0)


def test_pause_state_is_safe_before_the_viewer_exists(controls):
    """Buttons are wired up before the first render() builds the viewer; nothing may raise."""
    assert controls.viewer is None
    assert controls.paused is False
    controls._play()
    controls._pause()
