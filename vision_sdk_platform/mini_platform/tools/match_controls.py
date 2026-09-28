"""Clickable on-screen controls for the live MuJoCo viewer.

The buttons are drawn INTO the MuJoCo window rather than a side panel, and that is forced
by mushroom_rl's viewer: `MujocoViewer.render()` blocks in `while self._paused:
render_inner_loop(self)`, so a tk/matplotlib panel would have its own event loop frozen at
exactly the moment you need the Play button. That spin loop does keep calling
`glfw.poll_events()`, so a glfw callback on the render window is the one input path that
still works while paused.

Two hooks, both already-established patterns in this package: `custom_render_callback`
(a documented `MujocoViewer` constructor arg) paints the strip, and class-level patches of
`keyboard` / `mouse_button` route input -- the same monkeypatch approach `mini_play` already
uses for MuJoCo-3 compatibility.

Offline tool: never import from the runtime path.
"""

from dataclasses import dataclass
from typing import Callable

import glfw
import mujoco
import numpy as np

#: Button strip geometry in pixels from the window's bottom-left. MuJoCo's `MjrRect` uses the
#: OpenGL bottom-left origin; glfw's cursor uses a top-left origin, hence the flip in `click`.
BTN_W, BTN_H, GAP, MARGIN = 150, 40, 10, 14

#: The one control strip bound to this process's viewer. Resolving the viewer lazily through
#: this (rather than binding it at construction) matters: with `--paused` the viewer blocks
#: inside its very first `render()`, so there is no post-render moment to bind in -- the
#: Play button would be dead before the loop ever started.
_ACTIVE = None
_PATCHED = False


@dataclass
class Button:
    label: str
    action: Callable[[], None]
    colour: tuple           # (r, g, b) fill
    key: int | None = None  # glfw key that fires the same action, or None
    lit: Callable[[], bool] = lambda: False  # drawn brighter while this is true


def nudge_puck(base_env, speed):
    """Kick a stalled puck back into play by overwriting its planar velocity.

    Writing `qvel` directly is safe from a glfw callback -- it is a plain array store that
    the next `mj_step` picks up, whether that step happens now or after you un-pause.

    Table geometry: x spans +-0.508 (robot 1 defends -x, robot 2 defends +x), y spans +-0.254.
    """
    model, data = base_env._model, base_env._data
    dof_x = model.joint("puck_x").dofadr[0]
    dof_y = model.joint("puck_y").dofadr[0]
    pos = np.array([data.qpos[model.joint("puck_x").qposadr[0]],
                    data.qpos[model.joint("puck_y").qposadr[0]]])
    vel = np.array([data.qvel[dof_x], data.qvel[dof_y]])

    # TODO(human): choose the unit direction to kick the puck in.
    direction = np.array([1.0, 0.0])

    data.qvel[dof_x], data.qvel[dof_y] = speed * direction


class MatchControls:
    """PLAY / PAUSE / RESET / HIT PUCK, plus a live status strip above them."""

    def __init__(self, env, nudge_speed=1.0):
        global _ACTIVE
        self._env = env
        self._nudge_speed = nudge_speed
        self.reset_requested = False
        self.nudges = 0
        self._rects = []

        model = env.base_env._model
        self._puck_dofs = [model.joint(f"puck_{a}").dofadr[0] for a in ("x", "y")]

        self.buttons = [
            Button("PLAY   spc", self._play, (0.13, 0.50, 0.20),
                   lit=lambda: not self.paused),
            Button("PAUSE  spc", self._pause, (0.62, 0.45, 0.05),
                   lit=lambda: self.paused),
            Button("RESET    r", self._reset, (0.20, 0.34, 0.58), key=glfw.KEY_R),
            Button("HIT PUCK x", self._nudge, (0.62, 0.16, 0.16), key=glfw.KEY_X),
        ]
        _ACTIVE = self
        _install_patches()

    # ------------------------------------------------------------------ viewer access
    @property
    def viewer(self):
        """The live viewer, or None before the first `render()` created it."""
        return self._env.base_env._viewer

    @property
    def paused(self):
        v = self.viewer
        return bool(v is not None and v._paused)

    # --------------------------------------------------------------------- actions
    def _play(self):
        if self.viewer is not None:
            self.viewer._paused = False

    def _pause(self):
        if self.viewer is not None:
            self.viewer._paused = True

    def _reset(self):
        # The main loop cannot see this flag while the viewer spins in its pause loop, so a
        # reset necessarily resumes; hit PAUSE again once the new episode is up.
        self.reset_requested = True
        self._play()

    def _nudge(self):
        nudge_puck(self._env.base_env, self._nudge_speed)
        self.nudges += 1

    def take_reset(self):
        """True once per RESET press; clears the flag."""
        pending, self.reset_requested = self.reset_requested, False
        return pending

    # ----------------------------------------------------------------------- input
    def click(self, window):
        """Dispatch a left-press at the cursor. True if it landed on a button."""
        x, y = glfw.get_cursor_pos(window)
        y = glfw.get_window_size(window)[1] - y      # glfw top-left -> MuJoCo bottom-left
        for rect, button in self._rects:
            if (rect.left <= x < rect.left + rect.width
                    and rect.bottom <= y < rect.bottom + rect.height):
                button.action()
                return True
        return False

    # ------------------------------------------------------------------- rendering
    def draw(self, viewport, context):
        """`custom_render_callback`: paint the strip over the finished frame."""
        self._rects = []
        x = MARGIN
        for button in self.buttons:
            rect = mujoco.MjrRect(x, MARGIN, BTN_W, BTN_H)
            mujoco.mjr_rectangle(rect, *button.colour, 0.95 if button.lit() else 0.55)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT, rect,
                               button.label, None, context)
            self._rects.append((rect, button))
            x += BTN_W + GAP

        speed = float(np.linalg.norm(self._env.base_env._data.qvel[self._puck_dofs]))
        status = mujoco.MjrRect(MARGIN, MARGIN + BTN_H + GAP, x - MARGIN - GAP, BTN_H)
        mujoco.mjr_rectangle(status, 0.0, 0.0, 0.0, 0.55)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, status,
            f"{'PAUSED' if self.paused else 'RUNNING'}   puck {speed:5.2f} m/s"
            f"   nudges {self.nudges}", None, context)


def draw_active(viewport, context):
    """`custom_render_callback` for the viewer, resolved through `_ACTIVE`.

    Passing this instead of a bound method breaks the chicken-and-egg: `MatchControls` needs
    the env, but the env wants the callback in its `viewer_params` at construction.
    """
    if _ACTIVE is not None:
        _ACTIVE.draw(viewport, context)


def _install_patches():
    """Route glfw key/click events through `_ACTIVE` before the viewer's own handlers."""
    global _PATCHED
    if _PATCHED:
        return
    from mushroom_rl.utils.mujoco.viewer import MujocoViewer

    orig_keyboard = MujocoViewer.keyboard
    orig_mouse_button = MujocoViewer.mouse_button

    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.RELEASE and _ACTIVE is not None:
            for button in _ACTIVE.buttons:
                if button.key is not None and key == button.key:
                    button.action()
                    return
        orig_keyboard(self, window, key, scancode, act, mods)

    def mouse_button(self, window, button, act, mods):
        if (_ACTIVE is not None and button == glfw.MOUSE_BUTTON_LEFT
                and act == glfw.PRESS and _ACTIVE.click(window)):
            return      # swallowed: leaves button_left False, so no camera drag starts
        orig_mouse_button(self, window, button, act, mods)

    MujocoViewer.keyboard = keyboard
    MujocoViewer.mouse_button = mouse_button
    _PATCHED = True
