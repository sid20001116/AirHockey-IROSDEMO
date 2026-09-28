"""Mujoco-free description of the mini table in the EKF's own frame (offline EKF tooling).

:class:`~mini_platform.estimation.kalman_filter.PuckTracker` reads exactly six numbers out of
the full ``env_info`` dict: the table length/width/goal-width, the puck and mallet radii, the
control ``dt``, and the x-offset of the frame the puck state is expressed in. Building that
dict through :func:`mini_platform.make` pulls in mujoco + mushroom_rl and a whole MuJoCo
scene, which the *offline* EKF tools (``tools/fit_mini_table``, ``tools/ekf_replay``) and the
*camera-only* roll capture have no use for -- and which is not installed on a box that only
has a camera. This module reproduces those numbers directly.

The copy cannot drift: ``tests/test_mini_env.py`` pins every constant here against the live
``env_info`` / ``baseline_mini`` values, and fails in the sim env if either moves.

Frames
------
The camera calibration puts the WORLD origin at the **table centre**, +x toward the far end,
+z up out of the surface (``hw/calibrate_robot.py`` phase 2). The agent's EKF runs in the
virtual end-mount frame **V**, whose world pose is ``baseline_mini.virtual_base_frame(1, z)``
-- for agent 1 a pure **translation** by ``(-V_SETBACK, 0)``. World -> V is therefore
``xy + (V_SETBACK, 0)``:

* the two frames share their axes, so every **velocity**, every per-axis **damping**
  constant and every rim-local **bounce** quantity is numerically identical in both;
* only positions shift, by a constant.

That is what lets the roll captures be logged in world xy, fitted in world xy, and the fits
used verbatim by the EKF in V. (The measured ``base_frame`` never enters: the runtime maps
world -> robot -> V, and that composite collapses to ``T_wv^-1`` -- the robot frame cancels.)
"""

import numpy as np

#: Table geometry, metres. Mirrors ``envs/env_base.py`` (rim inner edges of the 40x20 in
#: table) -- pinned equal by ``tests/test_mini_env.py``.
TABLE_LENGTH = 1.016
TABLE_WIDTH = 0.508
GOAL_WIDTH = 0.172                  # MEASURED on the real table 2026-09-15 (was 0.122351); keep in step with assets/mini_table.xml rims and the other copy in envs/env_base.py
PUCK_RADIUS = 0.025
MALLET_RADIUS = 0.025

#: Control period, s. ``env_info['dt']`` = MuJoCo timestep (1 ms) x n_intermediate_steps (20).
DT = 0.02

#: Origin of the virtual planning frame V behind the own goal line, metres. Mirrors
#: ``agents/baseline_mini.V_SETBACK``.
V_SETBACK = 0.62094


def v_env_info(agent_id=1, dt=DT):
    """The subset of ``env_info`` :class:`PuckTracker` reads, expressed in frame V.

    ``base_frame`` carries only what the tracker uses of it -- ``abs(T[0, 3])``, the distance
    from the frame origin to the table centre -- so the returned 4x4 is
    ``virtual_base_frame(1, 0)`` exactly.
    """
    if agent_id != 1:
        raise ValueError("only agent 1 is supported offline (the hardware rig is single-arm; "
                         "agent 2's V frame adds a 180 deg rotation that the world->V shift "
                         "helpers below do not model)")
    T = np.eye(4)
    T[0, 3] = -V_SETBACK
    return {
        'puck': {'radius': PUCK_RADIUS},
        'mallet': {'radius': MALLET_RADIUS},
        'dt': float(dt),
        'table': {'length': TABLE_LENGTH, 'width': TABLE_WIDTH, 'goal_width': GOAL_WIDTH},
        'robot': {'base_frame': [T]},
    }


def world_to_v(xy):
    """World table-centre xy (the capture frame) -> frame V. Pure translation; (N, 2) or (2,)."""
    return np.asarray(xy, dtype=float) + np.array([V_SETBACK, 0.0])


def v_to_world(xy):
    """Frame V -> world table-centre xy. Inverse of :func:`world_to_v`."""
    return np.asarray(xy, dtype=float) - np.array([V_SETBACK, 0.0])


def rim_lines(puck_radius=PUCK_RADIUS):
    """The four rim lines the bounce model reflects off, in WORLD coordinates.

    Same construction as ``AirHockeyTable.__init__`` (the puck-centre lines, i.e. the rims
    inset by one puck radius), minus the frame offset. Returned as
    ``{'x-': v, 'x+': v, 'y-': v, 'y+': v}`` -- the puck-centre coordinate at contact with
    the near/far short rails and the two long rails.
    """
    return {
        'x-': -TABLE_LENGTH / 2 + puck_radius,
        'x+': TABLE_LENGTH / 2 - puck_radius,
        'y-': -TABLE_WIDTH / 2 + puck_radius,
        'y+': TABLE_WIDTH / 2 - puck_radius,
    }
