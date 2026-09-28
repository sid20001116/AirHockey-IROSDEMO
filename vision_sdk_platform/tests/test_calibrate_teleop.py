"""hw.calibrate_teleop: base frame + tool length fitted from teleop touches at taped marks.

The joint configurations are the real arm's, read off the rig on 2026-09-15, so the wrist tilts
in these tests are the ones the fit has to cope with: ~42 deg leaning BACK with the mallet on the
goal line, ~39 deg leaning OUT at the table centre, ~2 deg at the taught park. The world marks
are synthesised from a known base pose and tool length, and the fit must recover both.
"""

import numpy as np
import pytest

from mini_platform import make
from mini_platform.core.kinematics import forward_kinematics
from mini_platform.hw.calibrate_teleop import (MAX_RESIDUAL_MM, contact_point, fit_base_frame,
                                               tilt_deg)

Q_GOAL = [-0.1251, -1.9237, 1.2572, -1.6322, -0.1111, -1.8018]
Q_CENTRE = [0.0633, -2.1869, 1.7267, -0.4338, 0.1308, -1.6521]
Q_PARK = [0.0691, -1.6401, 1.2167, -1.1676, -0.0296, 1.6753]

L_TRUE = 0.092          # what the 2026-09-15 goal/centre pair implied
YAW_TRUE = np.radians(-0.7)


def _fk(env_info, q):
    pos, rot = forward_kinematics(env_info["robot"]["robot_model"], env_info["robot"]["robot_data"],
                                  np.asarray(q, float))
    return np.array(pos, float), np.array(rot, float)


@pytest.fixture(scope="module")
def touches():
    env_info = make("mini-defend").env_info
    qs = [Q_GOAL, Q_CENTRE, Q_PARK,
          np.add(Q_PARK, [0.35, 0, 0, 0, 0, 0]),       # park swung toward one rail
          np.add(Q_PARK, [-0.35, 0, 0, 0, 0, 0])]      # and toward the other
    pos, rot = map(np.array, zip(*(_fk(env_info, q) for q in qs)))
    T = np.eye(4)
    T[:2, :2] = [[np.cos(YAW_TRUE), -np.sin(YAW_TRUE)], [np.sin(YAW_TRUE), np.cos(YAW_TRUE)]]
    T[:3, 3] = [-0.62, 0.004, -0.05]
    contacts = pos + L_TRUE * rot[:, :, 2]
    world = (contacts @ T[:3, :3].T + T[:3, 3])[:, :2]
    return env_info, pos, rot, world, T


def test_tilts_are_the_real_ones(touches):
    _, _, rot, _, _ = touches
    assert tilt_deg(rot[0]) == pytest.approx(42.1, abs=0.2)     # goal: leaning back
    assert tilt_deg(rot[1]) == pytest.approx(39.4, abs=0.2)     # centre: leaning out
    assert tilt_deg(rot[2]) == pytest.approx(2.1, abs=0.2)      # park: nearly vertical


def test_fit_recovers_base_pose_and_tool_length(touches):
    _, pos, rot, world, T = touches
    T_fit, info = fit_base_frame(pos, rot, world)
    assert T_fit[:2, 3] == pytest.approx(T[:2, 3], abs=1e-6)
    assert np.radians(info["yaw_deg"]) == pytest.approx(YAW_TRUE, abs=1e-6)
    assert info["tool_length"] == pytest.approx(L_TRUE, abs=1e-6)
    assert info["residual_m"].max() < 1e-6
    assert T_fit[:3, 2] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)    # upright by construction


def test_a_fixed_tool_length_is_respected(touches):
    _, pos, rot, world, T = touches
    T_fit, info = fit_base_frame(pos, rot, world, tool_length=L_TRUE)
    assert info["tool_length"] == L_TRUE and info["tool_length_fitted"] is False
    assert T_fit[:2, 3] == pytest.approx(T[:2, 3], abs=1e-6)


def test_ignoring_the_tool_leaves_centimetre_residuals(touches):
    """The 2026-09-15 failure: jaw points at 40 deg tilt are not where the mallet touches."""
    _, pos, rot, world, _ = touches
    _, info = fit_base_frame(pos, rot, world, tool_length=0.0)
    assert info["residual_m"].max() > 0.020


def test_a_mis_measured_mark_shows_in_the_residuals(touches):
    _, pos, rot, world, _ = touches
    bad = world.copy()
    bad[3, 0] += 0.030                                  # one mark taped 30 mm off
    _, info = fit_base_frame(pos, rot, bad)
    assert info["residual_m"].max() * 1000 > MAX_RESIDUAL_MM


def test_tool_length_is_refused_when_the_wrist_never_changed_tilt(touches):
    """Near-vertical touches cannot tell a longer tool from a shifted base."""
    _, pos, rot, world, _ = touches
    with pytest.raises(ValueError, match="tool length"):
        fit_base_frame(pos[2:], rot[2:], world[2:])
    # ... but with the length supplied the same touches are fine
    _, info = fit_base_frame(pos[2:], rot[2:], world[2:], tool_length=L_TRUE)
    assert info["residual_m"].max() < 1e-6


def test_contact_point_moves_down_the_tool_axis(touches):
    _, pos, rot, _, _ = touches
    assert contact_point(pos[2], rot[2], 0.0) == pytest.approx(pos[2])
    c = contact_point(pos[0], rot[0], 0.1)
    assert np.linalg.norm(c - pos[0]) == pytest.approx(0.1)
    assert c[2] < pos[0][2]                             # the tool points down
