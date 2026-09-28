"""The table frame and hand-eye checks calibrate_robot builds from mallet touches.

Pure geometry: the touches are synthesised from the mount the sim is built for
(``assets/arm1.xml``) and the recovered frame is compared back against it -- no arm, no camera.

Each regression here is a failure seen on the real rig:

  * 2026-09-14 -- the legacy centre/far-end/side protocol gave a frame rotated 90 deg, because
    from the END mount the far end is 1.135 m out against a 0.740 m reach.
  * 2026-09-15 -- a tape-measured table-centre touch was recorded 92 mm short, and both rail
    touches rode up ~36 mm, tilting a fitted table plane 10.2 deg (+33 mm down-table once
    planarised): reach-probe read +125 mm with the mallet in the goal mouth. The same run's
    hand-eye put the camera 0.43 m BELOW the arm base and nothing flagged it.
"""

import numpy as np
import pytest

from mini_platform import make
from mini_platform.hw.calibrate_robot import (check_hand_eye, homog, inv_homog, report_vs_sim,
                                              sim_camera_relative_to_base,
                                              world_from_goal_and_rails, world_from_touches)

#: Grasp-frame height above the surface with the mallet flat (calibrate_robot --surface-drop).
DROP = 0.060


def yaw_deg(T):
    return np.degrees(np.arctan2(T[1, 0], T[0, 0]))


@pytest.fixture(scope="module")
def rig():
    env_info = make("mini-defend").env_info
    T_sim = np.asarray(env_info["robot"]["base_frame"][0], float)    # world <- base
    W2B = inv_homog(T_sim)
    half_l = env_info["table"]["length"] / 2
    rail_y = env_info["table"]["width"] / 2 - env_info["mallet"]["radius"]

    def touch(x_w, y_w, rise=0.0):
        """Grasp point, in the BASE frame, for a mallet resting at this world xy (+ ``rise``)."""
        return W2B[:3, :3] @ np.array([x_w, y_w, DROP + rise]) + W2B[:3, 3]

    goal = touch(-half_l, 0.0)
    left, right = touch(-0.277, rail_y), touch(-0.277, -rail_y)
    return env_info, T_sim, touch, goal, left, right


def recover(env_info, g, l, r):
    T_bw, info = world_from_goal_and_rails(g, l, r, half_length=env_info["table"]["length"] / 2,
                                           surface_drop=DROP)
    return inv_homog(T_bw), info


def test_goal_and_rails_recover_the_sim_mount_exactly(rig):
    env_info, T_sim, _, goal, left, right = rig
    rec, info = recover(env_info, goal, left, right)

    assert rec[:3, 3] == pytest.approx(T_sim[:3, 3], abs=1e-9)
    assert yaw_deg(rec) == pytest.approx(yaw_deg(T_sim), abs=1e-9)
    assert rec[:3, 2] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    expect_span = env_info["table"]["width"] - 2 * env_info["mallet"]["radius"]
    assert info["rail_span"] == pytest.approx(expect_span, abs=1e-9)
    assert info["flipped"] is False
    assert report_vs_sim(rec, env_info) is True


def test_the_goal_mouth_maps_onto_the_planners_goal_line(rig):
    """What reach-probe reports: dist-to-goal = world_x + L/2, and it must be 0 at the mouth."""
    env_info, _, _, goal, left, right = rig
    rec, _ = recover(env_info, goal, left, right)
    world_x = (rec[:3, :3] @ goal + rec[:3, 3])[0]
    assert world_x + env_info["table"]["length"] / 2 == pytest.approx(0.0, abs=1e-9)


def test_rails_riding_up_no_longer_tilt_or_move_the_frame(rig):
    """2026-09-15: both rail touches logged ~36 mm high. A fitted plane tilted 10.2 deg."""
    env_info, T_sim, touch, goal, _, _ = rig
    rail_y = env_info["table"]["width"] / 2 - env_info["mallet"]["radius"]
    rec, info = recover(env_info, goal, touch(-0.277, rail_y, rise=0.036),
                        touch(-0.277, -rail_y, rise=0.036))

    assert rec[:3, 3] == pytest.approx(T_sim[:3, 3], abs=1e-9)
    assert yaw_deg(rec) == pytest.approx(0.0, abs=1e-9)
    assert rec[:3, 2] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    # still reported, so the operator sees it happened
    assert info["left_rise"] == pytest.approx(0.036, abs=1e-9)
    assert info["touch_plane_tilt_deg"] > 5.0


def test_an_off_centre_goal_touch_does_not_move_the_origin(rig):
    """The centre line comes from the rail midpoint, so the goal touch only sets x."""
    env_info, T_sim, touch, _, left, right = rig
    rec, info = recover(env_info, touch(-env_info["table"]["length"] / 2, 0.02), left, right)
    assert rec[:3, 3] == pytest.approx(T_sim[:3, 3], abs=1e-9)
    assert info["goal_off_centre"] == pytest.approx(0.02, abs=1e-9)


def test_rails_given_in_the_wrong_order_are_flipped_not_mirrored(rig):
    env_info, T_sim, _, goal, left, right = rig
    rec, info = recover(env_info, goal, right, left)
    assert info["flipped"] is True
    assert rec[:3, 3] == pytest.approx(T_sim[:3, 3], abs=1e-9)
    assert yaw_deg(rec) == pytest.approx(yaw_deg(T_sim), abs=1e-9)


def test_both_touches_on_the_same_rail_is_refused(rig):
    env_info, _, touch, goal, left, _ = rig
    with pytest.raises(RuntimeError, match="OPPOSITE long rails"):
        recover(env_info, goal, left, touch(-0.24, 0.229))


def test_rails_reject_touch_noise_better_than_the_far_end_protocol(rig):
    """From behind its own goal the arm has only 113 mm of down-table baseline past the centre."""
    env_info, _, touch, goal, left, right = rig
    centre, far = touch(0.0, 0.0), touch(0.113, 0.0)
    rng = np.random.default_rng(0)
    jitter = lambda: rng.normal(0.0, 0.002, 3)          # 2 mm of touch repeatability

    rails, legacy = [], []
    for _ in range(200):
        rec, _ = recover(env_info, goal + jitter(), left + jitter(), right + jitter())
        rails.append(abs(yaw_deg(rec)))
        legacy.append(abs(yaw_deg(inv_homog(
            world_from_touches(centre + jitter(), far + jitter(), left + jitter(), DROP)))))

    assert np.percentile(rails, 95) < 1.0
    assert np.percentile(rails, 95) < 0.5 * np.percentile(legacy, 95)


def test_hand_eye_check_passes_the_sim_camera_and_fails_a_camera_below_the_base():
    expected = sim_camera_relative_to_base()
    pts = np.array([[0.25, 0.0, 0.10], [0.35, 0.1, 0.20], [0.30, -0.1, 0.30], [0.4, 0.0, 0.15]])
    err = np.full(len(pts), 0.8)

    assert expected == pytest.approx([0.62094, 0.298046, 0.545], abs=1e-3)
    assert check_hand_eye(homog(np.eye(3), expected), err, pts, expected) is True
    # the 2026-09-15 solve: fwd +0.10, left -0.25, up -0.43
    assert check_hand_eye(homog(np.eye(3), [0.1006, -0.2471, -0.4289]), err, pts, expected) is False
