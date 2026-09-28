"""Vision Part 4: agent-level integration of the camera puck source.

Slow, deliberately agent-scoped fixtures (constructing MiniBaselineAgent builds the
whole planning stack). GL-free tests pin the GT default and the restart semantics;
GL-gated tests run the vision agent closed-loop and pin the W -> R -> V yaw frame
composition (the assignment's own Part 4 sketch has a frame bug there: converting the
world yaw with _r2v.angle alone is off by the +90 deg base rotation).
"""

import mujoco
import numpy as np
import pytest

import mini_platform
from mini_platform.agents.baseline_mini import MiniBaselineAgent
from mini_platform.envs.env_base import AirHockeyBase
from mini_platform.tools.vision_eval import sight_line_blocked


@pytest.fixture(scope="module")
def env():
    return mini_platform.make("mini-hit")


@pytest.fixture(scope="module")
def gt_agent(env):
    return MiniBaselineAgent(env.env_info, agent_id=1)


@pytest.fixture(scope="module")
def vision_agent(env, require_gl):
    agent = MiniBaselineAgent(env.env_info, agent_id=1, puck_obs_source="vision",
                              vision_env=env.base_env)
    yield agent
    agent._sensor.close()


def teleport(env, x, y, yaw=0.0):
    be = env.base_env
    for name, val in (("puck_x", x), ("puck_y", y), ("puck_yaw", yaw)):
        be._data.qpos[be._model.joint(name).qposadr[0]] = val
    mujoco.mj_forward(be._model, be._data)


# ---------------------------------------------------------------------------- GL-free

def test_gt_default_measurement(env, gt_agent):
    """The GT path must be exactly the old behaviour: the V-frame puck pose with no
    per-measurement covariance."""
    np.random.seed(0)
    obs = env.reset()
    meas, meas_R = gt_agent.get_puck_measurement(obs)
    assert meas_R is None
    assert np.array_equal(meas, gt_agent.get_puck_pos(obs))


def test_restart_with_none_resets_to_prior(env, gt_agent):
    """A dropout on the restart step seeds the tracker from the own-half prior."""
    st = gt_agent.state
    st.reset()
    q = np.zeros(env.env_info['robot']['n_joints'])
    st.update_observation(q, q, None)
    prior = gt_agent.agent_params['puck_prior']
    assert np.allclose(st.puck_tracker.state[[0, 1, 4]], prior)
    st.reset()


def test_vision_requires_vision_env(env):
    with pytest.raises(ValueError, match="requires vision_env"):
        MiniBaselineAgent(env.env_info, agent_id=1, puck_obs_source="vision")
    with pytest.raises(ValueError, match="unknown puck_obs_source"):
        MiniBaselineAgent(env.env_info, agent_id=1, puck_obs_source="camera")


# --------------------------------------------------------------------------- GL-gated

def test_vision_agent_runs_closedloop(env, vision_agent, vision_viewer):
    """100 draw_action/step cycles on mini-hit without exception, with the EKF within
    2 cm of GT (in V) on a healthy share of unoccluded steps."""
    be = env.base_env
    cam_pos = vision_agent._sensor.camera.pos
    bf = env.env_info['robot']['base_frame'][0]
    np.random.seed(0)
    obs = env.reset()
    vision_agent.episode_start()
    good = 0
    for step in range(100):
        gt_pose = np.array([be._data.joint(n).qpos[0]
                            for n in ("puck_x", "puck_y", "puck_yaw")])
        action = vision_agent.draw_action(obs)
        vision_viewer(vision_agent._sensor.render(),
                      f"vision agent closed-loop, step {step}", pause=0.01)
        occ = sight_line_blocked(be._model, be._data, cam_pos, gt_pose[:2])
        if not occ:
            gt_v = vision_agent._r2v.pos2(
                AirHockeyBase._puck_2d_in_robot_frame(gt_pose, bf))
            if np.linalg.norm(vision_agent.state.puck_tracker.state[:2] - gt_v) < 0.02:
                good += 1
        obs, reward, absorbing, info = env.step(action)
        if absorbing:
            break
    assert good >= 30


def test_yaw_frame_composition(env, vision_agent, vision_viewer):
    """The decisive pin for the W -> R -> V yaw path: at an unoccluded mid-table pose
    with nonzero yaw, the vision-path yaw_V must match the GT-path conversion of the
    qpos pose. A missing base-frame rotation errs by ~pi/2 here."""
    matched = 0
    for yaw in (0.7, 1.9, -1.1):
        teleport(env, 0.1, 0.05, yaw)
        meas, meas_R = vision_agent.get_puck_measurement(None)  # obs unused on vision
        vision_viewer(vision_agent._sensor.render(),
                      f"yaw frame composition: world yaw {yaw:+.2f}")
        assert meas is not None
        if len(np.atleast_1d(meas)) != 3:
            continue                       # marker unresolved at this yaw: 2-D fallback
        gt_pose_r = AirHockeyBase._puck_2d_in_robot_frame(
            np.array([0.1, 0.05, yaw]), env.env_info['robot']['base_frame'][0])
        yaw_v_gt = vision_agent._r2v.angle(gt_pose_r[2])
        wrapped = (meas[2] - yaw_v_gt + np.pi) % (2 * np.pi) - np.pi
        assert abs(wrapped) < 0.15, (yaw, meas[2], yaw_v_gt)
        matched += 1
    assert matched >= 2
