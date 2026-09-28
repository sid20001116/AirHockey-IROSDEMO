import mujoco
import numpy as np

from mini_platform.envs.env_base import AirHockeyBase, N_JOINTS


class AirHockeyDouble(AirHockeyBase):
    """
    Base class for two agent air hockey tasks on the mini platform.

    Both arms are the same model, so one home configuration serves both, each in its own
    base frame.
    """

    def __init__(self, gamma=0.99, horizon=500, viewer_params={}, **kwargs):
        super().__init__(gamma=gamma, horizon=horizon, n_agents=2, viewer_params=viewer_params, **kwargs)

        self._compute_init_state()

        self.filter_ratio = 0.274
        self.q_pos_prev = np.zeros(N_JOINTS * self.env_info["n_agents"])
        self.q_vel_prev = np.zeros(N_JOINTS * self.env_info["n_agents"])

    def get_ee(self, robot=1):
        """
        Getting the ee properties from the current internal state of the selected robot.

        Args:
            robot: ID of robot, either 1 or 2

        Returns: ([pos_x, pos_y, pos_z], [ang_vel_x, ang_vel_y, ang_vel_z, lin_vel_x, lin_vel_y, lin_vel_z])
        """
        return self._read_data(f"robot_{robot}/ee_pos"), self._read_data(f"robot_{robot}/ee_vel")

    def get_joints(self, obs, agent=None):
        """
        Get joint position and velocity of the robots.
        Can choose the robot with agent = 1 / 2. If agent is None both are returned.
        """
        agents = [agent] if agent else [1, 2]
        q_pos = np.zeros(N_JOINTS * len(agents))
        q_vel = np.zeros(N_JOINTS * len(agents))

        for k, a in enumerate(agents):
            for i in range(N_JOINTS):
                q_pos[k * N_JOINTS + i] = self.obs_helper.get_from_obs(obs, f"robot_{a}/joint_{i + 1}_pos")[0]
                q_vel[k * N_JOINTS + i] = self.obs_helper.get_from_obs(obs, f"robot_{a}/joint_{i + 1}_vel")[0]

        return q_pos, q_vel

    def _create_observation(self, obs):
        # Filter the joint velocity
        q_pos, q_vel = self.get_joints(obs)
        q_vel_filter = self.filter_ratio * q_vel + (1 - self.filter_ratio) * self.q_vel_prev
        self.q_pos_prev = q_pos
        self.q_vel_prev = q_vel_filter

        for a in (1, 2):
            for i in range(N_JOINTS):
                self.obs_helper.get_from_obs(obs, f"robot_{a}/joint_{i + 1}_vel")[:] = \
                    q_vel_filter[(a - 1) * N_JOINTS + i]

        # Wrap puck's rotation angle to [-pi, pi)
        yaw_angle = self.obs_helper.get_from_obs(obs, "puck_yaw_pos")
        self.obs_helper.get_from_obs(obs, "puck_yaw_pos")[:] = (yaw_angle + np.pi) % (2 * np.pi) - np.pi
        return obs

    def _modify_observation(self, obs):
        new_obs = obs.copy()

        puck_pos, puck_vel = self.get_puck(new_obs)

        for a in (1, 2):
            frame = self.env_info['robot']['base_frame'][a - 1]
            pos = self._puck_2d_in_robot_frame(puck_pos, frame)
            vel = self._puck_2d_in_robot_frame(puck_vel, frame, type='vel')

            prefix = "" if a == 1 else "robot_2/"
            self.obs_helper.get_from_obs(new_obs, prefix + "puck_x_pos")[:] = pos[0]
            self.obs_helper.get_from_obs(new_obs, prefix + "puck_y_pos")[:] = pos[1]
            self.obs_helper.get_from_obs(new_obs, prefix + "puck_yaw_pos")[:] = pos[2]

            self.obs_helper.get_from_obs(new_obs, prefix + "puck_x_vel")[:] = vel[0]
            self.obs_helper.get_from_obs(new_obs, prefix + "puck_y_vel")[:] = vel[1]
            self.obs_helper.get_from_obs(new_obs, prefix + "puck_yaw_vel")[:] = vel[2]

            opponent_pos = self.obs_helper.get_from_obs(new_obs, f"robot_{a}/opponent_ee_pos")
            self.obs_helper.get_from_obs(new_obs, f"robot_{a}/opponent_ee_pos")[:] = \
                (np.linalg.inv(frame) @ np.concatenate([opponent_pos, [1]]))[:3]

        return new_obs

    def setup(self, obs):
        for a in (1, 2):
            for i, name in enumerate(self.arm_joint_names[a - 1]):
                self._data.joint(name).qpos = self.init_state[i]
                self.q_pos_prev[(a - 1) * N_JOINTS + i] = self.init_state[i]
                self.q_vel_prev[(a - 1) * N_JOINTS + i] = self._data.joint(name).qvel

        # Both mallets are free bodies -- put them back in their jaws on every reset.
        self._seat_mallet_in_gripper()

        super().setup(obs)

        # Update body positions, needed for _compute_universal_joint
        mujoco.mj_fwdPosition(self._model, self._data)

    def reward(self, state, action, next_state, absorbing):
        return 0
