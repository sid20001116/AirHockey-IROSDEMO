import numpy as np

from mini_platform.envs.env_base import TABLE_LENGTH, TABLE_WIDTH
from mini_platform.envs.env_single import AirHockeySingle


class AirHockeyDefend(AirHockeySingle):
    """
        Class for the air hockey defending task on the mini platform.
        The agent should stop the puck in its own half.
    """

    def __init__(self, gamma=0.99, horizon=500, viewer_params={}, **kwargs):
        # Same fractions of the table as the full-size env (0.411-0.770 of the half-length,
        # +/-0.771 of the half-width), and an incoming speed scaled with the table so the puck
        # takes the same time to arrive as it does at full size.
        self.init_velocity_range = (1 * 0.505226, 5 * 0.505226)

        # env_info is not built until super().__init__ runs, so read the module constants.
        half_length = TABLE_LENGTH / 2
        half_width = TABLE_WIDTH / 2
        self.start_range = np.array([[0.4107 * half_length, 0.7700 * half_length],
                                     [-0.7707 * half_width, 0.7707 * half_width]])  # Table Frame

        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

    def setup(self, obs):
        puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]

        lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
        angle = np.random.uniform(-0.75, 0.75)

        puck_vel = np.zeros(3)
        puck_vel[0] = -np.cos(angle) * lin_vel
        puck_vel[1] = np.sin(angle) * lin_vel
        puck_vel[2] = np.random.uniform(-10, 10, 1)

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_x_vel", puck_vel[0])
        self._write_data("puck_y_vel", puck_vel[1])
        self._write_data("puck_yaw_vel", puck_vel[2])

        super(AirHockeyDefend, self).setup(obs)

    def reward(self, state, action, next_state, absorbing):
        return 0

    def is_absorbing(self, state):
        puck_pos, puck_vel = self.get_puck(state)
        # If puck is over the middle line and moving towards opponent
        if puck_pos[0] > 0 and puck_vel[0] > 0:
            return True
        if np.linalg.norm(puck_vel[:2]) < 0.1:
            return True
        return super().is_absorbing(state)
