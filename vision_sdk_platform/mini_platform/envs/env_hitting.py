import numpy as np

from mini_platform.envs.env_double import AirHockeyDouble
from mini_platform.envs.env_base import N_JOINTS


class AirHockeyHit(AirHockeyDouble):
    """
        Class for the air hockey hitting task on the mini platform.
    """

    def __init__(self, opponent_agent=None, gamma=0.99, horizon=500, moving_init=True, viewer_params={}, **kwargs):
        """
            Constructor
            Args:
                opponent_agent(Agent, None): Agent which controls the opponent
                moving_init(bool, False): If true, initialize the puck with inital velocity.
        """

        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

        self.moving_init = moving_init
        half_length = self.env_info['table']['length'] / 2
        hit_width = self.env_info['table']['width'] / 2 - self.env_info['puck']['radius'] - \
                    self.env_info['mallet']['radius'] * 2

        # The full-size env spawns the puck between 0.719 and 0.205 of its own half-length
        # from the centre; the same fractions of the mini half-length are used here rather
        # than the absolute [-0.7, -0.2], which would be off the table. The far end (-0.104)
        # sits just outside the 0.08 m strip in front of the centreline that the arm cannot
        # reach (docs/optimal_control_implementation.md 0.3), so no episode starts with an
        # unplayable puck.
        self.hit_range = np.array([[-0.7186 * half_length, -0.2053 * half_length],
                                   [-hit_width, hit_width]])  # Table Frame

        # Scaled with the table so the puck takes the same time to cross it as at full size.
        self.init_velocity_range = (0, 0.5 * 0.505226)  # Table Frame

        # y-offset after 1 s of travel must stay inside the rail; 0.43 of the full-size
        # half-width, applied to the mini half-width.
        self._max_lateral_drift = 0.828 * self.env_info['table']['width'] / 2

        if opponent_agent is not None:
            self._opponent_agent = opponent_agent.draw_action
        else:
            self._opponent_agent = lambda obs: np.zeros(N_JOINTS)

    def setup(self, obs):
        # Initial position of the puck
        puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])

        if self.moving_init:
            while True:
                lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
                angle = np.random.uniform(-np.pi / 2 - 0.1, np.pi / 2 + 0.1)
                puck_vel = np.zeros(3)
                puck_vel[0] = -np.cos(angle) * lin_vel
                puck_vel[1] = np.sin(angle) * lin_vel
                puck_vel[2] = np.random.uniform(-2, 2, 1)
                if np.abs(puck_vel[1] * 1 + puck_pos[1]) < self._max_lateral_drift:
                    break

            self._write_data("puck_x_vel", puck_vel[0])
            self._write_data("puck_y_vel", puck_vel[1])
            self._write_data("puck_yaw_vel", puck_vel[2])

        super(AirHockeyHit, self).setup(obs)

    def reward(self, state, action, next_state, absorbing):
        return 0

    def is_absorbing(self, obs):
        # NOTE: this ends the episode on the puck's FIRST return, so a rally is impossible by
        # construction. That is the organizers' rule and it is kept, but it means the
        # two-player variants (mini-hit-2p) measure "can you score past a live defender once",
        # not "can you win a rally". Their episodes run ~225 steps against ~320 for the
        # scripted-wander version, purely because the defender returns the puck.
        puck_pos, puck_vel = self.get_puck(obs)
        # Stop if the puck bounces back on the opponents wall
        if puck_pos[0] > 0 and puck_vel[0] < 0:
            return True
        return super(AirHockeyHit, self).is_absorbing(obs)

    def _modify_observation(self, obs):
        obs = super()._modify_observation(obs)
        return np.split(obs, 2)[0]

    def _preprocess_action(self, action):
        opponents_obs = np.split(super()._modify_observation(self._obs), 2)[1]

        return action, self._opponent_agent(opponents_obs)
