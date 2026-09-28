# Vendored from the 2025 Robot Air Hockey Challenge (air_hockey_challenge/framework/air_hockey_challenge_wrapper.py).
# MIT License, Copyright (c) 2025 Atalay Donat, Puze Liu, Jonas Guenster, Davide Tateo.
# See ../../THIRD_PARTY.md. Modifications for the mini platform are marked MINI:.

from copy import deepcopy

from mini_platform.core.constraints import *
from mini_platform.core.transformations import robot_to_world
from mushroom_rl.core import Environment


class AirHockeyChallengeWrapper(Environment):
    def __init__(self, env, custom_reward_function=None, interpolation_order=3, **kwargs):
        """
        Environment Constructor

        Args:
            env [string]:
                The string to specify the running environments. Available environments: [3dof-hit, 3dof-defend].
                [7dof-hit, 7dof-defend, 7dof-prepare, tournament] will be available once the corresponding stage starts.
            custom_reward_function [callable]:
                You can customize your reward function here.
            interpolation_order (int, 3): Type of interpolation used, has to correspond to action shape. Order 1-5 are
                    polynomial interpolation of the degree. Order -1 is linear interpolation of position and velocity.
                    Set Order to None in order to turn off interpolation. In this case the action has to be a trajectory
                    of position, velocity and acceleration of the shape (20, 3, n_joints)
        """

        # MINI: this build serves the mini platform only, so the task table comes straight
        # from mini_platform.envs instead of the challenge's iiwa table.
        from mini_platform.envs import ENV_DICT as env_dict

        # Mini platform (40x20 in table + the 6-DOF metal arm). Resolved lazily so this
        # framework carries no import-time dependency on the mini_platform package.
        if isinstance(env, str) and env.startswith("mini-"):
            from mini_platform.envs import ENV_DICT as MINI_ENV_DICT
            env_dict.update(MINI_ENV_DICT)

        self.base_env = env_dict[env](interpolation_order=interpolation_order, **kwargs)
        self.env_name = env
        self.env_info = self.base_env.env_info

        if custom_reward_function:
            self.base_env.reward = lambda state, action, next_state, absorbing: custom_reward_function(self.base_env,
                                                                                                       state, action,
                                                                                                       next_state,
                                                                                                       absorbing)

        constraint_list = ConstraintList()
        constraint_list.add(JointPositionConstraint(self.env_info))
        constraint_list.add(JointVelocityConstraint(self.env_info))
        constraint_list.add(EndEffectorConstraint(self.env_info))
        constraint_list.add(LinkConstraint(self.env_info))

        self.env_info['constraints'] = constraint_list
        self.env_info['env_name'] = self.env_name

        super().__init__(self.base_env.info)

    def step(self, action):
        obs, reward, done, info = self.base_env.step(action)

        if "tournament" in self.env_name:
            info["constraints_value"] = list()
            for i in range(2):
                obs_agent = obs[i * int(len(obs) / 2): (i + 1) * int(len(obs) / 2)]
                info["constraints_value"].append(deepcopy(self.env_info['constraints'].fun(
                    obs_agent[self.env_info['joint_pos_ids']], obs_agent[self.env_info['joint_vel_ids']])))

            info["score"] = self.base_env.score
            info["faults"] = self.base_env.faults

        else:
            info["constraints_value"] = deepcopy(self.env_info['constraints'].fun(obs[self.env_info['joint_pos_ids']],
                                                                                  obs[self.env_info['joint_vel_ids']]))
            info["success"] = self.check_success(obs)

        return obs, reward, done, info

    def render(self, record=False):
        return self.base_env.render(record=record)

    def reset(self, state=None):
        return self.base_env.reset(state)
    
    def _get_obs(self, data):
        return self.base_env._get_obs(data)

    def check_success(self, obs):
        puck_pos, puck_vel = self.base_env.get_puck(obs)

        puck_pos, _ = robot_to_world(self.base_env.env_info["robot"]["base_frame"][0], translation=puck_pos)
        success = 0

        half_length = self.base_env.env_info['table']['length'] / 2
        half_width = self.base_env.env_info['table']['width'] / 2

        # The band the puck has to be settled in for defend/prepare to count, as a fraction of
        # the own half. The literals these replace (-0.8, -0.2, 0.39105) are the same bands on
        # the full-size table and would fall off the mini one.
        x_lo, x_hi = -0.8214 * half_length, -0.2053 * half_length
        y_max = 0.7535 * half_width

        if "hit" in self.env_name:
            if puck_pos[0] - half_length > 0 and \
                    np.abs(puck_pos[1]) - self.base_env.env_info['table']['goal_width'] / 2 < 0:
                success = 1

        elif "defend" in self.env_name:
            if x_lo < puck_pos[0] <= x_hi and puck_vel[0] < 0.1:
                success = 1

        elif "prepare" in self.env_name:
            if x_lo < puck_pos[0] <= x_hi and np.abs(puck_pos[1]) < y_max and puck_vel[0] < 0.1:
                success = 1
        return success