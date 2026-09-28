# Vendored from the 2025 Robot Air Hockey Challenge (baseline/baseline_agent/system_state.py).
# MIT License, Copyright (c) 2025 Atalay Donat, Puze Liu, Jonas Guenster, Davide Tateo.
# See ../../THIRD_PARTY.md. Modifications for the mini platform are marked MINI:.

import copy
from enum import Enum

import numpy as np

from mini_platform.core.kinematics import forward_kinematics, jacobian
from mini_platform.estimation.kalman_filter import PuckTracker


class TACTICS(Enum):
    __order__ = "INIT READY PREPARE DEFEND REPEL SMASH N_TACTICS"
    INIT = 0
    READY = 1
    PREPARE = 2
    DEFEND = 3
    REPEL = 4
    SMASH = 5
    N_TACTICS = 6


class SystemState:
    def __init__(self, env_info, agent_id, agent_params):
        self.env_info = env_info
        self.agent_id = agent_id
        self.agent_params = agent_params
        self.puck_tracker = PuckTracker(self.env_info, agent_id)
        self.robot_model = copy.deepcopy(self.env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(self.env_info['robot']['robot_data'])

        self.restart = True

        self.q_cmd = np.zeros(self.env_info['robot']['n_joints'])
        self.q_actual = np.zeros(self.env_info['robot']['n_joints'])

        self.dq_cmd = np.zeros(self.env_info['robot']['n_joints'])
        self.dq_actual = np.zeros(self.env_info['robot']['n_joints'])

        self.x_cmd = np.zeros(3)
        self.x_actual = np.zeros(3)

        self.v_cmd = np.zeros(3)
        self.v_actual = np.zeros(3)

        self.predicted_state = np.zeros(6)
        self.predicted_cov = np.eye(6)
        self.predicted_time = 0.
        self.estimated_state = np.zeros(6)

        self.tactic_current = TACTICS.READY
        self.is_new_tactic = True
        self.tactic_finish = True
        self.has_generated_stop_traj = False
        self.switch_tactics_count = self.agent_params['switch_tactics_min_steps']
        self.puck_static_count = 0
        self.puck_approaching_count = 0
        self.puck_transversal_moving_count = 0

        self.smash_finish = False

        self.trajectory_buffer = list()

    def reset(self):
        self.restart = True

        self.q_cmd = np.zeros(self.env_info['robot']['n_joints'])
        self.q_actual = np.zeros(self.env_info['robot']['n_joints'])

        self.dq_cmd = np.zeros(self.env_info['robot']['n_joints'])
        self.dq_actual = np.zeros(self.env_info['robot']['n_joints'])

        self.x_cmd = np.zeros(3)
        self.x_actual = np.zeros(3)

        self.v_cmd = np.zeros(3)
        self.v_actual = np.zeros(3)

        self.predicted_state = np.zeros(6)
        self.predicted_cov = np.eye(6)
        self.predicted_time = 0.
        self.estimated_state = np.zeros(6)

        self.tactic_current = TACTICS.READY
        self.is_new_tactic = True
        self.tactic_finish = True
        self.has_generated_stop_traj = False
        self.plan_new_trajectory = True
        self.switch_tactics_count = self.agent_params['switch_tactics_min_steps']
        self.puck_static_count = 0
        self.puck_approaching_count = 0
        self.puck_transversal_moving_count = 0

        self.smash_finish = False

        self.trajectory_buffer = list()

    def is_puck_static(self):
        return self.puck_static_count > 3

    def is_puck_approaching(self):
        # MINI: latch DEFEND on the FIRST approaching tick (~20 ms), not the vendored 4
        # (~80 ms) -- on a home surface the puck decelerates fast and rarely sustains speed
        # toward the goal for several ticks, so the arm was missing real approaches. Trades
        # latency for sensitivity: a single noisy velocity sample can trigger a defend now.
        return self.puck_approaching_count > 0

    def is_puck_transversal_moving(self):
        return self.puck_transversal_moving_count > 3

    # MINI: puck_state may now be None (vision dropout) or 2-D (no marker), with an
    # optional per-measurement covariance (Part 3). The GT path -- always 3-D, always
    # meas_R=None -- is byte-identical in behaviour.
    def update_observation(self, joint_pos_cur, joint_vel_cur, puck_state, meas_R=None,
                           meas_age=None):
        if self.restart:
            # MINI: with no detection at episode start the tracker seeds from the
            # own-half prior (only the vision agent sets 'puck_prior' -- a missing key
            # here is a wiring bug and should fail loudly); a 2-D first detection
            # seeds yaw 0.
            if puck_state is None:
                self.puck_tracker.reset(np.asarray(self.agent_params['puck_prior'],
                                                   dtype=float))
            elif len(puck_state) == 2:
                self.puck_tracker.reset(np.append(puck_state, 0.0))
            else:
                self.puck_tracker.reset(puck_state)
            self.q_cmd = joint_pos_cur
            self.dq_cmd = joint_vel_cur
            self.x_cmd, self.v_cmd = self.update_ee_pos_vel(self.q_cmd, self.dq_cmd)
            self.restart = False

        self.q_actual = joint_pos_cur
        self.dq_actual = joint_vel_cur
        self.x_actual, self.v_actual = self.update_ee_pos_vel(self.q_actual, self.dq_actual)

        # meas_age: how old the measurement is (s), when the sensor can say. Passed only when one
        # EXISTS, so the call keeps its vendored two-argument shape for the sim, for the GT path,
        # and for any stand-in tracker written against `step(measurement, R=None)`. Widening the
        # call unconditionally would break those for a value they would ignore anyway.
        if meas_age is None:
            self.puck_tracker.step(puck_state, meas_R)
        else:
            self.puck_tracker.step(puck_state, meas_R, age=meas_age)
        self.estimated_state = self.puck_tracker.state.copy()

        if np.linalg.norm(self.puck_tracker.state[2:4]) < self.agent_params['static_vel_threshold']:
            self.puck_static_count += 1
            self.puck_approaching_count = 0
            self.puck_transversal_moving_count = 0
        else:
            self.puck_static_count = 0
            puck_dir = self.puck_tracker.state[2:4] / np.linalg.norm(self.puck_tracker.state[2:4])
            if np.abs(np.dot(puck_dir, np.array([1., 0]))) < 0.15:
                self.puck_transversal_moving_count += 1
                self.puck_approaching_count = 0
            else:
                if self.puck_tracker.state[2] < 0 and self.puck_tracker.state[0] > self.agent_params['defend_range'][0]:
                    self.puck_approaching_count += 1

    def update_prediction(self, prediction_time, stop_line=0.):
        self.predicted_state, self.predicted_cov, self.predicted_time = \
            self.puck_tracker.get_prediction(prediction_time, stop_line)

    def update_ee_pos_vel(self, joint_pos, joint_vel):
        x_ee, _ = forward_kinematics(self.robot_model, self.robot_data, joint_pos)
        v_ee = jacobian(self.robot_model, self.robot_data, joint_pos)[:3,
               :self.env_info['robot']['n_joints']] @ joint_vel
        return x_ee, v_ee
