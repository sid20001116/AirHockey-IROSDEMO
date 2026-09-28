import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from mini_platform.core.kinematics import inverse_kinematics

from mini_platform import ASSET_DIR
from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from mushroom_rl.utils.spaces import Box

# ----------------------------------------------------------------------------------------
# Mini platform constants.
#
# Everything here is read off assets/mini_table.xml and assets/arm1.xml rather than
# scaled from the full-size platform, because the two rigs do not scale together, and the
# mounting is different in kind: the arm stands on the FLOOR (on a 5 mm plate) at the middle
# of a SHORT end, behind its own goal like the KUKA, but with its base 0.2109 m below the
# playing surface where the iiwa's is 0.1 m below it. In the double scene the arms face each
# other down the long axis: arm 1 at the home (-x) end, arm 2 at the away (+x) end.
# See docs/optimal_control_implementation.md 0.2-0.3.
#
# The MJCF stays the source of truth for anything MuJoCo can tell us (joint ranges, radii);
# these are the quantities MuJoCo does not model.
# ----------------------------------------------------------------------------------------

N_JOINTS = 6                        # metal_1/joint1..6. NOT 7 -- joint7/joint8 are the gripper.
TABLE_LENGTH = 1.016                # rim inner edges, 40 in
TABLE_WIDTH = 0.508                 # 20 in
GOAL_WIDTH = 0.172                  # MEASURED on the real table 2026-09-15 (was 0.122351); keep in step with assets/mini_table.xml rims and the other copy in estimation/mini_env.py
PUCK_RADIUS = 0.025           # assets/mini_table.xml `puck` geom
MALLET_RADIUS = 0.025               # assets/mallet1.xml head: 50 mm diameter
MALLET_HALF_HEIGHT = 0.015          # 30 mm tall

# --- the grip -------------------------------------------------------------------------
# The mallet is a FREE BODY. It is held only by friction between the gripper fingers and its
# 10 mm handle cube -- no weld, no equality, no joint. Everything below describes that grip.

# Distance from gripper_base to the grasp point, along the gripper's pointing axis (local +Z).
# Measured off the finger meshes: the inner faces track the joint travel exactly (inner face
# at y = |q7|) in the z band 0.060-0.070, and the fingers are ~18 mm wide there, so a 10 mm
# cube seats squarely. Nearer the knuckle the faces sit 2-3 mm proud and would grip on a
# corner instead.
GRASP_OFFSET = 0.065

# Half-width of the handle cube; the jaws open to exactly this to receive it.
HANDLE_HALF = 0.005

# Constant clamping force on joint7, N. joint7's actuatorfrcrange is +/-3 N and the grip is
# the only thing holding the mallet, so this runs at the limit. Positive closes (joint7's
# range is [-0.05, 0] with 0 = shut); joint8 follows through the equality.
GRIP_FORCE = 3.0

# How far the arm base sits BELOW the playing surface: the table stands 0.2159 m tall
# (assets/mini_table.xml, floor at z=-0.2159) and the base is on a 5 mm floor plate, so
# base z = -0.2109. Every robot-frame height below carries this offset.
BASE_BELOW_SURFACE = 0.2109

# Height of the grasp point above the PLAYING SURFACE. Set by the mallet's own geometry, not
# chosen: the head rests on the table (0.000-0.030), the post spans 0.030-0.055, and the
# handle cube is centred at 0.060.
GRASP_HEIGHT_ABOVE_SURFACE = 0.060

# The same height in the ROBOT'S OWN frame, which is what IK, the constraints and the
# planners all consume (`ee_desired_height` / `universal_height`). The base is on the floor,
# 0.2109 m below the surface, so this is no longer equal to the surface height.
GRASP_HEIGHT = GRASP_HEIGHT_ABOVE_SURFACE + BASE_BELOW_SURFACE

# Minimum wrist/elbow height for LinkConstraint, in the robot's frame. 0.035 m above the
# playing surface keeps every link clear of the mallet; the base-depth offset converts that
# to the base frame the constraint's forward kinematics work in.
LINK_CONSTRAINT_HEIGHT = 0.035 + BASE_BELOW_SURFACE

# Velocity limit = elementwise min of two independent bounds, because either can bind:
#
#   SDK vmax       [10, 10, 10, 10, 30, 30]  rad/s   motor_config.cpp :13 :34 :55 :76 :97 :118
#                  -- what the motor and its CAN encoding can actually do. Authoritative.
#   measured       [10.345, 18.876, 10.100, 13.657, 9.640, 9.960]
#                  -- worst case over 65 configurations of the peak velocity reached under
#                     saturated actuator force (artifacts/sysid/metal_arm_limits.json),
#                     i.e. what THIS plant can reach given its inertia and torque ceilings.
#
# The SDK binds on J1-J4; the plant binds on J5/J6, which cannot get near their 30 rad/s motor
# limit. Taking the min keeps the planner from asking for either an impossible speed or one the
# hardware would refuse.
#
# These moved when the MJCF torque ceilings were raised to the SDK's tmax -- J4 in particular
# went 4.395 -> 13.657 rad/s measured, because it had been running against a 10 N.m ceiling
# against a 30 N.m motor. Re-run tools/measure_joint_limits.py after ANY change to actuator
# force, damping or armature. See artifacts/reports/04_sdk_controller_port.md.
JOINT_VEL_LIMIT = np.array([10.0, 10.0, 10.0, 10.0, 9.640, 9.960])

# No SDK counterpart: the SDK bounds acceleration only inside its own trajectory time
# allocator, which is deliberately not ported (the platform's planner owns time allocation).
# So this stays purely measured. It is published in env_info and, as of today, never read at
# runtime -- no constraint checks acceleration.
JOINT_ACC_LIMIT = np.array([39.191, 283.664, 133.904, 261.759, 194.396, 198.820])

# Home pose of the grasp point in the robot's own frame. Robot +x points DOWN the table from
# behind the arm's own goal: the base centre stands 0.066 m out from the table's outer edge
# and the short-end rim is 0.0469404 m thick (2 x 0.0234702, assets/mini_table.xml
# rim_home_*), so the playing surface starts at robot-frame x = 0.1129 and x = 0.174 rests
# the mallet 0.061 m in from the near rim -- the same clearance the old side mount used --
# on the table's centre line (world (-0.44694, 0) for arm 1). z is the grasp height in the
# robot's frame, 0.2709 now that the base is on the floor.
HOME_POS = np.array([0.174, 0.0, GRASP_HEIGHT])

# Seed for the home-pose IK. Obtained by rejection-sampling configurations whose gripper +Z
# lies within 15 deg of world -Z (the vertical-mallet condition of
# docs/optimal_control_implementation.md 0.3), seeding from the closest, and solving. joint5
# and joint6 come out at ~0: with the wrist directly above the contact point the last two
# joints have almost nothing to do -- the ~1 DOF of redundancy that 0.2 warns about.
INIT_SEED = np.array([0.0014, -1.5243, 0.5668, -0.6133, 0.0, 0.0014])


class AirHockeyBase(MuJoCo):
    """
    Abstract class for all mini-platform AirHockey environments.

    Mirrors environments/iiwas/env_base.py, retargeted to the 40x20 in table and the 6-DOF
    metal arm.
    """

    def __init__(self, gamma=0.99, horizon=500, timestep=1 / 1000., n_intermediate_steps=20, n_substeps=1,
                 n_agents=1, viewer_params={}):
        """
            Constructor.

            Args:
                n_agents (int, 1): number of agent to be used in the environment (one or two)
        """

        self.n_agents = n_agents

        action_spec = []
        observation_spec = [("puck_x_pos", "puck_x", ObservationType.JOINT_POS),
                            ("puck_y_pos", "puck_y", ObservationType.JOINT_POS),
                            ("puck_yaw_pos", "puck_yaw", ObservationType.JOINT_POS),
                            ("puck_x_vel", "puck_x", ObservationType.JOINT_VEL),
                            ("puck_y_vel", "puck_y", ObservationType.JOINT_VEL),
                            ("puck_yaw_vel", "puck_yaw", ObservationType.JOINT_VEL)]

        additional_data = list(observation_spec)

        collision_spec = [("puck", ["puck"]),
                          ("rim", ["rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r", "rim_left", "rim_right"]),
                          ("rim_short_sides", ["rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r"])]

        if not 1 <= self.n_agents <= 2:
            raise ValueError('n_agents should be 1 or 2')

        scene = os.path.join(ASSET_DIR, "mini_single.xml" if self.n_agents == 1 else "mini_double.xml")

        self.arm_joint_names = []
        self.gripper_joint_names = []

        for r in range(1, self.n_agents + 1):
            arm = [f"metal_{r}/joint{j + 1}" for j in range(N_JOINTS)]
            self.arm_joint_names.append(arm)
            self.gripper_joint_names.append((f"metal_{r}/joint7", f"metal_{r}/joint8"))

            action_spec += arm

            if r == 2:
                # The second agent gets its own copy of the puck observation, expressed in
                # its own base frame by _modify_observation.
                observation_spec += [("robot_2/puck_x_pos", "puck_x", ObservationType.JOINT_POS),
                                     ("robot_2/puck_y_pos", "puck_y", ObservationType.JOINT_POS),
                                     ("robot_2/puck_yaw_pos", "puck_yaw", ObservationType.JOINT_POS),
                                     ("robot_2/puck_x_vel", "puck_x", ObservationType.JOINT_VEL),
                                     ("robot_2/puck_y_vel", "puck_y", ObservationType.JOINT_VEL),
                                     ("robot_2/puck_yaw_vel", "puck_yaw", ObservationType.JOINT_VEL)]

            observation_spec += [(f"robot_{r}/joint_{j + 1}_pos", arm[j], ObservationType.JOINT_POS)
                                 for j in range(N_JOINTS)]
            observation_spec += [(f"robot_{r}/joint_{j + 1}_vel", arm[j], ObservationType.JOINT_VEL)
                                 for j in range(N_JOINTS)]

            if self.n_agents == 2:
                # Must close each agent's block: AirHockeyHit._modify_observation splits the
                # observation into two equal halves, one per agent.
                observation_spec += [(f"robot_{r}/opponent_ee_pos", f"mallet_{3 - r}",
                                      ObservationType.BODY_POS)]

            # ee_pos / ee_vel read the MALLET, which is a free body: this is where the mallet
            # actually is, not where the arm commanded it. The two differ by however much the
            # grip slips, which is the whole point of gripping rather than welding.
            # `gripper_pos` is the jaw opening, i.e. whether the mallet is still in there.
            additional_data += [
                (f"robot_{r}/gripper_pos", f"metal_{r}/joint7", ObservationType.JOINT_POS),
                (f"robot_{r}/gripper_vel", f"metal_{r}/joint7", ObservationType.JOINT_VEL),
                (f"robot_{r}/ee_pos", f"mallet_{r}", ObservationType.BODY_POS),
                (f"robot_{r}/ee_vel", f"mallet_{r}", ObservationType.BODY_VEL),
                (f"robot_{r}/mallet_rot", f"mallet_{r}", ObservationType.BODY_ROT),
                (f"robot_{r}/grasp_rot", f"metal_{r}/gripper_base", ObservationType.BODY_ROT)]

            collision_spec += [(f"robot_{r}/ee", [f"metal_{r}/ee"])]

        self.env_info = dict()
        self.env_info['table'] = {"length": TABLE_LENGTH, "width": TABLE_WIDTH, "goal_width": GOAL_WIDTH}
        self.env_info['puck'] = {"radius": PUCK_RADIUS}
        self.env_info['mallet'] = {"radius": MALLET_RADIUS}
        self.env_info['n_agents'] = self.n_agents
        self.env_info['robot'] = {
            "n_joints": N_JOINTS,
            "ee_desired_height": GRASP_HEIGHT,
            "joint_vel_limit": np.vstack([-JOINT_VEL_LIMIT, JOINT_VEL_LIMIT]),
            "joint_acc_limit": np.vstack([-JOINT_ACC_LIMIT, JOINT_ACC_LIMIT]),
            "base_frame": [],
            "universal_height": GRASP_HEIGHT,
            "link_constraint_height": LINK_CONSTRAINT_HEIGHT,
            "control_frequency": 50,
        }

        self.env_info['puck_pos_ids'] = [0, 1, 2]
        self.env_info['puck_vel_ids'] = [3, 4, 5]
        self.env_info['joint_pos_ids'] = list(range(6, 6 + N_JOINTS))
        self.env_info['joint_vel_ids'] = list(range(6 + N_JOINTS, 6 + 2 * N_JOINTS))
        if self.n_agents == 2:
            base = 6 + 2 * N_JOINTS
            self.env_info['opponent_ee_ids'] = [base, base + 1, base + 2]
        else:
            self.env_info['opponent_ee_ids'] = []

        max_joint_vel = ([np.inf] * 3 + list(JOINT_VEL_LIMIT)) * self.n_agents

        super().__init__(scene, action_spec, observation_spec, gamma, horizon, timestep, n_substeps,
                         n_intermediate_steps, additional_data, collision_spec, max_joint_vel, **viewer_params)

        # Robot-only model at the origin. It must carry exactly N_JOINTS joints: PositionControl
        # calls mj_mulM() on it with an N_JOINTS-length acceleration and inverse_kinematics
        # returns its whole qpos, so a model that also carries the passive striker joints
        # silently produces wrong-shaped actions.
        robot_model = mujoco.MjModel.from_xml_path(os.path.join(ASSET_DIR, "metal_arm_only.xml"))
        robot_model.body('metal_1/base_link').pos = np.zeros(3)
        robot_data = mujoco.MjData(robot_model)
        assert robot_model.nq == N_JOINTS, \
            f"metal_arm_only.xml must expose exactly {N_JOINTS} joints, got {robot_model.nq}"

        self.env_info['dt'] = self.dt
        self.env_info["robot"]["joint_pos_limit"] = np.array(
            [self._model.joint(name).range for name in self.arm_joint_names[0]]).T
        self.env_info["robot"]["robot_model"] = robot_model
        self.env_info["robot"]["robot_data"] = robot_data
        self.env_info["rl_info"] = self.info

        frame_T = np.eye(4)
        temp = np.zeros((9, 1))
        for r in range(1, self.n_agents + 1):
            mujoco.mju_quat2Mat(temp, self._model.body(f"metal_{r}/base_link").quat)
            frame_T[:3, :3] = temp.reshape(3, 3)
            frame_T[:3, 3] = self._model.body(f"metal_{r}/base_link").pos
            self.env_info['robot']['base_frame'].append(frame_T.copy())

        # Ids of the joint, which are controlled by the action space
        self.actuator_joint_ids = [self._model.joint(name).id for name in action_spec]

        # Grip bookkeeping. The gripper is deliberately outside the action space: the agent
        # controls the six arm joints and the grip is simply held shut, the way a bolted-on
        # tool would be.
        self._grip_ctrl_ids = [self._model.actuator(f"metal_{r}/joint7").id
                               for r in range(1, self.n_agents + 1)]
        self._grip_qpos_adr = [self._model.joint(f"metal_{r}/joint{j}").qposadr[0]
                               for r in range(1, self.n_agents + 1) for j in (7, 8)]
        self._mallet_qpos_adr = [self._model.joint(f"mallet_{r}/free").qposadr[0]
                                 for r in range(1, self.n_agents + 1)]
        self._mallet_dof_adr = [self._model.joint(f"mallet_{r}/free").dofadr[0]
                                for r in range(1, self.n_agents + 1)]
        self._gripper_base_ids = [self._model.body(f"metal_{r}/gripper_base").id
                                  for r in range(1, self.n_agents + 1)]

    def _compute_init_state(self):
        """Home configuration, shared by both arms (each in its own base frame).

        The mallet is held flat by the passive universal joint, so the home pose only has to
        put the ROD vertical: striker_joint_link's +Z (the gripper pointing axis) along world
        -Z, which is Rx(pi). Unlike the full-size env there is no rod-angle choice to make --
        the mount is 0.05 m, not 0.585 m, so the wrist sits directly above the contact point.
        """
        success, self.init_state = inverse_kinematics(self.env_info['robot']['robot_model'],
                                                      self.env_info['robot']['robot_data'],
                                                      HOME_POS,
                                                      R.from_euler('xyz', [np.pi, 0, 0]).as_matrix(),
                                                      initial_q=INIT_SEED)

        assert success is True, f"Home-pose IK failed for {HOME_POS}"

    def _modify_mdp_info(self, mdp_info):
        # Called from inside MuJoCo.__init__, so env_info is not populated yet -- read the
        # limits straight off the compiled model.
        q_range = np.array([self._model.joint(name).range for name in self.arm_joint_names[0]]).T
        # Puck (x, y) bounds in the robot's own frame. With the base beside a long rail the
        # playing surface is not symmetric about any robot axis, so transform the four
        # surface corners through the base frame and take the envelope, with a margin.
        body = self._model.body("metal_1/base_link")
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, body.quat)
        rot = rot.reshape(3, 3)
        corners = np.array([[sx * TABLE_LENGTH / 2, sy * TABLE_WIDTH / 2, 0.0]
                            for sx in (-1, 1) for sy in (-1, 1)])
        corners_r = (corners - body.pos) @ rot
        lo = corners_r.min(axis=0)[:2] - 0.05
        hi = corners_r.max(axis=0)[:2] + 0.05

        obs_low = np.array([lo[0], lo[1], -np.pi, -20., -20., -100., *q_range[0], *(-JOINT_VEL_LIMIT)])
        obs_high = np.array([hi[0], hi[1], np.pi, 20., 20., 100., *q_range[1], *JOINT_VEL_LIMIT])
        if self.n_agents == 2:
            obs_low = np.concatenate([obs_low, [lo[0], lo[1], -0.5]])
            obs_high = np.concatenate([obs_high, [hi[0], hi[1], 0.5]])
        mdp_info.observation_space = Box(obs_low, obs_high)
        return mdp_info

    def _simulation_pre_step(self):
        # Hold the jaws shut. mushroom_rl writes the agent's torques into ctrl[action_indices]
        # just before this call, so the gripper actuator has to be re-commanded every substep
        # or it would sit at whatever it was left at.
        self._data.ctrl[self._grip_ctrl_ids] = GRIP_FORCE

    def _seat_mallet_in_gripper(self):
        """Open the jaws to the handle and place the mallet between them, at rest.

        Called from `setup` after the arm has been posed. Without this the mallet would spawn
        at the world origin and the arm would have to go and pick it up.
        """
        mujoco.mj_fwdPosition(self._model, self._data)

        for i in range(self.n_agents):
            # Jaws open to exactly the handle half-width, so the cube starts in contact with
            # both pads rather than being squeezed out of an already-shut gripper.
            self._data.qpos[self._grip_qpos_adr[2 * i]] = -HANDLE_HALF
            self._data.qpos[self._grip_qpos_adr[2 * i + 1]] = HANDLE_HALF

            body = self._data.body(self._gripper_base_ids[i])
            rot = body.xmat.reshape(3, 3)
            grasp_pos = body.xpos + rot @ np.array([0.0, 0.0, GRASP_OFFSET])

            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.reshape(-1))

            adr = self._mallet_qpos_adr[i]
            self._data.qpos[adr:adr + 3] = grasp_pos
            self._data.qpos[adr + 3:adr + 7] = quat

            dof = self._mallet_dof_adr[i]
            self._data.qvel[dof:dof + 6] = 0.0

        mujoco.mj_fwdPosition(self._model, self._data)

    def grip_state(self, agent=1):
        """How the grip is actually doing: (slip, tilt_deg, jaw_opening).

        slip      distance from the mallet's handle centre to the gripper's grasp point (m).
                  0 means the mallet has not moved in the jaws at all.
        tilt_deg  angle between the mallet's axis and vertical.
        jaw       half-opening of the jaws (m); HANDLE_HALF means still on the handle.

        The mallet is held by friction, so none of these are guaranteed by construction --
        that is the point of measuring them.
        """
        i = agent - 1
        body = self._data.body(self._gripper_base_ids[i])
        grasp_pos = body.xpos + body.xmat.reshape(3, 3) @ np.array([0.0, 0.0, GRASP_OFFSET])

        mallet = self._data.body(f"mallet_{agent}")
        slip = float(np.linalg.norm(mallet.xpos - grasp_pos))
        axis_z = mallet.xmat.reshape(3, 3)[2, 2]
        tilt = float(np.degrees(np.arccos(np.clip(-axis_z, -1.0, 1.0))))
        jaw = float(abs(self._data.qpos[self._grip_qpos_adr[2 * i]]))
        return slip, tilt, jaw

    def is_absorbing(self, obs):
        boundary = np.array([self.env_info['table']['length'], self.env_info['table']['width']]) / 2
        puck_pos, puck_vel = self.get_puck(obs)

        if np.any(np.abs(puck_pos[:2]) > boundary) or np.linalg.norm(puck_vel) > 100:
            return True
        return False

    @staticmethod
    def _puck_2d_in_robot_frame(puck_in, robot_frame, type='pose'):
        if type == 'pose':
            puck_w = np.eye(4)
            puck_w[:2, 3] = puck_in[:2]
            puck_w[:3, :3] = R.from_euler("xyz", [0., 0., puck_in[2]]).as_matrix()

            puck_r = np.linalg.inv(robot_frame) @ puck_w
            puck_out = np.concatenate([puck_r[:2, 3],
                                       R.from_matrix(puck_r[:3, :3]).as_euler('xyz')[2:3]])

        if type == 'vel':
            rot_mat = robot_frame[:3, :3]

            vel_lin = np.array([*puck_in[:2], 0])
            vel_ang = np.array([0., 0., puck_in[2]])

            vel_lin_r = rot_mat.T @ vel_lin
            vel_ang_r = rot_mat.T @ vel_ang

            puck_out = np.concatenate([vel_lin_r[:2], vel_ang_r[2:3]])
        return puck_out

    def get_puck(self, obs):
        """
        Getting the puck properties from the observations
        Args:
            obs: The current observation

        Returns:
            ([pos_x, pos_y, yaw], [lin_vel_x, lin_vel_y, yaw_vel])

        """
        puck_pos = np.concatenate([self.obs_helper.get_from_obs(obs, "puck_x_pos"),
                                   self.obs_helper.get_from_obs(obs, "puck_y_pos"),
                                   self.obs_helper.get_from_obs(obs, "puck_yaw_pos")])
        puck_vel = np.concatenate([self.obs_helper.get_from_obs(obs, "puck_x_vel"),
                                   self.obs_helper.get_from_obs(obs, "puck_y_vel"),
                                   self.obs_helper.get_from_obs(obs, "puck_yaw_vel")])
        return puck_pos, puck_vel

    def get_ee(self):
        raise NotImplementedError

    def get_joints(self, obs):
        raise NotImplementedError

    def reward(self, obs, action, cur_obs, absorbing):
        return 0.0
