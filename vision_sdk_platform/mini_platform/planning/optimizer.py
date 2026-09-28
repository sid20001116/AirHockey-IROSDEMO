# Vendored from the 2025 Robot Air Hockey Challenge (baseline/baseline_agent/optimizer.py).
# MIT License, Copyright (c) 2025 Atalay Donat, Puze Liu, Jonas Guenster, Davide Tateo.
# See ../../THIRD_PARTY.md. Modifications for the mini platform are marked MINI:.

import copy
import time

import mujoco
import nlopt
import numpy as np
import osqp
import scipy.linalg
from scipy import sparse

from mini_platform.core.kinematics import (flatten_config, forward_kinematics,
                                           inverse_kinematics, jacobian, link_to_xml_name)


class TrajectoryOptimizer:
    # MINI: gain (1/s) of the flat-mallet levelling task, and the tilt below which it stops
    # pushing. 4 1/s corrects a 50 deg lean in roughly a second without fighting the position
    # task; the deadband keeps the QP from chattering on a mallet that is already flat.
    FLAT_GAIN = 4.0
    FLAT_DEADBAND = np.radians(1.0)
    # MINI: how hard the levelling task is allowed to lean on the redundancy. Tuned over 250-step
    # sim episodes at both park positions, judged on wrist tilt AND how well the mallet holds the
    # play plane -- the two trade against each other, because every bit of null-space motion the
    # levelling demands is motion the trajectory's trapezoidal integration has to absorb:
    #
    #   damping   cap        wrist tilt (med/p90)      worst off-plane
    #   0 (exact) none       1.8 / 25.7 deg            3.7 mm     <- levels hard, tracks badly
    #   0.1       1.5 rad/s  0.9 / 10.1 deg            0.5 mm     <- this
    #   1.0       1.5 rad/s  1.0 / 14.6 deg            0.24 mm
    #   (soft)    --         2.6 / 20.8 deg            0.14 mm    <- the first cut of this option
    #
    # Undamped is the trap: near a singularity of the tool-axis map the exact solve asks for an
    # enormous null-space velocity, the QP's joint bounds clip it, and clipping breaks the
    # POSITION task the null space was supposed to leave alone. The damping is what makes
    # "level the tool with the spare dimensions" hold as a statement about the real solver.
    FLAT_DAMPING = 0.1
    FLAT_MAX_NULL_VEL = 1.5

    def __init__(self, env_info, keep_mallet_flat=False):
        self.env_info = env_info
        # MINI: use the arm's redundancy to hold the mallet FLAT on the table.
        #
        # The tracking task here is position-only (`jac[:3]`), which on this 6-DOF arm leaves a
        # 3-dimensional null space, and the vendored code spends it pulling the joints toward a
        # FIXED anchor pose -- one that, wherever `_vertical_hit_anchor` finds nothing, the hit
        # solver failed to converge on and simply returned its seed for, i.e. a pose a third of
        # a metre from the point being struck. Nothing in any of that says which way the tool
        # points, and measured in sim the wrist drifts from 1 deg off vertical at homing to a
        # median of 55 deg within a few seconds -- the mallet plays on its edge, and its face is
        # nowhere near the puck's equator. Holding the tool AXIS vertical costs 2 of the 3 spare
        # dimensions (spin about the axis is left free -- a disc mallet does not care), so it is
        # free as far as the position task is concerned, and the anchor keeps the third. Where
        # vertical is out of reach -- e.g. anywhere on the centreline at full cross-table
        # stretch -- the pull simply gets as close as the arm allows, which is the
        # point: the wrist angle compensates instead of drifting.
        #
        # Off by default: it changes every trajectory the optimiser emits, so sim and the whole
        # test suite stay byte-identical unless it is asked for.
        self.keep_mallet_flat = keep_mallet_flat
        self.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(env_info['robot']['robot_data'])
        self.n_joints = self.env_info['robot']['n_joints']
        # MINI: the [10,1,10,1,10,10,1] table is SEVEN weights -- it belongs to the vendored 7-DOF
        # iiwa. Applying it on a 6-DOF arm makes `_solve_aqp` raise
        # `ValueError: matmul ... size 7 is different from 6` on its first call, because
        # `np.diag(anchor_weights)` is then 7x7 against a 6-row null-space basis. That went
        # unnoticed only because `MiniBaselineAgent` overwrites the attribute with `np.ones(6)`
        # immediately after construction, so every path that builds a TrajectoryOptimizer or a
        # TrajectoryGenerator DIRECTLY was broken (two tests hand-patch it for exactly this
        # reason). Size it to the arm instead: uniform weights, which on a 6-DOF arm is also the
        # honest choice -- the null space is ~1-dimensional, so there is no structure for the
        # KUKA's alternating pattern to express. The 7-DOF case keeps the vendored table.
        if self.n_joints == 7:
            self.anchor_weights = np.array([10., 1., 10., 1., 10., 10., 1.])
        else:
            self.anchor_weights = np.ones(self.n_joints)

    def optimize_trajectory(self, cart_traj, q_start, dq_start, q_anchor):
        joint_trajectory = np.tile(np.concatenate([q_start]), (cart_traj.shape[0], 1))
        if len(cart_traj) > 0:
            q_cur = q_start.copy()
            dq_cur = dq_start.copy()

            for i, des_point in enumerate(cart_traj):
                if self.keep_mallet_flat:
                    # the RAW anchor pull; _solve_aqp ranks it below levelling the tool
                    dq_anchor = None if q_anchor is None else (q_anchor - q_cur)
                elif q_anchor is None:
                    dq_anchor = 0
                else:
                    dq_anchor = (q_anchor - q_cur)

                success, dq_next = self._solve_aqp(des_point[:3], q_cur, dq_anchor)

                if not success:
                    return success, []
                else:
                    q_cur += (dq_cur + dq_next) / 2 * self.env_info['dt']
                    # q_cur += dq_next * self.env_info['dt']
                    dq_cur = dq_next
                    joint_trajectory[i] = q_cur.copy()
            return True, joint_trajectory
        else:
            return False, []

    def _flat_anchor_dq(self, q_cur, dq_secondary, N_J, b):
        """MINI: the null-space target that levels the tool FIRST and only then chases the anchor.

        Ranking the two matters, and it cannot be done by weighting them: the QP measures its
        error in JOINT space, so simply adding an anchor pull to a levelling pull lets the
        anchor -- whose size grows without bound as the arm travels away from it -- swamp a
        levelling term that saturates at ``FLAT_GAIN``. (Measured: the wrist went straight back
        to drifting, 57 deg, with the levelling term present but outvoted.) So the priority is
        built into the answer instead.

        The QP can only realise ``dq = b + N_J z``, so both tasks are expressed in ``z``:

          * levelling solves ``M z = P(omega_des - J_w b)``, ``M = P J_w N_J`` -- the tool
            rotation the redundancy has to supply on top of whatever the position task already
            causes, with ``P`` dropping spin about the tool axis (see below);
          * the anchor then moves only within ``I - pinv(M) M``, i.e. the freedom levelling did
            not need, so it cannot fight it.

        The result lies in ``b + range(N_J)`` by construction, so the QP reproduces it exactly
        (subject only to its joint bounds) and the ranking survives contact with the solver --
        whatever ``anchor_weights`` says.
        """
        n = self.n_joints
        _, rot = forward_kinematics(self.robot_model, self.robot_data, q_cur)
        z_tool = rot @ np.array([0., 0., 1.])
        tilt = float(np.arccos(np.clip(-z_tool[2], -1.0, 1.0)))
        omega = (np.zeros(3) if tilt < self.FLAT_DEADBAND
                 else self.FLAT_GAIN * np.cross(z_tool, np.array([0., 0., -1.])))
        jac_w = jacobian(self.robot_model, self.robot_data, q_cur)[3:6, :n]
        # Constrain the tool AXIS, not the whole orientation: spin about the axis leaves a disc
        # mallet exactly where it was, so asking for it would burn a degree of freedom for
        # nothing -- and it burns the LAST one. (J_w N_J is generically 3x3 and invertible, so a
        # full-orientation target consumes the entire redundancy, leaving the anchor a null
        # projector of exactly zero. That is measurable: the secondary term had no effect at all
        # until this projection went in.) P drops the component along the tool axis, so
        # levelling costs 2 of the 3 spare dimensions and the anchor inherits the third.
        P = np.eye(3) - np.outer(z_tool, z_tool)
        M = P @ jac_w @ N_J
        rhs = P @ (omega - jac_w @ b)
        z = M.T @ np.linalg.solve(M @ M.T + self.FLAT_DAMPING * np.eye(3), rhs)
        if dq_secondary is not None:
            # what is left of the anchor pull after levelling has taken what it needs
            resid = np.asarray(dq_secondary, float)[:n] - (b + N_J @ z)
            null_M = np.eye(N_J.shape[1]) - np.linalg.pinv(M) @ M
            z = z + null_M @ (N_J.T @ resid)                 # N_J's columns are orthonormal
        speed = np.linalg.norm(z)                # == ||N_J z||: N_J's columns are orthonormal,
        if speed > self.FLAT_MAX_NULL_VEL:       # so this is the null-space joint speed in rad/s
            z = z * (self.FLAT_MAX_NULL_VEL / speed)
        return b + N_J @ z

    def flatten_config(self, q_0, x_des, **kwargs):
        """MINI: level ``q_0`` in place, without letting the tool leave ``x_des``.

        Used to finish a configuration another solver chose (a hit anchor picked for velocity
        manipulability, say): the position and the posture it settled on are kept, and only
        the wrist tilt it never considered is taken out. Returns ``(reached, q, tilt)``.
        """
        return flatten_config(self.robot_model, self.robot_data, q_0, x_des,
                              n_joints=self.n_joints, **kwargs)

    def _solve_aqp(self, x_des, q_cur, dq_anchor):
        x_cur = forward_kinematics(self.robot_model, self.robot_data, q_cur)[0]
        jac = jacobian(self.robot_model, self.robot_data, q_cur)[:3, :self.n_joints]
        N_J = scipy.linalg.null_space(jac)
        b = np.linalg.lstsq(jac, (x_des - x_cur) / self.env_info['dt'], rcond=None)[0]

        if self.keep_mallet_flat:
            dq_anchor = self._flat_anchor_dq(q_cur, dq_anchor, N_J, b)

        P = (N_J.T @ np.diag(self.anchor_weights) @ N_J) / 2
        q = (b - dq_anchor).T @ np.diag(self.anchor_weights) @ N_J
        A = N_J.copy()
        u = np.minimum(self.env_info['robot']['joint_vel_limit'][1] * 0.9,
                       (self.env_info['robot']['joint_pos_limit'][1] * 0.92 - q_cur) / self.env_info['dt']) - b
        l = np.maximum(self.env_info['robot']['joint_vel_limit'][0] * 0.9,
                       (self.env_info['robot']['joint_pos_limit'][0] * 0.92 - q_cur) / self.env_info['dt']) - b

        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(P), q=q, A=sparse.csc_matrix(A), l=l, u=u, verbose=False, polish=False)

        result = solver.solve()
        if result.info.status == 'solved':
            return True, N_J @ result.x + b
        else:
            return False, b

    def solve_hit_config(self, x_des, v_des, q_0):
        reg = 1e-6
        dim = q_0.shape[0]
        opt = nlopt.opt(nlopt.LD_SLSQP, dim)

        def _nlopt_f(q, grad):
            if grad.size > 0:
                grad[...] = numerical_grad(_nlopt_f, q)
            f = v_des @ jacobian(self.robot_model, self.robot_data, q)[:3, :dim]
            return f @ f + reg * np.linalg.norm(q - q_0)

        def _nlopt_h(q, grad):
            if grad.size > 0:
                grad[...] = 2 * (forward_kinematics(self.robot_model, self.robot_data, q)[0] - x_des) @ \
                            jacobian(self.robot_model, self.robot_data, q)[:3, :dim]
            return np.linalg.norm(forward_kinematics(self.robot_model, self.robot_data, q)[0] - x_des) ** 2 - 1e-4

        opt.set_max_objective(_nlopt_f)
        opt.set_lower_bounds(self.env_info['robot']['joint_pos_limit'][0])
        opt.set_upper_bounds(self.env_info['robot']['joint_pos_limit'][1])
        opt.add_inequality_constraint(_nlopt_h)
        opt.set_ftol_abs(1e-6)
        opt.set_xtol_abs(1e-8)
        opt.set_maxtime(5e-3)

        success, x = inverse_kinematics(self.robot_model, self.robot_data, x_des, initial_q=q_0)
        if not success:
            raise NotImplementedError("Need to check")
        xopt = opt.optimize(x[:dim])
        return opt.last_optimize_result() > 0, xopt

    def solve_hit_config_ik_null(self, x_des, v_des, q_0, max_time=5e-3):
        t_start = time.time()
        reg = 0e-6
        dim = q_0.shape[0]
        IT_MAX = 1000
        eps = 1e-4
        damp = 1e-3
        progress_thresh = 20.0
        max_update_norm = 0.1
        i = 0
        TIME_MAX = max_time
        success = False

        dtype = self.robot_data.qpos.dtype

        self.robot_data.qpos = q_0

        q_l = self.robot_model.jnt_range[:, 0]
        q_h = self.robot_model.jnt_range[:, 1]
        lower_limit = (q_l + q_h) / 2 - 0.95 * (q_h - q_l) / 2
        upper_limit = (q_l + q_h) / 2 + 0.95 * (q_h - q_l) / 2

        name = link_to_xml_name(self.robot_model, 'ee')

        def objective(q, grad):
            if grad.size > 0:
                grad[...] = numerical_grad(objective, q)
            f = v_des @ jacobian(self.robot_model, self.robot_data, q)[:3, :dim]
            return f @ f + reg * np.linalg.norm(q - q_0)

        null_opt_stop_criterion = False
        while True:
            # forward kinematics
            mujoco.mj_fwdPosition(self.robot_model, self.robot_data)

            x_pos = self.robot_data.body(name).xpos

            err_pos = x_des - x_pos
            error_norm = np.linalg.norm(err_pos)

            f_grad = numerical_grad(objective, self.robot_data.qpos.copy())
            f_grad_norm = np.linalg.norm(f_grad)
            if f_grad_norm > max_update_norm:
                f_grad = f_grad / f_grad_norm

            if error_norm < eps:
                success = True
            if time.time() - t_start > TIME_MAX or i >= IT_MAX or null_opt_stop_criterion:
                break

            jac_pos = np.empty((3, self.robot_model.nv), dtype=dtype)
            mujoco.mj_jacBody(self.robot_model, self.robot_data, jac_pos, None, self.robot_model.body(name).id)

            update_joints = jac_pos.T @ np.linalg.inv(jac_pos @ jac_pos.T + damp * np.eye(3)) @ err_pos

            # Add Null space Projection
            null_dq = (np.eye(self.robot_model.nv) - np.linalg.pinv(jac_pos) @ jac_pos) @ f_grad
            null_opt_stop_criterion = np.linalg.norm(null_dq) < 1e-4
            update_joints += null_dq

            update_norm = np.linalg.norm(update_joints)

            # Check whether we are still making enough progress, and halt if not.
            progress_criterion = error_norm / update_norm
            if progress_criterion > progress_thresh:
                success = False
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            mujoco.mj_integratePos(self.robot_model, self.robot_data.qpos, update_joints, 1)
            self.robot_data.qpos = np.clip(self.robot_data.qpos, lower_limit, upper_limit)
            i += 1
        q_cur = self.robot_data.qpos.copy()

        return success, q_cur


def numerical_grad(fun, q):
    eps = np.sqrt(np.finfo(np.float64).eps)
    grad = np.zeros_like(q)
    for i in range(q.shape[0]):
        q_pos = q.copy()
        q_neg = q.copy()
        q_pos[i] += eps
        q_neg[i] -= eps
        grad[i] = (fun(q_pos, np.array([])) - fun(q_neg, np.array([]))) / 2 / eps
    return grad
