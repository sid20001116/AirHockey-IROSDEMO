"""The METAL arm SDK's low-level joint controller, ported natively into MuJoCo.

Per joint, this reproduces::

    HOST, every 2.5 ms (400 Hz -- can_manager.cpp:996-998):
        qd_ff  = qd_meas with |qd| < 0.05 zeroed              can_manager.cpp:1016-1021
        tau_ff = Kg*G(q) + Kc*C(q,qd_ff) + b*clamp(qd_ff)     kdl_solver.cpp:290-321, 438-468
        tau_ff = clip(tau_ff, -tmax, +tmax)                   dm_motor_writer.cpp:44-45
        q_des  = clip(q_ref, position_min, position_max)      dm_motor_writer.cpp:37-42
        -> packed into one CAN frame and HELD until the next tick

    MOTOR FIRMWARE, continuously (here: every 1 ms substep):
        tau = kp*(q_des - q) + kd*(qd_des - qd) + tau_ff
        tau = clip(tau, -tmax, +tmax)

The planner, the FSM, the optimizer and the ``(2, 6)`` 50 Hz action are untouched. This class
replaces ``PositionControl._controller`` and ``PositionControl._enforce_safety_limits``,
nothing else.
"""

import mujoco
import numpy as np

from mini_platform.core import sdk_params
from mini_platform.core.position_control import PositionControl


def _quantize(x, lo, hi, bits):
    """``dm_motor_writer.cpp:26-32`` ``float_to_uint``, followed by the firmware's inverse.

    DEVIATION 2: the SDK truncates into a ``uint16_t`` WITHOUT clamping, so an out-of-range
    argument wraps. We clamp first. Position and torque are clamped by the SDK before this call
    anyway (:37-45); only ``velocity`` can actually wrap there.
    """
    n = float((1 << bits) - 1)
    return lo + np.floor((np.clip(x, lo, hi) - lo) / (hi - lo) * n) * (hi - lo) / n


class SdkJointControl(PositionControl):
    """The SDK's ``NRT_JOINT_POSITION`` law, in place of the challenge stack's PD."""

    # ----------------------------------------------------------------------- construction
    def __init__(self, sdk_config=None, **kwargs):
        cfg = sdk_params.nrt_config() if sdk_config is None else sdk_config
        self.sdk = cfg

        # PositionControl owns the interpolator, the action-shape contract, prev_pos/prev_vel
        # and reset() -- all kept. Its OWN gains are never read here; zeros both satisfy the
        # signature and make a stray call to the legacy _controller inert.
        z = [0.] * sdk_params.N
        super().__init__(p_gain=z, d_gain=z, i_gain=z, **kwargs)

        na = self.n_agents
        n = self._num_env_joints

        def tile(v):
            # np.tile, NOT `list * n`: the legacy code relies on Python list repetition, which
            # is silently correct at n_agents == 1 and wrong for any array. Test T5 guards it.
            return np.tile(np.asarray(v, dtype=float), na)

        self._kp = tile(cfg.kp)
        self._kd = tile(cfg.kd)
        self._pmax = tile(cfg.pmax)
        self._vmax = tile(cfg.vmax)
        self._b = tile(cfg.b_viscous)
        self._vlo = tile(cfg.visc_vel_min)
        self._vhi = tile(cfg.visc_vel_max)

        if cfg.quantize:                                    # dm_motor_writer.cpp:52-53
            self._kp = _quantize(self._kp, 0.0, 500.0, 12)  # 25 -> 24.908425, 20 -> 19.902320
            self._kd = _quantize(self._kd, 0.0, 5.0, 12)    # 0.2 -> 0.199023

        # TWO torque ceilings, deliberately.
        #   _tmax_ff  clamps the FEEDFORWARD, host-side (dm_motor_writer.cpp:44-45).
        #   _tmax_out clamps the total. MuJoCo clips data.ctrl to the actuator's ctrlrange and
        #             the joint's actuatorfrcrange regardless; doing it here only makes the
        #             saturation observable instead of silent. DEVIATION 3.
        self._tmax_ff = tile(cfg.tmax)
        plant = np.asarray(self._model.actuator_ctrlrange)[self._action_indices]
        assert np.allclose(plant[:, 0], -plant[:, 1]), "asymmetric actuator ctrlrange"
        self._tmax_out = np.minimum(self._tmax_ff, plant[:, 1])

        # Command position clamp = intersection of the SDK's soft limits and the MJCF joint
        # range. MEASURED: the MJCF range is the narrower of the two on ALL SIX joints (it
        # rounds the SDK's limits down by <= 0.0025 rad), so this IS
        # env_info['robot']['joint_pos_limit'] -- the same array the planner and
        # JointPositionConstraint use. Commanding outside it would fight MuJoCo's own joint
        # limit constraint for no benefit.
        mjcf = np.tile(self.env_info['robot']['joint_pos_limit'], (1, na))
        self._pos_min = np.maximum(tile(cfg.pos_min), mjcf[0])
        self._pos_max = np.minimum(tile(cfg.pos_max), mjcf[1])

        # Raw plant addresses. qpos is indexed by jnt_qposadr and qvel by jnt_dofadr, NOT by
        # joint id. They coincide in mini_single/mini_double today -- this is correct
        # regardless of body ordering.
        jid = np.asarray(self.actuator_joint_ids)
        self._qadr = self._model.jnt_qposadr[jid]
        self._vadr = self._model.jnt_dofadr[jid]
        assert np.array_equal(self._qadr, jid) and np.array_equal(self._vadr, jid), (
            "PositionControl.reset (core/position_control.py:180-184) indexes qpos/qvel by "
            "JOINT ID. This model's qposadr/dofadr no longer coincide with joint ids, so the "
            "inherited reset is now wrong. Fix core/position_control.py first.")

        # PRIVATE inverse-dynamics MjData. self.robot_data IS env_info['robot']['robot_data'],
        # which MetalPositionHit's opponent generator also hands to inverse_kinematics once per
        # env step. The legacy _controller wrote through it 20x per step; a controller that
        # ticks at 400 Hz must not share it. The MODEL is shared (read-only), only data is ours.
        #
        # metal_arm_only.xml is the right feedforward model, not a new asset built from the
        # SDK's own URDF:
        #  * MEASURED over 49 IK-solved play poses against the true plant, the worst-case
        #    static droop of ANY candidate -- Kg*G_sdk_urdf, Kg*G_arm_only, G_sdk_urdf,
        #    G_arm_only -- is <= 0.0026 rad, about 1.6 mm at the ee. kp = 500 swamps the choice.
        #  * VERIFIED: dof_damping and dof_armature do NOT enter qfrc_bias (max diff 0.0), so
        #    this model is a clean G/C source despite carrying damping=2 / armature=0.1.
        #  * DECISIVE: its armature=0.1 is what makes mj_mulM match the plant. A model built
        #    from the SDK URDF would have armature=0 and under-compensate M(q)*qdd by 0.1*qdd
        #    -- 38 N.m at J2's measured peak qdd of 383 rad/s^2.
        self._dyn_m = self.robot_model
        self._dyn_d = mujoco.MjData(self._dyn_m)
        assert self._dyn_m.nv == self.n_robot_joints
        self._G = np.zeros(self.n_robot_joints)
        self._Bias = np.zeros(self.n_robot_joints)
        self._Mqdd = np.zeros(self.n_robot_joints)

        self._hold = self._hold_mask(self._n_intermediate_steps, self._timestep, cfg.hold_hz)
        self._substep = 0
        self._cmd_q = np.zeros(n)
        self._cmd_qd = np.zeros(n)
        self._cmd_tau = np.zeros(n)

        for a in (self._kp, self._kd, self._b, self._tmax_ff, self._tmax_out,
                  self._pos_min, self._pos_max, self._vmax, self._pmax):
            assert a.shape == (n,), f"gain array is {a.shape}, expected {(n,)}"

    # ----------------------------------------------------------------------- 400 Hz clock
    @staticmethod
    def _hold_mask(n_substeps, timestep, hold_hz):
        """Which substeps re-latch the held CAN command.

        The sim substep is 1 ms; the SDK control thread is 400 Hz = 2.5 ms. They do not divide,
        so a phase accumulator is used -- evaluated on the SAMPLE time. Substep ``j`` commands
        the interpolator sample at ``t = (j+1)*timestep``, so accumulating on ``j+1`` makes the
        LAST latch of the step land exactly on ``t = dt``, the segment endpoint the planner
        actually asked for::

            latch at substeps  2  4  7  9 12 14 17 19
            sample times (ms)  3  5  8 10 13 15 18 20
            gaps (ms)          2  3  2  3  2  3  2  3     (mean exactly 2.5)

        8 latches per 20 ms env step = 400 Hz exactly, phase-locked, no drift. The pattern
        repeats identically every step because ``hold_hz * dt = 8`` is an integer; that is
        asserted so a future change to ``dt`` cannot silently alias.

        ``hold_hz=500`` -> every 2 substeps, perfectly even, 25 % faster than the hardware.
        ``hold_hz=1000`` / ``None`` -> re-latch every substep. NOT the hardware: it removes the
        command staircase and hands the PD a 1 kHz reference. Upper bound only.
        """
        if hold_hz is None:
            return np.ones(n_substeps, dtype=bool)
        ticks = hold_hz * timestep * n_substeps
        assert abs(ticks - round(ticks)) < 1e-9, (
            f"hold_hz={hold_hz} does not divide the {n_substeps * timestep:g} s env step; "
            "the latch pattern would alias across steps")
        j = np.arange(n_substeps)
        return (np.floor((j + 1) * hold_hz * timestep + 1e-9)
                > np.floor(j * hold_hz * timestep + 1e-9))

    # ------------------------------------------------------------------ rigid-body terms
    def _rigid_body_terms(self, q, qd, qdd_ref):
        """``G(q)``, ``C(q,qd)qd`` and (optionally) ``M(q)@qdd_ref`` from one pipeline.

        MuJoCo exposes only the combined bias force. The split is exact, not an approximation:
        ``G := qfrc_bias at qvel = 0``, ``C := qfrc_bias(q,qd) - G(q)``.

        ``mj_forward`` is overkill on a model with no contacts, constraints or actuators. This
        pipeline is VERIFIED bit-identical to ``mj_forward``'s ``qfrc_bias`` (max |diff| = 0.0
        over 200 random states) and to ``mj_forward + mj_mulM`` for the inertia term, and it is
        3.6x faster: 9.46 us for all three terms vs 24.5 us for two ``mj_forward`` calls. At 8
        latches per env step that is 76 us -- against 240 us for the legacy controller, which
        calls ``mj_forward`` on this same model 20x per step. The port is CHEAPER than what it
        replaces.

        ``mj_rne`` reads ``d.cvel`` / ``d.cdof_dot``, NOT ``d.qvel``, so ``mj_comVel`` MUST be
        re-run after writing ``qvel`` -- skipping it silently returns ``G`` twice.
        """
        m, d = self._dyn_m, self._dyn_d
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_rne(m, d, 0, self._G)                  # G(q)
        if self.sdk.inertia_ff:
            mujoco.mj_crb(m, d)                          # qM, incl. dof_armature
            mujoco.mj_mulM(m, d, self._Mqdd, qdd_ref)
        d.qvel[:] = qd
        mujoco.mj_comVel(m, d)                           # position stages still valid
        mujoco.mj_rne(m, d, 0, self._Bias)               # G(q) + C(q,qd)qd
        return self._G, self._Bias - self._G, self._Mqdd

    # ------------------------------------------------------------------------- host tick
    def _host_tick(self, q_ref, qd_ref, qdd_ref, q, qd):
        cfg = self.sdk
        nj = self.n_robot_joints

        # can_manager.cpp:1016-1021. The deadband is applied to a LOCAL copy of the measured
        # velocity that is passed only to FeedforwardTorqueCompensation (:1025-1026). It never
        # reaches the kd term: kd is closed in motor firmware against the motor's own velocity
        # estimate, which the host neither transmits nor filters.
        qd_ff = np.where(np.abs(qd) < cfg.vel_deadband, 0.0, qd)

        tau = np.empty(self._num_env_joints)
        for i in range(self.n_agents):
            s = slice(i * nj, (i + 1) * nj)
            G, C, Mqdd = self._rigid_body_terms(q[s], qd_ff[s], qdd_ref[s])
            tau[s] = cfg.k_gravity * G + cfg.k_coriolis * C      # kdl_solver.cpp:317-318
            if cfg.inertia_ff:
                tau[s] += Mqdd                                    # DEVIATION 1

        # kdl_solver.cpp:438-468. The clamp is on the VELOCITY entering the term, not on the
        # torque, so J3 saturates at 0.8*0.52 = 0.416 N.m and J4 at 1.2*0.42 = 0.504 N.m.
        # The sign is +b*qd (kdl_solver.cpp:319 adds friction_torque with a '+'): this is
        # feedforward that CANCELS drag, NOT damping that adds to it. If you ever "fix" this
        # into a damping term you will double the effect with the wrong sign; test T3 guards it.
        tau += self._b * np.clip(qd_ff, self._vlo, self._vhi)

        q_cmd = np.clip(q_ref, self._pos_min, self._pos_max)      # dm_motor_writer.cpp:37-42
        qd_cmd = np.clip(qd_ref, -self._vmax, self._vmax)         # DEVIATION 2
        tau = np.clip(tau, -self._tmax_ff, self._tmax_ff)         # dm_motor_writer.cpp:44-45

        if cfg.quantize:                                          # dm_motor_writer.cpp:50-54
            q_cmd = _quantize(q_cmd, -self._pmax, self._pmax, 16)      # 1.917e-4 rad
            qd_cmd = _quantize(qd_cmd, -self._vmax, self._vmax, 12)    # 4.9e-3 / 1.5e-2
            tau = _quantize(tau, -self._tmax_ff, self._tmax_ff, 12)    # 0.0147 .. 0.0586

        self._cmd_q[:] = q_cmd
        self._cmd_qd[:] = qd_cmd
        self._cmd_tau[:] = tau

    # ----------------------------------------------------------------------- per substep
    def _compute_action(self, obs, action):
        j = self._substep
        self._substep += 1

        # Consume one interpolator sample UNCONDITIONALLY. The generator yields exactly
        # _n_intermediate_steps tuples and MuJoCo.step calls _compute_action exactly that many
        # times (mushroom mujoco.py; _recompute_action_per_step is True because we override
        # _compute_action). Gating next() on the hold mask desynchronises the pair and the
        # failure is SILENT -- every 20 ms segment gets truncated to its first 40 % and
        # _step_init throws the rest away. The 12 non-latched samples are computed and
        # discarded; that IS the zero-order hold. Test T6 guards it.
        q_ref, qd_ref, qdd_ref = next(self.traj)

        if self.sdk.safety == "ros":
            # Debug affordance only. It MUST be called here, once per SUBSTEP: it advances
            # prev_controller_cmd_pos by min_vel * self._timestep with _timestep hard-coded to
            # 1e-3 (core/position_control.py:84-85). Calling it 8x instead of 20x per step
            # silently turns the rate limit into 0.4 * v_lim; MEASURED, that triples J4's
            # tracking error (p99 0.035 -> 0.107 rad) and drops the mallet (1/60 -> ~3/60).
            q_ref, qd_ref = PositionControl._enforce_safety_limits(self, q_ref, qd_ref)

        # RAW plant state, deliberately NOT get_joints(obs).
        # env_single._create_observation runs a first-order IIR (ratio 0.274) on the joint
        # velocity at the 1 ms substep rate -- tau = 3.13 ms, a 51 Hz lag, 10 deg of phase at
        # J4's closed-loop 9.4 Hz. The SDK's kd is closed in MOTOR FIRMWARE from the motor's
        # own encoder-difference velocity: noisy (hence the 0.05 rad/s deadband) but not
        # 51 Hz-lagged. The agent still sees the filtered velocity -- _create_observation is
        # untouched -- so the planner is not perturbed and the A/B stays clean.
        q = self._data.qpos[self._qadr]
        qd = self._data.qvel[self._vadr]

        if self._hold[j]:
            self._host_tick(q_ref, qd_ref, qdd_ref, q, qd)

        tau = (self._kp * (self._cmd_q - q)
               + self._kd * (self._cmd_qd - qd)
               + self._cmd_tau)
        tau_out = np.clip(tau, -self._tmax_out, self._tmax_out)

        if self.debug:
            # controller_record has ZERO consumers in this repo, so this layout is free. It is
            # EXTENDED, not reshaped: the first five blocks are what the legacy
            # PositionControl._controller used to log.
            self.controller_record.append(np.concatenate([
                q_ref, q, qd_ref, qd, qdd_ref,
                self._cmd_q, self._cmd_qd, self._cmd_tau, tau, tau_out,
                [float(j)], [float(self._hold[j])]]))
        return tau_out

    # ------------------------------------------------------------------------- lifecycle
    def _step_init(self, obs, action):
        super()._step_init(obs, action)      # PositionControl builds self.traj
        self._substep = 0

    def reset(self, obs=None):
        obs = super().reset(obs)             # re-syncs prev_pos / prev_vel to the plant
        self._substep = 0
        q = self._data.qpos[self._qadr].copy()
        qd = self._data.qvel[self._vadr].copy()
        # Seed the hold with a REAL host tick. The first latch of the first step is at substep
        # 2, so without this the arm spends 2 ms with tau_ff = 0 -- i.e. with no gravity
        # compensation -- immediately after every reset.
        self._host_tick(q, np.zeros_like(q), np.zeros_like(q), q, qd)
        return obs

    def _controller(self, *a, **kw):
        raise NotImplementedError(
            "SdkJointControl implements the law in _compute_action, because the 400 Hz hold "
            "needs the substep index that _controller is not given. Reaching this means "
            "something is calling the legacy path.")

    # NOTE: _simulation_pre_step is deliberately NOT defined here. AirHockeyBase's
    # (envs/env_base.py:275-279) writes GRIP_FORCE = 3.0 N into data.ctrl[_grip_ctrl_ids] every
    # substep and does not call super(). Shadowing it leaves the grip actuator at 0.0 and the
    # mallet -- a FREE body held only by friction -- falls out of the jaws within a few
    # substeps, with no exception and plausible-looking numbers. Test T7 guards it.
