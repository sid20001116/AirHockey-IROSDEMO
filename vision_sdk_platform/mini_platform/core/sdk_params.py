"""Numeric tables transcribed from the METAL arm SDK.

Source of truth: ``mini_platform/metal-python-ros/metal_sdk/native/``. Every constant carries
a file:line citation. Nothing here is tuned, fitted or scaled -- if a value disagrees with the
C++, the C++ is right and this file is wrong.

DEVIATIONS FROM THE SDK, all of them here and all opt-out:

1. ``inertia_ff=True`` -- the SDK computes ``M(q)*qdd`` (``kdl_solver.cpp:329``) and never
   calls it. Measured on this plant: removing it takes defend mallet drops from 1/60 to 7/60
   and turns the ``ee_constr`` z margin from -0.0009 to +0.0195. The sim's ``M`` is 84-99.5 %
   armature on J4-J6, so this term is far larger here than on the real arm. Set ``False`` for
   the literal SDK law.
2. ``qd_cmd`` is CLAMPED to +/-vmax before quantisation. ``dm_motor_writer.cpp:53`` does not
   clamp velocity, so an out-of-range value wraps through ``float_to_uint``. Reproducing an
   integer wrap would inject garbage, not fidelity.
3. The output torque is clipped to ``min(SDK tmax, the MJCF's actuator ctrlrange)``. MuJoCo
   clips ``data.ctrl`` to ``ctrlrange`` regardless; doing it here only makes the saturation
   observable instead of silent.

NOT PORTED: the SDK's own 400 Hz quintic re-planner (``trajectory_time_allocator.h``,
``polynomial_interpolation.cpp``) and therefore its ~27 ms transport lag. The user's planner
and ``PositionControl``'s cubic interpolator keep the 50 Hz ``(2, 6)`` contract.
"""

from dataclasses import dataclass, field

import numpy as np

DEG = np.pi / 180.0
N = 6

# ---- config/motor_config.cpp, kMotorInfos[0..5]  (field order: config/motor_config.h:21-55)
MIT_KP    = np.array([300., 500., 500., 400.,  25.,  20.])  # mit_kp        :17 :38 :59 :80 :101 :122
MIT_KD    = np.array([  5.,   5.,   5.,   4.,  0.2,  0.2])  # mit_kd        :18 :39 :60 :81 :102 :123
FOLLOW_KP = np.array([200., 500., 400., 200.,  20.,  20.])  # follow_mit_kp :19 :40 :61 :82 :103 :124
FOLLOW_KD = np.array([  3.,   5.,   5.,   2.,  0.1,  0.1])  # follow_mit_kd :20 :41 :62 :83 :104 :125
TMAX      = np.array([ 30., 120.,  30.,  30.,  20.,  20.])  # tmax  [N.m]   :14 :35 :56 :77 :98 :119
VMAX      = np.array([ 10.,  10.,  10.,  10.,  30.,  30.])  # vmax  [rad/s] :13 :34 :55 :76 :97 :118
PMAX      = np.full(N, 6.28)                                # pmax  [rad]   :12 :33 :54 :75 :96 :117
POS_MIN   = np.array([-160., -180.,   0., -123., -85., -145.]) * DEG   # :15 :36 :57 :78 :99  :120
POS_MAX   = np.array([ 160.,    0., 180.,   81.,  85.,  145.]) * DEG   # :16 :37 :58 :79 :100 :121

# Informational only. reduction_ratio is used at exactly one site, dm_motor_reader.cpp:102, to
# turn phase current into an output-torque estimate. It never touches position, velocity, kp,
# kd or the commanded torque -- that path is 1:1. (config/motor_config.h:52)
REDUCTION = np.array([40., 48., 40., 10., 10., 10.])

# The header comment at motor_config.h:30 says these limits are degrees. They are radians --
# the source writes them as `-160 * M_PI / 180`. The comment is wrong, the values are right.

GRIPPER_KP, GRIPPER_KD, GRIPPER_TMAX = 20.0, 0.1, 20.0      # kMotorInfos[6] :143 :144 :140
GRIPPER_POS_MIN, GRIPPER_POS_MAX     = 0.0, 2.4             # :141 :142
GRIPPER_STROKE_CLAMP_MM              = 100.0                # can_manager.cpp:581

# ---- kinodynamic/kdl_solver.cpp ---------------------------------------------------------
K_GRAVITY  = np.array([1.2, 1.15, 1.1, 1.15, 1.0, 1.0])     # gravity_torque_coe   :290
K_CORIOLIS = np.array([1.1, 1.15, 1.0, 1.0,  1.1, 1.1])     # coriolis_torque_coe  :291
B_VISCOUS  = np.array([0.15, 0.3025, 0.52, 0.42, 0.015, 0.015])                  # :441-442

# The clamp is on the VELOCITY entering the term, not on the resulting torque, so J3 saturates
# at 0.8 * 0.52 = 0.416 N.m and J4 at 1.2 * 0.42 = 0.504 N.m.   :450-456
VISC_VEL_MIN = np.array([-np.inf, -np.inf, -0.8, -1.2, -np.inf, -np.inf])
VISC_VEL_MAX = np.array([ np.inf,  np.inf,  0.8,  1.2,  np.inf,  np.inf])

# Present but COMMENTED OUT in ComputeFriction (:443-444, :462-464). Recorded because it is the
# only hardware statement about stiction we have. DO NOT feed this forward.
B_COULOMB_DISABLED = np.array([0.0, 0.1, 0.2, 0.3, 0.03, 0.0])

GRAVITY = np.array([0.0, 0.0, -9.81])                       # :105

VEL_DEADBAND = 0.05   # rad/s, MEASURED velocity, FEEDFORWARD PATH ONLY. can_manager.cpp:1016-1021
CONTROL_HZ   = 400.0  # can_manager.cpp:996-998

# ---- trajectory/trajectory_time_allocator.h:76-79 -- recorded, NOT ported. The sim's planner
# owns time allocation. Kept so the omission is explicit and reversible.
TRAJ_LIMITS  = dict(vmax=10.0, amax=3000.0, jmax=500.0, Tmin=0.0)   # streaming (< 190 ms apart)
POINT_LIMITS = dict(vmax=3.14, amax=13.0,   jmax=20.0,  Tmin=0.5)   # point-to-point; note Tmin!
K_V, K_A, K_J = 1.875, 5.7735, 60.0                                 # :30-32; K_J is DISCARDED at :68

# ---- Hardware mapping ---------------------------------------------------------------------
# UNVERIFIED AGAINST HARDWARE. Nothing in metal-python-ros establishes a joint sign or zero
# map: GetJointPosition() returns the motor angle verbatim (can_manager.cpp:320-322), so the
# SDK itself assumes identity. But test/calib_direction.py exists precisely because J2 was
# suspect, and J6's zero is a per-machine quantity (SaveJ6ZeroPosition, can_manager.cpp:625).
# If a calibration run ever contradicts this, it is a one-line change here.
# Do NOT import the teleop leader's joint_directions -- those belong to a different arm.
JOINT_SIGN   = np.ones(N)
JOINT_OFFSET = np.zeros(N)

# ---- PLANT, not controller ----------------------------------------------------------------
# These describe the MJCF, and they are CAD placeholders (tools/measure_joint_limits.py
# docstring). The closed-loop zeta table in the port's docs depends on them, and
# JOINT_VEL_LIMIT / JOINT_ACC_LIMIT were measured with them in place.
PLANT_DAMPING   = np.full(N, 2.0)
PLANT_ARMATURE  = np.full(N, 0.1)
PLANT_CTRLRANGE = np.array([30., 100., 30., 10., 10., 10.])


@dataclass
class SdkConfig:
    """The complete specification of the control law. SdkJointControl reads nothing else."""

    kp:           np.ndarray = field(default_factory=lambda: MIT_KP.copy())
    kd:           np.ndarray = field(default_factory=lambda: MIT_KD.copy())
    tmax:         np.ndarray = field(default_factory=lambda: TMAX.copy())
    vmax:         np.ndarray = field(default_factory=lambda: VMAX.copy())
    pmax:         np.ndarray = field(default_factory=lambda: PMAX.copy())
    pos_min:      np.ndarray = field(default_factory=lambda: POS_MIN.copy())
    pos_max:      np.ndarray = field(default_factory=lambda: POS_MAX.copy())
    k_gravity:    np.ndarray = field(default_factory=lambda: K_GRAVITY.copy())
    k_coriolis:   np.ndarray = field(default_factory=lambda: K_CORIOLIS.copy())
    b_viscous:    np.ndarray = field(default_factory=lambda: B_VISCOUS.copy())
    visc_vel_min: np.ndarray = field(default_factory=lambda: VISC_VEL_MIN.copy())
    visc_vel_max: np.ndarray = field(default_factory=lambda: VISC_VEL_MAX.copy())
    vel_deadband: float = VEL_DEADBAND

    # ---------------- port knobs. Defaults = the SDK, except inertia_ff. ----------------
    #: None -> re-latch every substep (NOT the hardware; upper bound only).
    hold_hz:    float | None = CONTROL_HZ
    #: 16-bit q, 12-bit qd / tau / kp / kd, exactly as the CAN frame carries them.
    quantize:   bool = True
    #: DEVIATION 1. False = the literal SDK law.
    inertia_ff: bool = True
    #: "sdk" = hard position clamp only | "ros" = additionally the legacy k=20 governor.
    safety:     str = "sdk"


def nrt_config(**kw) -> SdkConfig:
    """NRT_JOINT_POSITION -- the SDK's default mode (``can_manager.h:202``).

    Both of its live branches use ``MitKp()/MitKd()``: trajectory tracking
    (``can_manager.cpp:1393-1394``) and the exhausted-trajectory hold (``:1424-1425``).
    """
    return SdkConfig(**kw)


def follow_config(**kw) -> SdkConfig:
    """RT_JOINT_POSITION / leader-follower gain set (``can_manager.cpp:1171-1173, 1206-1208``).

    33 % softer on J1 and 50 % softer on J3/J4 than NRT. Provided for completeness; the sim
    defaults to NRT because that is what the SDK boots into.
    """
    return SdkConfig(kp=FOLLOW_KP.copy(), kd=FOLLOW_KD.copy(), **kw)
