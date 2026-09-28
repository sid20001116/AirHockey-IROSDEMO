# Vendored from the 2025 Robot Air Hockey Challenge (air_hockey_challenge/utils/kinematics.py).
# MIT License, Copyright (c) 2025 Atalay Donat, Puze Liu, Jonas Guenster, Davide Tateo.
# See ../../THIRD_PARTY.md. Modifications for the mini platform are marked MINI:.

import time

import mujoco
import numpy as np


def forward_kinematics(mj_model, mj_data, q, link="ee"):
    """
    Compute the forward kinematics of the robots.

    IMPORTANT:
        For the iiwa we assume that the universal joint at the end of the end-effector always leaves the mallet
        parallel to the table and facing down. This assumption only makes sense for a subset of robot configurations
        where the mallet can be parallel to the table without colliding with the rod it is mounted on. If this is the
        case this function will return the wrong values.

    Coordinate System:
        All translations and rotations are in the coordinate frame of the Robot. The zero point is in the center of the
        base of the Robot. The x-axis points forward, the z-axis points up and the y-axis forms a right-handed
        coordinate system

    Args:
        mj_model (mujoco.MjModel):
            mujoco MjModel of the robot-only model
        mj_data (mujoco.MjData):
            mujoco MjData object generated from the model
        q (np.array):
            joint configuration for which the forward kinematics are computed
        link (string, "ee"):
            Link for which the forward kinematics is calculated. When using the iiwas the choices are
            ["1", "2", "3", "4", "5", "6", "7", "ee"].

    Returns
    -------
    position: numpy.ndarray, (3,)
        Position of the link in robot's base frame
    orientation: numpy.ndarray, (3, 3)
        Orientation of the link in robot's base frame
    """

    return _mujoco_fk(q, link_to_xml_name(mj_model, link), mj_model, mj_data)


def inverse_kinematics(mj_model, mj_data, desired_position, desired_rotation=None, initial_q=None,
                       link="ee", max_time=None):
    """
    Compute the inverse kinematics of the robots.

    IMPORTANT:
        For the iiwa we assume that the universal joint at the end of the end-effector always leaves the mallet
        parallel to the table and facing down. This assumption only makes sense for a subset of robot configurations
        where the mallet can be parallel to the table without colliding with the rod it is mounted on. If this is
        the case this function will return the wrong values.

    Coordinate System:
        All translations and rotations are in the coordinate frame of the Robot. The zero point is in the center of the
        base of the Robot. The x-axis points forward, the z-axis points up and the y-axis forms a right-handed
        coordinate system

    Args:
        mj_model (mujoco.MjModel):
            mujoco MjModel of the robot-only model
        mj_data (mujoco.MjData):
            mujoco MjData object generated from the model
        desired_position (numpy.ndarray, (3,)):
            The desired position of the selected link.
        desired_rotation (optional, numpy.array, (3,3)):
            The desired rotation of the selected link.
        initial_q (numpy.ndarray, None):
            The initial configuration of the algorithm, if set to None it will take the initial configuration of the
            mj_data.
        link (str, "ee"):
            Link for which the inverse kinematics is calculated. When using the iiwas the choices are
            ["1", "2", "3", "4", "5", "6", "7", "ee"].
    """
    q_init = np.zeros(mj_model.nq)
    if initial_q is None:
        q_init = mj_data.qpos
    else:
        q_init[:initial_q.size] = initial_q

    q_l = mj_model.jnt_range[:, 0]
    q_h = mj_model.jnt_range[:, 1]
    lower_limit = (q_l + q_h) / 2 - 0.95 * (q_h - q_l) / 2
    upper_limit = (q_l + q_h) / 2 + 0.95 * (q_h - q_l) / 2

    desired_quat = None
    if desired_rotation is not None:
        desired_quat = np.zeros(4)
        mujoco.mju_mat2Quat(desired_quat, desired_rotation.reshape(-1, 1))

    return _mujoco_clik(desired_position, desired_quat, q_init, link_to_xml_name(mj_model, link), mj_model,
                        mj_data, lower_limit, upper_limit, max_time=max_time)


def jacobian(mj_model, mj_data, q, link="ee"):
    """
    Compute the Jacobian of the robots.

    IMPORTANT:
        For the iiwa we assume that the universal joint at the end of the end-effector always leaves the mallet
        parallel to the table and facing down. This assumption only makes sense for a subset of robot configurations
        where the mallet can be parallel to the table without colliding with the rod it is mounted on. If this is the
        case this function will return the wrong values.

    Coordinate System:
        All translations and rotations are in the coordinate frame of the Robot. The zero point is in the center of the
        base of the Robot. The x-axis points forward, the z-axis points up and the y-axis forms a right-handed
        coordinate system

    Args:
        mj_model (mujoco.MjModel):
            mujoco MjModel of the robot-only model
        mj_data (mujoco.MjData):
            mujoco MjData object generated from the model
        q (numpy.ndarray):
            joint configuration for which the forward kinematics are computed
        link (string, "ee"):
            Link for which the forward kinematics is calculated. When using the iiwas the choices are
            ["1", "2", "3", "4", "5", "6", "7", "ee"].

    Returns
    -------
    numpy.ndarray, (6, num_joints):
        The Jacobian matrix for the robot kinematics.
    """
    return _mujoco_jac(q, link_to_xml_name(mj_model, link), mj_model, mj_data)


# MINI: this build has only the 6-DOF metal arm. link6 is the wrist roll and gripper_base is
# rigidly attached to it, so "7" -- the link the constraint set calls the wrist -- resolves to
# gripper_base. "ee" is the GRASP FRAME: the mallet is a free body held by friction and is not
# in the kinematic chain at all.
_METAL_LINKS = {
    "1": "metal_1/link1",
    "2": "metal_1/link2",
    "3": "metal_1/link3",
    "4": "metal_1/link4",
    "5": "metal_1/link5",
    "6": "metal_1/link6",
    "7": "metal_1/gripper_base",
    "ee": "metal_1/striker_joint_link",
}


def _has_body(mj_model, name):
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name) != -1


def link_to_xml_name(mj_model, link):
    """Resolve a platform-independent link label to a body name in the metal arm."""
    if not _has_body(mj_model, "metal_1/striker_joint_link"):
        raise ValueError("Expected a metal_1/ kinematic chain with a striker_joint_link body; "
                         "this build of mini_platform has no iiwa support.")
    return _METAL_LINKS[link]


def _mujoco_fk(q, name, model, data):
    data.qpos[:len(q)] = q
    mujoco.mj_fwdPosition(model, data)
    return data.body(name).xpos.copy(), data.body(name).xmat.reshape(3, 3).copy()


def _mujoco_jac(q, name, model, data):
    data.qpos[:len(q)] = q
    dtype = data.qpos.dtype
    jac = np.empty((6, model.nv), dtype=dtype)
    jac_pos, jac_rot = jac[:3], jac[3:]
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_jacBody(model, data, jac_pos, jac_rot, model.body(name).id)
    return jac


def _mujoco_clik(desired_pos, desired_quat, initial_q, name, model, data, lower_limit, upper_limit,
                 max_time=None):
    # MINI: optional wall-clock budget. A reachable target converges in ~20 iterations (<1 ms);
    # an UNREACHABLE one grinds all IT_MAX=1000 iterations (~30 ms). The vertical-hit-anchor
    # search (baseline_mini) probes up to 12 mostly-unreachable orientations per replan, so
    # those 30 ms failures stacked into 300 ms+ freezes of the whole agent loop. A max_time cap
    # kills the grind without touching converging solves (they finish long before the budget).
    IT_MAX = 1000
    t_start = time.perf_counter()
    eps = 1e-4
    damp = 1e-3
    progress_thresh = 20.0
    max_update_norm = 0.1
    rot_weight = 1
    i = 0

    dtype = data.qpos.dtype

    data.qpos = initial_q

    neg_x_quat = np.empty(4, dtype=dtype)
    error_x_quat = np.empty(4, dtype=dtype)

    if desired_pos is not None and desired_quat is not None:
        jac = np.empty((6, model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, model.nv), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if desired_pos is not None:
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif desired_quat is not None:
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError("Desired Position and desired rotation is None, cannot compute inverse kinematics")

    while True:
        # forward kinematics
        mujoco.mj_fwdPosition(model, data)

        x_pos = data.body(name).xpos
        x_quat = data.body(name).xquat

        error_norm = 0
        if desired_pos is not None:
            err_pos[:] = desired_pos - x_pos
            error_norm += np.linalg.norm(err_pos)

        if desired_quat is not None:
            mujoco.mju_negQuat(neg_x_quat, x_quat)
            mujoco.mju_mulQuat(error_x_quat, desired_quat, neg_x_quat)
            mujoco.mju_quat2Vel(err_rot, error_x_quat, 1)
            error_norm += np.linalg.norm(err_rot) * rot_weight

        if error_norm < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        if max_time is not None and (time.perf_counter() - t_start) > max_time:
            success = False
            break

        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, model.body(name).id)

        hess_approx = jac.T.dot(jac)
        joint_delta = jac.T.dot(err)

        hess_approx += np.eye(hess_approx.shape[0]) * damp
        update_joints = np.linalg.solve(hess_approx, joint_delta)

        update_norm = np.linalg.norm(update_joints)

        # Check whether we are still making enough progress, and halt if not.
        progress_criterion = error_norm / (update_norm + 1e-16)
        if progress_criterion > progress_thresh:
            success = False
            break

        if update_norm > max_update_norm:
            update_joints *= max_update_norm / update_norm

        mujoco.mj_integratePos(model, data.qpos, update_joints, 1)
        data.qpos = np.clip(data.qpos, lower_limit, upper_limit)
        i += 1
    q_cur = data.qpos.copy()

    return success, q_cur


# MINI: ----------------------------------------------------------- flat-mallet kinematics ---
# This arm GRIPS its mallet rigidly, so wrist tilt IS mallet tilt: the tool's pointing axis
# (ee-frame +Z) has to lie along world -Z for the head to sit flat on the table. Nothing in the
# vendored stack says so -- the tracking QP tracks position only, and the hit solvers maximise
# velocity manipulability -- because the KUKA's mallet self-levels on a passive universal joint.
#
# The three helpers below are what lets the controllers and planners treat flatness as an
# OBJECTIVE they trade against the position task, rather than as the all-or-nothing IK
# constraint used until now, which simply fails wherever a vertical wrist is out of reach (the
# whole centreline, at this arm's cross-table stretch) and then leaves the orientation entirely
# uncontrolled. Position always stays the hard task: flatness is only ever bought with the
# arm's redundancy, never by leaving the commanded point.

Z_DOWN = np.array([0., 0., -1.])


def tool_tilt(mj_model, mj_data, q, link="ee"):
    """MINI: ``(tilt rad, rotation error)`` of the tool's pointing axis away from straight down.

    The error vector is ``z_tool x z_down``: its magnitude is sin(tilt) and its direction is the
    axis to rotate the tool about to level it -- i.e. exactly the angular velocity the angular
    half of the Jacobian consumes, so it can be handed straight to a least-squares solve.
    """
    _, rot = forward_kinematics(mj_model, mj_data, q, link)
    z_tool = rot @ np.array([0., 0., 1.])
    return float(np.arccos(np.clip(-z_tool[2], -1.0, 1.0))), np.cross(z_tool, Z_DOWN)


def _shrunk_limits(mj_model, n):
    """The 0.95-shrunk joint box the IK/optimiser paths all keep inside."""
    q_l = mj_model.jnt_range[:n, 0]
    q_h = mj_model.jnt_range[:n, 1]
    mid, half = (q_l + q_h) / 2, 0.95 * (q_h - q_l) / 2
    return mid - half, mid + half


def flatten_config(mj_model, mj_data, q_0, x_des, n_joints=None, link="ee", pos_tol=1e-4,
                   max_iter=60, max_time=12e-3, step=4.0, damp=1e-3, max_update_norm=0.1):
    """MINI: the FLATTEST configuration that still reaches ``x_des``, descending from ``q_0``.

    Position is the hard task (damped least squares onto ``x_des``); the tilt is reduced only
    inside that task's null space, so the mallet never leaves the commanded point in order to
    get flatter. Where a vertical wrist exists this converges to one; where it does not, it
    stops at the least tilt the arm can hold at that point -- which is the whole idea: the
    wrist compensates as far as the geometry allows instead of the solve failing outright.

    Returns ``(reached, q, tilt)``. ``reached`` reports whether the POSITION converged; False
    means ``x_des`` itself is out of reach and ``q`` should not be used.
    """
    n = int(n_joints if n_joints is not None else mj_model.nv)
    q = np.asarray(q_0, float)[:n].copy()
    x_des = np.asarray(x_des, float)
    lo, hi = _shrunk_limits(mj_model, n)
    deadline = time.perf_counter() + max_time
    best_q, best_tilt = None, np.inf

    for _ in range(max_iter):
        x_cur, _ = forward_kinematics(mj_model, mj_data, q, link)
        err_pos = x_des - x_cur
        tilt, err_rot = tool_tilt(mj_model, mj_data, q, link)
        if np.linalg.norm(err_pos) < pos_tol and tilt < best_tilt:
            best_q, best_tilt = q.copy(), tilt
        if time.perf_counter() > deadline:
            break

        jac = jacobian(mj_model, mj_data, q, link)[:, :n]
        jac_p, jac_w = jac[:3], jac[3:6]
        dq = jac_p.T @ np.linalg.solve(jac_p @ jac_p.T + damp * np.eye(3), err_pos)
        # the levelling step, projected so it cannot move the tool off x_des
        null_p = np.eye(n) - np.linalg.pinv(jac_p) @ jac_p
        dq = dq + step * (null_p @ np.linalg.lstsq(jac_w, err_rot, rcond=None)[0])

        norm = np.linalg.norm(dq)
        if norm < 1e-7:
            break
        if norm > max_update_norm:
            dq *= max_update_norm / norm
        q = np.clip(q + dq, lo, hi)

    if best_q is not None:
        return True, best_q, best_tilt
    x_cur, _ = forward_kinematics(mj_model, mj_data, q, link)
    tilt, _ = tool_tilt(mj_model, mj_data, q, link)
    return bool(np.linalg.norm(x_des - x_cur) < pos_tol), q, tilt


# Yaw offsets tried when probing for a strictly vertical wrist, in order. The mallet head is a
# disc, so every yaw plays identically -- these are purely different ways for the ARM to hold
# the same flat mallet, and which of them is reachable varies across the table.
FLAT_YAW_PROBES = (0., np.pi / 2, -np.pi / 2, np.pi, np.pi / 4, -np.pi / 4)


def solve_flat_config(mj_model, mj_data, x_des, q_0, yaw_hint=None, n_joints=None, link="ee",
                      max_time=40e-3, probe_time=4e-3, seeds=None):
    """MINI: a configuration at ``x_des`` holding the mallet as flat as the arm allows.

    Two tiers, cheapest first:

    1. **strict** -- full mallet-down IK (orientation constrained, only wrist yaw free),
       trying ``yaw_hint`` first and then the other yaws. This is the same probe the hit
       anchor has always used, and where it lands the tilt is zero.
    2. **graded** -- ``flatten_config`` from a position-only IK seed, for every point where
       no vertical wrist exists at all.

    Returns ``(reached, q, tilt)`` with ``q`` of length ``n_joints``.
    """
    from scipy.spatial.transform import Rotation as _R      # local: keep import cost off fk/jac

    n = int(n_joints if n_joints is not None else mj_model.nv)
    x_des = np.asarray(x_des, float)
    q_0 = np.asarray(q_0, float)
    seeds = (q_0,) if seeds is None else tuple(seeds)
    psi0 = 0.0 if yaw_hint is None else float(yaw_hint)
    deadline = time.perf_counter() + max_time

    # Tier 1 gets 40% of the budget; whatever is left funds the levelling descent, so
    # the WHOLE call stays inside max_time however many probes miss. (An unreachable
    # orientation grinds to its own cap, and there are up to len(FLAT_YAW_PROBES) of them --
    # uncapped, that alone froze the agent loop for 300 ms+ on the replans that need it.)
    probe_deadline = time.perf_counter() + 0.4 * max_time
    for dpsi in FLAT_YAW_PROBES:
        rot = (_R.from_euler('z', psi0 + dpsi) * _R.from_euler('x', np.pi)).as_matrix()
        for seed in seeds:
            ok, q = inverse_kinematics(mj_model, mj_data, x_des, rot, initial_q=seed,
                                       link=link, max_time=probe_time)
            if ok:
                q = np.asarray(q, float)[:n]
                return True, q, tool_tilt(mj_model, mj_data, q, link)[0]
            if time.perf_counter() > probe_deadline:
                break
        else:
            continue
        break

    # No vertical wrist here. Reach the point however the arm can, then level what is left.
    # The descent is a local one, so it is run from every seed on hand and the flattest result
    # kept -- different arm postures at the same point level out to genuinely different tilts.
    ok, q_ik = inverse_kinematics(mj_model, mj_data, x_des, initial_q=q_0, link=link,
                                  max_time=probe_time)
    starts = [np.asarray(q_ik, float)[:n]] if ok else []
    starts += [np.asarray(s, float)[:n] for s in seeds]
    # Floor the descent's share: a levelling run given only a millisecond returns its own
    # seed, which is the UNLEVELLED pose -- far worse than overrunning the budget slightly.
    budget = max(4e-3, (deadline - time.perf_counter()) / max(1, len(starts)))
    best = (False, np.asarray(starts[0] if starts else q_0, float)[:n], np.inf)
    for q_seed in starts:
        reached, q, tilt = flatten_config(mj_model, mj_data, q_seed, x_des, n_joints=n,
                                          link=link, max_time=budget)
        if reached and tilt < best[2]:
            best = (True, q, tilt)
    if best[0]:
        return best
    return flatten_config(mj_model, mj_data, starts[0] if starts else q_0, x_des,
                          n_joints=n, link=link, max_time=budget)
