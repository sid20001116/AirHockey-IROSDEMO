"""Hardware runtime: run the air-hockey brain on the physical arm + real camera.

This replaces ``mushroom_rl``'s ``env.step`` with a real-time loop:

    read SDK joint feedback -> agent.draw_action(obs) -> (2,6) joint target
    -> cubic-interpolate across the 20 ms tick -> stream to the vendor SDK at ~stream_hz
    -> watchdog (joint error codes, position limits) every stream frame.

The agent, planner, FSM and EKF are the unmodified sim ones (``MiniBaselineAgent``); only the
puck sensor (real camera) and the actuation path (vendor ``metal_sdk``) are swapped.

Staged bring-up (``--stage``), in the safe order from the plan -- never jump to the last one:

    grav    gravity-compensation hold; the arm just holds itself (no camera, no arm URDF move)
    table-height  teleop: rest the gripped mallet on the table, sample FK -> the on-table play
            height (--ee-height) so the mallet stays in contact instead of floating/swooping
    reach-probe   teleop: read the mallet's live distance to its own goal line -> the safe
            --min-goal-dist keep-out (stops the arm reaching back and tipping the table)
    set-origin    teleop: touch the mallet to a known world point (own-goal-line centre) ->
            correct a base_frame whose origin is offset (dist-to-goal != 0 at the goal)
    home    NRT move to the HOME config and hold (verifies feedback + a commanded move)
    reach   slow NRT move HOME -> a test config -> HOME (verifies tracking)
    vision  arm parked at HOME; print live puck measurements from the real camera (no play)
    play    full closed-loop play (needs --calib and a real camera)

Always exits by handing the arm to GRAVITY_COMPENSATION so it holds instead of collapsing.

Requires the cp310 runtime env (mujoco + mushroom_rl + metal_sdk). Use ``--mock`` to exercise
the loop logic without an arm (still needs mujoco/mushroom for ``env_info``).
"""

import argparse
import os
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

# Default arm URDF: the vendor's self-contained copy (relative mesh paths). It is the exact
# model metal_sdk/example/single_arm_control.py uses and the one the SDK's gravity comp is
# validated against, and it needs no ROS sourcing. Resolved relative to the repo so it works
# regardless of how metal_sdk was installed. (The ROS metal_description URDF is more detailed
# but uses package:// meshes and a different mass distribution -- pass --urdf to use it.)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_URDF = os.path.join(_REPO_ROOT, "metal_sdk", "example", "urdf", "metal_with_gripper.urdf")
#: Default home for ``--record-video`` mp4s. Repo-anchored for the same reason DEFAULT_URDF is:
#: runs are launched from wherever, and a video that lands next to the shell's cwd is a video
#: nobody finds again.
VIDEO_DIR = os.path.join(_REPO_ROOT, "final-md", "videos")

from mini_platform import make
from mini_platform.agents.baseline_mini import (MiniBaselineAgent, V_SETBACK, GOAL_LINE_INSET,
                                                HOME_PARK_XY,
                                                virtual_base_frame, PlanarXform)
from mini_platform.core.kinematics import forward_kinematics, inverse_kinematics
from mini_platform.envs.env_base import GRASP_HEIGHT, HOME_POS, INIT_SEED, N_JOINTS
from mini_platform.core.kinematics import solve_flat_config
from mini_platform.hw.sdk_backend import HardwareFault, MetalArm, MockMetalArm, N_ARM
from mini_platform.fsm.system_state import TACTICS


# ------------------------------------------------------------------- agent + observations
class HwBaselineAgent(MiniBaselineAgent):
    """``MiniBaselineAgent`` with an externally-supplied (real-camera) puck sensor.

    Built with ``puck_obs_source="gt"`` so it does NOT construct a MuJoCo ``PuckVisionSensor``,
    then switched to the vision path with the injected sensor. All V-frame logic is unchanged.
    """

    def __init__(self, env_info, sensor, agent_id=1, only_tactic=None, **kwargs):
        super().__init__(env_info, agent_id=agent_id, only_tactic=only_tactic,
                         puck_obs_source="gt", **kwargs)
        self._puck_obs_source = "vision"
        self._sensor = sensor
        #: Agent ticks dropped on the last draw to catch up with the wall clock (telemetry).
        self.last_skip = 0
        #: Set by :meth:`attach_streamer` once the RT stream exists (it is built after the agent).
        self._streamer = None
        self._resync_thresh = None
        #: Cumulative --resync-cmd corrections (telemetry).
        self.resync_count = 0

    def attach_streamer(self, streamer, resync_thresh=None):
        """Let the agent see what the RT stream actually wrote, for ``--resync-cmd``.

        Wired here rather than through ``__init__`` because the streamer does not exist yet when
        the agent is constructed -- homing builds it afterwards, and under ``--rt-home`` it is
        built inside the homing routine.
        """
        self._streamer = streamer
        self._resync_thresh = resync_thresh

    def _maybe_resync(self):
        """Re-seed the plan from what the stream actually wrote, once the two have drifted apart.

        Clearing ``trajectory_buffer`` is what makes this bite rather than a no-op: the vendored
        ``draw_action`` pops that buffer straight back into ``q_cmd``
        (``agents/baseline_agent.py:97``), so a corrected ``q_cmd`` sitting behind a stale buffer
        would be overwritten in the same tick, before any tactic planned against it. An empty
        buffer instead sends the vendored ``while`` loop through ``apply()``, which replans from
        the corrected seed -- the same path taken every time a plan runs out normally.
        """
        if self._streamer is None or self._resync_thresh is None:
            return
        q_w, qd_w = self._streamer.last_written()
        target = _resync_target(self.state.q_cmd, q_w, self._resync_thresh)
        if target is None:
            return
        self.state.q_cmd = target
        self.state.dq_cmd = np.asarray(qd_w, float)[:N_ARM].copy()
        self.state.x_cmd, self.state.v_cmd = self.state.update_ee_pos_vel(
            self.state.q_cmd, self.state.dq_cmd)
        self.state.trajectory_buffer = list()
        self.resync_count += 1

    def draw_action(self, obs, skip=0):
        """``BaselineAgent.draw_action``, but catching up with the WALL CLOCK.

        The vendored loop pops exactly one waypoint and steps the EKF one fixed 20 ms tick per
        call, however long the tick really took. Every overrun therefore stretches the plan
        against real time: measured 13-19% behind over a whole run, and per strike a median
        88-106 ms (bezier) and 259-478 ms (VPSTO), worst 775 ms. Because the filter's step is a
        fixed dt too, its velocity is biased by the same ratio exactly while stalls cluster --
        which is during strikes.

        ``skip`` is how many whole agent ticks the wall clock ate beyond this one. Coast the
        tracker that many ticks first -- a measurement-free step, which is what those ticks
        would have been, and it keeps the tactic counters advancing on real ticks -- then run
        the normal update, then drop the waypoints those ticks would have consumed so the
        command returned is the one due NOW instead of one from ``skip`` ticks ago.

        The vendored bookkeeping stays consistent: the tactics already assume the buffer drains
        before the next apply, and VPSTO's ``_vp_tick`` counts waypoints handed out, skipped
        ones included. ``skip=0`` is the vendored path, call for call.
        """
        skip = max(0, int(skip))
        self.last_skip = skip
        for _ in range(skip):
            # No measurement: this tick had no camera frame of its own, and passing one would
            # fuse the same detection twice.
            self.state.update_observation(self.get_joint_pos(obs), self.get_joint_vel(obs), None)
        self._maybe_resync()
        action = super().draw_action(obs)
        if skip:
            # Keep at least one waypoint queued so the streamer still has a lookahead to glide
            # along while the next plan is computed.
            drop = min(skip, max(0, len(self.state.trajectory_buffer) - 1))
            if drop:
                self.state.q_cmd, self.state.dq_cmd = self.state.trajectory_buffer[drop - 1]
                self.state.trajectory_buffer = self.state.trajectory_buffer[drop:]
                self.state.x_cmd, self.state.v_cmd = self.state.update_ee_pos_vel(
                    self.state.q_cmd, self.state.dq_cmd)
                action = np.vstack([self.state.q_cmd, self.state.dq_cmd])
        return action


def _resync_target(q_cmd, q_written, thresh):
    """The command to re-seed the agent's plan from, or ``None`` to leave it alone.

    ``q_cmd`` is where the agent's plan BELIEVES the arm is; ``q_written`` is the post-cap
    setpoint the streamer last actually wrote. The two diverge whenever the velocity/acceleration
    caps throttle a step -- measured at 27% of ticks, growing 70 -> 238 mrad across the first
    240 ms of a rally -- because the agent keeps planning from ``state.trajectory_buffer`` while
    the streamer re-anchors each new queue to its own throttled position (``_q0 = _q_last``). So
    every plan is seeded from a point the arm never reached, and the error compounds for as long
    as the engagement lasts. It collapses to zero at idle, which is why a restart appears to fix
    it: the startup ``restart`` branch is the only command-to-reality sync in the whole run.

    ``q_written`` is deliberately the COMMAND and never ``arm.q()``. The measured position carries
    a standing ~11 mrad of gravity droop (J5 -8.85, J3 +4.59 on this mount), so re-seeding from
    feedback would fold that droop into the command and the arm would sag a little further on
    every single resync -- a runaway, not a fix.
    """
    if thresh is None or thresh <= 0 or q_written is None:
        return None
    q_written = np.asarray(q_written, float)[:N_ARM]
    if not np.all(np.isfinite(q_written)):
        return None
    if np.linalg.norm(np.asarray(q_cmd, float)[:N_ARM] - q_written) <= float(thresh):
        return None
    return q_written.copy()


def only_tactic_for(task):
    if "defend" in task:
        return "defend"
    if "prepare" in task:
        return "prepare"
    return None


def make_obs(env_info, q, qd):
    """Assemble the agent observation from arm feedback.

    Layout (single agent): [puck(6), q(6), qd(6)]. In the vision path the puck fields are
    unused (the injected sensor supplies the measurement), so they stay zero -- only the
    joint block, which ``draw_action`` reads for FK/planning, is filled.
    """
    n = env_info["rl_info"].observation_space.shape[0]
    obs = np.zeros(n)
    obs[env_info["joint_pos_ids"]] = np.asarray(q, float)[:N_JOINTS]
    obs[env_info["joint_vel_ids"]] = np.asarray(qd, float)[:N_JOINTS]
    return obs


def home_config(env_info, target_xy=None, flat=False):
    """Joint angles that park the mallet at ``target_xy`` ON the play plane, wrist vertical.

    z is the play plane (``env_info`` ``ee_desired_height``): on hardware that is the
    calibrated real table-contact height (``--ee-height``), so home rests the mallet on the
    table instead of at the model height where it floats. In sim the two are equal.

    ``target_xy`` defaults to ``HOME_POS`` -- the arm's own station, where a vertical wrist is
    comfortably reachable and this is a plain constrained IK solve. Callers that want the TAUGHT
    park (``HOME_PARK_XY``, see ``hw/capture_home.py``) pass it explicitly with ``flat=True``:
    it is measured against the real table height, so at the model height -- which is where the
    sim and every un-``--ee-height``-ed stage sit -- it has no vertical solution.

    ``flat=True`` accepts a target where a vertical wrist does NOT exist (anywhere on the
    centreline, at this arm's cross-table stretch) and homes to the FLATTEST configuration that
    reaches it instead of refusing. That is what lets ``--home goal`` bring up directly at its
    park spot: without it the arm had to home to its own station and let the Init tactic glide
    across, which is the same cross-table move made faster and under the play loop rather than
    slowly under the vendor's NRT interpolator. Returns the tilt it settled for via
    ``home_config.last_tilt`` (rad).
    """
    rm = env_info["robot"]["robot_model"]
    rd = env_info["robot"]["robot_data"]
    xy = HOME_POS[:2] if target_xy is None else np.asarray(target_xy, float)
    home = np.array([xy[0], xy[1], env_info["robot"]["ee_desired_height"]])
    # Extrinsic "xyz" composes as Rz(c) @ Ry(b) @ Rx(a), so the trailing +pi/2 is a 90 deg CCW
    # rotation about WORLD z applied on top of the wrist-down Rx(pi): same vertical wrist, mallet
    # yawed a quarter turn.
    ok, q = inverse_kinematics(rm, rd, home,
                               R.from_euler("xyz", [np.pi, 0, np.pi / 2]).as_matrix(),
                               initial_q=INIT_SEED)
    if ok:
        home_config.last_tilt = 0.0
        return np.asarray(q, float)[:N_JOINTS]
    if not flat:
        raise RuntimeError("HOME-pose IK failed")
    # No vertical wrist here -- take the flattest pose that does reach the point. The budget is
    # generous because this runs once, before the RT stream, with nothing waiting on it.
    reached, q, tilt = solve_flat_config(rm, rd, home, INIT_SEED, n_joints=N_JOINTS,
                                         max_time=0.5, probe_time=20e-3)
    if not reached:
        raise RuntimeError(f"HOME-pose IK failed: {np.round(home, 3)} is not reachable at all "
                           f"(not just with a vertical wrist). A narrowed joint band "
                           f"(--j1-limit) can be what puts it out of reach -- the IK keeps "
                           f"inside robot_model.jnt_range.")
    home_config.last_tilt = tilt
    return np.asarray(q, float)[:N_JOINTS]


home_config.last_tilt = 0.0


def set_play_height(env_info, z):
    """Move the whole controller onto the planar play height ``z`` (robot-frame grasp-frame z).

    Sets ``ee_desired_height`` -- the plane every trajectory is generated on and the centre of
    the optimiser's z-constraint band -- plus ``universal_height``. Home pose, the
    EndEffectorConstraint band, the hit anchors and the Bezier plane all read these, so this
    single call drops the mallet onto the given plane. Must run BEFORE ``home_config`` and
    before the agent is built (the optimiser reads the band at construction time).
    """
    old = float(env_info["robot"]["ee_desired_height"])
    z = float(z)
    env_info["robot"]["ee_desired_height"] = z
    env_info["robot"]["universal_height"] = z
    dz = z - old
    print(f"play height: ee_desired_height {old:.4f} -> {z:.4f} m (robot frame, {dz*1000:+.1f} mm)")
    if dz < -0.05:
        print(f"  WARNING: {(-dz)*1000:.0f} mm BELOW the model height. If this is a bad "
              f"measurement the arm will drive the mallet into the table -- re-check "
              f"--stage table-height before running play.")


def resolve_ee_height(args):
    """The play height to apply, or None: ``--ee-height`` (float) or ``--ee-height-npz`` (key
    'z'), plus ``--contact-offset`` (0 = light contact; positive lifts the mallet a hair)."""
    z = None
    if args.ee_height is not None:
        z = float(args.ee_height)
    elif args.ee_height_npz and os.path.exists(args.ee_height_npz):
        z = float(np.load(args.ee_height_npz)["z"])
    if z is None:
        return None
    return z + float(args.contact_offset)


#: Index of the base (shoulder) joint. Its MJCF stops are +-2.79 rad (+-160 deg) -- far wider
#: than the sweep the arm can actually use at the end mount, where the cable run and the table
#: rim bound it well inside the model's range. Hence --j1-limit.
J_BASE = 0


def narrow_joint_pos_limit(env_info, j, lo, hi, flag="--j1-limit"):
    """Clip joint ``j``'s allowed POSITION band to ``[lo, hi]`` rad, everywhere the stack reads it.

    TWO arrays have to move together, because the planner and the IK read DIFFERENT ones:

      * ``env_info['robot']['joint_pos_limit']`` -- the tracking QP's position bound
        (``optimizer._solve_aqp`` bounds dq so that ``q + dq*dt`` stays inside 0.92x this band),
        the nlopt bounds in ``solve_hit_config``, ``JointPosConstraint``, and the streamer's
        final ``clamp_pos``;
      * ``robot_model.jnt_range`` -- the box every IK path keeps inside
        (``inverse_kinematics``, ``flatten_config`` / ``solve_flat_config``: the home pose and
        every hit anchor, which is where a strike's whole posture is chosen).

    Narrowing only the first would leave the anchors free to aim at a configuration the QP then
    refuses to track -- a plan-vs-command deficit, the failure mode this controller already pays
    for elsewhere. Narrowing both puts the band where the motion is CHOSEN, which is what "clip
    in the planner" means; the streamer's ``clamp_pos`` is then only a backstop.

    The band can only ever NARROW: it is intersected with the band already in place, so this can
    never ask the arm past a stop the model declares, and two calls compose.

    Note the QP's margin is MULTIPLICATIVE (``* 0.92``), so it tightens a band that straddles
    zero -- which the base joint's does, home sits at ~0 -- and would loosen one that does not.
    ``clamp_pos`` is the exact bound either way, and the IK paths shrink about the band's own
    MIDPOINT (``kinematics._shrunk_limits``), so they are correct for any band.
    """
    lo, hi = float(lo), float(hi)
    name = f"J{j + 1}"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise SystemExit(f"{flag}: LO and HI must both be finite, got {lo} {hi}")
    if lo >= hi:
        raise SystemExit(f"{flag}: LO must be below HI, got {lo} >= {hi}")
    # Units guard, in the spirit of min_goal_dist_error: the band is in RADIANS, but the readout
    # an operator eyeballs at the rig is natural to think of in degrees, and degrees would sail
    # straight through as a band wider than the arm.
    if max(abs(lo), abs(hi)) > 2 * np.pi:
        raise SystemExit(
            f"{flag} {lo:g} {hi:g} is out of range (+-2pi). It is in RADIANS -- did you mean "
            f"{np.radians(lo):.4f} {np.radians(hi):.4f} (i.e. {lo:g} {hi:g} DEGREES)?")

    pos_limit = env_info["robot"]["joint_pos_limit"]
    cur_lo, cur_hi = float(pos_limit[0, j]), float(pos_limit[1, j])
    new_lo, new_hi = max(lo, cur_lo), min(hi, cur_hi)
    if new_lo >= new_hi:
        raise SystemExit(
            f"{flag} [{lo:+.4f}, {hi:+.4f}] does not overlap {name}'s stops "
            f"[{cur_lo:+.4f}, {cur_hi:+.4f}] rad -- no configuration would be left.")
    if lo < cur_lo - 1e-9 or hi > cur_hi + 1e-9:
        print(f"  WARNING: {flag} [{lo:+.4f}, {hi:+.4f}] is WIDER than {name}'s existing stops "
              f"[{cur_lo:+.4f}, {cur_hi:+.4f}] rad; keeping the stops. This flag only narrows.")

    pos_limit[0, j], pos_limit[1, j] = new_lo, new_hi
    env_info["robot"]["robot_model"].jnt_range[j] = (new_lo, new_hi)
    print(f"{name} position limit: [{cur_lo:+.4f}, {cur_hi:+.4f}] -> [{new_lo:+.4f}, {new_hi:+.4f}] "
          f"rad  ([{np.degrees(new_lo):+.1f}, {np.degrees(new_hi):+.1f}] deg), sweep "
          f"{np.degrees(new_hi - new_lo):.1f} deg. Binds the QP, the IK anchors and the stream.")
    return np.array([new_lo, new_hi])


#: Slow-mode default joint-speed cap (rad/s, ~23 deg/s) for at-home development.
SLOW_MAX_JOINT_VEL = 0.4

#: Slow-mode default joint-ACCELERATION cap (rad/s^2) for at-home development. Chosen so the
#: arm can still reach the 0.4 rad/s slow speed in ~0.13 s (0.4/3) -- gentle, no visible jerk.
SLOW_MAX_JOINT_ACCEL = 3.0


def vpsto_kwargs_from_args(args):
    """Constructor kwargs for ``planning.vpsto_striker.VpstoStriker`` from the parsed CLI args.

    One place so the flags and the constructor signature cannot drift apart: the striker plans in
    the CARTESIAN plane, so it takes mallet vel_lim (m/s) / acc_lim (m/s^2) -- an earlier
    joint-space formulation took vel_frac/accel_cap, and passing those raised TypeError at agent
    construction. ``tests/test_vpsto_wiring.py`` pins this against the real signature.
    """
    vel_lim, acc_lim, _ = resolve_vpsto_limits(args)
    return dict(vel_lim=vel_lim, acc_lim=acc_lim,
                smooth=args.vpsto_smooth,
                max_iter=args.vpsto_max_iter, use_cma=(args.vpsto_max_iter > 0))


def joint_cap_arg(text):
    """argparse type for the RT-stream caps: ONE value for every joint, or ``N_ARM`` of them.

    Fix 6d wants J1 raised ALONE. Sideways mallet speed is reach x the base-joint cap, and J1 is
    the arm's best lateral lever (0.741 m/rad at full stretch, against J5's 0.164), but raising the
    single scalar cap would also raise the envelope the strike path is planned inside --
    ``resolve_vpsto_limits`` derives its Cartesian limits from the cap, and taking the SMALLEST
    entry keeps that envelope honest while the base joint alone gets its head.

    Returns a float for one value, so every existing caller and printout is untouched, and an
    ``N_ARM`` vector for six.
    """
    parts = [p for p in str(text).replace(",", " ").split() if p]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number, or {N_ARM} comma-separated numbers (J1..J{N_ARM}), got {text!r}")
    if len(vals) == 1:
        cap = float(vals[0])
    elif len(vals) == N_ARM:
        cap = np.asarray(vals, float)
    else:
        raise argparse.ArgumentTypeError(
            f"expected 1 or {N_ARM} values (J1..J{N_ARM}), got {len(vals)}: {text!r}")
    try:
        _cap_vector(cap)                   # the same finite/positive rule the streamer enforces
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    return cap


def _cap_str(cap, fmt="{:.2f}"):
    """A cap rendered for the startup banner: one number, or all six when they differ."""
    v = np.asarray(cap, float).reshape(-1)
    if v.size == 1 or np.allclose(v, v[0]):
        return fmt.format(float(v[0]))
    return "[" + " ".join(fmt.format(x) for x in v) + "]"


def resolve_max_joint_vel(args):
    """The RT-stream joint-speed cap (rad/s), or None: ``--max-joint-vel`` if given, else the
    slow-mode default when ``--slow`` is set, else no cap (full planned speed).

    A float when one value was given (the common case, and what every banner and downstream
    consumer assumes), an ``N_ARM`` vector when six were -- fix 6d's per-joint cap.
    """
    if args.max_joint_vel is not None:
        cap = np.asarray(args.max_joint_vel, float)
        return float(cap) if cap.ndim == 0 else cap.copy()
    if args.slow:
        return SLOW_MAX_JOINT_VEL
    return None


#: Cartesian VPSTO limits derived from the joint-speed cap, when neither --vpsto-vel-lim nor
#: --vpsto-acc-lim is given. 0.3 m/s of mallet speed per rad/s of joint cap is this arm's measured
#: worst-direction speed map at the end mount (0.33 m/s at the defend line under --max-joint-vel
#: 1.0, 0.11-0.22 deeper in, where the arm folds up over its own base); the acceleration is a
#: swing that reaches that speed in ~0.1 s.
VPSTO_VEL_PER_JOINT_VEL = 0.3
VPSTO_ACC_LIM_CAPPED = 2.5
#: With no joint cap at all there is nothing to derive FROM, so the shipped defaults stand.
VPSTO_VEL_LIM_UNCAPPED = 1.5
VPSTO_ACC_LIM_UNCAPPED = 8.0


def resolve_vpsto_limits(args):
    """``(vel_lim m/s, acc_lim m/s^2, names derived)`` for the VPSTO strike path.

    An explicit ``--vpsto-vel-lim`` / ``--vpsto-acc-lim`` always wins. Otherwise the limits are
    derived from the joint-speed cap the RT stream is going to enforce regardless, because the
    shipped 1.5 m/s / 8 m/s^2 are 4-8x what this arm can do (0.19-0.25 m/s sideways at
    ``--max-joint-vel`` 1.0): VPSTO planned sprints that the streamer then throttled, which put
    81-96% of DEFEND ticks over the joint cap and left the solver chasing paths nobody could
    track (draw p99 ~60 ms). Planning inside the real envelope is the point of a
    feasibility-aware planner; handing it a fictional one throws that away.
    """
    vel, acc, derived = args.vpsto_vel_lim, args.vpsto_acc_lim, []
    vmax = None
    if vel is None or acc is None:      # only consult the cap when something needs deriving
        vmax = resolve_max_joint_vel(args)
        if vmax is not None:
            vmax = float(np.min(np.asarray(vmax, float)))     # a per-joint cap: the slowest joint
    if vel is None:
        vel = (VPSTO_VEL_LIM_UNCAPPED if vmax is None
               else round(VPSTO_VEL_PER_JOINT_VEL * vmax, 3))
        derived.append("vel_lim")
    if acc is None:
        acc = VPSTO_ACC_LIM_UNCAPPED if vmax is None else VPSTO_ACC_LIM_CAPPED
        derived.append("acc_lim")
    return float(vel), float(acc), tuple(derived)


def resolve_max_joint_accel(args):
    """The RT-stream joint-ACCELERATION cap (rad/s^2), or None: ``--max-joint-accel`` if given,
    else the slow-mode default when ``--slow`` is set, else no cap.

    Separate from the velocity cap: telemetry showed commanded joint accel spiking to ~238 rad/s^2
    (jerk / table-shake) even under a 1.8 rad/s velocity cap, because the streamer rate-limited
    velocity but not its rate of change. Bounding the per-write change in commanded velocity caps
    the acceleration the arm actually feels. Default off (byte-identical stream) unless requested."""
    if getattr(args, "max_joint_accel", None) is not None:
        cap = np.asarray(args.max_joint_accel, float)
        return float(cap) if cap.ndim == 0 else cap.copy()
    if args.slow:
        return SLOW_MAX_JOINT_ACCEL
    return None


def min_goal_dist_error(min_goal_dist, half_l):
    """Return an error string if ``min_goal_dist`` (metres) is out of range, else None.

    Guards the metres/millimetres slip: ``--min-goal-dist`` is in METRES but the reach-probe
    prints the distance in mm, so passing e.g. ``30`` (meaning 30 mm) sets a 30 m keep-out that
    trips the watchdog on the first tick. Anything past the table half-length is a units error.
    """
    if 0.0 <= min_goal_dist <= half_l:
        return None
    return (f"--min-goal-dist {min_goal_dist:g} m is out of range (0..{half_l:.2f} m, the table "
            f"half-length). It is in METRES -- did you mean {min_goal_dist / 1000:.3f} "
            f"(i.e. {min_goal_dist:g} mm)? The reach-probe prints mm; divide by 1000.")


def mallet_base_v(env_info, agent_id=1):
    """The arm base's own xy in the V frame -- the centre of its reach disc."""
    return r2v_xform(env_info, agent_id).pos2(np.zeros(2))


def goal_line_v(env_info):
    """V-frame x of the agent's OWN goal line (the down-table extreme toward its own goal).

    Smaller V-x is toward the own goal; this is the smallest x the play region ever reaches.
    """
    return V_SETBACK - env_info["table"]["length"] / 2 + GOAL_LINE_INSET


def r2v_xform(env_info, agent_id=1):
    """The real-base -> virtual-frame planar transform, standalone (for stages with no agent)."""
    T_wr = env_info["robot"]["base_frame"][agent_id - 1]
    T_wv = virtual_base_frame(agent_id, T_wr[2, 3])
    return PlanarXform(T_wv, T_wr)


def v2r_xform(env_info, agent_id=1):
    """The inverse of `r2v_xform`: virtual planning frame -> the arm's own base frame."""
    T_wr = env_info["robot"]["base_frame"][agent_id - 1]
    T_wv = virtual_base_frame(agent_id, T_wr[2, 3])
    return PlanarXform(T_wr, T_wv)


def shift_base_origin(T, world_now_xy, ref_xy):
    """Return ``T`` with its world-xy translation shifted so a point that currently maps to
    ``world_now_xy`` instead maps to the known ``ref_xy`` (rotation, tilt and z untouched).

    calibrate_robot fixes the world origin at the table centre; when the arm cannot reach the
    true centre the origin lands off by a constant offset and every world position -- defense,
    the keep-out line, the reach-probe distance -- shifts by it. A single touch at a KNOWN
    reachable point (e.g. the own-goal-line centre) recovers that offset.
    """
    T = np.asarray(T, float).copy()
    T[:2, 3] += np.asarray(ref_xy, float) - np.asarray(world_now_xy, float)
    return T


def mallet_x_v(env_info, r2v, q):
    """V-frame down-table x of the mallet (grasp frame) for joint config ``q``.

    This is the axis the own-goal keep-out guards: the watchdog trips when the *measured*
    mallet crosses the hard floor, catching the arm before it drives into the near rim.
    """
    pos, _ = forward_kinematics(env_info["robot"]["robot_model"],
                                env_info["robot"]["robot_data"],
                                np.asarray(q, float)[:N_JOINTS], link="ee")
    return float(r2v.pos2(pos[:2])[0])


def mallet_reach_r(env_info, q):
    """Mallet distance from the arm's OWN base (m) -- the axis the base keep-out guards.

    The robot base frame is centred on the base, so this is just |FK xy|. Unlike ``mallet_x_v``
    it needs no r2v transform and no base_frame at all, so a mis-calibrated or mirrored frame
    cannot fool it -- worth having in the one check whose job is to stop the table tipping.
    """
    pos, _ = forward_kinematics(env_info["robot"]["robot_model"],
                                env_info["robot"]["robot_data"],
                                np.asarray(q, float)[:N_JOINTS], link="ee")
    return float(np.linalg.norm(pos[:2]))


# ----------------------------------------------------------------------------- math utils
def hermite(q0, v0, q1, v1, T, t):
    """Cubic-Hermite position at time ``t`` in [0, T] from (q0,v0) to (q1,v1)."""
    s = t / T
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return h00 * q0 + h10 * T * v0 + h01 * q1 + h11 * T * v1


def hermite_vel(q0, v0, q1, v1, T, t):
    """Velocity (d/dt) of :func:`hermite` at time ``t`` in [0, T], in the units of v0/v1."""
    s = t / T
    d00 = 6 * s ** 2 - 6 * s
    d10 = 3 * s ** 2 - 4 * s + 1
    d01 = -6 * s ** 2 + 6 * s
    d11 = 3 * s ** 2 - 2 * s
    return (d00 * q0 + d01 * q1) / T + d10 * v0 + d11 * v1


def clamp_pos(q, pos_limit, margin=0.0):
    """Clamp joint angles to [min+margin, max-margin] (pos_limit is 2x6: row0 min, row1 max)."""
    return np.clip(q, pos_limit[0] + margin, pos_limit[1] - margin)


def sleep_until(t_target):
    dt = t_target - time.perf_counter()
    if dt > 0:
        time.sleep(dt)


def wait_until_config(arm, q_target, timeout=20.0, tol=0.03):
    """Block until the arm reaches ``q_target`` (all joints within ``tol`` rad) or ``timeout``.

    Polls at 10 Hz and runs ``check_health`` each poll. Returns True on arrival, False on
    timeout -- bounded so a stuck/slow NRT move can never hang the run. Used to hold before
    starting closed-loop play until the slow homing move has actually settled.
    """
    t0 = time.perf_counter()
    q_target = np.asarray(q_target, float)[:N_ARM]
    while time.perf_counter() - t0 < timeout:
        arm.check_health()
        if np.max(np.abs(arm.q()[:N_ARM] - q_target)) < tol:
            return True
        time.sleep(0.1)
    return False


def wait_until_rest(arm, timeout=3.0, vel_tol=0.03):
    """Block until all joints are ~stationary (max |qd| < vel_tol rad/s) or timeout.

    ``wait_until_config`` accepts a ~1.7-deg arrival band, but if the NRT home move is still
    decelerating at hand-over, the RT PD must arrest that residual velocity the instant it engages
    -- part of the hand-off jerk. Waiting for rest first lets RT engage on a still arm.
    """
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if np.max(np.abs(arm.qd()[:N_ARM])) < vel_tol:
            return True
        time.sleep(0.05)
    return False


# --------------------------------------------------------------------------- RT streamer
def _cap_vector(cap):
    """A per-joint cap as an ``N_ARM`` vector, from a scalar, a sequence of six, or None.

    The streamer is the one place that has to honour a PER-JOINT limit (fix 6d raises J1 alone,
    because sideways mallet speed is reach x the base-joint cap), so it stores every cap in the
    vector form even when the operator gave one number.
    """
    if cap is None:
        return None
    v = np.asarray(cap, float).reshape(-1)
    if v.size == 1:
        v = np.full(N_ARM, float(v[0]))
    if v.size != N_ARM:
        raise ValueError(f"joint cap must be a scalar or {N_ARM} values, got {v.size}")
    if not np.all(np.isfinite(v)) or not np.all(v > 0):
        raise ValueError(f"joint caps must be finite and positive, got {v}")
    return v


#: Joints that cannot move the mallet, and so are excluded from the COUPLED scale factor.
#:
#: J6's position-Jacobian column is identically zero -- measured max ||jac_pos[:, J6]|| = 0.000000
#: m/rad over 2000 random configurations spanning the full joint-limit box, against 0.741 for J1,
#: the best lateral lever on the arm. It spins the mallet about its own axis, and a disc mallet is
#: symmetric under that; `TrajectoryOptimizer._flat_anchor_dq` already drops the same component for
#: the same reason ("spin about the axis is left free -- a disc mallet does not care").
#:
#: The coupling exists to keep a throttled step parallel to the planned CARTESIAN path (see
#: `_scale_to`). A joint that contributes nothing to that path cannot bend it, so letting it set
#: the scale throttles only the joints that DO carry it. That is not hypothetical: on goalie.npz
#: and resync_base.npz J6 is the largest planned mover on 44-58% of moving ticks (111 deg of travel
#: per run at up to 2.378 rad/s against a 1.4 cap, chasing strike-aligned wrist yaw), it binds
#: 52-72% of every over-cap tick, and on 10-13% of moving ticks it is the ONLY reason the vector is
#: scaled at all -- taxing the base joint for a spin nobody can see.
#:
#: Excluded joints are still clipped to their own caps, so the safety bound is exactly unchanged.
SPIN_JOINTS = (5,)


def _scale_to(vec, limit, mask=None):
    """The single factor in [0, 1] that pulls ``vec`` inside the per-joint box ``limit``.

    ``min_i(limit_i / |vec_i|)`` over the joints that are over their cap, so the scaled vector is
    PARALLEL to the original. That is the whole point: clipping each joint independently shortens
    only the ones that are over, which bends the command off the planned path. Replaying the
    logged plans through an independent clip reproduced the measured mallet lift (vp_base jaw-z
    p99 57 mm replayed vs 56 mm measured, corr 0.79); scaling the vector uniformly gives 9 mm.

    ``mask`` restricts which joints are allowed to SET the factor (see ``SPIN_JOINTS``); every
    joint is still scaled by whatever comes back.
    """
    mag = np.abs(np.asarray(vec, float))
    over = mag > limit
    if mask is not None:
        over = over & mask
    if not np.any(over):
        return 1.0
    return float(np.min(limit[over] / mag[over]))


class _JointStreamer:
    """Background thread that plays the agent's planned trajectory to the RT follower at a
    steady rate, DECOUPLED from how long the agent takes to think.

    Why a whole trajectory and not just the latest target: ``draw_action`` is bimodal -- most
    ticks are ~1 ms (pop the next buffered point), but a replan runs the trajectory optimiser
    and freezes for tens of ms (up to ~60 ms even after the IK-budget fix). If the stream only
    ever chased the single latest setpoint, then during a replan it would reach that point and
    SIT there until the plan updated, then lurch to the new one -- the table-shaking jerk. But
    the agent always hands us a short *lookahead* (the popped point + ``state.trajectory_buffer``,
    ~5 points = 100 ms of planned motion). So we play that whole plan forward at the true agent
    rate (one waypoint per ``wp_dt``, cubic-Hermite-interpolated at ``stream_hz``): while the
    agent is mid-replan the arm keeps gliding along the last good plan instead of freezing, and
    when a fresh plan arrives (normally every tick) it simply replaces the queue. Continuous by
    construction -- each new queue re-anchors at the stream's live position + velocity (C1).

    This is the in-process, real-time-safe decoupling the task needs; a ROS2 topic hop here
    would only add latency and jitter, not remove the freeze (the freeze is CPU-bound planning).
    """

    def __init__(self, arm, stream_hz, dt_agent, pos_limit, q_start, max_joint_vel=None,
                 max_joint_accel=None, limit_mode="coupled", lead_s=0.0, couple_spin=False):
        self._arm = arm
        self._period = 1.0 / stream_hz
        self._wp_dt = float(dt_agent)       # planned-waypoint spacing (agent tick, 20 ms)
        self._pos_limit = pos_limit
        # Hard per-joint velocity cap (rad/s) or None. When set, every streamed step moves no
        # further than vmax*period from the last COMMANDED position, so the arm cannot swing
        # faster than vmax no matter what the planner asks -- the safety limit for at-home dev.
        self._vmax = _cap_vector(max_joint_vel)
        # Hard per-joint acceleration cap (rad/s^2) or None. When set, the commanded joint
        # velocity may change by no more than amax*period per write, so the arm cannot jerk no
        # matter how discontinuous the plan is (a replan boundary can demand a step-change in
        # velocity; without this the arm feels it as a spike -- measured ~238 rad/s^2). Applied
        # AFTER the velocity cap, using the last commanded velocity as the reference, so it
        # preserves the vmax bound and re-anchors cleanly (no wind-up).
        self._amax = _cap_vector(max_joint_accel)
        #: How a throttled step is brought inside the caps. "coupled" (default) scales the whole
        #: joint vector by ONE factor, so the command keeps the planned DIRECTION; "per-joint" is
        #: the old independent clip, kept only as the escape hatch for one A/B.
        if limit_mode not in ("coupled", "per-joint"):
            raise ValueError(f"unknown limit_mode {limit_mode!r}")
        self._limit_mode = limit_mode
        #: Let the SPIN_JOINTS help set the coupled scale factor, as they did before fix 6d. Kept
        #: only as the A/B escape hatch: they cannot move the mallet, so all they can do is
        #: throttle the joints that can. Off => clipped to their own caps instead.
        self._couple_spin = bool(couple_spin)
        self._path_mask = np.ones(N_ARM, bool)
        self._path_mask[list(SPIN_JOINTS)] = False
        #: Seconds of PLAN to command ahead of the clock (Fix 8d), 0 = off. The RT follower is a
        #: position servo: it lags its setpoint, so the arm trails even a perfectly feasible plan
        #: (measured 60-180 ms). Commanding the point the plan reaches tau from now cancels a
        #: first-order lag of tau. Applied BEFORE the velocity/acceleration caps, so the lead can
        #: never smuggle a step past a limit -- it changes where the arm is asked to be, never how
        #: fast it may get there. Measure tau with `--stage track-test` before setting it: too
        #: large overshoots every reversal, which on a mallet is a skid.
        self._lead = max(0.0, float(lead_s))

        q0 = np.asarray(q_start, float)[:N_ARM].copy()
        self._lock = threading.Lock()
        self._q0 = q0                       # anchor the queue plays FROM (C0), + its velocity
        self._qd0 = np.zeros(N_ARM)
        self._wps = [(q0.copy(), np.zeros(N_ARM))]   # queue of (q, qd) waypoints
        self._t0 = None                     # perf_counter when the current queue started
        self._q_last = q0.copy()            # live commanded point/velocity (for re-anchoring)
        self._qd_last = np.zeros(N_ARM)
        self._q_written = q0.copy()         # last position actually streamed (rate-limit anchor)

        #: Writes to the arm that raised, and cycles that fell so far behind the clock that the
        #: pacing had to be re-based. Both are deliberately SWALLOWED in `_run` -- a transient SDK
        #: hiccup must not kill the RT thread -- but swallowing them silently means a stream that
        #: is failing half its writes looks identical to a healthy one. Counted here, surfaced in
        #: the telemetry (`stream_err`) and in the run's closing summary.
        self._write_errors = 0
        self._late_writes = 0
        self._last_write_error = None

        self._running = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def set_trajectory(self, waypoints):
        """Replace the played queue with ``waypoints`` (list of (q, qd)), planned from NOW.

        Re-anchors at the stream's current live position + velocity so the first segment is
        C1-continuous with whatever was already playing -- no snap when the plan updates.
        """
        wps = [(np.asarray(q, float)[:N_ARM].copy(), np.asarray(qd, float)[:N_ARM].copy())
               for (q, qd) in waypoints]
        if not wps:
            return
        # Reject a non-finite plan (NaN/Inf from a diverged estimate or failed IK): keep playing
        # the last good queue rather than latch NaN into the stream (the _run guard is the final
        # backstop, but dropping it here keeps _q0/_wps clean).
        if not all(np.all(np.isfinite(q)) and np.all(np.isfinite(qd)) for q, qd in wps):
            return
        now = time.perf_counter()
        with self._lock:
            self._q0 = self._q_last.copy()
            self._qd0 = self._qd_last.copy()
            self._wps = wps
            self._t0 = now

    def last_written(self):
        """``(q, qd)`` the streamer last actually WROTE to the arm, after its caps.

        The telemetry's ``q_cmd`` is what the agent PLANNED. This is what survived the velocity
        and acceleration limiters, and the gap between the two IS the throttling that takes the
        mallet off the planned path (CONTROLLER_FIX_PLAN.md Fix 1). A log that records only the
        plan cannot distinguish "the planner asked for something infeasible" from "the arm
        failed to track a feasible plan", which is the first question every A/B asks.
        """
        with self._lock:
            return self._q_written.copy(), self._qd_last.copy()

    def _bring_inside(self, vec, limit):
        """Bring ``vec`` inside the per-joint box ``limit``, however ``--stream-limit`` says to.

        One place for both caps, because the velocity step and the acceleration delta are brought
        inside their boxes the same way -- and a spin joint has to be left out of BOTH factors or
        it simply taxes the path joints through the acceleration limiter instead.
        """
        if self._limit_mode == "per-joint":
            return np.clip(vec, -limit, limit)
        if self._couple_spin:
            return vec * _scale_to(vec, limit)
        out = vec * _scale_to(vec, limit, mask=self._path_mask)
        spin = ~self._path_mask
        out[spin] = np.clip(vec[spin], -limit[spin], limit[spin])
        return out

    def _sample(self, now):
        """Cubic-Hermite (q, qd) command at wall time ``now`` along the current queue."""
        if self._t0 is None:
            return self._wps[0][0].copy(), np.zeros(N_ARM)
        elapsed = now - self._t0
        idx = int(elapsed // self._wp_dt)
        n = len(self._wps)
        if idx >= n:                                    # plan exhausted (planner slower than the
            return self._wps[-1][0].copy(), np.zeros(N_ARM)   # whole lookahead): hold last at rest
        tau = elapsed - idx * self._wp_dt
        q0, qd0 = (self._q0, self._qd0) if idx == 0 else self._wps[idx - 1]
        q1, qd1 = self._wps[idx]
        return (hermite(q0, qd0, q1, qd1, self._wp_dt, tau),
                hermite_vel(q0, qd0, q1, qd1, self._wp_dt, tau))

    def _run(self):
        next_t = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            with self._lock:
                q, qd = self._sample(now)
                if self._lead > 0.0:
                    # Command where the plan WILL be `lead` from now, at the plan's own velocity.
                    # Ahead of the caps on purpose: a lead that could push a step past vmax would
                    # be a speed limit with a hole in it.
                    q = q + qd * self._lead
                if self._vmax is not None:
                    # Never step more than vmax*period from the last commanded position: caps
                    # the joint speed at vmax. Re-anchoring reads _q_last (the capped command),
                    # so the next plan continues from where the arm actually is -- no wind-up.
                    # Scaled as a VECTOR (see _scale_to), so throttling slows the move down the
                    # planned path instead of bending it off the play plane -- over the joints
                    # that carry that path, which is not all of them (see SPIN_JOINTS).
                    max_step = self._vmax * self._period
                    step = self._bring_inside(q - self._q_written, max_step)
                    q = self._q_written + step
                    qd = step / self._period
                if self._amax is not None:
                    # Bound the change in commanded velocity per write to amax*period: caps the
                    # joint acceleration (the jerk the arm feels at replan boundaries). qd is the
                    # velocity implied by the (possibly vel-capped) sampled step; limit how fast it
                    # may change from the last commanded velocity (_qd_last), then re-integrate the
                    # position from that accel-limited velocity. Scaled as a vector for the same
                    # reason as the step above. The caps still compose: the result is a convex
                    # combination of _qd_last and a qd that is already <= vmax, so it is too.
                    dv_max = self._amax * self._period
                    qd = self._qd_last + self._bring_inside(qd - self._qd_last, dv_max)
                    q = self._q_written + qd * self._period
                # A non-finite command (NaN/Inf from a diverged estimate, a divide-by-zero in the
                # tactic, or a failed IK) must NEVER reach the arm -- clamp_pos(NaN)=NaN streams
                # straight through and drives the arm to a wild extend + hard fault. Hold the last
                # good command instead, and keep the anchors finite so the next plan re-anchors.
                if np.all(np.isfinite(q)):
                    # Clamp into the ANCHOR, not just into the outgoing command. A position limit
                    # that actually BINDS -- a narrowed --j1-limit band, or a `lead` that
                    # extrapolates past one -- would otherwise wind _q_written outside the band
                    # while the arm sat pinned at it, and the next re-anchor would continue from a
                    # pose the arm was never in (the velocity cap's comment above already claims
                    # this does not happen). A no-op while the band is the MJCF stops, which
                    # nothing reaches.
                    self._q_written = clamp_pos(np.asarray(q, float), self._pos_limit)
                    self._q_last = self._q_written.copy()
                    self._qd_last = np.asarray(qd, float).copy()
                    cmd = self._q_written.copy()
                else:
                    cmd = None
            try:
                if cmd is not None:
                    self._arm.stream_rt(cmd)
            except Exception as e:
                # A transient SDK hiccup must not kill the stream thread; the main loop's
                # check_health() is what turns a real fault into a clean gravity-comp catch.
                # Counted rather than merely ignored, so a persistently failing stream is
                # visible instead of looking exactly like a healthy one.
                self._write_errors += 1
                self._last_write_error = repr(e)
            next_t += self._period
            # Resync if we fell badly behind (GC pause, a long SDK write) so we pace the next
            # frame from now instead of firing a catch-up burst.
            if next_t < now - 4 * self._period:
                next_t = now + self._period
                self._late_writes += 1
            sleep_until(next_t)

    @property
    def write_errors(self):
        """How many `stream_rt` writes raised. Non-zero means the arm was not getting the stream."""
        return self._write_errors

    def health_summary(self):
        """A one-line report of what the stream swallowed, or None when it swallowed nothing."""
        if not self._write_errors and not self._late_writes:
            return None
        bits = []
        if self._write_errors:
            bits.append(f"{self._write_errors} failed RT write(s), last: {self._last_write_error}")
        if self._late_writes:
            bits.append(f"{self._late_writes} pacing re-base(s) (fell >4 periods behind)")
        return "; ".join(bits)

    def stop(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


# ----------------------------------------------------------------------------- build parts
def build_arm(args, q_start):
    if args.mock:
        return MockMetalArm(q0=q_start)
    return MetalArm(can_id=args.can_id, urdf_path=args.urdf,
                    arm_end_type=args.arm_end_type,
                    stream_include_gripper=args.stream_include_gripper)


def load_puck_profile(args):
    """Load the tuned real-puck colour gate, or warn and fall back to the sim default."""
    from mini_platform.vision.detector import PuckColorProfile
    if not args.puck_profile:
        print("WARNING: no --puck-profile; using the SIM red-puck gate, which will mis-detect "
              "a real puck. Tune one with: python -m mini_platform.hw.tune_puck")
        return None
    profile = PuckColorProfile.load(args.puck_profile)
    tuned = getattr(profile, "tuned_exposure", None)
    print(f"puck colour profile: {args.puck_profile} (mode={profile.mode}"
          f"{'' if tuned is None else f', tuned at exposure {tuned:g}'})")
    # A colour gate is only valid at the exposure it was fitted under: sat_min/val_min are fixed
    # thresholds, so running a profile at a different brightness costs detections silently --
    # and below ~18 detections/s this EKF cannot track a struck puck at all.
    run_exp = getattr(args, "cam_exposure", None)
    if tuned is not None and run_exp is not None and int(tuned) != int(run_exp):
        print(f"  WARNING: this profile was tuned at exposure {tuned:g} but the run pins "
              f"{run_exp}. Re-tune it at {run_exp}, or play at {tuned:g}:\n"
              f"    python -m mini_platform.hw.tune_puck "
              f"--camera-index {getattr(args, 'camera_index', 0)} "
              f"--cam-auto-exposure 1 --cam-exposure {run_exp} --cam-auto-priority 0 "
              f"--out {args.puck_profile}")
    elif tuned is not None and run_exp is None:
        print(f"  NB this profile was tuned at exposure {tuned:g}, but this run pins no "
              f"exposure -- the driver is free to move it. Pass --cam-auto-exposure 1 "
              f"--cam-exposure {tuned:g} --cam-auto-priority 0.")
    elif tuned is None and run_exp is not None:
        print(f"  NB this profile predates exposure stamping, so it cannot be checked against "
              f"the run's --cam-exposure {run_exp}. Re-tune to record it.")
    return profile


def video_path(args):
    """Where this run's mp4 goes: ``--record-video-path``, else an auto name in VIDEO_DIR.

    The auto name is the ``--log`` stem plus a timestamp (``clear_early_20260927_153012.mp4``),
    so a video is obviously paired with the telemetry it was taken with, and the timestamp means
    re-running the same ``--log`` name can never overwrite an earlier recording. That matters
    more here than for a log: :func:`warn_if_log_exists` exists because two baseline captures
    were already lost to a reused name, and a lost video cannot be re-derived at all.
    """
    if args.record_video_path:
        return args.record_video_path
    stem = os.path.splitext(os.path.basename(args.log))[0] if args.log else str(args.stage)
    return os.path.join(VIDEO_DIR, f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")


def attach_frame_recorder(cam, args):
    """Start the background mp4 recorder on ``cam`` if asked; return it (None when off).

    Recording is off unless asked for, and when it is on it is deliberately invisible to the
    control path -- the tee lives in the camera's grab thread and drops frames rather than ever
    making it wait (see ``hw/frame_recorder.py``). Closing the camera finalises the mp4.
    """
    if not (args.record_video or args.record_video_path):
        return None
    from mini_platform.hw.frame_recorder import FrameRecorder

    rec = FrameRecorder(video_path(args), fps=cam.props.get("fps") or 30.0)
    cam.attach_recorder(rec)
    print(f"recording the raw camera feed -> {rec.path}\n"
          f"  background encode at {rec.fps:g} fps; frames are DROPPED rather than made to "
          f"wait, so this cannot cost the run a frame or a tick. Summary printed at exit.")
    return rec


def build_sensor(args, env_info=None):
    """The real-camera puck sensor, with the play-area mask wired up when geometry is given.

    ``env_info`` supplies the table the mask is built from. Without it, or with
    ``--no-table-mask``, the sensor is exactly what it was before the mask existed.
    """
    from mini_platform.hw.real_camera import CameraCalibration, V4L2Camera
    from mini_platform.hw.real_sensor import RealPuckSensor

    if not args.calib:
        raise SystemExit("this stage needs --calib hw_calib.npz (run calibrate_extrinsics first)")
    calib = CameraCalibration.load(args.calib)
    cam = V4L2Camera(index=args.camera_index, resolution=calib.resolution,
                     fps=args.cam_fps, exposure=args.cam_exposure,
                     auto_exposure=args.cam_auto_exposure,
                     auto_priority=args.cam_auto_priority)
    print(f"camera: {cam.props['width']}x{cam.props['height']} {cam.props['fourcc']}, driver "
          f"reports {cam.props['fps']:.0f} fps; {_exposure_line(cam.props)}\n  What it DELIVERS "
          f"is the number that matters -- measure it with --stage cam-probe.")
    attach_frame_recorder(cam, args)
    table = None
    if env_info is not None and not args.no_table_mask:
        table = dict(length=env_info["table"]["length"], width=env_info["table"]["width"],
                     goal_width=env_info["table"]["goal_width"],
                     radius=env_info["puck"]["radius"])
    sensor = RealPuckSensor(cam, calib, puck_profile=load_puck_profile(args),
                            sigma_px=args.sigma_px, table=table)
    if table is None:
        print("table mask: OFF -- the detector takes the largest red blob ANYWHERE in the "
              "frame, so an off-table red object bigger than the puck replaces it (measured: "
              "158 frames across the 2026-09-15 logs).")
    else:
        sensor.render()          # force the lazy build now, so the banner is not a promise
        if sensor.mask_keep_frac is None:
            print("table mask: FAILED OPEN -- the table outline does not project into this "
                  "frame. Detection is unmasked, exactly as before; check --calib.")
        else:
            print(f"table mask: on, keeping {sensor.mask_keep_frac * 100:.1f}% of the frame "
                  f"(play area + both goal mouths). An off-table red object can no longer win "
                  f"blob selection. Disable with --no-table-mask.")
    return sensor


def planarize_base_frame(T):
    """Force a base_frame to a pure yaw about world +z (share z-axis, zero tilt).

    The agent's V-frame math (baseline_mini.PlanarXform) asserts the base and the virtual
    end-mount frame share their z-axis -- the mount is treated as upright. A hand-measured
    base_frame carries a fraction of a degree of tilt (table-normal noise), which trips that
    assert (tol 1e-9). Keep the heading (base x projected to horizontal), origin and height;
    drop the tilt. Returns (T_planar, tilt_deg_removed)."""
    T = np.asarray(T, float).copy()
    z_axis = T[:3, 2]
    tilt_deg = float(np.degrees(np.arccos(np.clip(z_axis[2] / np.linalg.norm(z_axis), -1, 1))))
    x = T[:3, 0].copy()
    x[2] = 0.0
    nx = np.linalg.norm(x)
    if nx < 1e-6:                       # base x nearly vertical: take heading from the y-axis
        y = T[:3, 1].copy()
        y[2] = 0.0
        y /= np.linalg.norm(y)
        x = np.array([y[1], -y[0], 0.0])
    else:
        x /= nx
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    Tp = np.eye(4)
    Tp[:3, 0], Tp[:3, 1], Tp[:3, 2], Tp[:3, 3] = x, y, z, T[:3, 3]
    return Tp, tilt_deg


def load_base_frame(env_info, path):
    """Override env_info base_frame with a measured 4x4 world<-base transform (.npz key 'T').

    The measured frame is planarized (tilt removed) so the agent's PlanarXform accepts it --
    a hand/vision calibration is never perfectly upright to 1e-9.
    """
    if not path:
        return
    T, tilt = planarize_base_frame(np.load(path)["T"])
    env_info["robot"]["base_frame"][0] = T
    print(f"base_frame[0] overridden from {path} (planarized, removed {tilt:.2f} deg tilt)")


# --------------------------------------------------------------------------------- stages
def stage_grav(arm):
    """Back-drivable hold, with a live joint readout.

    The readout is what you teach a joint band from: push the arm by hand to each end of the
    sweep it may actually use, read J1 off, and hand the two numbers to ``--j1-limit LO HI``.
    Printed in rad (the flag's unit) and deg (the one that is easy to eyeball).
    """
    print("gravity-compensation hold. Ctrl-C to exit.")
    print("  (teleop to a limit and read J1 -- those are the numbers --j1-limit takes, in rad)")
    arm.set_mode_gravity()
    try:
        while True:
            arm.check_health()
            q = np.asarray(arm.q(), float)[:N_ARM]
            print(f"  q (rad): {np.round(q, 4)}   J1: {q[0]:+.4f} rad "
                  f"({np.degrees(q[0]):+6.1f} deg)   ", end="\r")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()


def stage_home(arm, env_info, args):
    # The taught park, not the vendored station: this stage exists to verify the pose the arm
    # will actually hold in play. flat=True because the park is measured at the real table
    # height -- without --ee-height it sits at the model height, where a vertical wrist may not
    # reach it, and refusing to home would be a worse answer than homing a few degrees tilted.
    q_home = home_config(env_info, target_xy=HOME_PARK_XY, flat=True)
    print(f"HOME config (rad): {np.round(q_home, 4)}")
    if not args.mock:
        arm.set_gripper(args.gripper_mm)
    arm.set_mode_nrt()
    arm.move_nrt(q_home, velocity_ratio=args.velocity_ratio)
    print("moving to HOME (NRT). Ctrl-C to exit; the arm is caught under gravity comp.")
    try:
        while True:
            arm.check_health()
            err = np.round(arm.q() - q_home, 4)
            print(f"  q-err: {err}", end="\r")
            time.sleep(0.25)
    except KeyboardInterrupt:
        print()


def stage_reach(arm, env_info, args):
    q_home = home_config(env_info)
    pos_limit = env_info["robot"]["joint_pos_limit"]
    # A small, safe test displacement on the wrist joints only.
    q_test = clamp_pos(q_home + np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.3]), pos_limit, margin=0.05)
    if not args.mock:
        arm.set_gripper(args.gripper_mm)
    arm.set_mode_nrt()
    try:
        for name, q in [("HOME", q_home), ("TEST", q_test), ("HOME", q_home)]:
            print(f"NRT move -> {name}: {np.round(q, 3)}")
            arm.move_nrt(q, velocity_ratio=args.velocity_ratio)
            for _ in range(int(3.0 / 0.1)):
                arm.check_health()
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def stage_table_height(arm, env_info, args):
    """Teleop calibration of the on-table play height (``--ee-height``).

    Hand the arm to gravity compensation (back-drivable), rest the gripped mallet FLAT on the
    table, and sample. Each sample reads the joint feedback and computes the grasp-frame height
    with the SAME forward kinematics the planner and constraints use, so the mean is exactly the
    ``ee_desired_height`` that makes "mallet resting on the table" the play plane -- no floating,
    no swoop. Sample a few spread-out spots so table tilt or a loose grip shows up as spread.
    Writes the height to an npz and prints the ``--ee-height`` value to reuse.

    This is teleop, so it reads the keyboard with ``input()`` -- run it in your own terminal.
    """
    from mini_platform.core.kinematics import forward_kinematics
    rm = env_info["robot"]["robot_model"]
    rd = env_info["robot"]["robot_data"]
    nominal = GRASP_HEIGHT      # model on-table height; the "vs model" reference below

    arm.set_mode_gravity()
    print("gravity-comp teleop height calibration (the arm is back-drivable).")
    print("  Rest the gripped mallet FLAT on the table, hold it still, and press ENTER to")
    print("  sample. Do a few spots spread across the play area. Type 'q'+ENTER when done.\n")
    zs, xys = [], []
    try:
        while True:
            s = input(f"  sample {len(zs) + 1} (ENTER=sample, q=done): ").strip().lower()
            if s == "q":
                break
            arm.check_health()
            pos, _ = forward_kinematics(rm, rd, arm.q()[:N_JOINTS], link="ee")
            zs.append(float(pos[2]))
            xys.append(np.asarray(pos[:2], float).copy())
            print(f"    z = {pos[2]:.4f} m   xy = {np.round(pos[:2], 3)}   "
                  f"[{(pos[2] - nominal) * 1000:+.0f} mm vs model]")
    except (KeyboardInterrupt, EOFError):
        print()

    if not zs:
        print("no samples taken; nothing written.")
        return
    zs = np.array(zs)
    z_mean, z_std = float(zs.mean()), float(zs.std())
    spread = float(zs.max() - zs.min())
    print(f"\n  samples: {len(zs)}   mean z = {z_mean:.4f} m   std = {z_std * 1000:.1f} mm   "
          f"spread = {spread * 1000:.1f} mm")
    if spread > 0.01:
        print(f"  NOTE: {spread * 1000:.0f} mm spread across spots -- table tilt or an "
              f"inconsistent grip/contact. The mean is a compromise plane; re-sample if it "
              f"looks off (the play plane is flat, so a real tilt can't be fully removed here).")
    out = args.ee_height_npz or "ee_height.npz"
    np.savez(out, z=z_mean, samples=zs, xy=np.array(xys), nominal=nominal)
    print(f"  wrote {out}")
    print(f"\n  Use it:  --ee-height {z_mean:.4f}   (or --ee-height-npz {out})")


def stage_reach_probe(arm, env_info, args):
    """Teleop readout for the own-goal keep-out (``--min-goal-dist``).

    The arm goes to gravity compensation. Hand-move the mallet to the furthest-back spot toward
    its OWN goal that is still SAFE (does not push the near rim), and read ``dist-to-goal`` --
    that is the ``--min-goal-dist`` to use. Prints the mallet position live in the world, robot
    and V frames plus its distance to the own goal line. Ctrl-C to stop.
    """
    r2v = r2v_xform(env_info)
    T_wr = env_info["robot"]["base_frame"][0]
    gl_v = goal_line_v(env_info)
    rm = env_info["robot"]["robot_model"]
    rd = env_info["robot"]["robot_data"]

    # The jaw point FK reports is not where the mallet touches the table once the wrist tilts --
    # from the end mount it sits ~40 deg off vertical at the goal line and ~6 cm away from the
    # contact. A frame fitted by hw.calibrate_teleop carries the jaw->contact length; use it.
    tool_len = 0.0
    if args.base_frame:
        _bf = np.load(args.base_frame)
        if "tool_length" in _bf.files:
            tool_len = float(_bf["tool_length"])
    if tool_len:
        print(f"reading the MALLET CONTACT: {tool_len*1000:.0f} mm down the tool axis from the jaw "
              f"point (tool_length from {args.base_frame}) -- right at any wrist tilt.")
    else:
        print("reading the JAW point (this frame has no tool_length). With the wrist tilted that is "
              "several cm from where the mallet touches -- calibrate with hw.calibrate_teleop.")

    arm.set_mode_gravity()
    print("gravity-comp reach probe. Move the mallet toward the OWN goal to the furthest SAFE")
    print("spot; the 'dist-to-goal' reading there (in METRES) is your --min-goal-dist -- pass the")
    print("m value, NOT the mm one. Ctrl-C to stop.")
    print("FRAME CHECK: as you push the mallet TOWARD your own goal, dist-to-goal must DECREASE")
    print("(-> 0 at the goal line). If it INCREASES, the base_frame is mirrored -- toggle")
    print(f"--base-frame (currently {'ON: '+args.base_frame if args.base_frame else 'OFF (sim mount)'})")
    print("and re-run; use whichever makes dist decrease. The keep-out + defense depend on it.")
    print("ALSO: fold the wrist back TOWARD THE BASE until it is about to push the near rim --")
    print("the smallest 'reach' you get to safely is your --base-keepout. Unlike dist-to-goal,")
    print("reach does not depend on base_frame at all, so it is the trustworthy one.\n")
    try:
        while True:
            arm.check_health()
            q = arm.q()[:N_JOINTS]
            pos_r, rot_r = forward_kinematics(rm, rd, q, link="ee")
            tilt = float(np.degrees(np.arccos(np.clip(-rot_r[2, 2], -1.0, 1.0))))
            pos_r = pos_r + tool_len * rot_r[:, 2]          # jaw point -> mallet contact
            pos_w = T_wr[:3, :3] @ pos_r + T_wr[:3, 3]
            xv = float(r2v.pos2(pos_r[:2])[0])
            dist = xv - gl_v                         # >0 in front of the own goal line
            print(f"  world xy {np.round(pos_w[:2], 3)}  robot xy {np.round(pos_r[:2], 3)}  "
                  f"V-x {xv:+.3f}   dist-to-goal {dist:+.3f} m ({dist * 1000:+.0f} mm)   "
                  f"reach {float(np.linalg.norm(pos_r[:2])):.3f} m   tilt {tilt:4.1f} deg   ",
                  end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()


def stage_set_origin(arm, env_info, args):
    """Re-reference a base_frame's world ORIGIN to a reachable known point (teleop).

    Fixes the constant down-table offset left when calibrate_robot's origin touch could not
    reach the true table centre (the mallet at the own goal reads e.g. 357 mm instead of 0).
    Rest the gripped mallet at a KNOWN world point -- default the own-goal-line centre,
    world [-L/2, 0] -- and this shifts the base_frame translation so FK(that pose) maps there,
    leaving rotation/tilt untouched. Writes a corrected npz. Needs --base-frame (the frame to
    fix); override the reference with --ref-world-xy X Y.

    THE CAMERA CALIBRATION NEEDS THE SAME SHIFT. Moving the world origin under base_frame alone
    leaves the extrinsic describing a DIFFERENT origin, and the puck then lands in the robot
    frame off by exactly this delta -- silently, because both files individually look fine. Pass
    ``--calib in.npz --out-calib out.npz`` and the identical xy shift is applied to ``C_world``
    (``R_wc`` is a rotation and is untouched; the plane intersection shifts with the ray origin,
    so the two stay a consistent pair). Nothing is written to the camera file unless
    ``--out-calib`` is given.
    """
    if not args.base_frame:
        raise SystemExit("--stage set-origin needs --base-frame (the frame whose origin to fix)")
    T = np.load(args.base_frame)["T"].astype(float).copy()
    half_l = env_info["table"]["length"] / 2
    ref = (np.array(args.ref_world_xy, float) if args.ref_world_xy is not None
           else np.array([-half_l, 0.0]))          # own-goal-line centre (measured world frame)
    rm, rd = env_info["robot"]["robot_model"], env_info["robot"]["robot_data"]

    arm.set_mode_gravity()
    print("gravity-comp origin fix (arm back-drivable).")
    print(f"  Rest the gripped mallet at the KNOWN world point {np.round(ref, 3)} m")
    print("  (default = own-goal-line CENTRE). Hold it still, then press ENTER; 'q' aborts.\n")
    try:
        if input("  capture (ENTER) / abort (q): ").strip().lower() == "q":
            print("aborted; nothing written.")
            return
    except (EOFError, KeyboardInterrupt):
        print("\naborted; nothing written.")
        return
    arm.check_health()
    pos_r, _ = forward_kinematics(rm, rd, arm.q()[:N_JOINTS], link="ee")
    world_now = (T[:3, :3] @ pos_r + T[:3, 3])[:2]
    delta = ref - world_now
    print(f"\n  mallet maps to world {np.round(world_now, 3)} -> target {np.round(ref, 3)}  "
          f"(origin shift {np.round(delta * 1000, 0)} mm)")
    base_old = T[:2, 3].copy()
    T = shift_base_origin(T, world_now, ref)
    out = args.out or "base_frame_originfixed.npz"
    np.savez(out, T=T)
    print(f"  base world xy {np.round(base_old, 3)} -> {np.round(T[:2, 3], 3)}   wrote {out}")

    # The extrinsic describes the SAME world origin, so it has to move with it -- see the
    # docstring. Opt-in, because overwriting a calibration behind the operator's back is worse
    # than leaving the inconsistency visible.
    if args.out_calib:
        if not args.calib:
            raise SystemExit("--out-calib needs --calib (the camera file to shift)")
        cal = np.load(args.calib)
        C_old = cal["C_world"].astype(float)
        C_new = C_old.copy()
        C_new[:2] += delta
        np.savez(args.out_calib, K=cal["K"], dist=cal["dist"], R_wc=cal["R_wc"],
                 C_world=C_new, resolution=cal["resolution"])
        print(f"  camera C_world {np.round(C_old[:2], 3)} -> {np.round(C_new[:2], 3)}   "
              f"wrote {args.out_calib}")
    elif args.calib:
        print("  NOTE: --calib given but not --out-calib, so the camera file was NOT shifted. "
              "It now describes a different world origin than the frame just written, which "
              "puts the puck off by the shift above. Re-run with --out-calib.")

    tail = f" --calib {args.out_calib}" if args.out_calib else ""
    print(f"\n  Use it:  --base-frame {out}{tail}   then re-check --stage reach-probe "
          f"(dist should be ~0 at the goal line).")


def _annotate(cv2, frame_rgb, det, world_xy, extra_lines=None):
    """Draw the detection overlay on an RGB frame; return a BGR image for cv2.imshow.

    ``extra_lines`` (optional) are printed as extra status text below the world-xy line --
    used by the play view to show the FSM tactic and the estimated puck state.
    """
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if det is None:
        cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)  # red border = dropout
        cv2.putText(bgr, "no detection", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        u0, v0, u1, v1 = det.bbox
        cv2.rectangle(bgr, (int(u0), int(v0)), (int(u1), int(v1)), (0, 255, 0), 1)
        cu, cv = int(round(det.centroid[0])), int(round(det.centroid[1]))
        cv2.drawMarker(bgr, (cu, cv), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)      # green centroid
        if det.marker_centroid is not None:
            mu, mv = int(round(det.marker_centroid[0])), int(round(det.marker_centroid[1]))
            cv2.drawMarker(bgr, (mu, mv), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)  # yaw
        txt = "world xy: --" if world_xy is None else \
            f"world xy: {world_xy[0]:+.3f} {world_xy[1]:+.3f} m   px: {det.area}"
        cv2.putText(bgr, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y = 52
    for line in (extra_lines or []):
        cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        y += 24
    return bgr


def measure_follower_lag(t, cmd, meas, freq):
    """``(lag_s, amplitude_ratio, diag)`` for a sine tracked at known ``freq``.

    Both series are fitted to ``A*sin(wt) + B*cos(wt) + offset + slope*t`` by least squares:

      * the amplitude is ``hypot(A, B)``, NOT the peak of the signal. Peak-of-absolute was the
        first implementation and it was WRONG on the rig: gravity droop and non-sinusoidal
        content ride on top of the sine and the peak picks both up. It inflated three real runs
        by 1.4-1.7x and reported a 1.3x "overshoot" for an arm that was in fact attenuating to
        0.9x -- which then drove the advice to add a lead that made things worse. The explicit
        ``slope*t`` term absorbs the droop instead of letting it masquerade as motion.
      * the lag is the PHASE difference, so it is continuous instead of quantised to the sample
        grid the old cross-correlation used -- one 20 ms agent tick, against a real lag of ~40 ms.

    A sine rather than a step because a step measures a rise time no lead term can undo.
    ``diag`` carries the fitted drift (rad over the whole run) and the residual RMS: the two
    numbers that say whether a sine describes the motion at all.
    """
    t = np.asarray(t, float)
    cmd = np.asarray(cmd, float)
    meas = np.asarray(meas, float)
    if len(t) < 8 or float(freq) <= 0 or np.allclose(cmd, cmd[0]):
        return float("nan"), float("nan"), {}
    w = 2 * np.pi * float(freq)
    basis = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t), t])

    def _fit(y):
        c, *_ = np.linalg.lstsq(basis, y, rcond=None)
        resid = y - basis @ c
        return (float(np.hypot(c[0], c[1])), float(np.arctan2(c[1], c[0])), float(c[3]),
                float(np.sqrt(np.mean(resid ** 2))))

    amp_cmd, phase_cmd, _, _ = _fit(cmd)
    amp_meas, phase_meas, slope, resid = _fit(meas)
    if amp_cmd <= 0:
        return float("nan"), float("nan"), {}
    # Wrapped to +-half a period: a lag beyond that is not a lag, it is a different cycle.
    lag = ((phase_cmd - phase_meas + np.pi) % (2 * np.pi) - np.pi) / w
    return lag, amp_meas / amp_cmd, {"drift": slope * (t[-1] - t[0]), "residual": resid,
                                     "amp_cmd": amp_cmd, "amp_meas": amp_meas}


def stage_track_test(arm, env_info, args):
    """Oscillate ONE joint through the RT streamer and measure how far the arm trails it.

    Fix 8d. The arm lags even a feasible plan by 60-180 ms, and until that lag has a number
    there is nothing to set ``--stream-lead-ms`` from. A small slow sine about the home pose is
    the safest probe that still excites the servo: it never leaves the neighbourhood of a pose
    the arm already holds, and the streamer's own caps still apply.
    """
    j = int(args.track_joint)
    if not (0 <= j < N_ARM):
        raise SystemExit(f"--track-joint {j} is out of range (0..{N_ARM - 1})")
    amp, freq = float(args.track_amp), float(args.track_freq)
    duration = float(args.duration or 8.0)
    agent_hz = 50                      # stage_play fixes the same rate; keep the two in step
    dt_agent = 1.0 / agent_hz
    q_home = home_config(env_info)

    if not args.mock:
        arm.set_gripper(args.gripper_mm)
    arm.set_mode_nrt()
    arm.move_nrt(q_home, velocity_ratio=args.velocity_ratio)

    pos_limit = env_info["robot"]["joint_pos_limit"]
    q_start = np.asarray(arm.q(), float)[:N_ARM]
    vmax = resolve_max_joint_vel(args)
    amax = resolve_max_joint_accel(args)
    # PRE-FLIGHT. This probe only measures the FOLLOWER while the streamer is not saturating. The
    # sine needs amp*2*pi*freq of joint speed; as that approaches --max-joint-vel the rate limiter
    # sets the phase instead of the servo, and a --stream-lead-ms offset (qd*tau, which can be
    # several times the per-write budget vmax/stream_hz) is clipped away entirely -- so the lead
    # reads as doing nothing while the overshoot grows. Measured on the rig 2026-09-17: amp 0.08
    # at 0.75 Hz under --slow wants 0.377 rad/s against a 0.4 cap, and returned 1.31x amplitude
    # with a lag that did not budge when a 20 ms lead was applied.
    v_peak = amp * 2 * np.pi * freq
    a_peak = amp * (2 * np.pi * freq) ** 2
    # A per-joint cap (fix 6d) warns on its TIGHTEST entry: the probe sweeps one joint and this is
    # a "your sine will hit the limiter" guard, so the conservative bound is the right one.
    v_cap = None if vmax is None else float(np.min(np.asarray(vmax, float)))
    a_cap = None if amax is None else float(np.min(np.asarray(amax, float)))
    if v_cap is not None and v_peak > 0.5 * v_cap:
        print(f"  WARNING: this sine needs {v_peak:.3f} rad/s against a --max-joint-vel of "
              f"{v_cap:.2f} ({100 * v_peak / v_cap:.0f}% of it).\n  The RT stream will rate-limit "
              f"it, so what you measure is the LIMITER, not the follower -- and --stream-lead-ms "
              f"cannot\n  take effect at all (its offset exceeds the {v_cap / args.stream_hz:.4f} "
              f"rad per-write budget). Use --track-amp\n  {0.5 * v_cap / (2 * np.pi * freq):.3f} "
              f"or less at this cap and frequency, or run at a higher --max-joint-vel.")
    if a_cap is not None and a_peak > 0.5 * a_cap:
        print(f"  WARNING: sine peak accel {a_peak:.2f} rad/s^2 against --max-joint-accel "
              f"{a_cap:.2f}; lower --track-freq or --track-amp.")
    streamer = _JointStreamer(arm, args.stream_hz, dt_agent, pos_limit, q_start,
                              max_joint_vel=vmax, max_joint_accel=amax,
                              limit_mode=args.stream_limit,
                              couple_spin=args.stream_couple_spin,
                              lead_s=args.stream_lead_ms / 1000.0)
    arm.stream_rt(q_start)
    arm.set_mode_rt()
    arm.stream_rt(q_start)
    streamer.start()

    print(f"track test: joint {j}, {amp:.3f} rad at {freq:.2f} Hz for {duration:.0f}s "
          f"(lead {args.stream_lead_ms:.0f} ms). Ctrl-C to stop early.")
    ts, cmd, meas = [], [], []
    t0 = time.perf_counter()
    next_t = t0
    try:
        while True:
            now = time.perf_counter()
            el = now - t0
            if el > duration:
                break
            # A lookahead of sine waypoints, exactly as the agent hands the streamer a plan.
            wps = []
            for k in range(1, 6):
                tk = el + k * dt_agent
                q = q_start.copy()
                qd = np.zeros(N_ARM)
                q[j] = q_start[j] + amp * np.sin(2 * np.pi * freq * tk)
                qd[j] = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * tk)
                wps.append((q, qd))
            streamer.set_trajectory(wps)
            ts.append(el)
            cmd.append(q_start[j] + amp * np.sin(2 * np.pi * freq * el))
            meas.append(float(np.asarray(arm.q(), float)[j]))
            arm.check_health()
            next_t = max(next_t + dt_agent, time.perf_counter())
            sleep_until(next_t)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()

    lag, ratio, diag = measure_follower_lag(ts, cmd, meas, freq)
    rms = float(np.sqrt(np.mean((np.asarray(meas) - np.asarray(cmd)) ** 2)))
    # Attenuation is what a position servo that merely trails SHOULD do -- it is not a fault, and
    # calling it "undershoots" read like one.
    shape = ("OVERSHOOTS -- resonance, not lag" if ratio > 1.05 else
             "attenuates, as a trailing servo should" if ratio < 0.98 else "tracks")
    print(f"\n  follower lag  : {lag * 1000:+.1f} ms   (the delay --stream-lead-ms cancels)")
    print(f"  amplitude     : {ratio:.2f}x commanded -- {shape}")
    print(f"  tracking RMS  : {rms * 1000:.1f} mrad")
    print(f"  fit quality   : drift {diag.get('drift', float('nan')) * 1000:+.1f} mrad over the "
          f"run, {diag.get('residual', float('nan')) * 1000:.1f} mrad off the sine")
    # The amplitude ratio decides whether the lag is even readable. A position servo that merely
    # TRAILS attenuates (ratio <= 1); above 1 the run is saturating the rate limiter or resonating,
    # and the phase measured is the limiter's. Suggesting a lead from that number is how the rig
    # run of 2026-09-17 ended up doubling its overshoot, so the advice is gated on it.
    lead_now = float(args.stream_lead_ms)
    if ratio > 1.05:
        print(f"  -> NOT a usable lag reading. A follower that merely trails ATTENUATES (ratio "
              f"<= 1.00);\n     above that the rate limiter or a resonance is setting the phase, "
              f"not the servo, so do NOT\n     set --stream-lead-ms from it. Lower --track-amp "
              f"(or --track-freq), or raise --max-joint-vel,\n     until the ratio sits at or "
              f"below 1.00 -- then re-read the lag. If the 'off the sine'\n     residual above is "
              f"a large fraction of the amplitude, the joint is not tracking a sine at all.")
    elif lag > 0:
        print(f"  -> try --stream-lead-ms {lead_now + lag * 1000 / 2:.0f} (half the measured lag "
              f"added to the {lead_now:.0f} ms already applied),\n     then re-run: the lag "
              f"should fall and the amplitude stay <= 1.05x.")
    else:
        print("  -> no lag to cancel: the fitted phase is level with or ahead of the command.")
    if args.log:
        np.savez(args.log, t=np.asarray(ts), cmd=np.asarray(cmd), meas=np.asarray(meas),
                 joint=j, amp=amp, freq=freq, lag=lag, amp_ratio=ratio,
                 stream_lead_ms=float(args.stream_lead_ms))
        print(f"  telemetry: {len(ts)} samples -> {args.log}")


def _exposure_line(props):
    """The three exposure controls as read BACK from the device, with the frame-rate verdict.

    Read-back rather than requested, because a refused write is exactly the failure worth
    catching: on this rig `--cam-auto-exposure 0.25` was rejected, the menu stayed on AUTO, and
    every `--cam-exposure` after it was silently ignored.
    """
    def fmt(v):
        return "n/a" if v is None else str(v)

    auto = props.get("exposure_auto")
    pri = props.get("exposure_auto_priority")
    bits = [f"exposure_auto {fmt(auto)}"
            + (" (MANUAL)" if auto == 1 else " (AUTO -- --cam-exposure is ignored)"
               if auto == 3 else ""),
            f"exposure_absolute {fmt(props.get('exposure_absolute'))}",
            f"auto_priority {fmt(pri)}"
            + (" (frame rate held)" if pri == 0 else
               " (DRIVER MAY DROP FRAMES -- pass --cam-auto-priority 0)" if pri == 1 else "")]
    return ", ".join(bits)


def _cam_probe_view(cam):
    """Live window for trading exposure against frame rate. Returns ``(fps, frames, elapsed)``.

    Exposure is a visual judgement -- is the puck a disc or a streak, is the frame still bright
    enough for the colour gate to find it at all -- and getting it wrong is paid in DETECTIONS,
    not in brightness. Restarting the process once per candidate value makes that a slow loop, so
    these keys change the controls on the RUNNING camera and the overlay shows what each one costs
    in delivered fps immediately.

    The fps here counts DISTINCT camera frames (by `frame_seq`), not loop passes, and the window
    resets after every control change so the average reflects the new setting instead of blending
    across it.
    """
    import cv2

    from mini_platform.hw.real_camera import (V4L2_CID_EXPOSURE_ABSOLUTE, V4L2_CID_EXPOSURE_AUTO,
                                              V4L2_CID_EXPOSURE_AUTO_PRIORITY,
                                              V4L2_EXPOSURE_APERTURE_PRIORITY,
                                              V4L2_EXPOSURE_MANUAL, v4l2_control)

    def ctl(cid, value=None):
        return v4l2_control(cam.device, cid, value)

    win = "cam-probe   [ ] exposure   a auto   p priority   q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    exp = ctl(V4L2_CID_EXPOSURE_ABSOLUTE)
    auto = ctl(V4L2_CID_EXPOSURE_AUTO)
    pri = ctl(V4L2_CID_EXPOSURE_AUTO_PRIORITY)
    stamps, last_seq, fps = [], None, 0.0
    t0, seq0 = time.perf_counter(), cam.frame_seq
    print("  live view: [ ] exposure   a = manual/auto   p = auto-priority   q = quit")
    print("  roll a puck past while you watch: you want the SHORTEST exposure whose blob is "
          "still a disc.")
    while True:
        frame, seq, _ = cam.read_rgb_stamped()
        now = time.perf_counter()
        if seq != last_seq:
            last_seq = seq
            stamps.append(now)
            if len(stamps) > 90:
                del stamps[0]
        if len(stamps) > 1:
            fps = (len(stamps) - 1) / max(1e-6, stamps[-1] - stamps[0])
        if frame is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            ms = None if exp is None else exp * 0.1          # UVC exposure_absolute: 100 us units
            lines = [f"{fps:5.1f} fps delivered       mean level {gray.mean():5.1f}",
                     (f"exposure {exp}  (~{ms:.1f} ms)   blur @2.8 m/s {2.8 * ms:4.0f} mm"
                      if ms is not None else "exposure n/a"),
                     f"auto {auto} ({'MANUAL' if auto == V4L2_EXPOSURE_MANUAL else 'AUTO'})"
                     f"   auto_priority {pri}{'   <- may drop frames' if pri == 1 else ''}",
                     f"sharpness {sharp:7.0f}   (higher = crisper)"]
            cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 12 + 18 * len(lines)), (0, 0, 0), -1)
            for i, txt in enumerate(lines):
                cv2.putText(bgr, txt, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(win, bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("]"), ord("+"), ord("=")) and exp is not None:
            exp = ctl(V4L2_CID_EXPOSURE_ABSOLUTE, max(exp + 1, int(exp * 1.25)))
            stamps.clear()
        elif key in (ord("["), ord("-"), ord("_")) and exp is not None:
            exp = ctl(V4L2_CID_EXPOSURE_ABSOLUTE, max(1, min(exp - 1, int(exp / 1.25))))
            stamps.clear()
        elif key == ord("a"):
            auto = ctl(V4L2_CID_EXPOSURE_AUTO,
                       V4L2_EXPOSURE_APERTURE_PRIORITY if auto == V4L2_EXPOSURE_MANUAL
                       else V4L2_EXPOSURE_MANUAL)
            exp = ctl(V4L2_CID_EXPOSURE_ABSOLUTE)            # auto moves it -- re-read the truth
            stamps.clear()
        elif key == ord("p"):
            pri = ctl(V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0 if pri else 1)
            stamps.clear()
    cv2.destroyWindow(win)
    return fps, cam.frame_seq - seq0, time.perf_counter() - t0


def stage_cam_probe(args):
    """Measure what the camera actually DELIVERS, and say whether it is enough.

    Fix 8b. ``v4l2-ctl --all`` answers what the driver is configured for; this answers what
    arrives, which is the number the EKF lives or dies by, and it needs no root and no
    v4l-utils. Reports the negotiated format alongside, because a camera that silently fell back
    to YUYV is the other way to lose half the frame rate.
    """
    from mini_platform.hw.real_camera import CameraCalibration, V4L2Camera

    res = (640, 480)
    if args.calib:
        res = tuple(CameraCalibration.load(args.calib).resolution)
    seconds = float(args.duration or 5.0)
    cam = V4L2Camera(index=args.camera_index, resolution=res, fps=args.cam_fps,
                     exposure=args.cam_exposure, auto_exposure=args.cam_auto_exposure,
                     auto_priority=args.cam_auto_priority)
    attach_frame_recorder(cam, args)
    try:
        p = cam.props
        print(f"camera {args.camera_index} ({cam.device}): {p['width']}x{p['height']} "
              f"{p['fourcc']}, driver reports {p['fps']:.0f} fps\n  {_exposure_line(p)}")
        if args.view:
            fps, frames, elapsed = _cam_probe_view(cam)
            # Re-read after the interactive session: `props` was captured at construction, and the
            # blur verdict below has to describe the exposure actually settled on.
            from mini_platform.hw.real_camera import (V4L2_CID_EXPOSURE_ABSOLUTE,
                                                      V4L2_CID_EXPOSURE_AUTO,
                                                      V4L2_CID_EXPOSURE_AUTO_PRIORITY,
                                                      v4l2_control)
            p = dict(p, exposure_auto=v4l2_control(cam.device, V4L2_CID_EXPOSURE_AUTO),
                     exposure_absolute=v4l2_control(cam.device, V4L2_CID_EXPOSURE_ABSOLUTE),
                     exposure_auto_priority=v4l2_control(cam.device,
                                                         V4L2_CID_EXPOSURE_AUTO_PRIORITY))
            print(f"\n  settled on: {_exposure_line(p)}")
            print(f"  -> --cam-auto-exposure {p['exposure_auto']} "
                  f"--cam-exposure {p['exposure_absolute']} "
                  f"--cam-auto-priority {p['exposure_auto_priority']}")
        else:
            print(f"  measuring delivered rate for {seconds:.0f}s ...")
            t0 = time.perf_counter()
            seq0 = cam.frame_seq
            while time.perf_counter() - t0 < seconds:
                time.sleep(0.05)
            frames = cam.frame_seq - seq0
            elapsed = time.perf_counter() - t0
            fps = cam.delivered_fps
            fps = frames / elapsed if fps is None else fps
        print(f"\n  DELIVERED: {fps:.1f} fps ({frames} frames in {elapsed:.1f}s)")
        # Thresholds from this repo's OWN logs rather than a round number: s2s1 tracked well at
        # 21.8 detections/s (filter-vs-measurement p90 1.5 mm, 1.7% delocalized), vp_calm failed
        # at 11.4 (52.6 mm, 10.3%). So ~18 is the working floor, not 28.
        if fps >= 18.0:
            print("  -> workable. The repo's healthy log ran at 21.8 detections/s; the broken "
                  "one at 11.4. What matters from here is DETECTIONS per second, not frames: "
                  "every frame the colour gate misses costs the same as a frame the camera "
                  "never sent.")
        elif fps >= 12.0:
            print("  -> marginal. Hold the rate with --cam-auto-priority 0 and shorten the "
                  "exposure (--cam-auto-exposure 1 --cam-exposure <n>); at this rate motion "
                  "blur, not brightness, is what loses detections.")
        else:
            print("  -> BELOW 12 fps: this filter cannot track a struck puck at this rate (a "
                  "2.8 m/s puck moves ~250 mm between frames, the NIS gate rejects it and the "
                  "estimate freezes). Fix this before tuning any planner: check the format is "
                  "MJPG, pass --cam-auto-priority 0, and set a fixed exposure.")
        exp = p.get("exposure_absolute")
        if exp and fps > 0:
            # UVC exposure_absolute is conventionally in 100 us units -- verify on your driver,
            # but if it holds, this is how far the puck smears WITHIN one frame. The puck is
            # 50 mm across: once the smear approaches that, the blob stops being a disc, the
            # centroid is dragged along the motion, and the colour gate starts missing it.
            for v in (2.8, 1.5):
                smear = v * exp * 1e-4
                print(f"    blur at {v} m/s: ~{smear * 1000:.0f} mm "
                      f"({'FINE' if smear < 0.015 else 'MARGINAL' if smear < 0.03 else 'TOO MUCH'}"
                      f" against a 50 mm puck)")
            print(f"    frame period {1000 / fps:.0f} ms; exposure is "
                  f"{exp * 1e-4 * fps * 100:.0f}% of it")
    finally:
        cam.close()


def stage_vision(arm, env_info, sensor, args):
    """Park at HOME and stream live puck measurements from the real camera (no closed loop).

    With ``--view``, also opens an annotated live camera window (detected puck outline +
    centroid/marker crosses + world coords; red border on dropout). Press q/ESC or Ctrl-C.
    """
    q_home = home_config(env_info)
    if not args.mock:
        arm.set_gripper(args.gripper_mm)
    arm.set_mode_nrt()
    arm.move_nrt(q_home, velocity_ratio=args.velocity_ratio)

    if not args.view:
        print("vision open-loop. Roll the puck by hand; arm stays parked. Ctrl-C to exit.")
        try:
            while True:
                arm.check_health()
                m = sensor.measure()
                if m is None:
                    print("  puck: <no detection>            ", end="\r")
                else:
                    yaw = "n/a" if m.yaw is None else f"{m.yaw:+.3f}"
                    print(f"  puck world xy: {np.round(m.pos_xy, 4)}  yaw: {yaw}  "
                          f"px: {m.n_pixels}   ", end="\r")
                time.sleep(0.05)
        except KeyboardInterrupt:
            print()
        return

    # Annotated live view. One capture per loop feeds both the overlay (pixel-space) and the
    # displayed world coords (raw back-projection of the centroid -- no bias/covariance; the
    # full measurement path with bias correction is what feeds the EKF in --stage play).
    import cv2
    from mini_platform.vision.detector import detect_puck

    print("vision open-loop + live view. Roll the puck by hand; q/ESC or Ctrl-C to quit.")
    win = "metal puck detection"
    try:
        while True:
            arm.check_health()
            frame = sensor.render()                 # (H,W,3) RGB, one capture
            det = detect_puck(frame, sensor._puck_profile,
                              backend=getattr(sensor, "_detect_backend", "cv2"))
            world_xy = None
            if det is not None:
                P = sensor.camera.pixel_to_plane(*det.centroid)
                world_xy = None if P is None else P[:2]
            # Detect on the MASKED frame, show the RAW one: a mask aimed wrongly is invisible
            # if the viewer only ever sees what survived it.
            cv2.imshow(win, _annotate(cv2, getattr(sensor, "_last_raw_frame", None)
                                      if getattr(sensor, "_last_raw_frame", None) is not None
                                      else frame, det, world_xy))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):               # q or ESC
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


def _play_view_setup(args):
    """Open the live play window if --view; return (cv2, win) or (None, None)."""
    if not args.view:
        return None, None
    import cv2
    win = "metal play view"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("live play view on. Watch: is the puck detected (green box)? which tactic? "
          "Press q/ESC in the window to stop.")
    return cv2, win


def _approach_reason(agent):
    """Replicate SystemState's instantaneous approach classification for the overlay.

    Mirrors update_observation (system_state.py): velocity is estimated_state[2:4], the
    puck must move faster than static_vel_threshold, not purely sideways, and toward the
    arm's own goal (vx<0 in the V frame) beyond defend_range[0]. Returns a short reason so
    you can see WHICH condition is blocking DEFEND.
    """
    est = agent.state.estimated_state
    x, vx, vy = est[0], est[2], est[3]
    speed = float(np.hypot(vx, vy))
    thr = agent.agent_params["static_vel_threshold"]
    dr0 = agent.agent_params["defend_range"][0]
    if speed < thr:
        why = f"STATIC (v {speed:.2f} < {thr:.2f})"
    elif abs(vx) / speed < 0.15:
        why = "TRANSVERSAL (sideways)"
    elif not (vx < 0 and x > dr0):
        why = f"WRONG DIR/RANGE (vx {vx:+.2f}, x {x:+.2f} vs {dr0:+.2f})"
    else:
        why = "APPROACHING ok"
    return x, vx, vy, speed, why


def _smash_reason(agent):
    """Why the slow-puck smash is or is not firing, for the overlay.

    Mirrors MiniTacticMixin.can_smash, in the same order, so the line names the FIRST condition
    that blocks it. Returns None when the feature is off, so the overlay gains a line only when
    there is something to say.
    """
    ap = agent.agent_params
    if not ap.get("smash_slow"):
        return None
    st = agent.state
    est = st.estimated_state
    speed = float(np.hypot(est[2], est[3]))
    vel = float(ap["smash_slow_vel"])
    ticks = int(ap["smash_slow_ticks"])
    n = int(getattr(st, "_slow_count", 0))
    lo, hi = float(ap["hit_range"][0]), float(ap["smash_slow_max_x"])
    lateral = (agent.env_info["table"]["width"] / 2 - agent.env_info["puck"]["radius"]
               - 2 * agent.env_info["mallet"]["radius"])
    if speed >= vel:
        return f"NOT SLOW (v {speed:.2f} >= {vel:.2f})"
    if n < ticks:
        return f"DWELL {n}/{ticks}"
    if st.is_puck_approaching():
        return "APPROACHING -> DEFEND owns it"
    if not lo < est[0] < hi:
        return f"OUT OF BAND (x {est[0]:+.3f} vs [{lo:+.3f}, {hi:+.3f}])"
    if abs(est[1]) >= lateral:
        return f"TOO WIDE (|y| {abs(est[1]):.3f} >= {lateral:.3f})"
    return "smash ok"


#: `defend_phase` telemetry encoding. -1 = the clearing block is off.
CLEAR_PHASES = {"plant": 0, "cock": 1, "strike": 2}


def _clear_reason(agent):
    """What the clearing block is doing this tick, for the overlay.

    Mirrors MiniDefend._clear_target's own order, so the line names the FIRST thing stopping a
    return stroke. Returns None when the feature is off, so the overlay gains a line only when
    there is something to say.
    """
    ap = agent.agent_params
    if not ap.get("defend_clear"):
        return None
    st = agent.state
    gen = agent.traj_generator
    P = np.asarray(st.predicted_state[:2], float)
    floor = float(gen.bound_points[:, 0].min())
    d_max = float(P[0]) - (floor + float(ap["defend_clear_margin"]))
    d = max(min(float(ap["defend_clear_standoff"]), d_max), 0.0)
    d_avail = float(P[0]) - float(st.x_cmd[0])
    T = max(float(st.predicted_time), float(ap["max_plan_steps"]) * gen.dt)
    early = float(getattr(st, "defend_early_line", np.nan))
    pre = "" if not np.isfinite(early) else f"early@{early:+.3f} "
    if d_avail <= 0.01:
        return pre + f"PLANT (no run-up: d_avail {d_avail:+.3f})"
    if d <= 0.0:
        return pre + f"PLANT (no room: d_max {d_max:+.3f} at x {P[0]:+.3f})"
    budget = float(ap.get("defend_clear_max_path") or 0.0)
    if budget > 0.0 and d_avail > budget * T:
        return pre + (f"PLANT (too far for the time: needs {d_avail / max(T, 1e-9):.2f} > "
                      f"{budget:.2f} m/s over {T:.2f} s)")
    phase = getattr(st, "defend_clear_phase", None)
    if phase == "cock":
        return pre + f"COCK d {d:.3f} (floor {floor:+.3f})"
    if phase == "strike":
        v = min(2.0 * d_avail / max(T, 1e-9), float(ap["defend_clear_speed"]))
        return pre + f"STRIKE v {v:.2f} -> +x"
    return pre + f"ready: d_avail {d_avail:+.3f}, d {d:.3f}"


def _play_view_draw(cv2, win, agent, sensor):
    """Draw one annotated frame from the sensor's cached capture + the agent's FSM state.

    Returns True to keep going, False if the user pressed q/ESC. Uses the frame the agent
    already rendered this tick (sensor._last_frame) -- no extra camera read.
    """
    # The RAW capture when the play-area mask is on, so the operator sees what the camera saw
    # (including whatever the mask removed) rather than only what survived it.
    frame = getattr(sensor, "_last_raw_frame", None)
    if frame is None:
        frame = getattr(sensor, "_last_frame", None)
    if frame is None:
        return True
    det = getattr(sensor, "_last_detection", None)

    # Back-project the detected centroid DIRECTLY (independent of measure()), so a failing
    # extrinsic is visible even when measure() returns None and the EKF stays frozen.
    world_xy, ray_z, meas_status = None, None, "no detection"
    if det is not None:
        cam = sensor.camera
        d = cam.pixel_to_ray(*det.centroid)
        ray_z = float(d[2])
        P = cam.intersect_z_plane(d)
        world_xy = None if P is None else P[:2]
        if P is None:
            meas_status = "backproj MISS (ray not downward -> BAD EXTRINSIC)"
        elif getattr(agent, "_last_measurement", None) is None:
            meas_status = "backproj ok, measure None (marker/jac gate)"
        else:
            meas_status = "measure ok -> EKF updating"

    st = agent.state
    est = st.estimated_state
    x, vx, vy, speed, why = _approach_reason(agent)
    lines = [
        f"tactic: {st.tactic_current.name}   approaching(latched): {st.is_puck_approaching()}",
        f"meas: {meas_status}   ray_z={'--' if ray_z is None else f'{ray_z:+.2f}'}",
        f"puck est (V): x={est[0]:+.3f} y={est[1]:+.3f}  vel vx={vx:+.2f} vy={vy:+.2f} |v|={speed:.2f}",
        f"defend? {why}",
    ]
    smash_why = _smash_reason(agent)
    if smash_why is not None:
        lines.append(f"smash?  {smash_why}")
    clear_why = _clear_reason(agent)
    if clear_why is not None:
        lines.append(f"clear?  {clear_why}")
    cv2.imshow(win, _annotate(cv2, frame, det, world_xy, extra_lines=lines))
    return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)


def _lookahead_waypoints(action, agent, pos_limit):
    """The plan from NOW: the popped command point + the remaining agent lookahead buffer.

    ``draw_action`` returns the just-popped (q, dq) and leaves the rest of the planned
    trajectory in ``state.trajectory_buffer``. Handing the streamer the whole thing lets it
    keep gliding along the plan through a replan freeze instead of holding one stale target.
    """
    wps = [(clamp_pos(np.asarray(action[0], float)[:N_ARM], pos_limit),
            np.asarray(action[1], float)[:N_ARM])]
    for wp in agent.state.trajectory_buffer:
        wps.append((clamp_pos(np.asarray(wp[0], float)[:N_ARM], pos_limit),
                    np.asarray(wp[1], float)[:N_ARM]))
    return wps


def _gate_ok(agent):
    """Would DEFEND's geometric gate accept the intercept being aimed at this tick?

    The zone half of ``MiniTacticMixin.should_defend``, evaluated read-only: ``should_defend``
    itself MUTATES ``predicted_time`` on its miss branch, so calling it to fill in a telemetry
    column would change the run the column exists to observe.

    What it buys offline is the difference between "DEFEND is aimed somewhere its own gate
    allows" and "DEFEND latched and then chased its target out of the zone" -- the defect Fix 3
    is about. Recovering that from a log without it means re-deriving the zone from the flags the
    run was launched with, which is exactly the bookkeeping that goes wrong between runs.

    Asks the DEFEND tactic its own question rather than re-deriving the zone here, so a column
    that claims to report the gate cannot drift away from the gate.
    """
    defend = agent.tactics_processor[TACTICS.DEFEND.value]
    return bool(defend._in_zone(defend._gate_state()))


def _telemetry_row(t, draw_ms, q_meas, qd_meas, action, agent, streamer=None, sensor=None,
                   arm=None):
    """One diagnostic row: timing + commanded/measured joints + puck estimate/measurement.

    Enough to separate the two failure modes offline: joint-command jerk (compare consecutive
    q_cmd / the loop dt), and the wrong defend spot (compare the puck WORLD measurement and the
    V-frame estimate/intercept against where the puck actually was). ``q_meas``/``qd_meas`` are
    the feedback already read for the observation this tick -- reused here rather than issuing
    two more CAN reads, which would otherwise contend with the RT stream and add loop jitter.

    The rest of the columns are step 0c of CONTROLLER_FIX_PLAN.md: the things no amount of
    offline work recovers from a log that lacks them, so every later A/B depends on them being
    recorded from the first run rather than added once a number looks wrong.

      * ``q_streamed`` -- what the RT streamer actually WROTE, after its caps. ``q_cmd`` is only
        what the agent PLANNED, so on its own a log cannot tell an infeasible plan from an arm
        that failed to track a feasible one.
      * ``frame_seq`` / ``frame_age`` -- which camera frame this tick saw and how old it was.
        "No new frame" and "a frame with no puck in it" are indistinguishable in a measurement
        column that only ever says NaN, and they call for opposite fixes.
      * ``meas_age`` -- how stale the measurement the EKF actually FUSED was, from the camera's
        own capture stamp (Fix 8c). Distinct from ``frame_age``, which is the age of the newest
        frame at the moment the row was written: one says how late the evidence was, the other
        how long ago the camera last spoke, and only the first moves the estimate.
      * ``gate_ok`` -- see :func:`_gate_ok`.
      * ``vp_solved`` / ``vp_fallback`` -- cumulative, so a VPSTO fallback RATE is a number.
      * ``skipped`` -- agent ticks dropped to catch up with the wall clock (Fix 1e). 0 on an
        on-time tick and under ``--no-wall-clock``; a nonzero column IS the loop overrunning, and
        its distribution per tactic is what says whether the plan is keeping up with real time.
      * ``deficit`` -- ``|q_cmd - q_streamed|``, the gap the plan has opened on what the streamer
        could actually write. This is what ``--resync-cmd`` acts on, so the threshold is only
        readable next to the series it thresholds.
      * ``resync`` -- cumulative ``--resync-cmd`` corrections, so a resync RATE is a number.
      * ``rotor_temp`` / ``error_code`` -- per-joint, from the health poll's own readback. A log
        without them cannot distinguish "the arm got hot over a long match" from a software
        regression, which is the first question a long-run complaint asks.
    """
    m = getattr(agent, "_last_measurement", None)
    meas_world = m.pos_xy if m is not None else (np.nan, np.nan)
    est = agent.state.estimated_state
    gen = getattr(agent, "traj_generator", None)
    q_streamed = (streamer.last_written()[0] if streamer is not None
                  else np.full(N_ARM, np.nan))
    age = getattr(sensor, "frame_age", None)
    seq = getattr(sensor, "frame_seq", None)
    q_str = np.asarray(q_streamed, float)[:N_ARM]
    deficit = (float(np.linalg.norm(np.asarray(action[0], float)[:N_ARM] - q_str))
               if np.all(np.isfinite(q_str)) else np.nan)
    temps = getattr(arm, "last_temps", None)
    codes = getattr(arm, "last_codes", None)
    return dict(
        t=t, draw_ms=draw_ms, tactic=int(agent.state.tactic_current.value),
        q_cmd=np.asarray(action[0], float)[:N_ARM].copy(),
        q_meas=np.asarray(q_meas, float)[:N_ARM].copy(),
        qd_meas=np.asarray(qd_meas, float)[:N_ARM].copy(),
        meas_world=np.asarray(meas_world, float),
        est_v=np.asarray(est[:4], float).copy(),                 # x, y, vx, vy in the V frame
        intercept_v=np.asarray(agent.state.predicted_state[:2], float).copy(),
        q_streamed=np.asarray(q_streamed, float)[:N_ARM].copy(),
        frame_seq=-1 if seq is None else int(seq),
        frame_age=np.nan if age is None else float(age),
        meas_age=(lambda a: np.nan if a is None else float(a))(
            getattr(agent, "_last_meas_age", None)),
        gate_ok=bool(_gate_ok(agent)),
        defend_phase=(CLEAR_PHASES.get(getattr(agent.state, "defend_clear_phase", None), -1)
                      if agent.agent_params.get("defend_clear") else -1),
        # NaN when no early line is in play this tick (feature off, puck not head-on, already
        # past it, or not reachable in time) -- so the column says WHICH ticks got one.
        early_line=float(getattr(agent.state, "defend_early_line", np.nan)),
        vp_solved=int(getattr(gen, "vp_solved", 0)),
        vp_fallback=int(getattr(gen, "vp_fallback", 0)),
        skipped=int(getattr(agent, "last_skip", 0)),
        stream_err=int(getattr(streamer, "write_errors", 0)),
        deficit=deficit,
        resync=int(getattr(agent, "resync_count", 0)),
        rotor_temp=(np.full(N_ARM, np.nan) if temps is None
                    else np.asarray(temps, float)[:N_ARM].copy()),
        error_code=(np.full(N_ARM, -1, dtype=int) if codes is None
                    else np.asarray(codes, int)[:N_ARM].copy()),
    )


def warn_if_log_exists(path):
    """Warn BEFORE a run that ``--log path`` will overwrite an existing capture.

    Logs are the only record of what the rig did, and the A/B protocol compares a run against one
    taken days earlier -- so silently overwriting one destroys the baseline it was going to be
    compared against. This has already cost two: `goalie.npz` was overwritten twice, and
    `CONTROLLER_FIX_PLAN.md` still refers to a version of it that no longer exists. Warn at
    startup, not at save time, so there is something to do about it other than regret.
    """
    if not path or not os.path.exists(path):
        return
    try:
        old = np.load(path, allow_pickle=True)
        n = len(old["t"]) if "t" in old.files else 0
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
        detail = f"{n} ticks, written {when}"
    except Exception:
        detail = "unreadable"
    print(f"WARNING: --log {path} already exists ({detail}) and WILL be overwritten at the end "
          f"of this run.\n  Baselines are compared across days -- prefer a dated name, e.g. "
          f"--log {os.path.splitext(os.path.basename(path))[0]}_"
          f"{time.strftime('%m%d_%H%M')}.npz")


def _save_telemetry(path, rows, meta=None):
    """Write the per-tick columns, plus any run-CONSTANT metadata as scalars.

    The caps live in ``meta`` rather than in a column repeated N times: a log has to say what
    envelope it was taken under (the over-cap metrics in ``tools.analyze_telem --controller``
    are meaningless without it), but it is one number per run, not one per tick.
    """
    if not rows:
        return
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez(path, **cols, **(meta or {}))
    print(f"telemetry: {len(rows)} ticks -> {path}")


def _home_trajectory(q_from, q_to, dt, duration):
    """Eased (raised-cosine) joint-space waypoints q_from -> q_to over ``duration`` s, spaced
    ``dt``, with zero velocity at both ends. The path is joint-linear (the same path the vendor
    NRT interp takes to the same target) -- this only makes it smooth and gives it a soft start/stop.
    """
    q_from = np.asarray(q_from, float)[:N_ARM]
    q_to = np.asarray(q_to, float)[:N_ARM]
    dq = q_to - q_from
    n = max(2, int(round(duration / dt)))
    wps = []
    for i in range(n + 1):
        s = i / n
        ease = 0.5 * (1.0 - np.cos(np.pi * s))                 # 0->1, zero slope at both ends
        d_ease = 0.5 * np.pi / (n * dt) * np.sin(np.pi * s)    # d(ease)/dt for the waypoint velocity
        wps.append((q_from + dq * ease, dq * d_ease))
    return wps


def _rt_home_and_engage(arm, q_home, dt_agent, pos_limit, args):
    """Home entirely in RT: engage RT ONCE from the current resting pose (follower primed to it, so
    the PD engages steady), then stream a smooth eased glide to ``q_home``. No NRT move and no
    NRT->RT mode switch, so neither the NRT decel jerk nor the vendor engage transient occurs
    mid-run. The glide path is joint-linear -- identical to the vendor NRT move to the same q_home
    -- hence as rail-safe as NRT homing; speed is capped by --slow. Returns the started streamer
    with the arm settled at home."""
    q_cur = arm.q()[:N_ARM]
    vmax = resolve_max_joint_vel(args)
    amax = resolve_max_joint_accel(args)
    if vmax is not None:
        print(f"SLOW MODE: RT joint speed capped at {_cap_str(vmax)} rad/s "
              f"(~{_cap_str(np.degrees(vmax), '{:.0f}')} deg/s).")
    if amax is not None:
        print(f"RT joint accel capped at {_cap_str(amax, '{:.1f}')} rad/s^2.")
    streamer = _JointStreamer(arm, args.stream_hz, dt_agent, pos_limit, q_cur,
                              max_joint_vel=vmax, max_joint_accel=amax,
                              limit_mode=args.stream_limit,
                              couple_spin=args.stream_couple_spin,
                              lead_s=args.stream_lead_ms / 1000.0)
    print(f"RT homing (no NRT switch): smooth glide to the hockey home over ~{args.rt_home_time:.1f}s. "
          f"Path is joint-linear (same as the NRT move) -- keep a hand on the e-stop.")
    arm.stream_rt(q_cur)                   # prime the follower = current BEFORE engaging RT
    arm.set_mode_rt()
    arm.stream_rt(q_cur)                   # hold current through the engage
    streamer.start()
    if args.rt_settle > 0:
        sleep_until(time.perf_counter() + args.rt_settle)     # let the RT law lock on, at rest
    streamer.set_trajectory(_home_trajectory(q_cur, q_home, dt_agent, args.rt_home_time))
    if not wait_until_config(arm, q_home, timeout=max(20.0, args.rt_home_time * 3.0)):
        print("  WARNING: RT home glide did not reach the home pose within the timeout; starting "
              "play from the current pose.")
    wait_until_rest(arm)
    print("RT home complete; starting closed-loop play.")
    return streamer


def stage_play(arm, agent, env_info, sensor, args):
    """Full closed-loop play: a background thread streams the planned trajectory to the RT
    follower while this loop runs the (bursty) agent.

    With ``--view`` also opens a live annotated camera window (puck detection + the FSM's
    current tactic and estimated puck state) -- the fastest way to see whether the arm is
    idle because it sees no approaching puck vs a geometry problem. With ``--log PATH`` it
    writes per-tick telemetry (timing, commanded vs measured joints, puck estimate) for
    offline diagnosis.
    """
    agent_hz = 50
    dt_agent = 1.0 / agent_hz
    pos_limit = env_info["robot"]["joint_pos_limit"]
    # Resolved once and recorded in the telemetry (step 0c): the over-cap metrics in
    # `tools.analyze_telem --controller` cannot be computed without knowing the run's envelope,
    # and remembering it per log file is exactly the bookkeeping that drifts between runs.
    cap_vel = resolve_max_joint_vel(args)
    cap_accel = resolve_max_joint_accel(args)

    # Own-goal keep-out watchdog. The planner is already floored (--min-goal-dist, applied to
    # the agent), so plans do not target behind the line; this is the independent backstop --
    # if the MEASURED mallet still crosses a hard floor (planner floor minus a margin), a bad
    # puck estimate or IK overshoot is driving the arm toward its own goal / the near rim, so
    # catch it under gravity comp before it tips the table.
    keepout_v = None
    if not args.no_goal_keepout:
        plan_floor = goal_line_v(env_info) + args.min_goal_dist
        keepout_v = plan_floor - args.goal_keepout_hard_margin
        print(f"own-goal keep-out: plan floor V-x {plan_floor:+.3f} m, hard catch V-x "
              f"{keepout_v:+.3f} m (min-goal-dist {args.min_goal_dist*1000:.0f} mm, hard margin "
              f"{args.goal_keepout_hard_margin*1000:.0f} mm). Disable with --no-goal-keepout.")
        print("  NB the keep-out side depends on the base_frame: verify with --stage reach-probe "
              "(dist must DECREASE toward your own goal) BEFORE trusting this.")
    base_keepout_hard = None
    if args.base_keepout:
        base_keepout_hard = args.base_keepout - args.base_keepout_hard_margin
        print(f"base keep-out: plan radius {args.base_keepout:.3f} m, hard catch "
              f"{base_keepout_hard:.3f} m from the arm's own base. This is the disc the own-goal "
              f"x floor was really aiming at, and unlike it this check does not depend on "
              f"base_frame. Measure your own value with --stage reach-probe.")

    # Spin up the background vision thread (ThreadedPuckSensor) so the agent's per-tick
    # measure() returns instantly instead of blocking ~40 ms on capture+detect. Stopped by
    # sensor.close() in main()'s finally.
    if hasattr(sensor, "start"):
        sensor.start()

    if not args.mock:
        arm.set_gripper(args.gripper_mm)

    # Homing (both paths are gentle and independent of --slow's play cap):
    #  * default (NRT): a vendor NRT interpolated move to home, then hand over to RT. The hand-over
    #    is a mode switch, and the vendor RT law has an intrinsic engage transient we can only damp
    #    (prime the follower to the current pose + settle), not fully remove.
    #  * --rt-home: NO NRT move and NO mode switch -- engage RT once from the current resting pose
    #    and stream a smooth eased glide to home. Path is joint-linear (identical to the NRT move to
    #    the same q_home), so as rail-safe as NRT homing, but continuous (no decel jerk, no switch).
    #    Where: --home goal parks on the centreline, where no vertical wrist exists, so ask for
    #    the FLATTEST pose that reaches it instead of refusing (flat=True). The arm then brings
    #    up at its park spot; before this it had to home to its own station and let the Init
    #    tactic glide across under the play loop -- the same cross-table move, faster and less
    #    supervised than the vendor's NRT interpolator doing it. A --home-x/--home-y park takes
    #    the same route for the same reason: an arbitrary spot has no guarantee of a vertical-wrist
    #    solution either (those two flags are already ROBOT-frame, so all that is needed here is
    #    the x_home the agent settled on, mapped back). Plain --home base is unchanged (its own
    #    station, where a vertical wrist is comfortable, so the strict IK stays the check it was).
    # Home to the agent's OWN park, whatever chose it -- the taught default, --home goal, or
    # --home-x/-y -- and after the own-goal keep-out has had its say, so the arm brings up exactly
    # where the Ready tactic will hold it rather than at a spot the agent then slides it off.
    # flat=True throughout: a vertical wrist is not guaranteed at an arbitrary park, and where one
    # does exist (the taught park included) the strict solve is tried first anyway.
    home_xy = np.asarray(v2r_xform(env_info).pos2(
        np.asarray(agent.agent_params["x_home"], float)[:2]), float)
    q_home = home_config(env_info, target_xy=home_xy, flat=True)
    print(f"homing straight to the park (robot xy {np.round(home_xy, 3)} m), wrist "
          f"{np.degrees(home_config.last_tilt):.1f} deg off vertical -- the flattest this arm "
          f"can hold at that reach.")
    if args.rt_home:
        streamer = _rt_home_and_engage(arm, q_home, dt_agent, pos_limit, args)
    else:
        print(f"homing to the hockey start pose (NRT, velocity-ratio {args.home_velocity_ratio} -- "
              f"slow by default). Waiting for arrival...")
        arm.set_mode_nrt()
        arm.move_nrt(q_home, velocity_ratio=args.home_velocity_ratio)
        if not wait_until_config(arm, q_home):
            print("  WARNING: home move did not settle within the timeout; starting play from the "
                  "current pose anyway (RT stream begins at the measured position, so no jump).")
        wait_until_rest(arm)   # come fully to rest before the RT hand-over (residual velocity at
                               # the switch is arrested by the RT PD as a jerk otherwise)
        # Hand over to RT holding EXACTLY the current pose. The vendor RT law is a bare PD,
        # tau = kp*(q_des - q) + ..., with q_des = last streamed setpoint; on set_mode_rt() that
        # buffer is stale until the first stream lands, so prime it to the current pose BEFORE and
        # after the switch (commanding the current pose is inherently safe -- no motion), then hold
        # --rt-settle s so the law is steady before the agent moves the arm.
        q_start = arm.q()[:N_ARM]
        vmax = resolve_max_joint_vel(args)
        amax = resolve_max_joint_accel(args)
        if vmax is not None:
            print(f"SLOW MODE: RT joint speed capped at {_cap_str(vmax)} rad/s "
                  f"(~{_cap_str(np.degrees(vmax), '{:.0f}')} deg/s). The arm is rate-limited and "
                  f"cannot swing fast regardless of the plan.")
        if amax is not None:
            print(f"RT joint accel capped at {_cap_str(amax, '{:.1f}')} rad/s^2 "
                  f"(smooths replan-boundary jerk).")
        streamer = _JointStreamer(arm, args.stream_hz, dt_agent, pos_limit, q_start,
                                  max_joint_vel=vmax, max_joint_accel=amax,
                                  limit_mode=args.stream_limit,
                                  couple_spin=args.stream_couple_spin,
                                  lead_s=args.stream_lead_ms / 1000.0)
        arm.stream_rt(q_start)             # pre-engage: seed q_des = current so RT engages steady
        arm.set_mode_rt()
        arm.stream_rt(q_start)             # post-engage: hold current through the switch
        streamer.start()
        if args.rt_settle > 0:
            sleep_until(time.perf_counter() + args.rt_settle)
            print(f"RT engaged and held at home for {args.rt_settle:.2f}s; starting play.")

    # Let the agent see the stream's post-cap position so --resync-cmd can pull its plan back onto
    # what the arm was actually commanded to do. Both homing paths have built the streamer by here.
    agent.attach_streamer(streamer,
                          resync_thresh=(args.resync_thresh if args.resync_cmd else None))
    agent.episode_start()

    # Poll joint error/temperature codes at health_hz, NOT every agent tick. Every check is
    # two CAN reads that share the one bus lock with the RT stream writes; at 50 Hz they were a
    # measurable chunk of the loop's jitter tail. The arm's own kHz driver does the real-time
    # over-current/-temperature protection -- this supervisory poll only needs to catch a fault
    # within ~1/health_hz s to hand the arm to gravity comp, so 10 Hz is ample and far quieter.
    health_every = max(1, round(agent_hz / max(1e-6, args.health_hz)))

    warn_if_log_exists(args.log)
    rows = [] if args.log else None
    cv2, win = _play_view_setup(args)
    print(f"closed-loop play at agent {agent_hz} Hz / stream {args.stream_hz} Hz "
          f"(trajectory streamed on a background thread; health poll {args.health_hz:g} Hz). "
          f"Ctrl-C to exit.")
    t0 = time.perf_counter()
    next_t = t0
    ticks = 0
    t_prev_tick = None                  # start of the previous agent tick, for the wall clock
    try:
        while True:
            t_draw = time.perf_counter()
            q_meas, qd_meas = arm.q(), arm.qd()
            if keepout_v is not None and mallet_x_v(env_info, agent._r2v, q_meas) < keepout_v:
                raise HardwareFault(
                    f"mallet crossed the own-goal keep-out (V-x < {keepout_v:+.3f} m) -- the arm "
                    f"is reaching toward its own goal; catching under gravity comp before it "
                    f"tips the table. Raise --min-goal-dist / check the puck calibration.")
            if base_keepout_hard is not None and mallet_reach_r(env_info, q_meas) < base_keepout_hard:
                raise HardwareFault(
                    f"mallet folded back inside the base keep-out (reach < "
                    f"{base_keepout_hard:.3f} m) -- the wrist is coming over its own base and "
                    f"will push the near rim; catching under gravity comp before it tips the "
                    f"table. Raise --base-keepout.")
            obs = make_obs(env_info, q_meas, qd_meas)
            # How many whole agent ticks the previous pass ate beyond its own (Fix 1e). The
            # agent otherwise advances one waypoint and one fixed EKF step per call however long
            # the tick really took, so every overrun stretches the plan against real time.
            skip = 0
            if args.wall_clock and t_prev_tick is not None:
                skip = max(0, int(round((t_draw - t_prev_tick) / dt_agent)) - 1)
            t_prev_tick = t_draw
            action = agent.draw_action(obs, skip=skip)    # (2, 6); also refreshes the sensor
            draw_ms = (time.perf_counter() - t_draw) * 1e3
            streamer.set_trajectory(_lookahead_waypoints(action, agent, pos_limit))

            if ticks % health_every == 0:
                arm.check_health()
            if rows is not None:
                rows.append(_telemetry_row(time.perf_counter() - t0, draw_ms,
                                           q_meas, qd_meas, action, agent,
                                           streamer=streamer, sensor=sensor, arm=arm))
            if cv2 is not None and ticks % 3 == 0 and not _play_view_draw(cv2, win, agent, sensor):
                break

            ticks += 1
            if args.duration and (time.perf_counter() - t0) > args.duration:
                break
            # Pace the agent loop at ~agent_hz. When draw_action already overran the tick this
            # is a no-op (and we don't bank the lag); the stream thread has kept the arm
            # moving regardless.
            next_t = max(next_t + dt_agent, time.perf_counter())
            sleep_until(next_t)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        health = streamer.health_summary()
        if health:
            print(f"WARNING: the RT stream swallowed errors during this run -- {health}")
        if cv2 is not None:
            cv2.destroyAllWindows()
        if rows is not None:
            _save_telemetry(args.log, rows, meta=dict(
                cap_joint_vel=np.nan if cap_vel is None else np.asarray(cap_vel, float),
                cap_joint_accel=np.nan if cap_accel is None else np.asarray(cap_accel, float),
                agent_tick=float(dt_agent), stream_hz=float(args.stream_hz),
                planner=str(args.planner), goalie=bool(args.goalie),
                # What the RUN was configured with. Without these a log cannot say whether e.g.
                # the spin-joint exclusion or --resync-cmd was active, so an A/B pair months
                # apart is uninterpretable -- and the caps alone do not pin the behaviour.
                stream_limit=str(args.stream_limit),
                couple_spin=bool(args.stream_couple_spin),
                # The band the base joint was allowed, as APPLIED (post-intersection), not as
                # typed: a run taken inside a 60 deg sweep is not comparable with one taken
                # inside the model's 320 deg, and nothing else in the log would say which it was.
                j1_limit=np.asarray(pos_limit[:, J_BASE], float),
                resync_cmd=bool(args.resync_cmd),
                rendezvous_from_measured=bool(args.rendezvous_from_measured),
                # The camera and the mallet pose are just as much "what the run was" as the
                # caps are, and leaving them out already cost an A/B: a colour profile is only
                # valid at the exposure it was fitted under, and when a run came back slower
                # than its baseline NOTHING in the log could say what exposure it had pinned,
                # or whether --flat-mallet was on. Both had in fact changed. np.nan is "left
                # to the driver", which is a different state from any pinned value.
                flat_mallet=bool(args.flat_mallet),
                smash_slow=bool(args.smash_slow),
                smash_slow_vel=(np.nan if args.smash_slow_vel is None
                                else float(args.smash_slow_vel)),
                smash_slow_ticks=(np.nan if args.smash_slow_ticks is None
                                  else float(args.smash_slow_ticks)),
                defend_clear=bool(args.defend_clear),
                defend_clear_speed=(np.nan if args.defend_clear_speed is None
                                    else float(args.defend_clear_speed)),
                defend_clear_standoff=(np.nan if args.defend_clear_standoff is None
                                       else float(args.defend_clear_standoff)),
                defend_clear_max_path=(np.nan if args.defend_clear_max_path is None
                                       else float(args.defend_clear_max_path)),
                defend_early_frac=(np.nan if args.defend_early_frac is None
                                   else float(args.defend_early_frac)),
                defend_early_cos=(np.nan if args.defend_early_cos is None
                                  else float(args.defend_early_cos)),
                camera_index=int(args.camera_index),
                puck_profile=str(args.puck_profile or ""),
                cam_fps=np.nan if args.cam_fps is None else float(args.cam_fps),
                cam_exposure=(np.nan if args.cam_exposure is None
                              else float(args.cam_exposure)),
                cam_auto_exposure=(np.nan if args.cam_auto_exposure is None
                                   else float(args.cam_auto_exposure)),
                cam_auto_priority=(np.nan if args.cam_auto_priority is None
                                   else float(args.cam_auto_priority))))
    print(f"\nran {ticks} agent ticks in {time.perf_counter() - t0:.1f}s")


# ------------------------------------------------------------------------------------ main
#: What `--goalie` turns on. Each entry is only applied when the operator left that flag alone,
#: so any one of them can still be tuned or switched off individually alongside --goalie.
GOALIE_DEFAULTS = {
    "ekf_own_goal": True,
    # 0.35 was measured at the SIDE mount, where the goal mouth sat 0.36 m out from the base and
    # ran 0.39-0.41 m/s. From the end mount the same band is 0.11-0.33 m/s (the arm folds up over
    # its own base to reach it), so the old budget over-promises exactly where defence happens.
    "rendezvous_vel": 0.25,
    "defend_abort_dist": 0.02,
    "defend_block": "centre",
    "defend_centre_deadband": 0.03,
    "defend_give_up_ticks": 10,
    "fresh_horizon": True,
    "gate_unled": True,
    "defend_hold_target": True,
}

#: Of those, the ones declared `store_true`, whose "operator said nothing" sentinel is False
#: rather than None. The rest default to None, so for them False is an EXPLICIT `--no-...` and
#: must survive --goalie.
GOALIE_STORE_TRUE = ("ekf_own_goal",)


def apply_goalie_defaults(args):
    """Expand `--goalie` into the individual knobs, without overriding anything typed.

    Deliberately does NOT touch --max-joint-vel / --max-joint-accel: these are logic fixes and
    the tuned smooth envelope stays as it is. It also leaves the base keep-out alone -- that one
    changes where the arm is ALLOWED to go, so it is opted into separately.
    """
    if not getattr(args, "goalie", False):
        return args
    for name, value in GOALIE_DEFAULTS.items():
        current = getattr(args, name)
        unset = (current is False) if name in GOALIE_STORE_TRUE else (current is None)
        if unset:
            setattr(args, name, value)
    return args


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["grav", "home", "reach", "vision", "play",
                             "table-height", "reach-probe", "set-origin",
                             "cam-probe", "track-test"])
    ap.add_argument("--task", default="mini-hit-sdk",
                    help="platform task (selects the tactic); default mini-hit-sdk")
    ap.add_argument("--mock", action="store_true", help="use MockMetalArm (no hardware)")
    # arm / SDK
    ap.add_argument("--can-id", default="can0")
    ap.add_argument("--urdf", default=DEFAULT_URDF,
                    help="arm URDF (default: the self-contained vendor metal_with_gripper.urdf)")
    ap.add_argument("--arm-end-type", type=int, default=1, help="0 none, 1 gripper")
    ap.add_argument("--stream-include-gripper", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="append gripper stroke to the RT follower vector. Default: auto -- "
                         "ON when a gripper is attached (arm-end-type != 0), which this SDK "
                         "build requires (a 6-vector core-dumps). Use --no-stream-include-gripper "
                         "to force 6.")
    ap.add_argument("--gripper-mm", type=float, default=8.0,
                    help="gripper stroke to clamp the mallet handle (calibrate to the real mallet)")
    # play plane: keep the gripped mallet on the table instead of floating above it.
    ap.add_argument("--ee-height", type=float, default=None,
                    help="override the play plane: robot-frame z (m) of the grasp frame where "
                         "the mallet rests on the table. Measure it with --stage table-height. "
                         "Drops the whole controller (home, strikes, defends) onto the real "
                         "surface so the mallet stays in contact instead of floating/swooping.")
    ap.add_argument("--ee-height-npz", default=None,
                    help="npz holding the calibrated play height (key 'z'). --stage table-height "
                         "writes it; other stages read it when --ee-height is not given.")
    ap.add_argument("--contact-offset", type=float, default=0.0,
                    help="added to the calibrated play height (m). 0 = light contact (default); "
                         "a small positive value lifts the mallet a hair off the table if it binds.")
    ap.add_argument("--init-lift", type=float, default=0.0,
                    help="vertical lift (m) of the episode-start home pose above the play plane. "
                         "Default 0 keeps the mallet on the table (no start swoop); the sim "
                         "agent uses 0.06.")
    # own-goal keep-out: stop the arm driving the mallet past its own goal line into the near
    # rim (which shoves the table). Measure a safe value with --stage reach-probe.
    ap.add_argument("--min-goal-dist", type=float, default=0.06,
                    help="closest the mallet may come to its OWN goal line, in METRES (the "
                         "reach-probe prints mm, so divide by 1000: 30 mm -> 0.03). down-table. "
                         "Floors the planner's back-reach (Bezier near bound, hit/defend/prepare "
                         "ranges, home) so the arm never swings toward its own goal / the near "
                         "rim. Default 0.06 (~the home rest spot). Measure a safe value with "
                         "--stage reach-probe; raise it if the arm still reaches too far back.")
    ap.add_argument("--out-calib", default=None,
                    help="(set-origin) write a camera calibration carrying the SAME origin shift "
                         "as the corrected base frame, from --calib. Without it the extrinsic "
                         "keeps describing the old origin and every puck measurement lands off "
                         "by the shift -- silently, since both files look fine on their own.")
    ap.add_argument("--goal-keepout-hard-margin", type=float, default=0.03,
                    help="(play) how far past the planner floor the MEASURED mallet may drift "
                         "toward the own goal before the watchdog catches the arm under gravity "
                         "comp (m). Default 0.03.")
    ap.add_argument("--no-goal-keepout", action="store_true",
                    help="disable the own-goal keep-out (planner floor + watchdog). Not "
                         "recommended on hardware -- it is what stops the arm tipping the table.")
    # DEFEND sensitivity. The FSM defends as soon as the ESTIMATED puck speed toward the goal
    # crosses this threshold for a SINGLE tick (see is_puck_approaching). On a home surface
    # (no air) a hand-pushed puck decelerates fast and only briefly exceeds the vendored
    # scaled speed (0.4*TABLE_SCALE ~ 0.20 m/s), so it was missing real approaches -- hence
    # the low hardware default below. Too low and camera-position noise reads as motion
    # (spurious defends / twitching); raise it if that happens.
    ap.add_argument("--static-vel-threshold", type=float, default=0.02,
                    help="(play) puck speed (m/s, V frame) above which the FSM counts the puck "
                         "as MOVING and can start defending. Hardware default 0.02 (very "
                         "sensitive, for gentle at-home pushes; the sim uses the table-scaled "
                         "~0.20). Raise toward 0.05-0.10 if noise triggers spurious defends.")
    ap.add_argument("--switch-tactics-min-steps", type=int, default=None,
                    help="(play) how many ticks the FSM must hold a tactic before it may switch "
                         "again while mid-motion (the vendored default is 15 ~= 300 ms, which "
                         "can delay committing to DEFEND). Lower (e.g. 5) reacts sooner; too low "
                         "lets the tactic thrash (DEFEND<->READY) and adds jerk.")
    ap.add_argument("--defend-line-frac", type=float, default=None,
                    help="(play) where the DEFEND line sits, as a fraction of the table "
                         "half-length out from the OWN goal (vendored default 0.271, deep near "
                         "the goal). LARGER (e.g. 0.4-0.5) intercepts the puck EARLIER/further "
                         "out -- more reaction time AND less cross-table reach for far-side "
                         "shots (the puck hasn't spread to the post yet). Floored by the "
                         "own-goal keep-out.")
    ap.add_argument("--defend-width-frac", type=float, default=None,
                    help="(play) half-width the defender will chase across the goal mouth, as a "
                         "fraction of the table half-width (vendored default 0.867, which leaves "
                         "the near/far posts undefended). Raise toward 1.0-1.1 so far-side / "
                         "near-post shots trigger a defend at all.")
    # Slow-puck offence. The vendored smash needs `is_puck_static()`, and the hardware default
    # above is 0.02 m/s -- so a puck creeping at 0.05-0.15 m/s in front of the arm is claimed by
    # NOTHING: not the smash (not static), not DEFEND (not approaching), not PREPARE (not static).
    # The FSM sits in READY and watches it. These open exactly that gap, and nothing else.
    ap.add_argument("--smash-slow", action="store_true",
                    help="(play) strike a puck that is SLOW rather than dead still, once it has "
                         "settled in our own half, aiming at the enemy goal centre. Fills the gap "
                         "left by --static-vel-threshold 0.02: under that a puck is 'moving' to "
                         "the FSM but too slow for any tactic to claim, so it just sits there. "
                         "DEFEND still wins any approaching puck. Off = the vendored behaviour, "
                         "byte-identically.")
    ap.add_argument("--smash-slow-vel", type=float, default=None,
                    help="(play) puck speed (m/s) below which --smash-slow strikes. Default 0.3. "
                         "Must be ABOVE --static-vel-threshold or it does nothing at all (under "
                         "it the vendored static test already fires); the banner warns if so.")
    ap.add_argument("--smash-slow-ticks", type=int, default=None,
                    help="(play) consecutive slow ticks before --smash-slow fires. Default 5 "
                         "(0.1 s at 50 Hz). The dwell separates a settled puck from one briefly "
                         "slow mid-bounce. Note the FSM's own hysteresis "
                         "(--switch-tactics-min-steps, 15 ticks) usually paces entry into SMASH, "
                         "so a dwell under that rarely binds; raise it to make the settle "
                         "requirement real.")
    ap.add_argument("--smash-slow-max-x", type=float, default=None,
                    help="(play) up-table bound (V-frame x, m) on the --smash-slow band. Default "
                         "0.4, which chases ~45 mm past the DEFEND plant line (0.355) while "
                         "staying inside the vendored hit range. Lower it to 0.355 to keep the "
                         "arm strictly on its post; raise toward 0.511 to attack further out. "
                         "Floored by --min-goal-dist.")
    ap.add_argument("--goalie", action="store_true",
                    help="(play) MASTER SWITCH for the deep-defence fixes. Turns on "
                         "--ekf-own-goal, --rendezvous-vel 0.25, --defend-abort-dist 0.02, "
                         "--defend-block centre, --defend-centre-deadband 0.03, "
                         "--defend-give-up-ticks 10, --defend-hold-target, --fresh-horizon and "
                         "--gate-unled, unless you "
                         "set one of those explicitly. Together they stop the arm swinging "
                         "FORWARD at a phantom rebound and let it keep defending the last 138 mm "
                         "in front of its own goal -- which at the current end mount is also the "
                         "SLOWEST band of the workspace, 0.11-0.33 m/s, because the arm has to "
                         "fold up over its own base to reach it. "
                         "Off = the vendored behaviour, byte-identically.")
    ap.add_argument("--ekf-own-goal", action="store_true",
                    help="(play) treat our OWN goal mouth as a hole in the EKF bounce model. The "
                         "vendored score test can never fire for a V-frame x, so a puck headed "
                         "into our goal is predicted to bounce off the near rim and come back "
                         "up-table -- the arm then swings forward at a target that does not "
                         "exist. Measured: a puck at V-x 0.20 closing at 1 m/s predicts to V-x "
                         "0.64 moving +0.52 m/s. Pucks OUTSIDE the mouth still bounce (they "
                         "really do). Off = vendored.")
    ap.add_argument("--rendezvous-vel", type=float, default=None,
                    help="(play) Cartesian speed budget (m/s) for the earliest-feasible "
                         "intercept. Unset, the agent asks 'where is the puck max-prediction-time "
                         "from now' -- 1.0 s by default -- and plants there, which for a deep "
                         "puck is somewhere up-table. Set, it asks for the SOONEST reachable "
                         "intercept instead. 0.25 (the --goalie default) is the honest budget at "
                         "the end mount: measured worst-direction mallet speed at --max-joint-vel "
                         "1.0 is 0.33 m/s at the defend line and 0.11-0.22 deeper, where the arm "
                         "folds over its own base. It was 0.39-0.41 at the side mount, where the "
                         "goal mouth was a 0.36 m reach instead of 0.11.")
    ap.add_argument("--defend-abort-dist", type=float, default=None,
                    help="(play) how close to the own goal line (m) the agent keeps TRACKING a "
                         "puck before it abandons it and goes home. Splits the give-up line off "
                         "--defend-line-frac, which today is both the line the mallet stands on "
                         "AND the line past which nothing is defended -- two jobs that want "
                         "opposite values. Try 0.02. Floored by the own-goal keep-out.")
    ap.add_argument("--defend-block", choices=["mallet", "centre", "contact"], default=None,
                    help="(play) where the mallet plants relative to the predicted crossing. "
                         "'mallet' (vendored) offsets one mallet radius toward the centreline, "
                         "which leaves only 25 mm of contact margin on the BASE side against "
                         "75 mm on the other -- the 'swung but missed on the base side' case. "
                         "'centre' plants on the crossing point (50 mm each way). 'contact' is "
                         "the Prepare convention and is WORSE here (zero base-side margin).")
    ap.add_argument("--defend-centre-deadband", type=float, default=None,
                    help="(play) latch which side of the puck to tap while the predicted "
                         "crossing is within this |y| (m) of the centreline. The vendored "
                         "sign(y) flips the tap direction for a shot straight down the middle. "
                         "Try 0.03. 0 = vendored.")
    ap.add_argument("--defend-give-up-ticks", type=int, default=None,
                    help="(play) consecutive failing evaluations before DEFEND drops back to "
                         "READY and glides home (vendored 4). Raise (e.g. 10) so a deep puck is "
                         "not abandoned mid-approach.")
    ap.add_argument("--fresh-horizon", action=argparse.BooleanOptionalAction, default=None,
                    help="(play) hand the prediction its full horizon every tick. The vendored "
                         "caller feeds its own previous OUTPUT back in, a monotone ratchet; once "
                         "it decays below 0.1 s no defend can fire at all.")
    ap.add_argument("--gate-unled", action=argparse.BooleanOptionalAction, default=None,
                    help="(play) gate the tactics on the UN-led intercept and apply --lead-time "
                         "only to the planner target. Without this, any --lead-time above ~20 ms "
                         "drags the gated intercept below the defend line and disables DEFEND "
                         "entirely.")
    ap.add_argument("--defend-hold-target", action=argparse.BooleanOptionalAction, default=None,
                    help="(play) once DEFEND has latched, keep blocking the last intercept that "
                         "passed its own gate instead of following the estimate out of the "
                         "defend zone. Measured 19-50%% of DEFEND ticks aiming outside the zone; "
                         "in one run the target ran out to V-x 0.78 with the arm chasing it to "
                         "0.66 -- up-table, away from the goal, which is the opposite of "
                         "defending. Still bounded by --defend-give-up-ticks: an out-of-zone "
                         "intercept fails the gate, so the defend ends on schedule either way.")
    ap.add_argument("--defend-clear", action="store_true",
                    help="(play) RETURN an approaching puck up-table instead of stopping it. The "
                         "vendored block aims [0, +-1] at 0.05 m/s -- a sideways tap with no "
                         "up-table component -- so DEFEND is structurally a block. That is "
                         "invisible for an angled puck, which gets swatted by the momentum of "
                         "the cross-table traverse, and total for a head-on one, whose crossing "
                         "lands on the mallet's own line: measured 0.050 m/s of path speed with "
                         "zero travel against 0.703 m/s over 0.149 m. This stands the mallet off "
                         "down-table, then drives it THROUGH the crossing along +x. NOT part of "
                         "--goalie: it moves the mallet toward our own goal to cock.")
    ap.add_argument("--defend-clear-speed", type=float, default=None,
                    help="(play) terminal mallet speed (m/s) along +x at the crossing, with "
                         "--defend-clear. Default 0.25, matching --rendezvous-vel's own budget. "
                         "Cheap: peak path speed is set by the standoff, not by this (measured "
                         "identical across 0.15-0.30).")
    ap.add_argument("--defend-clear-standoff", type=float, default=None,
                    help="(play) run-up distance (m) down-table of the crossing, with "
                         "--defend-clear. THIS is the safety knob -- measured peak path speed "
                         "0.75 / 1.19 / 1.64 m/s at 0.04 / 0.06 / 0.08 against a 1 m/s puck. "
                         "Default 0.05. Bounded by the Bezier routing floor, so it can never "
                         "route behind the own-goal keep-out; lower it, not the speed, if the "
                         "arm is demanding more than it can track.")
    ap.add_argument("--defend-clear-margin", type=float, default=None,
                    help="(play) keep the cock this far (m) above the Bezier routing floor. "
                         "Default 0.02.")
    ap.add_argument("--defend-clear-max-path", type=float, default=None,
                    help="(play) mean path speed (m/s) a clearing stroke may demand of the arm; "
                         "above it DEFEND plants instead of swinging. Default 0.25. Measured on "
                         "the rig without it: the commit asked 3.58 m/s of mallet speed "
                         "(6.05 rad/s of joint against a 1.4 cap) and delivered 0.03 m/s, i.e. a "
                         "plant with a 0.39 rad command deficit. 0 disables the check.")
    ap.add_argument("--defend-early-frac", type=float, default=None,
                    help="(play) meet a NEAR-HEAD-ON puck this far up-table instead of at the "
                         "plant line, as a fraction of the own half from the goal line (same "
                         "convention as --defend-line-frac; 1.0 is the halfway line). The plant "
                         "line 0.251 sits BEHIND the taught park 0.306, so a normal defend has "
                         "negative run-up and the clearing block can never stroke: measured "
                         "d_avail p50 -0.010 m, only 31.8 percent of DEFEND ticks with any "
                         "forward room. Meeting the puck earlier buys run-up AND time. Try 0.6 "
                         "(V-x 0.418). Capped at the forward reach and self-limited by the "
                         "--rendezvous-vel reach test, so it never chases a line it cannot make.")
    ap.add_argument("--defend-early-cos", type=float, default=None,
                    help="(play) |vx|/|v| above which a puck counts as near-head-on for "
                         "--defend-early-frac. Default 0.90 (about 26 deg). Measured on the rig: "
                         "0.90 covers 48 percent of approaching ticks, 0.80 covers 71 percent.")
    ap.add_argument("--workspace-clamp", action=argparse.BooleanOptionalAction, default=True,
                    help="(play) clamp the predicted intercept into the arm's reachable play "
                         "region (--min-goal-dist back, the hit range forward, the table "
                         "half-width sideways) before any tactic plans to it. The clamp used to "
                         "sit behind --lead-time / --intercept-max-jump / --rendezvous-vel, so "
                         "the PLAINEST command line was the one running unclamped: a "
                         "velocity-spike extrapolation sent the target past the far rim "
                         "(measured V-x ~1.0) and the arm slewed out to reach it, into its joint "
                         "limits and an over-current fault. --no-workspace-clamp restores the old "
                         "behaviour for one A/B.")
    ap.add_argument("--rendezvous-from-measured", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="(play) run --rendezvous-vel's reachability test from where the mallet "
                         "actually IS, not from where it was last commanded to be. The RT stream "
                         "caps every step, so the command runs ahead of the arm -- a DEFEND "
                         "median of 16.5 mm and a p99 of 138 mm on goalie.npz, always in the "
                         "direction of travel -- and testing 'can I get there in time' from that "
                         "point makes every intercept look up to ~14 cm more reachable than it "
                         "is, worst exactly when the arm is moving hardest. Only a measured "
                         "distance is taken, never a commanded position, so this cannot "
                         "accumulate gravity droop the way an encoder resync would. "
                         "OFF by default, like every other behavioural flag here. It is a strictly "
                         "MORE CONSERVATIVE reachability test, so it makes the arm decline "
                         "intercepts it used to accept -- correct in principle, but on the rig it "
                         "reads as hesitancy, and it shipped on by default once, which changed how "
                         "the arm played without anyone asking for it. Ask for it explicitly.")
    ap.add_argument("--ready-guard", action=argparse.BooleanOptionalAction, default=None,
                    help="(play) while idle, hold the mallet on the line from the puck to our "
                         "own goal centre instead of parking on a fixed spot. When DEFEND gives "
                         "up the vendored Ready retreats to the park over a hard 1.0 s minimum "
                         "and cannot re-latch DEFEND for 300 ms -- while a 1 m/s puck crosses "
                         "the whole band in front of the goal in 138 ms. Works with either "
                         "--planner.")
    ap.add_argument("--ready-guard-radius", type=float, default=None,
                    help="(play) how far out from the own goal CENTRE the guard stands (m, "
                         "default 0.10). Larger cuts more angle but leaves more lateral ground "
                         "to cover, which this arm is slow at.")
    ap.add_argument("--ready-guard-drift", type=float, default=None,
                    help="(play) only re-aim the guard when the point moves more than this (m, "
                         "default 0.02). Stops EKF jitter turning the guard into chatter.")
    ap.add_argument("--ready-t-stop-min", type=float, default=None,
                    help="(play) floor on the READY glide duration (s, vendored 1.0). Try 0.25 "
                         "WITH --ready-guard; keep it >= 0.2 or the Bezier finishes every tick "
                         "and the tactic thrashes. Do NOT lower it without the guard -- a fast "
                         "retreat is worse than a slow one.")
    ap.add_argument("--base-keepout", type=float, default=None,
                    help="(play) minimum mallet reach from the arm's OWN base (m). A disc "
                         "keep-out, for a mount where the fold-over hazard IS a disc. LEAVE IT "
                         "OFF AT THE CURRENT END MOUNT: the base stands behind its own goal on "
                         "the table axis, so the hazard is a half-plane again (--min-goal-dist), "
                         "and the old side-mount value 0.26 would forbid the own goal line (0.113 "
                         "m of reach), the ready-guard point (0.213) and the defend plant line "
                         "(0.251) -- the whole defensive band. If you want one, MEASURE IT with "
                         "--stage reach-probe and expect well under 0.15.")
    ap.add_argument("--base-keepout-hard-margin", type=float, default=0.03,
                    help="(play) how far inside --base-keepout the MEASURED mallet may get "
                         "before the watchdog catches the arm under gravity comp.")
    ap.add_argument("--velocity-ratio", type=int, default=3, help="NRT move speed 1..10")
    ap.add_argument("--home-velocity-ratio", type=int, default=1,
                    help="(play) NRT speed 1..10 for the initial slow move to the hockey home "
                         "pose before closed-loop play begins. Default 1 (slowest/safest); "
                         "raise it once you trust the homing sweep.")
    # safety: cap how fast the arm swings (at-home development).
    ap.add_argument("--max-joint-vel", type=joint_cap_arg, default=None,
                    help="(play) HARD cap on commanded joint speed in the RT stream (rad/s). "
                         "Rate-limits every joint so the arm cannot swing fast no matter what "
                         "the planner asks -- the safety limit for developing at home. Try "
                         "0.3-0.5. Accepts ONE value, or six comma-separated (J1..J6) to cap each "
                         "joint on its own: sideways mallet speed is reach x the BASE-joint cap, "
                         "and J1 is the best lateral lever on the arm (0.741 m/rad vs J5's 0.164), "
                         "so e.g. '2.5,1.4,1.4,1.4,1.4,1.4' frees the base to reach the posts "
                         "while leaving the rest of the envelope alone -- the Cartesian strike "
                         "limits are derived from the SMALLEST entry, so they do not inflate with "
                         "it. Ladder J1 up (1.4 -> 1.8 -> 2.0 -> 2.5) watching rotor_temp, table "
                         "shake and faults. Default: no cap (full planned speed).")
    ap.add_argument("--max-joint-accel", type=joint_cap_arg, default=None,
                    help="(play) HARD cap on commanded joint ACCELERATION in the RT stream "
                         "(rad/s^2): bounds the per-write change in commanded velocity, so a "
                         "discontinuous plan (replan boundary) can no longer jerk the arm. "
                         "Independent of --max-joint-vel, and takes one value or six the same "
                         "way. Telemetry showed ~238 rad/s^2 spikes "
                         "even under a velocity cap; try 40-80 to bring it under the ~80 target "
                         "(the arm's measured joint_acc_limit is the ceiling -- and J1's is the "
                         "LOWEST of the six at 39.2, so a raised J1 speed cap still accelerates "
                         "into it at the same rate). Default: no cap.")
    ap.add_argument("--j1-limit", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="HARD position band for J1, the base joint, in RADIANS (e.g. "
                         "--j1-limit -0.60 0.55). The MJCF stops are +-2.79 rad (+-160 deg), far "
                         "wider than the sweep the arm can use at the end mount, so the planner "
                         "is free to route the base somewhere the rig cannot go. This clips it "
                         "where the motion is CHOSEN -- the tracking QP's position bound, the "
                         "nlopt hit-config bounds and every IK anchor (home pose included) -- "
                         "and again at the stream as a backstop, so a plan is never made that the "
                         "command then has to clip. Teach the two numbers with --stage grav: "
                         "teleop to each end of the usable sweep and read J1 off the live "
                         "readout. Only ever NARROWS (intersected with the stops). The home/park "
                         "pose has to lie inside the band -- if it does not, the home IK fails at "
                         "startup, before the arm moves. Default: the MJCF stops.")
    ap.add_argument("--slow", action="store_true",
                    help=f"convenience slow mode for at-home dev: sets --max-joint-vel "
                         f"{SLOW_MAX_JOINT_VEL} rad/s and --max-joint-accel "
                         f"{SLOW_MAX_JOINT_ACCEL} rad/s^2 (each unless given) AND forces the NRT "
                         f"--velocity-ratio to 1, so every stage moves slowly and smoothly.")
    ap.add_argument("--stream-hz", type=int, default=250, help="RT follower stream rate (<=400)")
    ap.add_argument("--stream-limit", choices=["coupled", "per-joint"], default="coupled",
                    help="(play) how the RT stream brings a step inside --max-joint-vel / "
                         "--max-joint-accel. 'coupled' (default) scales the whole joint vector "
                         "by one factor, so a throttled move slows down ALONG the planned path. "
                         "'per-joint' clips each joint independently, which shortens only the "
                         "joints that are over and bends the command off the path -- replaying "
                         "the logged plans that way reproduces the measured mallet lift (jaw-z "
                         "p99 57 mm vs 9 mm coupled). Kept only as the A/B escape hatch.")
    ap.add_argument("--stream-couple-spin", action="store_true",
                    help="(play) let the wrist SPIN joint (J6) help set the coupled scale factor "
                         "again, as it did before fix 6d. J6's position Jacobian is identically "
                         "zero -- it turns the mallet about its own axis and cannot move it -- so "
                         "by default it is clipped to its own cap instead of throttling the joints "
                         "that do carry the path. Measured on goalie/resync_base: J6 is the "
                         "largest planned mover on 44-58%% of moving ticks, binds 52-72%% of every "
                         "over-cap tick, and on 10-13%% of them is the ONLY reason the whole vector "
                         "was scaled down. Restore the old coupling only for an A/B.")
    ap.add_argument("--stream-lead-ms", type=float, default=0.0,
                    help="(play) command the plan this many ms AHEAD of the clock (q + qd*tau) "
                         "to cancel the RT follower's own lag -- the arm trails even a feasible "
                         "plan by 60-180 ms. Applied inside the speed/accel caps, so it cannot "
                         "push a step past them. Measure tau first with --stage track-test and "
                         "start at half of it; too much overshoots every reversal. Default 0 "
                         "(off).")
    ap.add_argument("--resync-cmd", action="store_true",
                    help="(play) re-seed the agent's plan from what the RT stream ACTUALLY wrote "
                         "whenever the two drift more than --resync-thresh apart. The agent plans "
                         "from its own trajectory buffer while the streamer re-anchors to its own "
                         "throttled position, so under a speed cap every plan starts from a point "
                         "the arm never reached -- measured 70 -> 238 mrad across the first 240 ms "
                         "of a rally, collapsing to zero only at idle. This is the mid-run half of "
                         "what restarting the script does. Off by default.")
    ap.add_argument("--resync-thresh", type=float, default=0.15,
                    help="(play) joint-space gap (rad) that triggers --resync-cmd. Default 0.15: "
                         "above the ~0.12 rad normal working deficit and below the measured rally "
                         "peak, so it corrects the runaway without firing on every throttled tick.")
    ap.add_argument("--wall-clock", action=argparse.BooleanOptionalAction, default=True,
                    help="(play) advance the agent by the number of ticks the WALL CLOCK "
                         "actually took, not one per call. Without it a tick that overran still "
                         "pops one waypoint and steps the EKF one fixed 20 ms, so the plan drifts "
                         "behind real time -- measured 13-19%% over a run, and a median 259-478 "
                         "ms per VPSTO strike. --no-wall-clock restores the old behaviour.")
    ap.add_argument("--rt-settle", type=float, default=0.4,
                    help="(play) seconds to hold the home pose in RT after the NRT->RT switch, "
                         "before closed-loop play starts, so the RT law engages smoothly instead "
                         "of lurching on the first plan (the post-homing jerk). 0 disables.")
    ap.add_argument("--rt-home", action="store_true",
                    help="(play) home entirely in RT: engage RT once from the resting pose and "
                         "glide smoothly to home, with NO NRT move and NO NRT->RT switch -- avoids "
                         "both the NRT decel jerk and the vendor mode-switch transient. Path is "
                         "joint-linear (same as the NRT move); use with --slow.")
    ap.add_argument("--rt-home-time", type=float, default=3.0,
                    help="(play, --rt-home) seconds for the smooth RT glide to the home pose.")
    ap.add_argument("--health-hz", type=float, default=10.0,
                    help="(play) rate to poll joint error/temperature codes. Each poll is two "
                         "CAN reads that contend with the RT stream, so lower = smoother; the "
                         "SDK's own loop handles instantaneous protection and a fault is still "
                         "caught within ~1/health_hz s. Default 10.")
    ap.add_argument("--duration", type=float, default=None, help="play seconds (default: until Ctrl-C)")
    ap.add_argument("--log", default=None,
                    help="(play stage) write per-tick telemetry .npz (timing, commanded vs "
                         "measured joints, puck estimate) for offline diagnosis")
    # camera / vision
    ap.add_argument("--calib", default=None, help="hw_calib.npz from calibrate_extrinsics")
    ap.add_argument("--camera-index", type=int, default=4)
    # Camera RATE (Fix 8b). Below ~20 detections/s the EKF cannot track a struck puck at all:
    # a 2.8 m/s puck moves ~250 mm between frames, the NIS gate rejects the measurement as an
    # outlier and the estimate freezes while the puck crosses the table. No planner tuning fixes
    # that, so these come before any of it.
    ap.add_argument("--cam-fps", type=float, default=None,
                    help="(vision/play/cam-probe) frame rate to request from the camera. V4L2 "
                         "picks the nearest it supports; --stage cam-probe reports what is "
                         "actually delivered. Default: leave the driver's own setting alone.")
    ap.add_argument("--cam-auto-exposure", type=float, default=None,
                    help="(vision/play/cam-probe) the V4L2 exposure MENU, set by ioctl: "
                         "**1 = MANUAL, 3 = auto**. (0.25/0.75 are accepted and translated, but "
                         "this rig's driver wants the raw 1/3 -- 0.25 writes 0, which it "
                         "rejects, leaving exposure on AUTO and silently ignoring "
                         "--cam-exposure.) Set this to 1 before --cam-exposure, or the exposure "
                         "does nothing.")
    ap.add_argument("--cam-auto-priority", type=int, default=None, choices=[0, 1],
                    help="(vision/play/cam-probe) V4L2 exposure_auto_priority. **0 = hold the "
                         "frame rate**, 1 = let the driver DROP frames to buy a longer exposure. "
                         "This is the usual reason a camera that advertises 30 fps delivers 20, "
                         "and OpenCV cannot reach it at all (no CAP_PROP), so it is set by "
                         "ioctl. Measured on the rig's Sonix KD-USB: 20 fps at priority 1.")
    ap.add_argument("--cam-exposure", type=float, default=None,
                    help="(vision/play/cam-probe) CAP_PROP_EXPOSURE, in whatever units the "
                         "driver uses (often log2 steps). Set --cam-auto-exposure to manual "
                         "first or this is ignored. Lower = shorter = faster and less motion "
                         "blur, at the cost of a darker frame (re-tune --puck-profile after).")
    ap.add_argument("--record-video", action="store_true",
                    help="(play/vision/cam-probe) record the RAW camera feed to an mp4 in "
                         "final-md/videos (default OFF). Encoded on a background thread off the "
                         "camera's grab thread, and frames are dropped rather than waited on, so "
                         "it cannot cost a camera frame or a control tick. Writes a "
                         "*_frames.npz sidecar of per-frame stamps + camera frame_seq, which is "
                         "also a --log column, so a video frame joins to the tick that saw it.")
    ap.add_argument("--record-video-path", default=None,
                    help="explicit mp4 path for --record-video (and implies it). Default: "
                         "final-md/videos/<--log stem, else stage>_<date_time>.mp4.")
    ap.add_argument("--track-joint", type=int, default=1,
                    help="(track-test) which joint to oscillate, 0-based. Default 1.")
    ap.add_argument("--track-amp", type=float, default=0.08,
                    help="(track-test) sine amplitude in rad. Default 0.08 (~4.6 deg).")
    ap.add_argument("--track-freq", type=float, default=0.75,
                    help="(track-test) sine frequency in Hz. Default 0.75.")
    ap.add_argument("--puck-profile", default=None,
                    help="tuned real-puck colour gate JSON (from mini_platform.hw.tune_puck)")
    ap.add_argument("--sigma-px", type=float, default=0.186)
    ap.add_argument("--no-table-mask", action="store_true",
                    help="(vision/play) disable the play-area pixel mask. The mask zeroes "
                         "everything outside the table and its two goal mouths BEFORE "
                         "detection, so an off-table red object can never win the "
                         "largest-blob selection. Turn it off only to A/B against the "
                         "pre-mask behaviour.")
    # de-jerk (play): output-side smoothing of the predicted intercept. See MiniSystemState.
    ap.add_argument("--intercept-max-jump", type=float, default=None,
                    help="(play) cap the predicted intercept's per-tick L2 jump (m). Tames the "
                         "noisy-velocity intercept explosion (measured up to ~1.3 m/tick) that "
                         "drives joint jerk. Recommended 0.05; <=0 or unset disables.")
    ap.add_argument("--intercept-ema-alpha", type=float, default=None,
                    help="(play) optional EMA factor (0-1) pulling the predicted intercept toward "
                         "last tick's value; smaller = smoother. Try 0.4. Unset disables.")
    ap.add_argument("--max-puck-speed", type=float, default=None,
                    help="(play) physical cap (m/s) on the estimated puck speed on the vision "
                         "path; kills detection-outlier velocity spikes (measured 40-60 m/s) that "
                         "explode the intercept and false-trigger defend. Try 4.0. Unset disables.")
    ap.add_argument("--lead-time", type=float, default=0.0,
                    help="(play) latency lead (s): aim the intercept this far further along the "
                         "predicted puck trajectory, to compensate camera+compute+stream delay so "
                         "the arm stops chasing a stale puck (late/miss when far). Try 0.05-0.08. "
                         "0 (default) disables. A pure target shift -- the plan horizon is untouched.")
    ap.add_argument("--max-prediction-time", type=float, default=None,
                    help="(play) override the puck-prediction horizon cap (s, vendored 1.0). Larger "
                         "lets the arm commit to a farther-future intercept for fast/far pucks. "
                         "Raise only with a feasibility-aware planner (--planner vpsto); otherwise "
                         "the arm over-commits to a spot it cannot reach in time. Unset keeps 1.0.")
    ap.add_argument("--home", choices=["base", "goal"], default="base",
                    help="where the mallet PARKS between plays. 'base' (default) is the TAUGHT "
                         "park (baseline_mini.HOME_PARK_XY, currently PROVISIONAL -- re-teach it "
                         "with hw/capture_home after a re-mount). 'goal' parks "
                         "on the centreline in front of the own goal -- on the defend line, so "
                         "a shot down the middle needs no travel -- at the cost of idling at a "
                         "0.37 m cross-table reach, where the flattest wrist the arm can hold "
                         "is ~10 deg off vertical (no vertical solution exists there). Either "
                         "mode's park can be moved per axis with --home-x / --home-y.")
    ap.add_argument("--home-goal-dist", type=float, default=None,
                    help="with --home goal: park this far (m) up-table from the own GOAL LINE "
                         "instead of on the defend line. Floored by --min-goal-dist.")
    ap.add_argument("--home-x", type=float, default=None,
                    help="park the mallet at this ROBOT-BASE-frame x (m), overriding whatever "
                         "--home chose: straight out from the base, so this is the reach-out "
                         f"distance (the default base park is HOME_POS, x={HOME_POS[0]:.3f}). "
                         "The park is still floored by the own-goal keep-out and refused if it "
                         "leaves the table -- both of which live in table coordinates, so the "
                         "banner prints where the ask actually landed.")
    ap.add_argument("--home-y", type=float, default=None,
                    help="park the mallet at this ROBOT-BASE-frame y (m), overriding whatever "
                         "--home chose: sideways across the base, 0 = straight out from it "
                         "(the default base park). Sign follows the arm's own y axis, not the "
                         "table's. Combine with --home-x for an arbitrary park; either flag "
                         "alone moves only its own robot-frame axis and leaves the other at "
                         "whatever --home picked.")
    ap.add_argument("--flat-mallet", action="store_true",
                    help="spend the arm's redundancy on keeping the mallet FLAT on the table. "
                         "Covers the whole stack: the tracking QP levels the wrist every tick "
                         "(and spends what is left on the joint anchor, as before), the strike "
                         "anchors are levelled wherever a strictly vertical wrist is out of "
                         "reach, and homing goes to the flattest reachable pose. Default off = "
                         "the vendored behaviour, where nothing controls the tool axis at all "
                         "and the wrist drifts ~55 deg off vertical within seconds, so the "
                         "mallet plays on its edge. RECOMMENDED ON HARDWARE at the current end "
                         "mount: the tilt-aim compensation it replaces (baseline_mini."
                         "AIM_BIAS_COMP, 6.5 deg) was measured at the side mount and the lean it "
                         "cancels points at the base, whose direction changed with the re-mount, "
                         "so it is unvalidated. --flat-mallet squares the contact normal and "
                         "selects AIM_BIAS_FLAT = 0 instead.")
    ap.add_argument("--ekf-params", default=None,
                    help="npz of puck-motion constants measured on THIS table "
                         "(tools.fit_mini_table -> mini_table_ekf.npz): damping, the rim "
                         "collision rows and OBS_COV. Default off = the vendored full-size "
                         "KUKA fits, which bias every prediction on the mini table.")
    ap.add_argument("--vel-lpf-hz", type=float, default=None,
                    help="(play) low-pass the EKF output puck velocity at this cutoff (Hz) to steady "
                         "the intercept (the velocity is forward-integrated into it). Keep it LIGHT "
                         "(8-15 Hz); too low lags the velocity and reacts late. Unset disables. "
                         "NB the 24 mm fresh-frame position snap is better fixed by recalibration "
                         "(Stage 2 refit) or a higher --sigma-px; this targets velocity jitter.")
    ap.add_argument("--meas-age-lead", action="store_true",
                    help="(play) aim the intercept the measured MEASUREMENT AGE further along "
                         "the puck's predicted path. A camera measurement is a frame period + "
                         "detection + hand-off old by the time the EKF sees it, so the estimate "
                         "tracks where the puck WAS: measured, the filter sits a steady 72 mm "
                         "behind a 1.2 m/s puck. This leads the target by the age the camera "
                         "actually reports (smoothed), through the same bounce-aware prediction "
                         "the tactics already gate on. Like --lead-time but self-measuring; "
                         "needs a camera that stamps its frames. Off = vendored.")
    ap.add_argument("--meas-lag-comp", action="store_true",
                    help="(play) EXPERIMENTAL, and measured WORSE than leaving it off (124 mm "
                         "against 50 mm mean error on a 3-tick lag; tests/test_latency.py). "
                         "Fuses each measurement against the filter's own past state instead of "
                         "the current prediction -- which is circular (that state is only the "
                         "right one if the filter already tracks NOW), and around a rim bounce "
                         "the reference is on the wrong side of the discontinuity, so the gate "
                         "rejects and the filter re-locks with a bad velocity. Prefer "
                         "--meas-age-lead. Kept only so the approach can be re-tried.")
    ap.add_argument("--max-plan-steps", type=int, default=None,
                    help="(play) agent ticks between target refreshes (vendored 5 = 100 ms, "
                         "during which a 2 m/s puck moves 200 mm). 3 refreshes every 60 ms at "
                         "the cost of more replans on the CPU the vision thread needs -- watch "
                         "draw_ms and the detection rate together. Default: vendored.")
    ap.add_argument("--planner", choices=["bezier", "vpsto"], default="bezier",
                    help="(play) strike planner. 'bezier' (default) = the vendored Bezier + "
                         "per-waypoint QP-IK. 'vpsto' = a joint-space, vel/acc-feasible, "
                         "time-optimal VPSTO swing to the intercept (fixes the infeasible-plan lag "
                         "that makes the arm miss when the puck is far). Needs `cma` + vp-sto.")
    # NB VPSTO plans the strike in the CARTESIAN table plane (see planning/vpsto_striker.py),
    # so these are mallet limits in m/s and m/s^2 -- NOT joint limits. The joint-space caps are
    # --max-joint-vel / --max-joint-accel, which still bound the stream downstream of the QP.
    ap.add_argument("--vpsto-vel-lim", type=float, default=None,
                    help="(play, --planner vpsto) cartesian mallet speed limit (m/s) the strike "
                         "path is planned within. Raise (Stage 4) to hit harder / reach farther; "
                         "too high and the QP-IK must clip near a singularity. Default: derived "
                         f"from --max-joint-vel ({VPSTO_VEL_PER_JOINT_VEL} m/s per rad/s, this "
                         f"arm's measured worst-direction speed), or {VPSTO_VEL_LIM_UNCAPPED} "
                         "when there is no joint cap at all. The old fixed 1.5 was 4-8x what the "
                         "arm can do, so the streamer throttled every swing.")
    ap.add_argument("--vpsto-acc-lim", type=float, default=None,
                    help="(play, --planner vpsto) cartesian mallet acceleration limit (m/s^2) for "
                         "the strike path. Lower = smoother/slower swings. Default: "
                         f"{VPSTO_ACC_LIM_CAPPED} under a --max-joint-vel cap, else "
                         f"{VPSTO_ACC_LIM_UNCAPPED}.")
    ap.add_argument("--vpsto-smooth", type=float, default=0.0,
                    help="(play, --planner vpsto) smoothness weight. VPSTO minimises the strike "
                         "DURATION, so by default it saturates the vel/acc limits and the swing "
                         "is a sprint. A positive weight penalises acceleration effort, so the "
                         "planner accepts a slower path in exchange for controlled motion. Try "
                         "0.3-1.0; 0 (default) is pure minimum time.")
    ap.add_argument("--vpsto-max-iter", type=int, default=10,
                    help="(play, --planner vpsto) CMA iterations per strike solve (warm-started). "
                         "Lower = faster/less time-optimal; 0 uses the deterministic mean path. "
                         "Default 10: in isolation 30 iterations cost 12 ms p50 against 5 ms for "
                         "10, and on the rig that solve competes with the vision thread -- the "
                         "detection rate is worth more than the last few percent of path time.")
    ap.add_argument("--view", action="store_true",
                    help="(vision/play/cam-probe) open a live camera window. vision/play annotate "
                         "the puck detection (play also the FSM tactic + estimate); cam-probe "
                         "shows delivered fps, the exposure in ms, the motion blur that implies "
                         "and a sharpness readout, and lets you change the exposure controls on "
                         "the running camera ([ ] exposure, a manual/auto, p auto-priority, "
                         "q quit) -- then prints the flags to paste.")
    # geometry
    ap.add_argument("--base-frame", default=None, help="npz with key 'T' (4x4 world<-base)")
    ap.add_argument("--ref-world-xy", type=float, nargs=2, default=None,
                    metavar=("X", "Y"),
                    help="(set-origin) known world xy (m) the touched mallet should map to. "
                         "Default: the own-goal-line centre [-L/2, 0].")
    ap.add_argument("--out", default=None,
                    help="(set-origin) output npz for the origin-corrected base_frame "
                         "(default base_frame_originfixed.npz).")
    args = ap.parse_args(argv)
    if args.slow:
        args.velocity_ratio = 1     # slowest NRT interpolation for home/reach moves too
    apply_goalie_defaults(args)

    env_info = make(args.task).env_info
    load_base_frame(env_info, args.base_frame)

    # Narrow the base joint's band BEFORE anything reads it: the home pose (IK), the agent's
    # optimiser (constructed with env_info) and the RT streamer's clamp all take their copy from
    # here, so this is the one place that reaches every consumer.
    if args.j1_limit is not None:
        narrow_joint_pos_limit(env_info, J_BASE, *args.j1_limit, flag="--j1-limit")

    if args.stage == "play" and not args.no_goal_keepout:
        err = min_goal_dist_error(args.min_goal_dist, env_info["table"]["length"] / 2)
        if err:
            # raise SystemExit(err)
            pass
    # Recalibrate the play plane to the real table BEFORE anything reads the height (home
    # pose, the optimiser's z-constraint band, the agent's hit anchors).
    # (the table-height stage MEASURES the height, so it never applies an override.)
    ee_height = resolve_ee_height(args)
    if ee_height is not None and args.stage != "table-height":
        set_play_height(env_info, ee_height)

    q_home = home_config(env_info)
    arm = build_arm(args, q_start=q_home)

    sensor = None
    agent = None
    if args.stage in ("vision", "play"):
        sensor = build_sensor(args, env_info)
    if args.stage == "play":
        # Run vision (render + detect) on a background thread so it never blocks the
        # plan+stream loop. The cv2 detector backend keeps a frame to a few ms, but the
        # decoupling still guards against a stray slow frame re-anchoring the RT streamer
        # mid-tick (which turns into visible jerk).
        from mini_platform.hw.real_sensor import ThreadedPuckSensor
        sensor = ThreadedPuckSensor(sensor)
        ekf_params = None
        if args.ekf_params:
            from mini_platform.estimation.kalman_filter import EkfParams
            ekf_params = EkfParams.load(args.ekf_params)
        imj = args.intercept_max_jump
        agent = HwBaselineAgent(env_info, sensor, agent_id=1,
                                only_tactic=only_tactic_for(args.task),
                                init_lift=args.init_lift,
                                min_goal_dist=(None if args.no_goal_keepout
                                               else args.min_goal_dist),
                                intercept_max_jump=(imj if (imj is not None and imj > 0)
                                                    else None),
                                intercept_ema_alpha=args.intercept_ema_alpha,
                                max_puck_speed=args.max_puck_speed,
                                lead_time=args.lead_time,
                                max_prediction_time=args.max_prediction_time,
                                vel_lpf_hz=args.vel_lpf_hz,
                                ekf_params=ekf_params,
                                home=args.home,
                                home_goal_dist=args.home_goal_dist,
                                home_x=args.home_x,
                                home_y=args.home_y,
                                keep_mallet_flat=args.flat_mallet,
                                base_keepout=args.base_keepout,
                                ready_guard=bool(args.ready_guard),
                                ready_guard_radius=args.ready_guard_radius,
                                ready_guard_drift=args.ready_guard_drift,
                                ready_t_stop_min=args.ready_t_stop_min,
                                ekf_own_goal=args.ekf_own_goal,
                                rendezvous_vel=args.rendezvous_vel,
                                defend_abort_dist=args.defend_abort_dist,
                                defend_block=args.defend_block,
                                defend_centre_deadband=args.defend_centre_deadband,
                                defend_give_up_ticks=args.defend_give_up_ticks,
                                defend_fresh_horizon=bool(args.fresh_horizon),
                                defend_gate_unled=bool(args.gate_unled),
                                defend_hold_target=bool(args.defend_hold_target),
                                workspace_clamp=bool(args.workspace_clamp),
                                rendezvous_from_measured=bool(args.rendezvous_from_measured),
                                meas_lag_comp=bool(args.meas_lag_comp),
                                meas_age_lead=bool(args.meas_age_lead),
                                max_plan_steps=args.max_plan_steps,
                                smash_slow=bool(args.smash_slow),
                                smash_slow_vel=args.smash_slow_vel,
                                smash_slow_ticks=args.smash_slow_ticks,
                                smash_slow_max_x=args.smash_slow_max_x,
                                defend_clear=bool(args.defend_clear),
                                defend_clear_speed=args.defend_clear_speed,
                                defend_clear_standoff=args.defend_clear_standoff,
                                defend_clear_margin=args.defend_clear_margin,
                                defend_clear_max_path=args.defend_clear_max_path,
                                defend_early_frac=args.defend_early_frac,
                                defend_early_cos=args.defend_early_cos,
                                planner=args.planner,
                                vpsto_kwargs=vpsto_kwargs_from_args(args))
        if args.planner == "vpsto":
            vel_lim, acc_lim, derived = resolve_vpsto_limits(args)
            cap = resolve_max_joint_vel(args)
            how = ""
            if derived:
                src = (f"--max-joint-vel {cap}" if cap is not None
                       else "the no-joint-cap defaults")
                how = f" [{'+'.join(derived)} from {src}]"
            print(f"planner: VPSTO strike (cartesian vel_lim {vel_lim} m/s, acc_lim "
                  f"{acc_lim} m/s^2, smooth {args.vpsto_smooth}, "
                  f"{args.vpsto_max_iter} CMA iters){how}. "
                  f"Feasible time-optimal swings; bezier is the per-strike fallback.")
            if args.rendezvous_vel is not None and float(args.rendezvous_vel) > vel_lim:
                print(f"  WARNING: --rendezvous-vel {args.rendezvous_vel} m/s is above the "
                      f"strike path's own vel_lim {vel_lim} m/s, so the intercept is picked on "
                      f"a speed budget the planner will never spend -- every block is chosen "
                      f"too near and reached late. Lower it, or raise --vpsto-vel-lim.")
        if args.flat_mallet:
            print("flat mallet: the arm's redundancy now holds the tool axis vertical, in the "
                  "controller AND\n  the planners -- the tracking QP levels the wrist every "
                  "tick (then spends what is left on\n  the joint anchor as before), the strike "
                  "anchors are levelled wherever a strictly vertical\n  wrist is out of reach, "
                  "and homing goes to the flattest reachable pose. Measured in sim:\n  wrist "
                  "tilt median 54 -> ~2 deg. The mallet face meets the puck squarely instead of "
                  "on its rim.")
        custom_park = args.home_x is not None or args.home_y is not None
        if True:
            # EVERY park gets the same scrutiny in the banner -- where it ended up, how flat a
            # wrist reaches it, how far the arm is stretched holding it -- including the default,
            # which is no longer the vendored 0.17 m station but a taught pose that can sit
            # anywhere the operator put it. A park is unvetted by construction, so print the
            # numbers BEFORE the arm moves there.
            xh = agent.agent_params["x_home"]
            gl = goal_line_v(env_info)
            base_v = mallet_base_v(env_info)
            p_r = np.asarray(v2r_xform(env_info).pos2(np.asarray(xh, float)[:2]), float)
            reach = np.linalg.norm(p_r)          # robot frame is centred on the base
            park_reached, _, park_tilt = solve_flat_config(
                env_info["robot"]["robot_model"], env_info["robot"]["robot_data"],
                np.array([p_r[0], p_r[1], env_info["robot"]["ee_desired_height"]]),
                INIT_SEED, n_joints=N_JOINTS, max_time=0.5, probe_time=20e-3)
            if custom_park:
                # --home-x/--home-y are robot-frame, the own-goal keep-out is a TABLE-frame floor,
                # and the two frames are rotated: flooring the park moves BOTH robot coordinates.
                # Print the ask against where it ended up rather than a per-axis delta.
                asked_r = np.array([args.home_x if args.home_x is not None else p_r[0],
                                    args.home_y if args.home_y is not None else p_r[1]])
                if np.linalg.norm(asked_r - p_r) > 1e-6:
                    print(f"NB the park asked for (robot xy {np.round(asked_r, 3)}) reaches behind "
                          f"the own-goal keep-out; it was moved out to\n   robot xy "
                          f"{np.round(p_r, 3)} m (--min-goal-dist {args.min_goal_dist:g}).")
            centred = abs(xh[1]) <= 1e-6
            side = ("" if (centred or abs(base_v[1]) <= 1e-3) else
                    (", the arm's side" if xh[1] * base_v[1] > 0 else ", the far side"))
            head = ("WHERE --home-x/-y ASKED" if custom_park else
                    "IN FRONT OF THE GOAL" if args.home == "goal" else "THE TAUGHT DEFAULT")
            print(f"park: {head} at robot xy {np.round(p_r, 3)} m, {reach:.2f} m out from the "
                  f"base\n  = table V-x {xh[0]:+.3f}, y {xh[1]:+.3f} "
                  f"({(xh[0] - gl) * 1000:.0f} mm up-table from the own goal line, "
                  f"{abs(xh[1]) * 1000:.0f} mm off the centreline{side})." +
                  (" A centred shot needs no travel." if centred else ""))
            if not park_reached:
                # Only a hand-picked park can land here; say so plainly, because the bring-up
                # IK a few lines later refuses outright and the arm never leaves its rest pose.
                print(f"  BUT NO ARM CONFIGURATION REACHES IT -- a {reach:.2f} m stretch, and "
                      f"the table height pins the third axis.\n  Bring-up will refuse: pull the "
                      f"park in with a smaller --home-x / --home-y.")
            else:
                print(f"  The arm BRINGS UP there directly, at the flattest wrist that reaches it "
                      f"({np.degrees(park_tilt):.0f} deg off\n  vertical" +
                      ("" if np.degrees(park_tilt) < 1.0 else
                       " -- a strictly vertical wrist has no IK solution at that reach") + ").\n"
                      f"  It then idles at a {reach:.2f} m reach -- watch the first minute for "
                      f"droop or a held-current fault." +
                      ("" if args.flat_mallet else
                       "\n  WITHOUT --flat-mallet the wrist then drifts back to ~57 deg as soon "
                       "as play starts: pass both."))
        if ekf_params is not None:
            print(ekf_params.describe())
            if ekf_params.sigma_px is not None and abs(args.sigma_px - ekf_params.sigma_px) > 0.02:
                print(f"   NB that fit also measured sigma_px = {ekf_params.sigma_px:.3f} px, but "
                      f"this run uses --sigma-px {args.sigma_px:.3f}. On hardware the SENSOR's "
                      f"sigma_px sets the\n   measurement covariance the filter actually "
                      f"trusts (OBS_COV is only the sim/GT default), so pass "
                      f"--sigma-px {ekf_params.sigma_px:.3f} too.")
        if args.vel_lpf_hz is not None:
            print(f"EKF velocity LPF: {args.vel_lpf_hz:.1f} Hz cutoff (steadies the intercept).")
        if args.lead_time:
            print(f"latency lead: aiming {args.lead_time*1000:.0f} ms further along the predicted "
                  f"puck path (compensates sensor+compute+stream delay).")
        if args.meas_age_lead:
            print("measurement-age lead: ON -- the intercept is aimed the camera's own reported "
                  "measurement\n  age further along the predicted path, so the target is where "
                  "the puck will be rather than\n  where a stale frame said it was. Check the "
                  "meas_age column: NaN means the camera is not\n  stamping frames and this is "
                  "doing nothing.")
        if args.meas_lag_comp:
            print("measurement lag compensation: ON -- EXPERIMENTAL, and measured WORSE than off "
                  "(124 mm vs\n  50 mm mean error on a 3-tick lag). It is circular, and it "
                  "re-locks around rim bounces.\n  Prefer --meas-age-lead.")
        if args.stream_lead_ms:
            print(f"stream lead: commanding the plan {args.stream_lead_ms:.0f} ms ahead of the "
                  f"clock to cancel the RT\n  follower's lag. Inside the speed/accel caps. "
                  f"Watch for overshoot at reversals.")
        if args.max_plan_steps is not None:
            # NB dt_agent is a local of the play loop, not of main() -- this banner is the only
            # place in main() that wants the agent tick, and it crashed with a NameError the
            # first time anyone passed --max-plan-steps. Take it from env_info, which is what
            # the play loop derives its own agent_hz from.
            _dt_agent = 1.0 / env_info["robot"]["control_frequency"]
            print(f"replan cadence: targets refresh every {args.max_plan_steps} ticks "
                  f"({args.max_plan_steps * _dt_agent * 1000:.0f} ms), vendored 5 (100 ms). "
                  f"Watch draw_ms AND the detection rate.")
        if args.max_prediction_time is not None:
            print(f"prediction horizon cap: {args.max_prediction_time:.2f} s "
                  f"(vendored 1.0; larger = commit to farther-future intercepts).")
        if args.static_vel_threshold is not None:
            old = agent.agent_params["static_vel_threshold"]
            agent.agent_params["static_vel_threshold"] = args.static_vel_threshold
            print(f"defend sensitivity: static_vel_threshold {old:.3f} -> "
                  f"{args.static_vel_threshold:.3f} m/s (puck counts as moving above this)")
        # After the override above, so it compares against the threshold the run will ACTUALLY use.
        if args.smash_slow:
            _thr = float(agent.agent_params["static_vel_threshold"])
            _vel = float(agent.agent_params["smash_slow_vel"])
            _ticks = int(agent.agent_params["smash_slow_ticks"])
            _lo = float(agent.agent_params["hit_range"][0])
            _hi = float(agent.agent_params["smash_slow_max_x"])
            print(f"slow-puck smash: ON -- a puck under {_vel:.2f} m/s held for {_ticks} ticks "
                  f"in V-x [{_lo:+.3f}, {_hi:+.3f}]\n  is struck at the enemy goal centre. DEFEND "
                  f"still wins any approaching puck.")
            if _vel <= _thr:
                print(f"  WARNING: --smash-slow-vel {_vel:.2f} is at or below "
                      f"static_vel_threshold {_thr:.2f}, so every puck it could claim is already "
                      f"'static'\n  and the vendored smash has it. This flag is doing NOTHING -- "
                      f"raise it above {_thr:.2f}.")
        if args.switch_tactics_min_steps is not None:
            old = agent.agent_params["switch_tactics_min_steps"]
            agent.agent_params["switch_tactics_min_steps"] = args.switch_tactics_min_steps
            print(f"tactic hysteresis: switch_tactics_min_steps {old} -> "
                  f"{args.switch_tactics_min_steps} ticks (lower = commits to DEFEND sooner)")
        # DEFEND geometry overrides. own_half(f) = goal_line_v + f*half_length (test-pinned);
        # replicate it here so --defend-line-frac maps to the same V-x the agent uses.
        if args.defend_line_frac is not None or args.defend_width_frac is not None:
            gl = goal_line_v(env_info)
            half_l = env_info["table"]["length"] / 2
            half_w = env_info["table"]["width"] / 2
            dr = agent.agent_params["defend_range"]
            if args.defend_line_frac is not None:
                # keep the line in front of the own-goal keep-out floor, and below the far bound.
                floor = gl + (0.0 if args.no_goal_keepout else args.min_goal_dist)
                new_line = min(max(gl + args.defend_line_frac * half_l, floor), dr[1] - 1e-3)
                print(f"defend line: {dr[0]:+.3f} -> {new_line:+.3f} m V-x "
                      f"(frac {args.defend_line_frac}; larger = intercept earlier/further out)")
                dr[0] = new_line
            if args.defend_width_frac is not None:
                old = agent.agent_params["defend_width"]
                agent.agent_params["defend_width"] = args.defend_width_frac * half_w
                print(f"defend width: {old:+.3f} -> {agent.agent_params['defend_width']:+.3f} m "
                      f"(frac {args.defend_width_frac} of half-width; wider = covers the posts)")
        # Last, so it reads the plant line the run will ACTUALLY use: the two frac overrides
        # above mutate agent_params after the agent was built.
        if args.defend_clear:
            _floor = float(agent.traj_generator.bound_points[:, 0].min())
            _d = float(agent.agent_params["defend_clear_standoff"])
            _m = float(agent.agent_params["defend_clear_margin"])
            _v = float(agent.agent_params["defend_clear_speed"])
            _plant = float(agent.agent_params["defend_range"][0])
            _d_max = _plant - (_floor + _m)
            print(f"clearing block: ON -- DEFEND drives THROUGH the crossing at {_v:.2f} m/s "
                  f"along +x instead\n  of tapping 0.05 m/s sideways. Standoff {_d:.3f} m, "
                  f"never deeper than V-x {_floor + _m:+.3f}\n  (routing floor {_floor:+.3f} + "
                  f"margin {_m:.3f}). Watch the defend_phase column.")
            if _d > _d_max:
                print(f"  NOTE: standoff {_d:.3f} exceeds the {_d_max:.3f} m available at the "
                      f"plant line {_plant:+.3f},\n  so it is clipped there; deeper crossings "
                      f"fall back to the vendored plant.")
            _blk = agent.agent_params.get("defend_block_offset")
            if _blk != "centre":
                print(f"  NOTE: --defend-block {_blk} is ignored while clearing -- the stroke "
                      f"lands ON the crossing,\n  and there is no lateral tap to offset.")
            _park = float(agent.agent_params["x_home"][0])
            if args.defend_early_frac is None and _plant <= _park:
                print(f"  WARNING: the plant line {_plant:+.3f} is BEHIND the park {_park:+.3f}, "
                      f"so a normal defend has\n  NEGATIVE run-up and this will fall back to the "
                      f"vendored plant. Add --defend-early-frac 0.6\n  (or move the park with "
                      f"--home-x) or the flag does almost nothing.")
        # Independent of --defend-clear: meeting a head-on puck earlier helps the plain block too.
        if args.defend_early_frac is not None:
            _gl = goal_line_v(env_info)
            _half_l = env_info["table"]["length"] / 2
            _line = min(_gl + args.defend_early_frac * _half_l,
                        float(agent.agent_params["hit_range"][1]))
            _line = max(_line, float(agent.agent_params["defend_range"][0]))
            _cos = float(agent.agent_params["defend_early_cos"])
            _vel = agent.agent_params.get("rendezvous_vel") or 0.25
            print(f"early defend line: a puck with |vx|/|v| >= {_cos:.2f} is met at V-x "
                  f"{_line:+.3f}\n  instead of the plant line "
                  f"{float(agent.agent_params['defend_range'][0]):+.3f} -- but only when the arm "
                  f"can be there in\n  time at {float(_vel):.2f} m/s. Watch the early_line "
                  f"column: NaN means it never engaged.")

    if args.stage == "cam-probe":
        # Pure camera diagnostic: no arm, no CAN, no calibration required, so it can be run with
        # the rig powered down and answers the one question every other stage depends on.
        stage_cam_probe(args)
        return

    try:
        arm.init()
        if args.stage == "grav":
            stage_grav(arm)
        elif args.stage == "table-height":
            stage_table_height(arm, env_info, args)
        elif args.stage == "reach-probe":
            stage_reach_probe(arm, env_info, args)
        elif args.stage == "set-origin":
            stage_set_origin(arm, env_info, args)
        elif args.stage == "home":
            stage_home(arm, env_info, args)
        elif args.stage == "reach":
            stage_reach(arm, env_info, args)
        elif args.stage == "vision":
            stage_vision(arm, env_info, sensor, args)
        elif args.stage == "track-test":
            stage_track_test(arm, env_info, args)
        elif args.stage == "play":
            stage_play(arm, agent, env_info, sensor, args)
    except HardwareFault as e:
        print(f"\n!! HARDWARE FAULT: {e}\n   catching arm under gravity compensation.")
    finally:
        arm.catch_and_close()
        if sensor is not None:
            sensor.close()
        print("arm handed to gravity compensation; exiting.")


if __name__ == "__main__":
    main()
