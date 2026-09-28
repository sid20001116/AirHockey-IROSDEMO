"""BaselineAgent retargeted to the mini platform.

`baseline/baseline_agent/` is the organizers' agent and the standing comparison point for the
optimal-control stack, so it is never edited (docs/optimal_control_implementation.md Part 3).
The vendored copies under fsm/, planning/ and estimation/ are likewise left untouched; every
mini-platform adaptation lives in this module.

The vendored stack assumes an END-MOUNTED arm: its own frame table-aligned, x running down
the table toward the opponent goal, the rails at |y| = width/2, the base a fixed setback
behind its own goal line. That assumption is load-bearing everywhere -- the EKF's bounce
model, the Bezier table bounds, the tactic ranges, Smash's goal ray. Since 2026-09-10 the mini
arm MATCHES that assumption: it stands on the floor at the middle of its own SHORT end, behind
its own goal, facing down the table (arm 1 at the home end, arm 2 mirrored; assets/arm1.xml),
base frame axis-aligned with the world. Before that it was SIDE-MOUNTED beside a long rail with
its base frame rotated 90 deg to the table; the V-frame machinery below was built for that mount
and is kept deliberately -- it is what made re-mounting the arm a change to two XML poses.

Rather than rewrite the FSM, the whole agent PLANS IN A VIRTUAL END-MOUNT FRAME "V": the
frame the old layout had -- origin V_SETBACK behind the table centre on its own goal side,
x toward the opponent, at the real base's height. In V every vendored assumption holds
verbatim. A single fixed planar transform V<->R (R = the real base frame) is applied at the
four frame crossings, and nowhere else:

    in,  R->V:  `get_puck_pos` / `get_puck_vel`   (observations feeding the EKF)
    in,  R->V:  `SystemState.update_ee_pos_vel`   (FK readback of the mallet)
    out, V->R:  `TrajectoryGenerator.optimize_trajectory`      (Cartesian traj -> IK/QP)
    out, V->R:  `TrajectoryGenerator.solve_anchor_pos[_ik_null]` (hit-config anchors)

plus the EKF's table placement, which reads base_frame -- its tracker is built against the
V frame instead. Both frames share their z origin, so z never transforms.

Feasibility of the whole play region was verified with position-only IK (the optimizer
tracks position and leaves orientation to the null space): 100% of the arm's own half is
reachable and the max centreline reach is 0.73 m, so reach is not the binding constraint --
SPEED is. Worst-direction mallet speed at --max-joint-vel 1.0 runs 0.41-0.42 m/s across the
middle of the own half and collapses to 0.11-0.22 m/s in the band in front of the own goal,
where the arm has to fold up over its own base. Watch that, and `grip_state` tilt/slip in
tools/mini_env_smoke.py -- not IK failures.

Everything else in the baseline is already driven by `env_info`, including the whole Kalman
filter geometry. Its *fitted* constants (`damping`, `b_params`, `OBS_COV` in
`kalman_filter.py`) are still full-size fits -- refitting them is Phase 1 (T1.1-T1.3), and
until then predictions on the mini table are biased.

This exists to satisfy T0.4's acceptance gate and to keep a regression baseline running on
the mini env. It is not the optimal-control agent; that is `oc_agent.py` (T4.1).
"""

import time

import numpy as np
from scipy.spatial.transform import Rotation as _Rot

from mini_platform.core.kinematics import inverse_kinematics, forward_kinematics
from mini_platform.envs.env_base import (AirHockeyBase, GRASP_HEIGHT, HOME_POS,
                                                             INIT_SEED, N_JOINTS,
                                                             TABLE_LENGTH)
from mini_platform.vision.sensor import PuckVisionSensor
from mini_platform.agents.baseline_agent import BaselineAgent
from mini_platform.estimation.kalman_filter import AirHockeyTable, PuckTracker
from mini_platform.planning.bezier_planner import BezierPlanner
from mini_platform.fsm.system_state import SystemState, TACTICS
from mini_platform.fsm.tactics import Defend, Init, Prepare, Ready, Repel, Smash
from mini_platform.planning.trajectory_generator import TrajectoryGenerator

# Scale factor between the two tables, for the quantities that genuinely do scale with it
# (speeds, amplitudes).
TABLE_SCALE = 0.505226

# Setback of the VIRTUAL end-mount frame behind the table centre: the base position the
# vendored end-mounted layout had, kept as the planning frame's origin. 0.113 m behind the goal
# line, like the original scene. This is a property of the planning frame, NOT of the physical
# base, so it is a constant here rather than a base_frame lookup -- which is exactly what let the
# arm be re-mounted without touching any of this. Since 2026-09-10 the SIM base stands exactly
# here, so R <-> V is the identity in sim; on hardware it is not, because base_frame is measured.
V_SETBACK = 0.62094

#: Down-table inset (m) of the real own-goal line from the model's table extreme (length/2).
#: Measured on hardware: with this at 0, dist-to-goal reads a constant +0.07 m everywhere, so
#: the goal line the arm actually plays to sits at world x = -(length/2 - GOAL_LINE_INSET)
#: = -0.438, not -0.508. Applied wherever the own goal line is derived, so the reach-probe
#: readout and the play loop's ranges stay consistent.
GOAL_LINE_INSET = 0.07

#: The DEFAULT ("base") park, as ROBOT-BASE-frame xy (m) -- captured by teleoping the arm to the
#: spot and reading its own FK (``python -m mini_platform.hw.capture_home``), which is why it is
#: stored in the frame the arm measures in: the park then stays at the physical spot it was
#: taught even if ``base_frame`` is later recalibrated. Only its table coordinates are derived.
#: Moves only where the mallet idles between plays; the goal park, the strike ranges and the
#: own-goal keep-out are all untouched.
#:
#: Taught 2026-09-14 at the END mount: 0.245 m out from the base, essentially on its centreline,
#: held to 2.05 deg off vertical. That lands on the defend line, so a shot straight down the
#: middle needs no travel, and it is just clear of the fold-over zone.
#:
#: Re-teach after ANY re-mount. The 2026-09-10 capture (0.3200, 0.0025) was taken at the SIDE
#: mount, where those robot-frame numbers put the mallet at world (-0.474, +0.043), 34 mm in
#: front of its own goal line; from the end mount the identical numbers land at world
#: (-0.301, 0.000), 207 mm UP-TABLE of the goal line and past the defend line -- the arm would
#: idle up-table of where it defends. Do not park much deeper than this one without measuring:
#: worst-direction mallet speed falls from 0.33 m/s here to 0.11-0.22 m/s between here and the
#: goal mouth, where the arm folds up over its own base.
HOME_PARK_XY = np.array([0.2143, -0.0050])   # taught 2026-09-14, held to 2.05 deg off vertical

# Aim calibration for the rigid grip, rad. Even with the vertical hit anchor the wrist
# carries a few degrees of residual tilt at contact (it leans toward the base under the
# strike's dynamics), and a tilted head rotates the outgoing puck ~-1.5 deg per deg of
# tilt. Measured over 10 planned smashes (20 mini-hit episodes, sim ground truth): median
# outgoing-vs-commanded error -6.5 deg, so commands are pre-rotated to cancel it. In the
# agent's own V frame the lean, and therefore the sign, is the same for both SIDE mounts by the
# double scene's diagonal symmetry. Re-measure after any change to the grip, the mallet,
# or the strike speed (tools note: the measurement script lives in the T0.4 report).
#
# STALE at the end mount, and NOT re-measured. The wrist leans TOWARD THE BASE, and the base
# direction in V moved from "the +y rail" to "straight back down -x" when the arm was re-mounted
# on 2026-09-10 -- so both the sign and the size of this are unverified. Run hardware with
# --flat-mallet until it is re-measured: that holds the wrist vertical through the swing, makes
# the contact normal square, and selects AIM_BIAS_FLAT = 0 instead of this constant.
AIM_BIAS_COMP = np.radians(6.5)

# ... and it is a compensation for a TILT, so it belongs to the tilted arm and not to this one.
# With `--flat-mallet` the wrist is held vertical through the swing (SMASH tilt median 42 -> 0.2
# deg), the contact normal is square, and the pre-rotation above has nothing left to cancel --
# it becomes the error. Measured the same way as the constant it replaces (sim ground truth,
# outgoing vs commanded direction over 28 seeded episodes):
#
#   default,      6.5 deg comp   median  +0.65 deg   (n=15)   <- calibrated, as intended
#   --flat-mallet 6.5 deg comp   median  +7.90 deg   (n= 6)   <- now over-rotating by the comp
#   --flat-mallet no comp        median  +0.37 deg   (n= 5)   <- square contact needs none
#
# n is small (a seeded episode yields ~0.2 committed smashes), but the effect is ~8 sigma on the
# mean and matches the pre-rotation almost exactly, which is what it would be if the underlying
# bias were simply gone. Re-measure if the grip, the mallet or the strike speed changes.
AIM_BIAS_FLAT = 0.0


class PlanarXform:
    """A z-aligned rotation + xy translation between two frames sharing their z origin.

    Built from two world poses (4x4), maps FROM frame `src` TO frame `dst`. Both bases on
    this rig are upright (rotations about z only) and the virtual frame is defined at the
    real base's height, so the general SE(3) relation reduces to this planar form and z
    passes through untouched everywhere.
    """

    def __init__(self, T_w_dst, T_w_src):
        R_rel = T_w_dst[:3, :3].T @ T_w_src[:3, :3]
        t_rel = T_w_dst[:3, :3].T @ (T_w_src[:3, 3] - T_w_dst[:3, 3])
        assert abs(R_rel[2, 2] - 1) < 1e-9 and abs(t_rel[2]) < 1e-9, \
            "frames must share z axis and z origin"
        self.theta = np.arctan2(R_rel[1, 0], R_rel[0, 0])
        self.rot2 = R_rel[:2, :2].copy()
        self.t2 = t_rel[:2].copy()

    def pos2(self, p):
        return self.rot2 @ np.asarray(p)[:2] + self.t2

    def vec2(self, v):
        return self.rot2 @ np.asarray(v)[:2]

    def pos3(self, p):
        p = np.asarray(p)
        return np.concatenate([self.pos2(p), p[2:3]])

    def vec3(self, v):
        v = np.asarray(v)
        return np.concatenate([self.vec2(v), v[2:3]])

    def angle(self, a):
        return (a + self.theta + np.pi) % (2 * np.pi) - np.pi


def project_out_of_base(xy_v, v2r, r2v, radius):
    """Push a V-frame mallet target radially out of a keep-out disc around the arm's own base.

    The hazard the own-goal x floor was aiming at is the wrist folding back OVER ITS OWN BASE and
    shoving the near rim, which tips the table. At the SIDE mount that was a DISC around the base,
    not a half-plane in x -- and the half-plane guarded the wrong axis in both directions: it
    PERMITTED V (0.2506, +0.2202), 0.198 m from the base and the worst fold-over point in the
    whole defend band, while FORBIDDING the goal-mouth centre at 0.364 m, which was both the
    safest and the fastest place on that table for that arm.

    LEAVE THIS OFF AT THE END MOUNT (2026-09-10 onward). The base now stands behind its own goal,
    on the table's own axis, so the hazard is a half-plane in x again and `min_goal_dist` is the
    flag that guards it. A disc centred on the base now forbids the goal end of the table rather
    than the wedge beside it: at the side-mount value 0.26 it would forbid the own goal line
    (0.113 m of reach), the ready-guard point (0.213) AND the defend plant line (0.251) -- i.e.
    the entire defensive band. If you want one here, measure it with `--stage reach-probe` and
    expect a number well under 0.15.

    In frame R the base IS the origin, so the keep-out is a plain minimum-reach clamp. A falsy
    radius returns the target untouched.
    """
    if not radius:
        return xy_v
    p_r = v2r.pos2(xy_v)
    n = float(np.linalg.norm(p_r))
    if n >= radius:
        return xy_v
    return np.asarray(r2v.pos2(p_r * (radius / max(n, 1e-9))), dtype=float)


def virtual_base_frame(agent_id, z):
    """World pose of agent `agent_id`'s virtual end-mount frame, at base height `z`."""
    T = np.eye(4)
    if agent_id == 1:
        T[:3, 3] = [-V_SETBACK, 0., z]
    else:
        T[:3, :3] = np.diag([-1., -1., 1.])
        T[:3, 3] = [V_SETBACK, 0., z]
    return T


class MiniAirHockeyTable(AirHockeyTable):
    """AirHockeyTable whose OWN goal is a hole in the near rim rather than a wall.

    `check_collision` decides "scored" with ``state[0] + u[0] < -self.boundary[0, 0, 0]``. That
    threshold is NEGATIVE (here -0.13794), and a V-frame x is always >= 0.113, so it can never
    fire: only the OPPONENT goal (the `> boundary[0, 1, 0]` half) is ever detected. A puck headed
    into our own goal therefore falls through to `_check_collision_impl` and is REFLECTED off rim
    3 -- the near end rail -- straight back up-table.

    That phantom rebound is what makes the arm swing forward instead of at the puck. Measured on
    the real tracker, a puck at V (0.20, 0.00) closing at 1.0 m/s predicts one second out to
    V-x 0.642 travelling at +0.515 m/s -- past the table centre, moving away. The mallet is
    correctly chasing a target that does not exist.

    Fix: while the puck is inside the goal MOUTH, drop rim 3 from the candidate set so the puck
    flies through. Rims 0-2 keep their indices, so `local_rim_transform[collide_rim_idx]` stays
    aligned; `boundary` is restored in a `finally` because `check_collision` reads it again after
    this returns.

    Deliberately NOT reported as `score=True`: that path sets ``F = np.eye`` (`SystemModel.f`),
    which freezes position AND velocity, so `get_prediction` would never hit its stop line and
    would report the puck arriving a full horizon from now. A hole is the honest model -- the
    caller's stop line and the workspace clamp bound what comes out.

    This matters beyond `get_prediction`: `PuckTracker.predict` runs the collision model on every
    COAST tick, and at 11-22 Hz vision against the 50 Hz loop most ticks are coasts, so the
    phantom corrupts the filter's own state too, not just the lookahead.
    """

    def _check_collision_impl(self, state, u):
        if abs(state[1]) < self.goal_width / 2 and state[0] + u[0] < self.boundary[0, 0, 0]:
            saved = self.boundary
            self.boundary = saved[:3]
            try:
                return super()._check_collision_impl(state, u)
            finally:
                self.boundary = saved
        return super()._check_collision_impl(state, u)


class MiniSystemState(SystemState):
    """SystemState living in the virtual frame V.

    Two of the state's inputs arrive in other frames and are converted here:
      * the puck tracker's table geometry comes from base_frame, which now describes the
        side mount -- the tracker is rebuilt against the virtual frames, where the vendored
        bounce model is valid;
      * `update_ee_pos_vel` reads FK, which is in the real base frame R.
    """

    def __init__(self, env_info, agent_id, agent_params, r2v,
                 intercept_max_jump=None, intercept_ema_alpha=None, max_puck_speed=None,
                 lead_time=0.0, vel_lpf_hz=None, ekf_params=None, workspace_clamp=False,
                 meas_lag_comp=False, meas_age_lead=False):
        super().__init__(env_info, agent_id, agent_params)
        self._r2v = r2v
        # MINI: latency lead (s). The command the agent issues this tick does not reach the arm
        # until ~t_lead later (camera capture age ~1 frame + draw_action compute + RT-stream
        # buffering), during which the puck keeps moving -- so the arm chases a stale puck and
        # arrives late, worst when the puck is fast/far. We aim t_lead further along the predicted
        # puck trajectory to compensate. 0.0 (default) => exact vendored behaviour (sim/tests
        # byte-identical). Applied to predicted_state[:2] only -- predicted_time and the tactic
        # gating are untouched, so this is a pure target shift, not a horizon/plan change.
        self._lead_time = float(lead_time)
        # MINI: optional output-side rate-limit on the predicted intercept (predicted_state[:2]).
        # get_prediction forward-integrates the EKF velocity, so a noisy / sign-flipping velocity
        # estimate makes the intercept teleport up to ~1.3 m/tick (measured on hardware); that jump
        # IS the planner target, so it renders as joint jerk. Clamp the per-tick L2 step (and/or EMA
        # toward last tick) against a frozen per-tick anchor. Both None (the sim default) => the
        # exact vendored behaviour, so sim and the EKF/GT-regression tests stay byte-identical.
        self._intercept_max_jump = intercept_max_jump
        self._intercept_ema_alpha = intercept_ema_alpha
        self._intercept_last = None     # last tick's final intercept xy (anchor source)
        self._intercept_anchor = None   # anchor frozen for the current tick
        #: MINI: run the absolute workspace clamp in `intercept_at` even when none of the
        #: intercept knobs above are on. The clamp used to sit BEHIND the knobs-off early
        #: return, so the plainest hardware command line -- no --lead-time, no
        #: --intercept-max-jump, no --rendezvous-vel -- was the one that got no clamp at all,
        #: and an unreachable prediction (V-x 0.78 measured, against a 0.51 m hit range) went
        #: to the planner unbounded. False (the sim default) keeps GT/sim byte-identical;
        #: run_hw_match passes True.
        self._workspace_clamp = bool(workspace_clamp)
        #: MINI: un-led copy of predicted_state for the tactic gates (see update_prediction).
        self.predicted_gate_state = np.zeros(6)
        #: MINI: approach counter re-derived against the ABORT line rather than the plant line
        #: (see is_puck_approaching). Only read when `defend_abort_line` is set.
        self._approach_count = 0
        #: MINI: consecutive ticks the puck has been SLOW -- below `smash_slow_vel`, which sits
        #: ABOVE `static_vel_threshold`. Hardware runs that threshold at 0.02 m/s, so a puck
        #: creeping at 0.05-0.15 m/s is never `is_puck_static()` and no tactic claims it. Counted
        #: only when `smash_slow` is on; see MiniTacticMixin.can_smash.
        self._slow_count = 0
        #: MINI (Fix 8c): age (s) of the measurement about to be fused, parked here by
        #: `MiniBaselineAgent.get_puck_measurement` on its way past. The vendored
        #: `BaselineAgent.draw_action` calls `get_puck_measurement` and then
        #: `update_observation` on consecutive lines with no channel between them, and the age
        #: belongs to the measurement, not to the tick -- so rather than fork that method, the
        #: sensor's answer is left here and collected one line later. Cleared on use.
        self._pending_meas_age = None
        #: MINI (Fix 8c): lead the intercept by the MEASURED measurement age. The EKF fuses a
        #: frame that is a camera period + detection + hand-off old, so its estimate tracks where
        #: the puck WAS -- measured, a steady 72 mm behind a 1.2 m/s puck. Leading the target by
        #: that age recovers the miss without touching the filter, and it goes through the same
        #: bounce-aware prediction the tactics gate on (unlike `meas_lag_comp`, which compares
        #: against a historical state that can sit on the wrong side of a bounce).
        #: This is `--lead-time`, except the arm measures the delay instead of being told it.
        self._meas_age_lead = bool(meas_age_lead)
        #: Smoothed measurement age (s). Per-frame ages are noisy -- a dropped frame doubles one
        #: -- and the lead has to be steady or it is just another source of intercept jitter.
        self._age_ema = 0.0

        v_robot = dict(env_info['robot'])
        v_robot['base_frame'] = [
            virtual_base_frame(i + 1, env_info['robot']['base_frame'][i][2, 3])
            for i in range(len(env_info['robot']['base_frame']))]
        v_env_info = dict(env_info)
        v_env_info['robot'] = v_robot
        # MINI: `ekf_params` swaps the vendored KUKA puck-motion fits for ones measured on THIS
        # table (tools/fit_mini_table.py). None -> the vendored constants, byte-identically.
        self.puck_tracker = PuckTracker(v_env_info, agent_id, max_puck_speed=max_puck_speed,
                                        vel_lpf_hz=vel_lpf_hz, ekf_params=ekf_params,
                                        meas_lag_comp=meas_lag_comp)
        # MINI: open our own goal mouth in the bounce model -- see MiniAirHockeyTable. Off (the
        # sim/vendored default) leaves the tracker object-for-object as it was built above.
        if agent_params.get('ekf_own_goal'):
            _sys = self.puck_tracker.system
            _sys.table = MiniAirHockeyTable(
                v_env_info['table']['length'], v_env_info['table']['width'],
                v_env_info['table']['goal_width'], v_env_info['puck']['radius'],
                abs(v_env_info['robot']['base_frame'][agent_id - 1][0, 3]),
                v_env_info['dt'], ekf_params=_sys.params)

    def update_ee_pos_vel(self, joint_pos, joint_vel):
        x_ee, v_ee = super().update_ee_pos_vel(joint_pos, joint_vel)
        return self._r2v.pos3(x_ee), self._r2v.vec3(v_ee)

    # MINI: freeze the intercept clamp anchor once per tick. update_prediction runs >1x per tick
    # (should_defend / can_smash gating + Defend._update_tactic_impl), and every call must
    # rate-limit against the SAME anchor (last tick's final intercept) so they do not compound.
    # A restart clears the anchor so the first acquisition is never clamped.
    def update_observation(self, joint_pos_cur, joint_vel_cur, puck_state, meas_R=None,
                           meas_age=None):
        # MINI (Fix 8c): the measurement's age, as `get_puck_measurement` parked it on the way
        # past. Only acted on when the tracker was built with `meas_lag_comp`; otherwise it is
        # carried, logged and ignored. Cleared on use, so a tick whose measurement has no stamp
        # -- GT, a dropped frame -- cannot inherit the previous tick's age.
        if meas_age is None:
            meas_age = self._pending_meas_age
        self._pending_meas_age = None
        if meas_age is not None and self._meas_age_lead:
            # EMA, and capped: a stall can hand back an age of seconds, and leading a target by
            # seconds would throw the mallet across the table on one bad frame.
            self._age_ema = (1.0 - self.AGE_EMA_ALPHA) * self._age_ema \
                + self.AGE_EMA_ALPHA * min(float(meas_age), self.MAX_AGE_LEAD)
        was_restart = self.restart
        super().update_observation(joint_pos_cur, joint_vel_cur, puck_state, meas_R, meas_age)
        if was_restart:
            self._intercept_last = None
            self._approach_count = 0
            self._slow_count = 0
        self._intercept_anchor = self._intercept_last
        # MINI: the vendored counter (system_state.py) only ever increments while the puck is
        # up-table of defend_range[0] -- the PLANT line. A puck inside it can therefore never
        # re-latch "approaching", which is half of why nothing defends the last 138 mm. Re-derive
        # the same three-way test against the abort line instead. Unset -> vendored counter only.
        line = self.agent_params.get('defend_abort_line')
        if line is not None:
            v = self.puck_tracker.state[2:4]
            speed = float(np.linalg.norm(v))
            if speed < self.agent_params['static_vel_threshold'] or \
                    abs(v[0]) / max(speed, 1e-12) < 0.15:
                self._approach_count = 0
            elif v[0] < 0 and self.puck_tracker.state[0] > line:
                self._approach_count += 1
            elif v[0] > 0:
                # MINI: the puck is RECEDING, so the approach is over. Without this arm a head-on
                # puck matches NEITHER branch above once it turns around -- |vx|/speed ~ 1 so it
                # is never "transversal", speed > static_vel_threshold so it is never "static",
                # and vx > 0 so it never increments -- and `_approach_count` keeps its last
                # positive value indefinitely. `is_puck_approaching()` then stays True, which
                # pins `can_smash` at False for the rest of the run. An angled puck hid this: it
                # routinely dips under |vx|/speed < 0.15 on a rim bounce or a deflection and
                # takes the transversal reset. A successful clearing block MANUFACTURES the
                # jamming state (head-on, receding, above the static threshold), so this has to
                # land with it. Deliberately NOT reset for a puck still closing below the abort
                # line -- refusing that one is `_in_zone`'s job, not this counter's.
                self._approach_count = 0
        # MINI: dwell counter for the slow-puck smash. It cannot reuse `puck_static_count`, which
        # resets above `static_vel_threshold` -- 0.02 m/s on hardware, i.e. the very band this
        # exists to cover. Unset -> never counted, and `is_puck_slow` is always False.
        if self.agent_params.get('smash_slow'):
            if float(np.linalg.norm(self.puck_tracker.state[2:4])) < \
                    self.agent_params['smash_slow_vel']:
                self._slow_count += 1
            else:
                self._slow_count = 0

    def is_puck_approaching(self):
        if self.agent_params.get('defend_abort_line') is None:
            return super().is_puck_approaching()
        return self._approach_count > 0

    def is_puck_slow(self):
        """Slow -- not necessarily static -- and settled there for the dwell.

        The dwell separates "the puck has come to rest in front of us" from "the puck is
        momentarily slow mid-bounce", and it rides out the ~300 ms of wrong-sign velocity the EKF
        emits after a re-lock, which would otherwise read as a free hit.
        """
        if not self.agent_params.get('smash_slow'):
            return False
        return self._slow_count >= int(self.agent_params['smash_slow_ticks'])

    def mallet_ref_xy(self):
        """Where the mallet is taken to BE, for any "can I reach that in time" question.

        `x_cmd` is where the planner last told it to go, and on hardware that runs AHEAD of the
        arm: the RT streamer caps every step, so the command leads the measured mallet by a DEFEND
        median of 16.5 mm and a p99 of 138 mm, always in the direction of travel. Judging reach
        from it therefore flatters every candidate by up to ~14 cm, worst exactly when the arm is
        moving hardest. Only a measured DISTANCE is ever taken from `x_actual`, never a commanded
        position, so unlike an encoder resync this cannot accumulate gravity droop.

        `np.any` guards the pre-first-measurement state: `x_actual` is a literal zeros(3) until
        `update_observation` has run once, and that is the V-frame ORIGIN, not a mallet position
        -- measuring reach from there would accept anything.
        """
        if self.agent_params.get('rendezvous_from_measured') and np.any(self.x_actual):
            return np.asarray(self.x_actual[:2], dtype=float)
        return np.asarray(self.x_cmd[:2], dtype=float)

    def _rendezvous_rollout(self, horizon, stop_line, vel):
        """`get_prediction`, but stopping at the SOONEST intercept the mallet could reach.

        The vendored question is "where is the puck `horizon` from now", and Defend.ready() sets
        that horizon to max_prediction_time (1.0 s). For a puck already deep in our own half the
        honest answer is "somewhere up-table", so the mallet is planted up-table and the puck
        rolls past underneath it -- the arm swings FORWARD instead of at the puck. (Before
        MiniAirHockeyTable the answer was fiction as well: a phantom rebound off the near rim.)

        Ask instead for the smallest T with ``||P(T) - mallet_now|| <= vel * T`` -- the first
        moment the mallet could plausibly be where the puck will be. A near, slow puck then
        collapses to "hit it roughly where it is"; a fast far puck still runs out to the line
        crossing exactly as before, because no earlier T is reachable.

        `vel` is a deliberately conservative Cartesian budget: measured on this arm at
        --max-joint-vel 1.0 the guaranteed (worst-direction) mallet speed is 0.39-0.41 m/s across
        the whole deep zone, and only 0.12 m/s in the wedge beside the base -- which is a place we
        never want to promise we can reach.

        `rendezvous_from_measured` decides WHERE the mallet is taken to be, and it matters more
        than it looks. `x_cmd` is where the planner last told the mallet to go, and on hardware
        that runs AHEAD of the arm: the RT streamer caps every step, so the command leads the
        measured mallet by a DEFEND median of 16.5 mm and a p99 of 138 mm (`analyze_telem
        --controller`, goalie.npz). Asking "can I get there in time" from the commanded point
        therefore flatters every candidate intercept by up to ~14 cm, always in the direction of
        travel, and worst exactly when the arm is moving hardest -- so the rendezvous accepts
        intercepts the mallet cannot actually reach. `x_actual` is the FK of the joint positions
        handed to `update_observation` on this same tick, already in V. Note this is only a
        MEASURED DISTANCE, never a commanded position, so unlike an encoder-based resync it
        cannot accumulate gravity droop.

        The loop mirrors `PuckTracker.get_prediction` step for step (same stop-line break, same
        static-puck early exit, same accumulation of `predict_time`), so with `vel <= 0` it
        returns bit-for-bit what the vendored call would have.
        """
        tracker = self.puck_tracker
        dt = tracker.system.dt
        t_min = float(self.agent_params['max_plan_steps']) * float(self.env_info['dt'])
        mallet_xy = self.mallet_ref_xy()
        state_current = tracker.state.copy()
        P_current = tracker.P.copy()
        predict_time = 0.
        best = None                    # least-infeasible point, for when nothing is reachable
        for _ in range(round(horizon / dt)):
            state_next, P_next = tracker.predict(state_current, P_current)
            if state_next[0] < stop_line:
                break
            if np.linalg.norm(state_current[2:4]) < 1e-2 and np.linalg.norm(state_next[2:4]) < 1e-2:
                predict_time = horizon
                break
            predict_time += dt
            state_current = state_next
            P_current = P_next
            if vel > 0. and predict_time >= t_min:
                slack = float(np.linalg.norm(state_current[:2] - mallet_xy)) - vel * predict_time
                if slack <= 0.:
                    best = None
                    break
                if best is None or slack < best[0]:
                    best = (slack, state_current.copy(), P_current.copy(), predict_time)
        if best is not None:
            # Nothing on the predicted path is reachable in time -- a puck running down the
            # base-side rail, say, which rebounds and recedes faster than the mallet closes.
            # The vendored fallback is the position at the END of the horizon, which after a rim
            # rebound is up-table and then gets pinned to hit_range[1] by the workspace clamp:
            # the arm lunges at the far rail while the puck is beside it. Aim at the CLOSEST
            # approach instead -- the best attempt available, and it at least moves the mallet
            # the right way for the next replan.
            _, state_current, P_current, predict_time = best
        # Floor the PLAN duration at one plan block. `should_defend`'s fourth condition refuses
        # any crossing less than max_plan_steps*dt away, so a fast puck close to the line -- the
        # exact case a goalie exists for -- gets no defend at all and the mallet goes home
        # instead. The target stays the true crossing point; only the time the planner is given
        # to get there is floored, so the mallet commits and is merely late rather than absent.
        return state_current, P_current, max(predict_time, t_min)

    #: Smoothing on the measured age, and the ceiling on how far it may lead the target (s).
    #: 0.2 s is already ten camera frames; an age past it is a stall, not a latency.
    AGE_EMA_ALPHA = 0.3
    MAX_AGE_LEAD = 0.2

    @property
    def _effective_lead(self):
        """Seconds to aim ahead: the operator's `--lead-time` plus the measured frame age."""
        return self._lead_time + (self._age_ema if self._meas_age_lead else 0.0)

    def _knobs_off(self):
        return (self._intercept_max_jump is None and self._intercept_ema_alpha is None
                and not self._lead_time and not self._meas_age_lead
                and not self.agent_params.get('rendezvous_vel'))

    def intercept_at(self, prediction_time, stop_line=0.):
        """The intercept `update_prediction` would produce, WITHOUT mutating anything.

        Returns ``(state6, cov, time, gate_state6)``. Exists so the VPSTO rendezvous can iterate
        candidate arrival times through the SAME lead + clamp pipeline the tactic gates read --
        it used to call `tracker.get_prediction` directly, so under `--planner vpsto` the gate
        and the executed target were two different intercepts.
        """
        rv = self.agent_params.get('rendezvous_vel')
        if rv:
            st, cov, t = self._rendezvous_rollout(prediction_time, stop_line, float(rv))
        else:
            st, cov, t = self.puck_tracker.get_prediction(prediction_time, stop_line)
        st = np.array(st, dtype=float)
        #: MINI: the intercept the TACTIC GATES read, before any latency lead. `should_defend`
        #: compares the intercept x against the plant/abort line; the lead shifts it t_lead
        #: further DOWN-track (vx < 0 for an approaching puck) while the un-led x is already
        #: pinned AT that line by the stop line, so any lead beyond ~20 ms puts the gated x below
        #: the line and DEFEND never fires at all. Gate on this; aim at `predicted_state`.
        #: Identical to `predicted_state` whenever lead_time == 0. Carries the same absolute
        #: workspace clamp the target gets below, so the two are directly comparable.
        gate = st.copy()
        _hl = 0.5 * float(self.env_info['table']['length'])
        _hw = 0.5 * float(self.env_info['table']['width'])
        gate[0] = min(max(float(gate[0]), V_SETBACK - _hl),
                      float(self.agent_params['hit_range'][1]))
        gate[1] = min(max(float(gate[1]), -_hw), _hw)
        # Latency lead (MINI): shift the target t_lead further along the predicted puck
        # trajectory, using the velocity AT the predicted point. Recomputed from the fresh
        # predicted_state on every call, so the >1x-per-tick gating calls do not compound it.
        # A static puck (velocity ~0) gets ~0 lead, as intended.
        # ...plus, under `meas_age_lead`, the age the CAMERA reported for the frame this estimate
        # rests on: the filter tracks where the puck was when that frame was captured, so the
        # same extrapolation that covers command latency covers sensor latency too.
        lead = self._effective_lead
        if lead:
            st[:2] = st[:2] + st[2:4] * lead
        # MINI: the rendezvous counts as "enabled" here. It is an extrapolation like the lead
        # is, so it must reach the absolute workspace clamp below; without this the clamp is
        # skipped on exactly the command lines that need it most (no --lead-time, no
        # --intercept-max-jump), and an unreachable puck's fallback target -- V-x 0.70 on a
        # 0.51 m reach, measured -- is handed to the planner unbounded.
        knobs_off = self._knobs_off()
        if knobs_off and not self._workspace_clamp:
            return st, cov, t, gate                 # all disabled -> exact vendored prediction
        anchor = None if knobs_off else self._intercept_anchor
        if anchor is not None:
            xy = st[:2]
            if self._intercept_ema_alpha is not None:
                xy = anchor + self._intercept_ema_alpha * (xy - anchor)
            if self._intercept_max_jump is not None:
                d = xy - anchor
                n = float(np.linalg.norm(d))
                if n > self._intercept_max_jump:
                    xy = anchor + d * (self._intercept_max_jump / n)
            st[:2] = xy
        # Absolute workspace clamp (SAFETY): keep the intercept inside the arm's reachable play
        # region. A velocity-spike extrapolation otherwise sends it off-table (observed on hardware:
        # V-x ran to ~+1.0, past the far rim), and the arm slews out to reach it, ramping into its
        # joint limits -> over-current fault -> collapse. hit_range[1] is the intended forward reach;
        # the own-goal line is the back bound; y is the table half-width.
        hl = 0.5 * float(self.env_info['table']['length'])
        hw = 0.5 * float(self.env_info['table']['width'])
        st[0] = min(max(float(st[0]), V_SETBACK - hl),
                    float(self.agent_params['hit_range'][1]))
        st[1] = min(max(float(st[1]), -hw), hw)
        return st, cov, t, gate

    def update_prediction(self, prediction_time, stop_line=0.):
        self.predicted_state, self.predicted_cov, self.predicted_time, \
            self.predicted_gate_state = self.intercept_at(prediction_time, stop_line)
        if not self._knobs_off():
            self._intercept_last = self.predicted_state[:2].copy()


class MiniTrajectoryGenerator(TrajectoryGenerator):
    """TrajectoryGenerator planning in V, solving IK in R.

    The Bezier bounds are the old end-mount bounds (V frame, so the vendored corners work
    with V_SETBACK), and every Cartesian quantity headed for the optimizer -- which runs
    the robot model, i.e. frame R -- is converted on the way through.
    """

    #: Re-solve the VPSTO strike only when the predicted intercept has moved more than this (m)
    #: AND at least max_plan_steps ticks have passed -- a per-COMMIT (~10 Hz) cadence. Between
    #: solves every tick just resamples the cached solution (~1.4 ms), and the RT streamer glides
    #: along the ~200 ms lookahead across a solve, so a ~30-70 ms re-solve never stalls the arm.
    VP_DRIFT = 0.03

    #: Reuse the cached vertical-wrist anchor when the strike point has moved less than this (m).
    #: The anchor search is the expensive half of a re-solve -- up to ~35 ms by its own time
    #: budgets (6 yaws x 2 seeds x 8 ms), against ~5-12 ms for the CMA path solve -- and a
    #: centimetre of target motion does not change which IK branch reaches it. Well inside
    #: VP_DRIFT, so a re-solve triggered by drift alone still re-anchors.
    VP_ANCHOR_REUSE = 0.01

    def __init__(self, env_info, agent_params, state, v2r, min_x_v=None, use_vpsto=False,
                 vpsto_kwargs=None, keep_mallet_flat=False, r2v=None):
        self._v2r = v2r
        #: Inverse of `_v2r`, needed to hand a base-keep-out projection back in V. Optional so
        #: the generator can still be built standalone in tests that never use the keep-out.
        self._r2v = r2v
        # Own-goal keep-out floor (V-frame x), or None. Set before super().__init__ because it
        # calls _init_bezier_planner, which reads it. See MiniBaselineAgent's `min_goal_dist`.
        self._min_x_v = min_x_v
        super().__init__(env_info, agent_params, state, keep_mallet_flat=keep_mallet_flat)
        # VPSTO strike planner (opt-in, --planner vpsto). Lazy import so the default bezier path
        # never imports vpsto/cma (keeps sim/tests dependency-free). The cache carries the current
        # solution + a tick index; the ~30 ms CMA solve runs per-COMMIT, every tick resamples it.
        self.use_vpsto = bool(use_vpsto)
        self._vp = None
        self._vp_sol = None                 # cached VPSTO x-y path (CMA solved per-commit)
        self._vp_q_anchor = None            # vertical-wrist anchor for the QP redundancy
        self._vp_tick = 0                   # ticks since the cached path was solved
        self._vp_P_v = None                 # predicted intercept (V) the cached path targets
        self._vp_is_defend = None           # which TACTIC solved it (a block and a smash are
        #                                     opposite paths to the same point -- see reset_vpsto)
        self._vp_xT = None                  # strike point the cached anchor was solved for
        self._vp_anchor_vertical = False    # was that anchor a true vertical-wrist solution?
        #: Cumulative VPSTO strike outcomes, for the telemetry (run_hw_match step 0c). Falling
        #: back to bezier is the designed safety net, not an error -- but the RATE is what says
        #: whether `--planner vpsto` is planning the strikes or quietly sitting most of them out,
        #: and without a counter that is guesswork from the outside.
        self.vp_solved = 0
        self.vp_fallback = 0
        self._eeh = float(env_info['robot']['ee_desired_height'])
        if self.use_vpsto:
            from mini_platform.planning.vpsto_striker import VpstoStriker
            self._vp = VpstoStriker(env_info, **(vpsto_kwargs or {}))
            # ee_constr play box in the ROBOT frame (mirrors core/constraints.EndEffectorConstraint).
            # A VPSTO plan whose QP output leaves this box is rejected -> bezier fallback, so
            # --planner vpsto can never add an ee_constr violation the bezier path would not.
            T = np.asarray(env_info['robot']['base_frame'][0], float)
            hl = env_info['table']['length'] / 2
            hw2 = env_info['table']['width'] / 2
            corners = np.array([[sx * hl, sy * hw2, 0.] for sx in (-1, 1) for sy in (-1, 1)])
            corners_r = (corners - T[:3, 3]) @ T[:3, :3]
            r_m = env_info['mallet']['radius']
            self._box_xlb = corners_r[:, 0].min() + r_m
            self._box_ylb = corners_r[:, 1].min() + r_m
            self._box_yub = corners_r[:, 1].max() - r_m
            self._box_ztol = 0.02

    def _init_bezier_planner(self):
        half_length = self.env_info['table']['length'] / 2
        half_width = self.env_info['table']['width'] / 2
        r_m = self.env_info['mallet']['radius']

        x_near = -(half_length - r_m) + V_SETBACK
        if self._min_x_v is not None:
            x_near = max(x_near, self._min_x_v)   # never route the mallet behind the keep-out
        x_far = -0.1 * TABLE_SCALE + V_SETBACK
        y = half_width - r_m

        self.bound_points = np.array([[x_near, -y], [x_near, y], [x_far, y], [x_far, -y]])
        self.boundary_idx = np.array([[0, 1], [1, 2], [0, 3]])

        table_bounds = np.array([[self.bound_points[0], self.bound_points[1]],
                                 [self.bound_points[1], self.bound_points[2]],
                                 [self.bound_points[2], self.bound_points[3]],
                                 [self.bound_points[3], self.bound_points[0]]])
        return BezierPlanner(table_bounds, self.dt)

    def project_out_of_base(self, xy_v):
        """Base keep-out for a single mallet target; see the module-level function."""
        return project_out_of_base(xy_v, self._v2r, self._r2v,
                                   self.agent_params.get('base_keepout_radius'))

    def optimize_trajectory(self, cart_traj, q_start, dq_start, q_anchor):
        cart_traj = np.asarray(cart_traj)
        # Own-goal floor for the clearing block's cock, enforced on the WAYPOINTS and in frame V,
        # where the floor is an x half-plane (the base keep-out below is a disc, so that one is
        # done in R). Same argument as that clamp: the Bezier `boundary` only shapes control
        # points, it never clamps the curve -- and a terminal velocity commanded with
        # p_start == p_stop provably dips the curve 0.0667*v_hit BACKWARD, i.e. straight at our
        # own goal, 40 mm at 0.6 m/s. A no-op for every trajectory that already routes legally.
        if self.agent_params.get('defend_clear'):
            floor_v = float(self.bound_points[:, 0].min())
            below = cart_traj[:, 0] < floor_v
            if below.any():
                cart_traj = cart_traj.copy()
                cart_traj[below, 0] = floor_v
                # Strip the inward (-x) component of vel/accel: asking the QP to track motion
                # into a position it will never be given is how a plan turns into tracking error.
                for col in (3, 6):
                    cart_traj[below, col] = np.maximum(cart_traj[below, col], 0.0)
        cart_traj_r = np.empty_like(cart_traj)
        cart_traj_r[:, 0:2] = cart_traj[:, 0:2] @ self._v2r.rot2.T + self._v2r.t2
        cart_traj_r[:, 3:5] = cart_traj[:, 3:5] @ self._v2r.rot2.T
        cart_traj_r[:, 6:8] = cart_traj[:, 6:8] @ self._v2r.rot2.T
        cart_traj_r[:, [2, 5, 8]] = cart_traj[:, [2, 5, 8]]
        # Base keep-out, enforced on the WAYPOINTS. Projecting the target is not enough: the
        # Bezier `boundary` only shapes control points (compute_control_point intersects the
        # velocity rays with it), it never clamps the curve, so a route between two legal
        # endpoints can still sweep through the disc. This is the one funnel every Cartesian
        # trajectory passes through -- bezier, Init's cubic-linear and VPSTO alike -- and it is
        # already in frame R, where the disc is centred on the origin.
        radius = self.agent_params.get('base_keepout_radius')
        if radius:
            xy = cart_traj_r[:, 0:2]
            rad = np.linalg.norm(xy, axis=1)
            inside = rad < radius
            if inside.any():
                u = xy[inside] / np.maximum(rad[inside, None], 1e-9)   # outward unit radials
                cart_traj_r[inside, 0:2] = u * radius
                # Strip the inward radial component of vel/accel: asking the QP to track motion
                # into a position it will never be given is how a plan turns into tracking error.
                for cols in (slice(3, 5), slice(6, 8)):
                    w = cart_traj_r[inside, cols]
                    radial = np.sum(w * u, axis=1)
                    w -= np.minimum(radial, 0.0)[:, None] * u
                    cart_traj_r[inside, cols] = w
        return super().optimize_trajectory(cart_traj_r, q_start, dq_start, q_anchor)

    def _vertical_hit_anchor(self, hit_pos_r_2d, hit_dir_r_2d, q_0):
        """A hit-config anchor with the mallet VERTICAL at the hit point, if one exists.

        The vendored hit-config solvers maximize velocity manipulability and leave the
        orientation to the null space -- correct for the KUKA, whose mallet self-levels on
        a passive universal joint, but this arm GRIPS its mallet rigidly: wrist tilt is
        head tilt, and a tilted head's skewed contact normal rotates the shot (measured
        ~-1.5 deg of aim error per deg of tilt; at the ~40 deg the free-orientation anchors
        produce, every smash sprays wide). So the anchor is solved with the full
        mallet-down orientation constrained and only the wrist yaw free, trying the
        strike-aligned yaw first. Where no vertical config reaches the hit point, the
        caller falls back to the vendored manipulability anchor -- a strike still happens,
        it is just as scattered as before.
        """
        hit_pos_r = np.concatenate([hit_pos_r_2d, [self.env_info['robot']['ee_desired_height']]])
        # +90 deg CCW about world z on the wrist yaw. The tool axis stays vertical either way
        # (_flat_anchor_dq levels the AXIS and deliberately leaves spin about it free), so this
        # only turns the mallet about its own stem -- it is the yaw the play loop actually holds.
        psi0 = np.arctan2(hit_dir_r_2d[1], hit_dir_r_2d[0]) + np.pi / 2
        opt = self.optimizer
        # Budget each probe: a reachable orientation converges in <1 ms, an unreachable one
        # would otherwise grind ~30 ms to the iteration cap. Up to 12 probes here (6 yaws x 2
        # seeds), and most miss, so an uncapped search froze the agent loop for 300 ms+ on the
        # replans that need it -- the single biggest source of the on-hardware jerk. 8 ms is
        # ~10x the converged-solve time, so it never truncates a real solution.
        deadline = time.perf_counter() + 20e-3  # whole-search cap: when nothing is reachable,
        for dpsi in (0., np.pi / 2, -np.pi / 2, np.pi, np.pi / 4, -np.pi / 4):
            rot = (_Rot.from_euler('z', psi0 + dpsi) * _Rot.from_euler('x', np.pi)).as_matrix()
            for seed in (q_0, INIT_SEED):
                ok, q = inverse_kinematics(opt.robot_model, opt.robot_data, hit_pos_r,
                                           rot, initial_q=seed, max_time=8e-3)
                if ok:
                    return q
            if time.perf_counter() > deadline:  # stop probing and let the caller fall back to
                break                           # the vendored (non-vertical) anchor promptly
        return None

    @property
    def aim_bias(self):
        """MINI: the rigid-grip contact-bias pre-rotation to apply to a hit direction, rad.

        `AIM_BIAS_COMP` cancels the shot rotation a TILTED mallet head causes. Keeping the
        mallet flat removes the tilt, so it also removes what there was to cancel -- see the
        measurement next to the two constants.
        """
        return AIM_BIAS_FLAT if self.optimizer.keep_mallet_flat else AIM_BIAS_COMP

    def _flattest_hit_anchor(self, q, hit_pos_r_2d):
        """MINI: take the wrist tilt out of an anchor some other solver chose.

        The vendored fallbacks maximise velocity manipulability with the orientation entirely
        free, so where `_vertical_hit_anchor` finds nothing the strike is planned around a
        wrist that can be 60 deg over -- the mallet meets the puck on its rim, and a rim
        contact rotates the shot (~-1.5 deg of aim per deg of tilt). Levelling happens strictly
        inside the position task's null space, so the anchor still puts the mallet exactly
        where the strike solver asked; only the posture it never had an opinion about changes.

        Off unless the mallet is being kept flat, so the default anchor is the vendored one.
        """
        if not self.optimizer.keep_mallet_flat:
            return q
        x_des = np.concatenate([np.asarray(hit_pos_r_2d, float),
                                [self.env_info['robot']['ee_desired_height']]])
        reached, q_flat, _ = self.optimizer.flatten_config(q, x_des)
        return q_flat if reached else q

    def solve_anchor_pos(self, hit_pos_2d, hit_dir_2d, q_0):
        pos_r, dir_r = self._v2r.pos2(hit_pos_2d), self._v2r.vec2(hit_dir_2d)
        q = self._vertical_hit_anchor(pos_r, dir_r, q_0)
        self.last_anchor_vertical = q is not None
        if q is not None:
            return q
        return self._flattest_hit_anchor(super().solve_anchor_pos(pos_r, dir_r, q_0), pos_r)

    def solve_anchor_pos_ik_null(self, hit_pos_2d, hit_dir_2d, q_0):
        pos_r, dir_r = self._v2r.pos2(hit_pos_2d), self._v2r.vec2(hit_dir_2d)
        q = self._vertical_hit_anchor(pos_r, dir_r, q_0)
        self.last_anchor_vertical = q is not None
        if q is not None:
            return q
        return self._flattest_hit_anchor(
            super().solve_anchor_pos_ik_null(pos_r, dir_r, q_0), pos_r)

    # ---- VPSTO strike planner (opt-in) ------------------------------------------------------
    def reset_vpsto(self):
        """Drop the cached VPSTO path so the next strike solves from scratch.

        The cache is keyed on nothing but the intercept POSITION, so without this a tactic change
        inherits the previous tactic's path: a SMASH latching near where a DEFEND was blocking
        found `drift` under VP_DRIFT, skipped the solve and replayed the block -- ending at
        0.05 m/s sideways (a plant) instead of 0.61 m/s at the goal (a shot). The tactics call
        this on `is_new_tactic`; `plan_strike_vpsto` also re-solves whenever `is_defend` flips, so
        a caller that forgets cannot resurrect the bug.
        """
        self._vp_sol = None
        self._vp_q_anchor = None
        self._vp_P_v = None
        self._vp_xT = None
        self._vp_is_defend = None
        self._vp_anchor_vertical = False
        self._vp_tick = 0

    def _clamp_xy_v(self, xy_v):
        """The absolute workspace clamp (`MiniSystemState.intercept_at`'s), on a MALLET target.

        `intercept_at` clamps the INTERCEPT, but the VPSTO path is planned to the strike point,
        which the aim then offsets by up to a mallet radius further out -- and on the held-target
        path (`fixed_P_v`) `intercept_at` is not consulted at all. Same bounds, so a strike can
        never be sent where an intercept would not have been allowed: `hit_range[1]` forward, the
        own-goal line back, the table half-width across.
        """
        hl = 0.5 * float(self.env_info['table']['length'])
        hw = 0.5 * float(self.env_info['table']['width'])
        x = min(max(float(xy_v[0]), V_SETBACK - hl), float(self.agent_params['hit_range'][1]))
        y = min(max(float(xy_v[1]), -hw), hw)
        return np.array([x, y])

    def _vp_aim(self, P_v, is_defend, goal_pos_v):
        """Hit direction + mallet contact point (V frame) for a predicted intercept ``P_v``.
        Mirrors the tactic aiming: DEFEND blocks laterally toward the near post; SMASH aims from
        the intercept toward the opponent goal, with the same rigid-grip contact-bias pre-rotation
        MiniSmash applies (`aim_bias`).

        The DEFEND branch is the VENDORED block offset only. It ignores `--defend-block` and the
        centre deadband, so under `--planner vpsto` a `--goalie` run silently got the 25 mm
        base-side margin the flag exists to remove -- which is why `MiniDefend` now hands its own
        `aim_fn` (`_vp_aim_block`) to `plan_strike_vpsto` and this branch is the fallback for
        callers that do not.
        """
        P_v = np.asarray(P_v, float)
        if is_defend:
            hd = np.array([0.0, np.sign(P_v[1] + 1e-6)])
        else:
            hd = np.asarray(goal_pos_v, float) - P_v
            n = np.linalg.norm(hd)
            hd = hd / n if n > 1e-9 else np.array([1.0, 0.0])
            c, s = np.cos(self.aim_bias), np.sin(self.aim_bias)
            hd = np.array([c * hd[0] - s * hd[1], s * hd[0] + c * hd[1]])
        hit_pos = P_v - hd * self.env_info['mallet']['radius']
        return hit_pos, hd

    def _vp_rendezvous_solve(self, x_cmd, v_cmd, state, predicted_time, defend_line,
                             is_defend, goal_pos_v, hit_vel_mag, q_seed, fixed_P_v=None,
                             aim_fn=None):
        """Rendezvous + VPSTO path solve (the per-COMMIT step, CMA ~20-30 ms). The rendezvous picks
        the intercept xT (V, x-y) + terminal mallet velocity vT self-consistent with the puck
        arrival AND the mallet's feasible cartesian reach time (iterate P = puck-at-T,
        T = min feasible time to reach P), then VPSTO plans the time-optimal in-plane path. Also
        solves the vertical-wrist anchor for the QP redundancy.

        `aim_fn(P_v) -> (hit_pos_v, hit_dir_v)` overrides the built-in aim, so a tactic with its
        own block geometry (`--defend-block`, the centre deadband) plans the point it would
        actually have driven to. Returns (sol, q_anchor, P_v); `q_anchor` is None when the strike
        is DECLINED -- a smash with no vertical-wrist anchor sprays (measured as bad as -143 deg
        off the goal ray), so it hands back to the bezier path, which declines it as it always
        has. A block is never declined: a mis-angled mallet still blocks.
        """
        max_pt = float(self.agent_params['max_prediction_time'])
        T = float(np.clip(predicted_time, self.dt, max_pt))
        xT = vT = P_v = hit_dir_v = None
        for _ in range(2):
            # Through the agent's own pipeline, NOT tracker.get_prediction: the latency lead,
            # the intercept rate-limit and the absolute workspace clamp all live there, and the
            # tactic gates read their output. Going straight to the tracker meant the gate and
            # the executed target were two different intercepts on the same tick. `intercept_at`
            # is the non-mutating form, so iterating candidate T here cannot disturb the
            # predicted_state the caller already committed to.
            if fixed_P_v is None:
                st, _, _, _ = state.intercept_at(T, defend_line)
                P_v = np.asarray(st[:2], float)
            else:
                # The caller is HOLDING its target (see MiniDefend._defend_target): that point
                # does not move with T, so re-querying the intercept here would walk the
                # EXECUTED target straight back off the gate the hold exists to respect --
                # which is the bug, just one layer further down.
                P_v = np.asarray(fixed_P_v, float)
            xT, hit_dir_v = (aim_fn(P_v) if aim_fn is not None
                             else self._vp_aim(P_v, is_defend, goal_pos_v))
            # SAFETY: the same absolute clamp the intercept gets. The aim offsets the strike point
            # up to a mallet radius PAST the intercept, and the held-target path never consults
            # `intercept_at` at all, so without this the plainest command line -- no lead, no
            # rate limit -- plans the mallet outside the arm's reachable play region.
            xT = self._clamp_xy_v(xT)
            vT, _ = self._vp.clip_terminal_velocity(hit_dir_v * hit_vel_mag)
            if fixed_P_v is not None:
                break
            T_best = self._vp.min_feasible_T(x_cmd, v_cmd, xT, vT)   # cartesian min-time, <1 ms
            if abs(T_best - T) < 0.03:
                break
            T = float(np.clip(T_best, self.dt, max_pt))
        # Vertical-wrist anchor for the QP redundancy -- the expensive half of a re-solve (up to
        # ~35 ms of IK probing against ~5-12 ms for the path itself), so reuse it while the strike
        # point has barely moved. Cached WITH its xT and its verticality: read back on its own,
        # `last_anchor_vertical` describes whatever solved last, which after a bezier fallback is
        # a different anchor entirely.
        if (self._vp_q_anchor is not None and self._vp_xT is not None
                and float(np.linalg.norm(xT - self._vp_xT)) < self.VP_ANCHOR_REUSE):
            q_anchor, vertical = self._vp_q_anchor, self._vp_anchor_vertical
        else:
            q_anchor = self.solve_anchor_pos_ik_null(xT, hit_dir_v, q_seed)
            vertical = bool(self.last_anchor_vertical)
            self._vp_q_anchor, self._vp_xT, self._vp_anchor_vertical = q_anchor, xT, vertical
        if not is_defend and not vertical:
            return None, None, P_v              # decline the smash -- see the docstring
        sol = self._vp.plan(x_cmd, v_cmd, xT, vT)
        return sol, q_anchor, P_v

    def plan_strike_vpsto(self, q0, dq0, x_cmd, v_cmd, state, predicted_time, defend_line,
                          is_defend, goal_pos_v, hit_vel_mag, P_now_v, q_seed,
                          fixed_P_v=None, aim_fn=None):
        """Return a joint trajectory buffer for the current strike. The VPSTO x-y PATH is CMA-solved
        only per-COMMIT (a fresh commit, the intercept drifting > VP_DRIFT throttled to every
        max_plan_steps ticks, or the path running out); EVERY tick then samples that cached path at
        the current elapsed and re-runs the cheap null-space QP-IK from the CURRENT joint state --
        re-anchoring smoothly each tick like the bezier path, so no replan-boundary jump, while
        keeping z pinned to the plane. A safety gate rejects (-> bezier fallback) any tick whose QP
        output leaves the ee_constr box. Returns (ok, (k,2,ndof) buffer or None, pred_time_left)."""
        dt = self.dt
        k = int(self.agent_params['max_plan_steps'])                # per-tick QP buffer (as bezier)
        drift = np.inf if self._vp_P_v is None else float(np.linalg.norm(np.asarray(P_now_v, float)
                                                                         - self._vp_P_v))
        exhausted = self._vp_sol is not None and self._vp_tick * dt >= float(self._vp_sol.T_best)
        # A block and a smash are OPPOSITE paths to the same place, and the cache is keyed on the
        # intercept POSITION alone: a SMASH latching where a DEFEND had been blocking found the
        # drift under VP_DRIFT, skipped the solve and replayed the block -- ending at 0.05 m/s
        # sideways instead of 0.61 m/s at the goal. The tactics call reset_vpsto() on a new
        # tactic; keying the cache on the tactic too makes that structural rather than a promise.
        tactic_changed = (self._vp_is_defend is not None
                          and self._vp_is_defend != bool(is_defend))
        need_solve = (self._vp_sol is None or exhausted or tactic_changed
                      or (drift > self.VP_DRIFT
                          and self._vp_tick >= self.agent_params['max_plan_steps']))
        if need_solve:
            sol, q_anchor, P_v = self._vp_rendezvous_solve(
                x_cmd, v_cmd, state, predicted_time, defend_line, is_defend,
                goal_pos_v, hit_vel_mag, q_seed, fixed_P_v=fixed_P_v, aim_fn=aim_fn)
            # `q_anchor is None` is a DECLINED smash (no vertical wrist at the strike point);
            # `sol is None` a failed path solve. Either way the bezier path takes this strike.
            if sol is None or q_anchor is None:
                self.vp_fallback += 1
                return False, None, predicted_time
            self._vp_sol, self._vp_q_anchor, self._vp_P_v = sol, q_anchor, P_v
            self._vp_is_defend = bool(is_defend)
            self._vp_tick = 0
        # per-tick: sample the cached path from the current elapsed, QP-IK from the CURRENT state
        elapsed = self._vp_tick * dt
        P, V, A = self._vp.sample_xy(self._vp_sol, elapsed, k, dt)
        cart = np.zeros((len(P), 9))
        cart[:, 0:2] = P
        cart[:, 2] = self._eeh                                       # pin z to the play plane
        cart[:, 3:5] = V
        cart[:, 6:8] = A
        ok, jt = self.optimize_trajectory(cart, q0, dq0, self._vp_q_anchor)
        if not ok or len(jt) < 1:
            self.vp_fallback += 1
            return False, None, predicted_time
        jt = np.asarray(jt)
        # Safety gate: any waypoint leaving the ee_constr play box -> reject, keep bezier fallback.
        # The box carries no FORWARD bound (its far edge is the opponent's end of the table, not
        # a limit the arm should ever test), so the reach limit the intercept clamp uses applies
        # here too, mapped back into V. Without it an overreaching plan is caught by nothing and
        # the arm slews out toward its joint limits -- the `vp_base` V-x 0.66 case.
        m = 0.005
        x_ub_v = float(self.agent_params['hit_range'][1]) + m
        for w in jt:
            x = forward_kinematics(self.optimizer.robot_model, self.optimizer.robot_data, w[0])[0]
            if (x[0] < self._box_xlb + m or x[1] < self._box_ylb + m or x[1] > self._box_yub - m
                    or abs(x[2] - self._eeh) > self._box_ztol - m):
                self.vp_fallback += 1
                return False, None, predicted_time
            if self._r2v is not None and float(self._r2v.pos2(x[:2])[0]) > x_ub_v:
                self.vp_fallback += 1
                return False, None, predicted_time
        # `plan_strike_vpsto` runs once per apply(), i.e. once per max_plan_steps AGENT ticks
        # (the buffer has to drain first), and hands out that many waypoints -- which are then
        # consumed one per tick. Advancing the sampling cursor by ONE dt per call therefore
        # replayed the cached path at 1/max_plan_steps speed (1/5 = 100 ms of wall time per
        # 20 ms of path), with each new buffer starting behind where the last one ended. Advance
        # by the waypoints actually handed out, so `elapsed` is real elapsed time -- which also
        # makes `exhausted` and the VP_DRIFT re-solve throttle mean what their comments say.
        self._vp_tick += len(jt)
        leftover = max(0.0, float(self._vp_sol.T_best) - elapsed - k * dt)
        self.vp_solved += 1
        return True, jt, leftover


class MiniTacticMixin:
    """The vendored assumptions about `defend_range[0]` that a side-mounted goalie cannot keep.

    `defend_range[0]` is asked to be four things at once: the PLANT line the mallet drives to,
    the `get_prediction` stop line, the `should_defend` lower bound, and the `is_puck_approaching`
    gate. The first wants to sit far out (angle coverage, reaction margin); the last three want
    to sit deep (keep defending until the puck is actually in the goal). One number cannot be
    both, and on this rig the compromise leaves a 138 mm band in front of the goal -- the band
    that physically contains the arm's own base -- defended by nothing.

    This mixin splits them: `defend_range[0]` keeps its plant-line job, and a separate, deeper
    `defend_abort_line` takes over the three give-up roles. It also refreshes the prediction
    horizon (which the vendored code lets ratchet monotonically down until no defend can ever
    fire again) and lets the gates read the UN-led intercept.

    Every method returns `super()`'s result verbatim when its knob is unset, so an agent built
    with no new arguments runs the vendored path object-for-object.

    Applied to Ready/Prepare/Defend/Smash. `Init` never predicts and `Repel` is a stub, so both
    stay vendored.
    """

    #: Reach budget (m/s) for the early-line test when `--rendezvous-vel` is not set.
    DEFAULT_EARLY_VEL = 0.25

    @property
    def _abort_line(self):
        return self.agent_params.get('defend_abort_line')

    def _stop_line(self):
        """The x `get_prediction` should stop at, this tick.

        Vendored (`fsm/tactics.py:37-41`): the plant line while the puck is outside it, and
        ``0.`` once it is inside -- which for a V-frame x means "never stop", so a puck in the
        last 138 mm is rolled the whole horizon and the plan aims wherever it lands. Split: the
        plant line while outside, the ABORT line once inside, so the target stays the deepest
        line the mallet may legally reach.
        """
        early = self._early_line()
        if early is not None and self.state.estimated_state[0] >= early:
            return early
        plant = self.agent_params['defend_range'][0]
        if self.state.estimated_state[0] >= plant:
            return plant
        return 0. if self._abort_line is None else self._abort_line

    def _early_line(self):
        """`_early_line_impl`, recording the answer for the overlay and the telemetry column."""
        line = self._early_line_impl()
        self.state.defend_early_line = np.nan if line is None else float(line)
        return line

    def _early_line_impl(self):
        """Meet a NEAR-HEAD-ON puck out in front, instead of at the plant line. None = not now.

        The plant line sits at V-x 0.2506 and the taught park at 0.3064, so on this mount the
        crossing a normal defend solves for is 56 mm BEHIND the mallet. Everything that wants
        run-up is then impossible by construction: measured over a rig run, `d_avail` had p50
        -0.010 m and only 31.8% of DEFEND ticks had any forward room at all, so the clearing
        block fell back to the vendored plant at 59.3% of them -- and at every actual contact.

        Meeting the puck further up-table fixes both halves at once: it puts the crossing ahead
        of the mallet (run-up) and it leaves more time to get there (feasible speed). It is
        restricted to near-head-on pucks deliberately -- those are the ones aimed at our goal,
        and their path needs no rim-bounce model to predict, so committing early to it is cheap.
        An angled puck keeps the vendored line.

        Self-limiting by reachability: the line is only used when the arm could actually BE there
        when the puck arrives, judged with the same `rendezvous_vel` budget and from the same
        measured mallet. Without that test a far line is exactly the "arm swings forward at a
        phantom" regression that `--workspace-clamp` and `--defend-hold-target` exist to stop.
        """
        frac = self.agent_params.get('defend_early_frac')
        if frac is None:
            return None
        st = self.state.estimated_state
        vx, vy = float(st[2]), float(st[3])
        speed = float(np.hypot(vx, vy))
        if speed < float(self.agent_params['static_vel_threshold']) or vx >= 0.:
            return None
        if abs(vx) / max(speed, 1e-12) < float(self.agent_params['defend_early_cos']):
            return None                                  # not head-on enough; vendored line
        half_l = 0.5 * float(self.env_info['table']['length'])
        line = (V_SETBACK - half_l) + float(frac) * half_l
        # Never past the forward reach the workspace clamp would pull it back to anyway, and
        # never BEHIND the plant line -- this knob only ever moves the meeting point up-table.
        line = min(line, float(self.agent_params['hit_range'][1]))
        line = max(line, float(self.agent_params['defend_range'][0]))
        if float(st[0]) < line:
            return None                                  # already past it; vendored line
        t_to = (float(st[0]) - line) / max(-vx, 1e-6)
        if t_to <= 0.:
            return None
        vel = self.agent_params.get('rendezvous_vel') or self.DEFAULT_EARLY_VEL
        mallet = self.state.mallet_ref_xy()
        reach = float(np.hypot(line - mallet[0], (float(st[1]) + vy * t_to) - mallet[1]))
        if reach > float(vel) * t_to:
            return None                                  # cannot be there in time; do not chase
        return line

    def _zone_hi(self):
        """`should_defend`'s upper bound, raised to the early line when one is in play."""
        hi = float(self.agent_params['defend_range'][1])
        early = self._early_line()
        return hi if early is None else max(hi, early)

    def _horizon(self):
        """The horizon handed to `get_prediction`.

        `get_prediction` returns ``min(time_to_line, t)`` and the vendored caller feeds its own
        previous OUTPUT back in as `t` -- a monotone ratchet. `Ready.ready()` re-inflates it only
        on `is_new_tactic`, so while sitting in READY watching a puck come in it decays; once it
        is below ``max_plan_steps * dt`` (0.1 s) `should_defend`'s fourth condition refuses every
        defend, and the only escape needs the puck beyond `defend_range[1]`. Handing the full
        horizon each tick makes the returned time the TRUE time-to-line, which is what the
        planner's `t_plan` wanted in the first place.
        """
        if not self.agent_params.get('defend_fresh_horizon'):
            return self.state.predicted_time
        return float(self.agent_params['max_prediction_time'])

    def _update_prediction(self):
        if (self._abort_line is None and not self.agent_params.get('defend_fresh_horizon')
                and self.agent_params.get('defend_early_frac') is None):
            return super()._update_prediction()          # vendored path, untouched
        self.state.update_prediction(self._horizon(), self._stop_line())

    def _gate_state(self):
        if not self.agent_params.get('defend_gate_unled'):
            return self.state.predicted_state
        return getattr(self.state, 'predicted_gate_state', self.state.predicted_state)

    def _in_zone(self, ps):
        """The GEOMETRIC half of `should_defend`: is this intercept one we would block?

        Split out because it is the question a LATCHED defend has to keep asking (see
        `MiniDefend._defend_target`) and the one the telemetry reports per tick
        (`run_hw_match._gate_ok`). `should_defend` itself cannot be reused for either: it adds
        the approach and timing conditions, and it MUTATES `predicted_time` on its miss branch,
        so merely asking it the question changes the run.
        """
        lo = self._abort_line if self._abort_line is not None \
            else self.agent_params['defend_range'][0]
        return bool(lo <= ps[0] <= self._zone_hi()
                    and np.abs(ps[1]) <= self.agent_params['defend_width'])

    def should_defend(self):
        if (self._abort_line is None and not self.agent_params.get('defend_gate_unled')
                and self.agent_params.get('defend_early_frac') is None):
            return super().should_defend()               # vendored path, untouched
        ps = self._gate_state()
        hi = self._zone_hi()
        if self.state.is_puck_approaching():
            if self._in_zone(ps) \
                    and self.state.predicted_time >= \
                    self.agent_params['max_plan_steps'] * self.generator.dt:
                return True
            elif self.state.predicted_time < self.agent_params['max_prediction_time'] \
                    and ps[0] > hi:
                self.state.predicted_time += ((hi - ps[0])
                                              / ps[2])
                self.state.predicted_time = np.clip(self.state.predicted_time, 0,
                                                    self.agent_params['max_prediction_time'])
        return False

    def can_smash(self):
        """Vendored `can_smash`, plus the slow-but-not-static puck nothing on the rig claims.

        `Tactic.can_smash` requires `is_puck_static()`, and hardware runs `static_vel_threshold`
        at 0.02 m/s so DEFEND latches on gentle pushes. A puck drifting at 0.05-0.15 m/s is
        therefore static to nothing: the smash refuses it, `should_defend` refuses it unless it is
        closing on our goal, and `is_puck_stuck` refuses it too -- the FSM sits in READY and
        watches it. This adds exactly that case.

        It lives HERE rather than at the Ready dispatcher because `Smash.ready` and
        `Smash._update_tactic_impl` ask the same question: gate only the dispatcher and SMASH is
        entered and then dropped five ticks later.

        DEFEND keeps priority. The vendored dispatcher tests `can_smash` BEFORE `should_defend`,
        so an approaching puck is handed back here rather than reordering vendored code.
        """
        if not self.agent_params.get('smash_slow'):
            return super().can_smash()               # vendored path, untouched
        if super().can_smash():
            return True                              # the truly-static case, unchanged
        if not self.state.is_puck_slow():
            return False
        if self.state.is_puck_approaching():
            return False                             # an approaching puck belongs to DEFEND
        # The UN-led intercept: --lead-time compensates latency against a MOVING target, and this
        # puck is by construction nearly stationary, so leading it would only add jitter.
        ps = self._gate_state()
        lateral = (self.env_info['table']['width'] / 2 - self.env_info['puck']['radius']
                   - 2 * self.env_info['mallet']['radius'])
        return bool(self.agent_params['hit_range'][0] < ps[0]
                    < self.agent_params['smash_slow_max_x']
                    and np.abs(ps[1]) < lateral)


class MiniReady(MiniTacticMixin, Ready):
    """Ready that GUARDS the goal instead of parking on a fixed spot.

    When DEFEND gives up, the vendored Ready drives the mallet back to `x_home` at
    `default_linear_vel` with a hard 1.0 s floor on the glide, and the FSM cannot re-latch DEFEND
    for `switch_tactics_min_steps` ticks (300 ms) while it does. A 1 m/s puck crosses the whole
    138 mm band in front of the goal in 138 ms. Retreating is therefore strictly dominated: the
    arm cannot finish the retreat before the puck arrives, cannot react during it, and ends up
    further from the goal than it started.

    More importantly this arm cannot swing its way out of trouble. The guaranteed mallet speed at
    --max-joint-vel 1.0 is ~0.4 m/s in the fast part of the workspace and 0.12 m/s in the wedge
    beside the base, against pucks at 1-3 m/s. Being in the right place already is the only
    defence that scales, which is what this does: hold the mallet on the line from the puck to
    our own goal centre, at a fixed radius from that centre, and let the puck come to it.

    PLANNER-AGNOSTIC by construction: this is a target, not a trajectory. Ready plans its glide
    through `generator.bezier_planner` in both `--planner` modes (VPSTO is only ever used for
    strikes, in MiniDefend/MiniSmash), so the guard behaves identically either way.

    Off (`ready_guard` unset) every method here defers to the vendored Ready.
    """

    def _goal_centre_v(self):
        return np.array([V_SETBACK - 0.5 * float(self.env_info['table']['length']), 0.0])

    def _guard_target(self):
        """Where to WAIT: on the puck -> own-goal-centre line, `ready_guard_radius` out.

        Falls back to the park whenever the puck is not a thing we can sensibly stand in front
        of -- no estimate yet, or off the table (which is also what a zeroed `estimated_state`
        looks like before the first observation).
        """
        ap = self.agent_params
        home = np.asarray(ap['x_home'][:2], dtype=float)
        if not ap.get('ready_guard'):
            return home
        puck = np.asarray(self.state.estimated_state[:2], dtype=float)
        if not np.all(np.isfinite(puck)):
            return home
        goal = self._goal_centre_v()
        half_w = 0.5 * float(self.env_info['table']['width'])
        if not (goal[0] <= puck[0] <= goal[0] + float(self.env_info['table']['length'])
                and abs(puck[1]) <= half_w):
            return home
        d = puck - goal
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return home
        target = goal + d * (float(ap.get('ready_guard_radius', 0.10)) / n)
        # Deep bound. NOT the plant line: that is where the defender chooses to meet the puck,
        # not a legality limit, and flooring a WAITING position there would push the guard back
        # out of the band it exists to cover. The real floors are the Bezier routing box (which
        # already carries the own-goal keep-out) and, when the line is split, the abort line --
        # below which we would not defend anyway, so waiting there buys nothing.
        lo = float(self.generator.bound_points[:, 0].min())
        abort = ap.get('defend_abort_line')
        if abort is not None:
            lo = max(lo, float(abort))
        target[0] = max(float(target[0]), lo)
        w = float(ap['defend_width'])
        target[1] = min(max(float(target[1]), -w), w)
        return np.asarray(self.generator.project_out_of_base(target), dtype=float)

    def _guard_t_stop(self):
        """Glide duration to the guard point.

        The vendored 1.0 s floor is harmless on a retreat -- slow is fine when you are giving up
        -- but it is the whole cost on a guard move, which is a small correction that wants to
        finish before the next one. Floored by `ready_t_stop_min`; keep that >= 0.2 s or the
        Bezier drops into its terminal branch every tick (`t_final >= 2*max_plan_steps*dt`) and
        the tactic thrashes.
        """
        travel = float(np.linalg.norm(self._guard_xy - np.asarray(self.state.x_cmd[:2], float)))
        return max(travel / self.agent_params['default_linear_vel'],
                   float(self.agent_params.get('ready_t_stop_min', 1.0)))

    def ready(self):
        if not self.agent_params.get('ready_guard'):
            return super().ready()                       # vendored path, untouched
        if self.state.is_new_tactic:
            self.state.tactic_finish = False
            self.plan_new_trajectory = True
            self.replan_time = 0
            self.switch_count = 0
            self.state.predicted_time = self.agent_params['max_prediction_time']
            self._guard_xy = self._guard_target()
            self.t_stop = self._guard_t_stop()
            return True
        if len(self.state.trajectory_buffer) == 0:
            target = self._guard_target()
            # Re-aim only on a real move. `apply` re-aims the TAIL of the live curve
            # (update_bezier_curve), so tracking a drifting guard point costs nothing extra --
            # but re-aiming for sub-millimetre EKF jitter is how a guard turns into chatter.
            if (getattr(self, '_guard_xy', None) is None
                    or np.linalg.norm(target - self._guard_xy)
                    > float(self.agent_params.get('ready_guard_drift', 0.02))):
                self._guard_xy = target
            self.t_stop = self._guard_t_stop()
            return True
        return False

    def apply(self):
        if not self.agent_params.get('ready_guard'):
            return super().apply()                       # vendored path, untouched
        self.state.is_new_tactic = False
        target = np.asarray(self._guard_xy, dtype=float)

        for _ in range(10):
            if self.plan_new_trajectory:
                self.generator.bezier_planner.compute_control_point(
                    self.state.x_cmd[:2], self.state.v_cmd[:2], target, np.zeros(2), self.t_stop)
            else:
                if self.generator.bezier_planner.t_final > self.replan_time:
                    self.generator.bezier_planner.update_bezier_curve(
                        self.replan_time, target, np.zeros(2), self.t_stop)

            if self.generator.bezier_planner.t_final >= \
                    2 * self.agent_params['max_plan_steps'] * self.generator.dt:
                cart_traj = self.generator.generate_bezier_trajectory(
                    self.agent_params['max_plan_steps'])
            # MINI: floor the terminal branch at 2*dt. `ready()` resets `replan_time` to 0, so on
            # the first pass of every fresh plan `t_final >= 0` accepted a ZERO-length curve, and
            # `generate_bezier_trajectory()` then indexed an empty sample grid -> IndexError. The
            # else branch below (tactic_finish, return) is the vendored intent for a curve that
            # short; `MiniDefend.apply` already floors the same branch at 2*dt.
            elif self.generator.bezier_planner.t_final >= max(self.replan_time,
                                                              2 * self.generator.dt):
                self.state.tactic_finish = True
                cart_traj = self.generator.generate_bezier_trajectory()
            else:
                self.state.tactic_finish = True
                return

            success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
                cart_traj, self.state.q_cmd, self.state.dq_cmd,
                self.agent_params['joint_anchor_pos'])

            if success:
                self.replan_time = self.agent_params['max_plan_steps'] * self.generator.dt
                self.plan_new_trajectory = False
                self.t_stop = self.generator.bezier_planner.t_final - \
                    self.state.trajectory_buffer.shape[0] * self.generator.dt
                self.state.predicted_time = np.maximum(0, self.state.predicted_time)
                return
            self.t_stop *= 1.5
            self.replan_time = 0
            self.state.tactic_finish = False

        self.state.tactic_finish = True


class MiniPrepare(MiniTacticMixin, Prepare):
    pass


class MiniDefend(MiniTacticMixin, Defend):
    """Defend with a per-intercept anchor, solved vertical.

    Identical to `Defend.apply` except the anchor line the vendored code carries
    commented-out is enabled, routed through the vertical solver. The KUKA never needed
    it -- its mallet self-levels -- but with the rigid grip a far intercept swung on the
    home anchor flops the wrist past horizontal and torques the mallet out of the jaws
    (measured: defend drops rise from ~2/10 to 5/10 when swings start from the base-side
    home). The vertical anchor keeps the head down through the swing.
    """

    def _update_tactic_impl(self):
        """`Defend._update_tactic_impl`, with the stop line and the give-up count split out.

        The vendored version re-predicts with ``stop_line = defend_range[0]`` unconditionally,
        overwriting the `_update_prediction` that ran two lines earlier. Once the puck is inside
        the plant line that call breaks at i=0, so `predicted_time` comes back 0 -> the Bezier is
        handed ``t_plan = 0`` -> ``t_final = 0`` -> `break` -> `tactic_finish = True`. DEFEND
        therefore ENDS ITSELF the moment the puck goes deep, which is precisely when it is needed.
        """
        if (self._abort_line is None and not self.agent_params.get('defend_fresh_horizon')
                and self.agent_params.get('defend_early_frac') is None):
            return super()._update_tactic_impl()          # vendored path, untouched
        self.state.update_prediction(self._horizon(), self._stop_line())
        if not self.should_defend():
            self.switch_count += 1
        else:
            self.switch_count = 0
        if (self.switch_count > int(self.agent_params.get('defend_give_up_ticks', 4))
                or self.state.tactic_finish) and len(self.state.trajectory_buffer) == 0:
            self._set_tactic(TACTICS.READY)

    def ready(self):
        out = super().ready()
        if self.state.is_new_tactic:
            self._block_side = None          # re-latch the block side per intercept
            self._clear_phase = None         # and re-decide cock/strike per intercept
            self.generator.reset_vpsto()     # never inherit a SMASH's path (see reset_vpsto)
            # A brand-new DEFEND is here because READY's gate just accepted this intercept, so
            # it is in-zone by construction -- seed the hold with it (see _defend_target).
            self._held = np.asarray(self.state.predicted_state[:2], dtype=float).copy()
        return out

    #: Below this much available run-up (m) there is no stroke to plan, only a plant.
    CLEAR_MIN_RUNUP = 0.01

    def _clear_target(self, P, latch=True):
        """Where to plant, and how fast, to RETURN the puck up-table instead of stopping it.

        The vendored block aims `[0, +-1] * 0.05` -- a 5 cm/s sideways tap with no up-table
        component at all -- so DEFEND is structurally a block, never a return. That is invisible
        for an angled puck, whose crossing is displaced in y: the mallet has to traverse to reach
        it and the puck is swatted away by the momentum of the traverse. For a head-on puck the
        crossing lands on the mallet's own line, the traverse is ~0, and the puck simply dies on
        a stationary mallet. Measured on the same terminal command and time budget: 0.050 m/s of
        path speed with zero travel against 0.703 m/s over 0.149 m.

        So the fix is RUN-UP, not a bigger number. `bezier_planner` reserves
        `min_length = min(0.15/h_23, 1)` of the approach ray for the terminal leg, which makes
        its terminal slope `1/(3*min(0.15, h_23))` -- independent of the commanded hit speed.
        Raising the 0.05 alone buys nothing but a bigger BACKWARD excursion (measured
        0.0667 * v_hit, i.e. 40 mm toward our own goal at 0.6 m/s, uncommanded and unclamped --
        which is what the waypoint floor in `MiniTrajectoryGenerator.optimize_trajectory` is for).

        Two phases, latched once per apply like the tap side (see `_vp_aim_block`):

            cock   -- aim at `P - x_hat*d`, on the crossing's own line, `d` metres down-table,
                      terminal velocity zero. Still a block: it shares the crossing's y and sits
                      in the puck's path, 50 mm deeper. It does not open a lane.
            strike -- aim at the crossing `P` itself with terminal velocity `v*x_hat`, so the
                      mallet arrives MOVING up-table as the puck gets there.

        Commit when `2*d_avail/T >= v_clear`: the mean speed of a rest->v ramp is v/2, so that is
        exactly the instant the available run-up fills the remaining time at the commanded speed.
        The terminal point is `P` and there is no overshoot knob -- ending at `P + 0.05*x_hat`
        was measured to force the mallet across `P` at 0.68-1.6 m/s with double the a_max, while
        ending at `P` still gives contact at >= v_clear because mid-path speed exceeds the
        terminal value.

        Returns None wherever there is no room or no time, which lets `_block_target` fall
        through to the vendored plant -- every degrade lands on today's behaviour, never a whiff.
        """
        ap = self.agent_params
        u = np.array([1.0, 0.0])                    # up-table, away from our own goal
        # The Bezier ROUTING floor, not the plant line: it is already the max of the table rim
        # bound and the own-goal keep-out, so the standoff tracks --min-goal-dist without a
        # second copy of it. Read at use time -- --defend-line-frac and friends mutate
        # agent_params after construction.
        floor = float(self.generator.bound_points[:, 0].min())
        d_max = float(P[0]) - (floor + float(ap['defend_clear_margin']))
        d = max(min(float(ap['defend_clear_standoff']), d_max), 0.0)

        T = max(float(self.state.predicted_time),
                float(ap['max_plan_steps']) * self.generator.dt)
        d_avail = float(np.dot(P - np.asarray(self.state.x_cmd[:2], dtype=float), u))
        # Negative d_avail means the mallet is already up-table of the crossing: there is no
        # forward stroke to make and a backward one would drive at our own goal. Plant instead.
        if d_avail <= self.CLEAR_MIN_RUNUP:
            return None

        v_clear = float(ap['defend_clear_speed'])
        # Feasibility, learned the hard way on the rig: without this the commit fired on ramp
        # arithmetic alone and asked for 3.58 m/s of mallet speed -- 6.05 rad/s of joint speed
        # against a 1.4 cap -- which the streamer throttled to 0.03 m/s delivered, i.e. a plant
        # with extra tracking error and a 0.39 rad command deficit. The old
        # `T <= 2*max_plan_steps*dt` escape hatch FORCED exactly that case, so it is gone: no
        # time left is a reason to plant, never a reason to swing wildly.
        budget = float(ap.get('defend_clear_max_path') or 0.0)
        if budget > 0.0 and d_avail > budget * T:
            return None
        v_need = 2.0 * d_avail / max(T, 1e-9)
        strike = v_need >= v_clear
        if latch or getattr(self, '_clear_phase', None) is None:
            self._clear_phase = 'strike' if strike else 'cock'

        if self._clear_phase == 'cock':
            if d <= 0.0:
                return None                         # no room to cock (deep crossing) -> plant
            S = self.generator.project_out_of_base(P - u * d)
            return np.asarray(S, dtype=float), u, np.zeros(2)
        return np.asarray(P, dtype=float), u, u * float(np.clip(v_need, 0.0, v_clear))

    def _block_target(self, puck_pos_2d, latch=True):
        """Where the mallet plants for a block, and which way it taps.

        `latch=False` reuses the side already latched instead of re-deciding it, for the repeated
        aim calls inside ONE VPSTO rendezvous (see `_vp_aim_block`).

        The vendored offset is a LATERAL clear: plant at ``puck - hit_dir * mallet_radius``, half
        overlapping the predicted crossing on the CENTRELINE side, then nudge outward at
        0.05 m/s. Contact needs ``|y_actual - y_mallet| <= r_m + r_p = 0.050``, so for a puck
        crossing at +y that window is ``[y_p - 0.075, y_p + 0.025]`` -- 25 mm of margin on the
        base side against 75 mm on the other. A 30 mm y-error toward the base and the puck goes
        past untouched, which is the miss that gets reported as "it swung but missed on the base
        side".

            'mallet'  (k = r_m)        [y_p-0.075, y_p+0.025]   vendored; 25 mm base-side margin
            'centre'  (k = 0)          [y_p-0.050, y_p+0.050]   symmetric; 50 mm  <- recommended
            'contact' (k = r_m + r_p)  [y_p-0.100, y_p+0.000]   the Prepare convention; ZERO

        Note that copying `Prepare`'s ``r_m + r_p`` -- the obvious "the offset is too small" fix --
        makes the base-side hole WORSE, not better. The margin comes from centring, not from
        offsetting further.

        'centre' also removes the ``np.sign(y + 1e-6)`` discontinuity from the TARGET entirely
        (with k = 0 `hit_pos` no longer depends on the sign), leaving it only in the 0.05 m/s tap
        and the IK anchor. The deadband latches the side anyway, so a shot straight down the
        middle stops flipping the tap direction tick to tick.
        """
        if self.agent_params.get('defend_clear'):
            cleared = self._clear_target(np.asarray(puck_pos_2d[:2], dtype=float), latch=latch)
            # Record what actually happened, for the overlay and the `defend_phase` column.
            # Recomputing it downstream would be a second copy of this decision to keep in sync.
            self.state.defend_clear_phase = (self._clear_phase if cleared is not None
                                             else 'plant')
            if cleared is not None:
                return cleared
        mode = self.agent_params.get('defend_block_offset', 'mallet')
        dead = float(self.agent_params.get('defend_centre_deadband', 0.0))
        y = float(puck_pos_2d[1])
        if dead <= 0.0:
            side = np.sign(y + 1e-6)                                  # vendored
        else:
            if getattr(self, '_block_side', None) is None or (latch and abs(y) > dead):
                self._block_side = 1.0 if y >= 0.0 else -1.0
            side = self._block_side
        hit_dir_2d = np.array([0., side])
        r_m = self.env_info['mallet']['radius']
        r_p = self.env_info['puck']['radius']
        k = {'mallet': r_m, 'centre': 0.0, 'contact': r_m + r_p}[mode]
        hit_pos_2d = np.asarray(puck_pos_2d[:2], dtype=float) - hit_dir_2d * k
        hit_pos_2d = self.generator.project_out_of_base(hit_pos_2d)
        return hit_pos_2d, hit_dir_2d, hit_dir_2d * 0.05

    def _vp_aim_block(self, P_v):
        """VPSTO aim for a block: the plant point and tap direction `_block_target` would use.

        `_vp_aim`'s own DEFEND branch hard-codes the VENDORED offset, so under `--planner vpsto`
        a `--goalie` run planted 25 mm base-side of the crossing while the bezier path planted ON
        it -- `--defend-block centre` and the centre deadband never reached the executed target,
        which is the base-side miss those flags exist to close.

        The tap side is latched on the FIRST call of each apply: the rendezvous aims once per
        candidate arrival time, so a puck near the centreline would otherwise flip the side
        between iterations of a single solve, and the two iterations would plan opposite blocks.
        """
        hit_pos, hit_dir, _ = self._block_target(
            P_v, latch=getattr(self, '_vp_aim_latch', True))
        self._vp_aim_latch = False
        return np.asarray(hit_pos, float), np.asarray(hit_dir, float)

    def _defend_target(self):
        """The intercept this apply plans to, and whether it is the HELD one.

        A latched DEFEND re-reads `predicted_state` on every apply and never re-checks its own
        gate, so once the estimate wanders out of the defend zone the plan follows it out:
        measured 19-50% of DEFEND ticks aiming outside the zone, and in `vp_base` the target ran
        to V-x 0.78 with the arm chasing it to 0.66 -- up-table, away from the goal, which is the
        opposite of defending. The gate is what said a target was worth blocking, so when it
        stops holding, keep blocking the last target that DID pass it: to score, the puck still
        has to come back through there.

        The hold is bounded by the tactic itself, not by a timer -- an intercept out of the zone
        also fails `should_defend`, so `switch_count` climbs and DEFEND hands back to READY after
        `defend_give_up_ticks`. This only decides where the mallet waits until then.

        Off (`defend_hold_target` unset) this is `predicted_state[:2]`, as vendored.
        """
        target = np.asarray(self.state.predicted_state[:2], dtype=float)
        held = getattr(self, '_held', None)
        if not self.agent_params.get('defend_hold_target'):
            return target, False
        if held is None or self._in_zone(self._gate_state()):
            self._held = target.copy()
            return target, False
        return np.asarray(held, dtype=float), True

    def apply(self):
        self.state.is_new_tactic = False

        target, holding = self._defend_target()

        # Aim ONCE, ahead of either planner. Both paths have to plan the same (point, direction,
        # speed) or the gate and the executed block are two different targets -- the divergence
        # this file already had to fix once for --defend-block (see _vp_aim_block).
        prev_phase = getattr(self, '_clear_phase', None)
        hit_pos_2d, hit_dir_2d, hit_vel_2d = self._block_target(target)
        if (self.agent_params.get('defend_clear') and prev_phase == 'cock'
                and getattr(self, '_clear_phase', None) == 'strike'):
            # Cock -> strike: the live curve was shaped for a target DOWN-table of the crossing
            # and the fallback below only re-aims its tail (update_bezier_curve), so it would
            # stroke from the wrong shape. Start a fresh curve from the live commanded state --
            # the same two lines, for the same reason, as the VPSTO success path below.
            self.plan_new_trajectory = True
            self.replan_time = 0

        # VPSTO block: a feasible, time-optimal swing to the predicted line-crossing so the mallet
        # arrives on time (the bezier path's infeasible plan is what made a far block late). Small
        # terminal velocity = plant on the intercept, as the vendored defend does. ok=False (the
        # path solve failed, or the QP output left the play box) -> the vendored bezier path below
        # runs unchanged. A block is never DECLINED for want of a vertical wrist the way a smash
        # is: a mallet at the wrong angle still blocks, it just does not aim.
        if self.generator.use_vpsto:
            self._vp_aim_latch = True        # the rendezvous aims twice; latch the tap side once
            ok, buf, leftover = self.generator.plan_strike_vpsto(
                self.state.q_cmd, self.state.dq_cmd,
                self.state.x_cmd[:2], self.state.v_cmd[:2],
                self.state, self.state.predicted_time,
                self._stop_line(),
                is_defend=True, goal_pos_v=None,
                hit_vel_mag=float(np.linalg.norm(hit_vel_2d)),
                P_now_v=target, q_seed=self.state.q_cmd,
                fixed_P_v=(target if holding else None),
                aim_fn=self._vp_aim_block)
            if ok:
                self.state.trajectory_buffer = buf
                self.state.predicted_time = leftover
                # The bezier curve is STALE after a VPSTO tick -- it was last shaped for an older
                # target, and the fallback path only re-aims its TAIL (`update_bezier_curve`).
                # Leaving this False meant the first fallback after a VPSTO success re-aimed a
                # curve that had already run out, produced no trajectory, and the defend was
                # dropped. Start the fallback from the live commanded state instead.
                self.plan_new_trajectory = True
                self.replan_time = 0
                return

        self.q_anchor_pos = self.generator.solve_anchor_pos_ik_null(hit_pos_2d, hit_dir_2d, self.q_anchor_pos)

        for i in range(10):
            if self.plan_new_trajectory:
                self.generator.bezier_planner.compute_control_point(self.state.x_cmd[:2], self.state.v_cmd[:2],
                                                                    hit_pos_2d, hit_vel_2d, self.state.predicted_time)
            else:
                if self.generator.bezier_planner.t_final > self.replan_time:
                    self.generator.bezier_planner.update_bezier_curve(self.replan_time, hit_pos_2d,
                                                                      hit_vel_2d, self.state.predicted_time)

            if self.generator.bezier_planner.t_final >= 2 * self.agent_params['max_plan_steps'] * self.generator.dt:
                cart_traj = self.generator.generate_bezier_trajectory(self.agent_params['max_plan_steps'])
            elif self.generator.bezier_planner.t_final >= 2 * self.generator.dt:
                self.state.tactic_finish = True
                cart_traj = self.generator.generate_bezier_trajectory()
            else:
                break

            success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
                cart_traj, self.state.q_cmd, self.state.dq_cmd, self.q_anchor_pos)

            if success:
                self.replan_time = self.agent_params['max_plan_steps'] * self.generator.dt
                self.plan_new_trajectory = False
                self.state.predicted_time = self.generator.bezier_planner.t_final - \
                                            self.state.trajectory_buffer.shape[0] * self.generator.dt
                self.state.predicted_time = np.clip(self.state.predicted_time, 0,
                                                    self.agent_params['max_prediction_time'])
                return

            self.state.predicted_time += self.agent_params['max_plan_steps'] * self.generator.dt
            self.replan_time = 0
            self.state.tactic_finish = False

        self.state.tactic_finish = True


class MiniSmash(MiniTacticMixin, Smash):
    """Smash aimed at the mini table's goal instead of the full-size table's.

    Identical to `Smash.apply` except for `goal_pos`, which the baseline hard-codes. In the
    virtual frame the opponent goal centre is half the table ahead of the centreline plus
    the frame's own setback.
    """

    def ready(self):
        out = super().ready()
        if self.state.is_new_tactic:
            self.generator.reset_vpsto()     # never inherit a DEFEND's path (see reset_vpsto)
        return out

    def apply(self):
        self.state.is_new_tactic = False

        goal_pos = np.array([TABLE_LENGTH / 2 + V_SETBACK, 0.0])

        # VPSTO strike: a CARTESIAN, vel/acc-feasible, time-optimal swing to the intercept with
        # the terminal mallet velocity aimed at the goal (the rendezvous inside matches the lead to
        # the arm's real reach). With no vertical-wrist anchor at the strike point it returns
        # ok=False -> the vendored bezier path below runs and DECLINES the smash at its own
        # `last_anchor_vertical` check, exactly as when --planner vpsto is off. (That was the
        # documented behaviour long before it was the real one: the check here could not fire,
        # so a spraying smash was planned and executed under --planner vpsto.)
        if self.generator.use_vpsto:
            ok, buf, leftover = self.generator.plan_strike_vpsto(
                self.state.q_cmd, self.state.dq_cmd,
                self.state.x_cmd[:2], self.state.v_cmd[:2],
                self.state, self.state.predicted_time,
                self._stop_line(),
                is_defend=False, goal_pos_v=goal_pos, hit_vel_mag=self.hit_vel_mag,
                P_now_v=self.state.predicted_state[:2], q_seed=self.state.q_cmd)
            if ok:
                self.state.trajectory_buffer = buf
                self.state.predicted_time = leftover
                # See MiniDefend.apply: the bezier curve is stale after a VPSTO tick, so the
                # fallback has to start a new one from the live commanded state.
                self.plan_new_trajectory = True
                self.replan_time = 0
                return

        puck_pos_2d = self.state.predicted_state[:2]

        hit_dir_2d = goal_pos - puck_pos_2d
        hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)
        # Pre-rotate the command to cancel the measured rigid-grip contact bias (zero when the
        # mallet is being held flat -- there is then no tilt to cancel).
        c, s = np.cos(self.generator.aim_bias), np.sin(self.generator.aim_bias)
        hit_dir_2d = np.array([c * hit_dir_2d[0] - s * hit_dir_2d[1],
                               s * hit_dir_2d[0] + c * hit_dir_2d[1]])
        hit_vel_2d = hit_dir_2d * self.hit_vel_mag

        hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * (self.env_info['mallet']['radius'])
        hit_pos_2d = self.generator.project_out_of_base(hit_pos_2d)
        self.q_anchor_pos = self.generator.solve_anchor_pos_ik_null(hit_pos_2d, hit_dir_2d, self.q_anchor_pos)

        # A smash without a vertical wrist config sprays (measured as bad as -143 deg off
        # the goal ray): a shot guaranteed to miss is worse than declining, so hand back
        # to READY and wait for the puck to drift somewhere strikeable.
        if not getattr(self.generator, 'last_anchor_vertical', True):
            self.state.tactic_finish = True
            return

        for i in range(10):
            if self.plan_new_trajectory:
                self.generator.bezier_planner.compute_control_point(self.state.x_cmd[:2], self.state.v_cmd[:2],
                                                                    hit_pos_2d, hit_vel_2d, self.state.predicted_time)
            else:
                if self.generator.bezier_planner.t_final > self.replan_time:
                    self.generator.bezier_planner.update_bezier_curve(self.replan_time, hit_pos_2d,
                                                                      hit_vel_2d, self.state.predicted_time)

            if self.generator.bezier_planner.t_final >= 2 * self.agent_params['max_plan_steps'] * self.generator.dt:
                cart_traj = self.generator.generate_bezier_trajectory(self.agent_params['max_plan_steps'])
            # MINI: see MiniReady.apply -- a zero-length curve indexes an empty sample grid.
            elif self.generator.bezier_planner.t_final >= max(self.replan_time,
                                                              2 * self.generator.dt):
                self.state.tactic_finish = True
                cart_traj = self.generator.generate_bezier_trajectory()
            else:
                self.state.tactic_finish = True
                return

            success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
                cart_traj, self.state.q_cmd, self.state.dq_cmd, self.q_anchor_pos)

            if success:
                self.replan_time = self.agent_params['max_plan_steps'] * self.generator.dt
                self.plan_new_trajectory = False
                self.state.predicted_time = self.generator.bezier_planner.t_final - \
                                            self.state.trajectory_buffer.shape[0] * self.generator.dt
                self.state.predicted_time = np.maximum(0, self.state.predicted_time)
                return

            # MINI: restore the vendored retry contract (fsm/tactics.py:445-451). This loop had
            # kept only the decay, so `hit_vel_2d` was never recomputed from the decayed
            # magnitude, `predicted_time` never grew and `replan_time` was never reset: all ten
            # iterations re-planned the byte-identical infeasible curve and failed identically.
            # Without the post-loop `tactic_finish = True` SMASH was then left holding an empty
            # buffer with no give-up signal, so `_update_tactic_impl` never counted toward
            # handing back to READY.
            self.state.predicted_time += 2 * self.generator.dt
            self.hit_vel_mag *= 0.9
            hit_vel_2d = hit_dir_2d * self.hit_vel_mag
            self.replan_time = 0
            self.state.tactic_finish = False

        self.state.tactic_finish = True


class MiniBaselineAgent(BaselineAgent):
    # puck_obs_source / vision_env are EXPLICIT named parameters on purpose: AgentBase
    # accepts **kwargs and silently ignores unknown names, so a typo'd option would
    # otherwise vanish without a trace. "vision" sources the puck measurement from the
    # side camera on the live scene; env_info cannot reach that scene (its robot_model
    # is an arm-only FK scratchpad with no table, puck or cameras), hence vision_env.
    def __init__(self, env_info, agent_id=1, only_tactic=None, puck_obs_source="gt",
                 vision_env=None, init_lift=0.06, min_goal_dist=None,
                 intercept_max_jump=None, intercept_ema_alpha=None, max_puck_speed=None,
                 lead_time=0.0, max_prediction_time=None, vel_lpf_hz=None,
                 planner="bezier", vpsto_kwargs=None, ekf_params=None,
                 home="base", home_goal_dist=None, home_x=None, home_y=None,
                 keep_mallet_flat=False,
                 base_keepout=None, ready_guard=False, ready_guard_radius=None,
                 ready_guard_drift=None, ready_t_stop_min=None,
                 ekf_own_goal=False, rendezvous_vel=None, defend_abort_dist=None,
                 defend_block=None, defend_centre_deadband=None, defend_give_up_ticks=None,
                 defend_fresh_horizon=False, defend_gate_unled=False,
                 defend_hold_target=False, workspace_clamp=False,
                 rendezvous_from_measured=False,
                 meas_lag_comp=False, meas_age_lead=False, max_plan_steps=None,
                 smash_slow=False, smash_slow_vel=None, smash_slow_ticks=None,
                 smash_slow_max_x=None,
                 defend_clear=False, defend_clear_speed=None, defend_clear_standoff=None,
                 defend_clear_margin=None, defend_clear_max_path=None,
                 defend_early_frac=None, defend_early_cos=None, **kwargs):
        super().__init__(env_info, agent_id=agent_id, only_tactic=only_tactic, **kwargs)

        T_wr = env_info['robot']['base_frame'][agent_id - 1]
        T_wv = virtual_base_frame(agent_id, T_wr[2, 3])
        self._r2v = PlanarXform(T_wv, T_wr)   # real base frame  -> virtual frame
        self._v2r = PlanarXform(T_wr, T_wv)   # virtual frame    -> real base frame

        half_length = env_info['table']['length'] / 2                 # 0.508
        half_width = env_info['table']['width'] / 2                   # 0.254
        goal_line = V_SETBACK - half_length + GOAL_LINE_INSET         # 0.183, V frame

        def own_half(frac):
            """V-frame x at `frac` of the way from the own goal line to the centreline."""
            return goal_line + frac * half_length

        # Own-goal keep-out (hardware safety). The arm stands behind its own goal at the middle
        # of the near SHORT end, so a plan that routes the mallet toward its own goal drives it
        # head-on into the near rim -- with the mallet resting on the table, that push shoves the
        # whole table. `min_goal_dist` (m) is the closest the mallet may come to its own goal
        # line; it floors the Bezier near bound (below), the hit/defend/prepare ranges and the
        # home/init pose, so nothing the planner emits reaches behind it. None (the sim default)
        # keeps the original behaviour untouched. A half-plane in x is the right SHAPE for the
        # hazard at this mount -- see `project_out_of_base` for why the base-disc form exists and
        # why it has to stay off here.
        x_floor_v = None if min_goal_dist is None else goal_line + float(min_goal_dist)

        def floor_x(x):
            return x if x_floor_v is None else max(x, x_floor_v)

        # Ranges are the baseline's own fractions of the KUKA's half ([0.8, 1.3] etc. is
        # 0.271-0.784 of it), expressed on this table in the V frame. The vendored end-mount
        # strike-distance cap is gone: reach is a disc around the base, position-only IK covers
        # the whole band, and infeasible extremes surface as optimizer failures, which every
        # tactic already retries or abandons.
        hit_range = [floor_x(own_half(0.271)), own_half(0.784)]

        # Where the mallet PARKS: the Ready tactic drives it here whenever there is nothing
        # to do, and Init puts it here at episode start.
        #
        #   "base" (default) -- the TAUGHT park (HOME_PARK_XY, a base-frame position)
        #       expressed in V: the arm idles wherever it was teleoped to and moves only to
        #       intercept. PROVISIONAL at this mount -- see HOME_PARK_XY.
        #   "goal" -- on the centreline in front of the own goal, by default ON THE DEFEND
        #       LINE, so a shot straight down the middle needs no travel at all. At the end
        #       mount that is straight out from the base at 0.25 m and a vertical wrist DOES
        #       solve there (it solves along the centreline from the goal mouth out to ~0.32 m).
        #       At the side mount it was a 0.37 m CROSS-TABLE reach with no vertical-wrist
        #       solution anywhere on the centreline, so the mallet parked tilted on the
        #       position-only null-space QP. What costs here is SPEED, not posture: park deeper
        #       than the defend line and the worst-direction mallet speed falls from 0.33 m/s
        #       toward 0.11-0.22 as the arm folds up over its own base.
        #
        #   Either mode's xy can then be overridden per axis with `home_x` / `home_y` (ROBOT BASE
        #   frame, metres) for a park anywhere on the table -- see below.
        if home not in ("base", "goal"):
            raise ValueError(f"unknown home {home!r}; expected 'base' or 'goal'")
        if home == "goal":
            # Default: the defend line itself (hit_range/defend_range share their near bound),
            # i.e. the spot the defend tactic aims at for a centred shot. An explicit
            # home_goal_dist overrides it with a distance up-table from the own goal LINE.
            x_park = (hit_range[0] if home_goal_dist is None
                      else floor_x(goal_line + float(home_goal_dist)))
            x_home = np.array([x_park, 0.0, 0.0])
        else:
            x_home = self._r2v.pos3([HOME_PARK_XY[0], HOME_PARK_XY[1], HOME_POS[2]])
        # Per-axis park OVERRIDE (run_hw_match --home-x / --home-y), in the ROBOT BASE frame --
        # the frame the operator is standing in front of, where the origin IS the arm base, so
        # `hypot(x, y)` is the reach and y=0 is straight out from the base (the default base park
        # is exactly HOME_POS.xy there). Everything downstream plans in V, so convert once, here.
        #
        # The two frames need not be aligned -- they were rotated 90 deg at the side mount, and on
        # hardware base_frame is measured, never assumed -- so a robot x on its own does not name a
        # V axis and cannot be substituted into x_home directly. To
        # keep either flag usable alone, take the mode's park INTO the robot frame, replace the
        # axis that was given, and convert the pair back: `--home-y 0.05` then slides the park
        # sideways along the arm's own y and leaves its reach-out distance alone, which is what an
        # operator nudging the park by hand means by it.
        r_m = self.env_info['mallet']['radius']
        if home_x is not None or home_y is not None:
            p_r = np.asarray(self._v2r.pos2(x_home[:2]), float)      # mode default, robot frame
            if home_x is not None:
                p_r[0] = float(home_x)
            if home_y is not None:
                p_r[1] = float(home_y)
            xy_v = np.asarray(self._r2v.pos2(p_r), float)
            # Bounds-checked HERE because nothing downstream does, and in V because that is the
            # frame the TABLE is axis-aligned in -- a robot-frame ask can leave the table on both
            # coordinates at once, so report where it landed as well as what was asked for.
            x_lo, x_hi = V_SETBACK - half_length + r_m, V_SETBACK + half_length - r_m
            y_lim = half_width - r_m
            # Check the FLOORED x: reaching back past the own-goal keep-out is a clamp (the
            # documented behaviour of every other back-reach knob), not an error. Only a park
            # the floor cannot rescue -- through a side rail, or past the far end -- is refused.
            if not (x_lo <= floor_x(xy_v[0]) <= x_hi and abs(xy_v[1]) <= y_lim):
                raise ValueError(
                    f"the requested park (robot xy {np.round(p_r, 3)} m) is off the table: it "
                    f"lands at V-xy {np.round(xy_v, 3)}, outside the play region "
                    f"(x {x_lo:.3f}..{x_hi:.3f}, |y| <= {y_lim:.3f}). home_x/home_y are METRES "
                    f"in the ROBOT BASE frame -- x out from the base, y across it, origin at the "
                    f"base itself, so the reach asked for is {np.linalg.norm(p_r):.3f} m.")
            x_home = np.array([xy_v[0], xy_v[1], x_home[2]])
        #: True when --home-x/--home-y moved the park off its mode default. run_hw_match reads it
        #: to decide whether to bring up ON the park (which needs the flat-wrist IK fallback,
        #: since an arbitrary spot has no guarantee of a vertical solution) or at the arm station.
        self.home_xy_override = home_x is not None or home_y is not None
        # Pin home to the PLAY PLANE. HOME_POS.z is the model grasp height; on hardware the
        # play plane is recalibrated to the real table-contact height (env_info
        # ee_desired_height, set by run_hw_match --ee-height). If home kept the model z the
        # arm would park with the mallet FLOATING and only the strikes would reach the table,
        # so track the plane here. In sim ee_desired_height == HOME_POS.z, so this is a no-op.
        if base_keepout:
            x_home = np.concatenate([project_out_of_base(x_home[:2], self._v2r, self._r2v,
                                                         float(base_keepout)), x_home[2:3]])
        x_home = np.array([floor_x(x_home[0]), x_home[1],
                           self.env_info['robot']['ee_desired_height']])
        #: Where the mallet parks, and how it was chosen -- read by the runtime's banner and by
        #: run_hw_match's bring-up, which does NOT follow it (see home_config there).
        self.home_mode = home

        self.agent_params.update({
            # J6 carries the wrist yaw, and this anchor is what the QP's null space relaxes
            # toward whenever there is no strike anchor (READY, DEFEND hold). Left at INIT_SEED
            # it pulls the wrist back to the un-rotated yaw the moment play starts, undoing
            # home_config. -pi/2 on J6 IS the +90 deg CCW world-z rotation: the rotated home
            # solve returns J6 = -1.5693 where the un-rotated one returns 0.
            'joint_anchor_pos': INIT_SEED + np.array([0., 0., 0., 0., 0., -np.pi / 2]),
            # x_init is the episode-start home pose. The vendored KUKA lifted its self-levelling
            # mallet clear before homing (init_lift 0.06, the sim default). With a rigid grip
            # meant to stay in contact, hardware passes init_lift=0 so the arm homes ON the
            # table and never swoops down to strike -- the only vertical move in the whole plan.
            'x_init': x_home + np.array([0., 0., init_lift]),
            'x_home': x_home,
            'hit_range': hit_range,
            'defend_range': [floor_x(own_half(0.271)), own_half(0.476)],
            'defend_width': 0.867 * half_width,
            'prepare_range': hit_range,
            # Speeds scale with the table so events happen on the same timeline.
            'max_hit_velocity': 1.2 * TABLE_SCALE,
            'default_linear_vel': 0.6 * TABLE_SCALE,
            'static_vel_threshold': 0.4 * TABLE_SCALE,
            'transversal_vel_threshold': 0.1 * TABLE_SCALE,
            # Tracker seed when vision has not seen the puck at episode start: the
            # own-half centre in frame V. Set unconditionally (the dict is shared by
            # reference with MiniSystemState); only the vision path ever reads it.
            'puck_prior': np.array([own_half(0.5), 0.0, 0.0]),
            # --- goalie knobs. Every one of these is a no-op at the value set here, so an agent
            # built without the new arguments runs the vendored path object-for-object. ---
            #: Open our own goal mouth in the EKF bounce model (see MiniAirHockeyTable).
            'ekf_own_goal': bool(ekf_own_goal),
            #: Cartesian speed budget (m/s) for the earliest-feasible rendezvous, or None for the
            #: vendored fixed-horizon roll (see MiniSystemState._rendezvous_rollout).
            'rendezvous_vel': None if rendezvous_vel is None else float(rendezvous_vel),
            #: Run the rendezvous's reachability test from the MEASURED mallet (`x_actual`)
            #: instead of the commanded one (`x_cmd`). False here keeps sim and GT
            #: byte-identical -- in sim the two barely differ; `run_hw_match` passes True,
            #: because only on hardware does the streamer's cap leave the command running
            #: ahead of the arm (DEFEND median 16.5 mm, p99 138 mm).
            'rendezvous_from_measured': bool(rendezvous_from_measured),
            #: The deep give-up / track line, split off defend_range[0] (see MiniTacticMixin).
            #: defend_range[0] keeps its plant-line job; this takes the three "when do we stop
            #: caring" roles. None -> defend_range[0] does both, as vendored.
            'defend_abort_line': (None if defend_abort_dist is None
                                  else floor_x(goal_line + float(defend_abort_dist))),
            #: Where the mallet plants relative to the predicted crossing (see _block_target).
            'defend_block_offset': 'mallet' if defend_block is None else str(defend_block),
            'defend_centre_deadband': (0.0 if defend_centre_deadband is None
                                       else float(defend_centre_deadband)),
            'defend_fresh_horizon': bool(defend_fresh_horizon),
            'defend_give_up_ticks': (4 if defend_give_up_ticks is None
                                     else int(defend_give_up_ticks)),
            'defend_gate_unled': bool(defend_gate_unled),
            #: Keep blocking the last intercept that passed DEFEND's own gate, instead of
            #: following the estimate out of the defend zone (see MiniDefend._defend_target).
            'defend_hold_target': bool(defend_hold_target),
            #: Minimum mallet reach from the arm's OWN base (m), or None. Replaces the own-goal
            #: x half-plane with the shape the hazard actually has -- see project_out_of_base.
            'base_keepout_radius': None if base_keepout is None else float(base_keepout),
            #: Hold the mallet on the puck -> own-goal line while idle, instead of parking on a
            #: fixed spot -- see MiniReady. This arm cannot swing its way out of trouble
            #: (0.12-0.4 m/s against 1-3 m/s pucks), so standing in the right place is the only
            #: defence that scales.
            'ready_guard': bool(ready_guard),
            'ready_guard_radius': (0.10 if ready_guard_radius is None
                                   else float(ready_guard_radius)),
            'ready_guard_drift': (0.02 if ready_guard_drift is None
                                  else float(ready_guard_drift)),
            #: Floor on the READY glide duration. 1.0 is the vendored value; drop it only WITH
            #: the guard -- a faster retreat is worse than a slow one.
            'ready_t_stop_min': (1.0 if ready_t_stop_min is None else float(ready_t_stop_min)),
            #: Strike a puck that is SLOW rather than dead still, once it has settled in our own
            #: half. Off -> `can_smash` is the vendored one, so offence is unchanged.
            'smash_slow': bool(smash_slow),
            #: Speed (m/s) below which the puck counts as slow. Only does anything ABOVE
            #: `static_vel_threshold`: under it the vendored static test already fires. 0.3 clears
            #: the SIM's own 0.202 as well as the rig's 0.02, so the branch is live in both.
            'smash_slow_vel': (0.3 if smash_slow_vel is None else float(smash_slow_vel)),
            #: Consecutive slow ticks before a smash may fire (50 Hz, so 5 = 0.1 s). Shorter than
            #: `switch_tactics_min_steps` (15), which is what usually paces entry into SMASH.
            'smash_slow_ticks': (5 if smash_slow_ticks is None else int(smash_slow_ticks)),
            #: Up-table bound on the band. 0.4 reaches ~45 mm PAST the defend plant line
            #: (own_half(0.476) = 0.355) while staying inside the vendored hit range (0.511).
            #: Floored by the own-goal keep-out.
            'smash_slow_max_x': (floor_x(0.4) if smash_slow_max_x is None
                                 else floor_x(float(smash_slow_max_x))),
            #: Plant THROUGH the predicted crossing, up-table, instead of tapping sideways at
            #: 0.05 m/s -- see MiniDefend._clear_target. Off -> the vendored block, unchanged.
            'defend_clear': bool(defend_clear),
            #: Terminal mallet speed (m/s) along +x at the crossing. 0.25 matches --goalie's own
            #: rendezvous budget, the honest end-mount number; it is cheap because peak path
            #: speed is set by the STANDOFF, not by this (measured identical across 0.15-0.30).
            'defend_clear_speed': (0.25 if defend_clear_speed is None
                                   else float(defend_clear_speed)),
            #: Nominal run-up (m) down-table of the crossing. THIS is the safety knob: measured
            #: peak path speed 0.75 / 1.19 / 1.64 m/s at 0.04 / 0.06 / 0.08 against a 1 m/s puck.
            #: Bounded at use time by the Bezier routing floor plus the margin below.
            'defend_clear_standoff': (0.05 if defend_clear_standoff is None
                                      else float(defend_clear_standoff)),
            #: Keep the cock this far (m) above the routing floor.
            'defend_clear_margin': (0.02 if defend_clear_margin is None
                                    else float(defend_clear_margin)),
            #: Mean path speed (m/s) a clearing stroke may demand of the arm. 0 disables the
            #: check. Measured: without it the commit asked 3.58 m/s and delivered 0.03.
            'defend_clear_max_path': (0.25 if defend_clear_max_path is None
                                      else float(defend_clear_max_path)),
            #: Meet a near-head-on puck at this fraction of the own half instead of at the plant
            #: line (same convention as --defend-line-frac). None -> the vendored line, always.
            'defend_early_frac': (None if defend_early_frac is None
                                  else float(defend_early_frac)),
            #: |vx|/|v| above which a puck counts as near-head-on. 0.90 is about 26 degrees.
            'defend_early_cos': (0.90 if defend_early_cos is None else float(defend_early_cos)),
        })

        # Optional longer prediction horizon (default keeps the vendored 1.0 s). Larger lets the
        # arm commit to a farther-future intercept for fast/far pucks; safe to raise only once the
        # motion is feasibility-aware (Stage 3 VPSTO) -- otherwise the arm over-commits to a spot
        # it cannot reach in time.
        if max_prediction_time is not None:
            self.agent_params['max_prediction_time'] = float(max_prediction_time)
        # Replan cadence (Fix 8e). The tactics re-aim once per max_plan_steps ticks, so the
        # vendored 5 refreshes the target every 100 ms -- a 2 m/s puck moves 200 mm in that
        # time. Lowering it trades draw time for freshness; it is a knob rather than a new
        # default because the extra replans land on the same CPU the vision thread needs.
        if max_plan_steps is not None:
            self.agent_params['max_plan_steps'] = int(max_plan_steps)
        self.state = MiniSystemState(self.env_info, agent_id, self.agent_params, self._r2v,
                                     intercept_max_jump=intercept_max_jump,
                                     intercept_ema_alpha=intercept_ema_alpha,
                                     max_puck_speed=max_puck_speed,
                                     lead_time=lead_time, vel_lpf_hz=vel_lpf_hz,
                                     ekf_params=ekf_params,
                                     workspace_clamp=workspace_clamp,
                                     meas_lag_comp=meas_lag_comp,
                                     meas_age_lead=meas_age_lead)
        self.traj_generator = MiniTrajectoryGenerator(self.env_info, self.agent_params,
                                                      self.state, self._v2r, min_x_v=x_floor_v,
                                                      r2v=self._r2v,
                                                      use_vpsto=(planner == "vpsto"),
                                                      vpsto_kwargs=vpsto_kwargs,
                                                      keep_mallet_flat=keep_mallet_flat)
        # 6-DOF arm with ~1 DOF of redundancy: the null space is essentially one-dimensional,
        # so there is no structure for the KUKA's [10,1,10,1,10,10,1] pattern to express.
        self.traj_generator.optimizer.anchor_weights = np.ones(N_JOINTS)

        args = (self.env_info, self.agent_params, self.state, self.traj_generator)
        self.tactics_processor = [Init(*args),
                                  MiniReady(*args, only_tactic=only_tactic),
                                  MiniPrepare(*args),
                                  MiniDefend(*args),
                                  Repel(*args),
                                  MiniSmash(*args)]

        # Vision wiring LAST: __init__ rebuilds self.state above, so anything placed
        # earlier would be wired to a discarded SystemState.
        if puck_obs_source not in ("gt", "vision"):
            raise ValueError(f"unknown puck_obs_source {puck_obs_source!r}")
        if puck_obs_source == "vision" and vision_env is None:
            raise ValueError("puck_obs_source='vision' requires vision_env "
                             "(pass env.base_env)")
        self._puck_obs_source = puck_obs_source
        self._last_measurement = None
        #: Age (s) of the measurement fused on the last tick, or None. Logged per tick (Fix 8c):
        #: `frame_age` says how old the camera's NEWEST frame was when the telemetry row was
        #: written, which is a different question from how stale the thing the EKF actually
        #: fused was -- and it is the latter that moves the estimate.
        self._last_meas_age = None
        self._sensor = (PuckVisionSensor(vision_env._model, vision_env._data)
                        if puck_obs_source == "vision" else None)

    # The env reports the puck in the real base frame; the agent thinks in V. These two
    # accessors are the observation-side frame crossing.
    def get_puck_pos(self, obs):
        p = super().get_puck_pos(obs)
        return np.concatenate([self._r2v.pos2(p), [self._r2v.angle(p[2])]])

    def get_puck_vel(self, obs):
        v = super().get_puck_vel(obs)
        return np.concatenate([self._r2v.vec2(v), v[2:3]])

    def get_puck_measurement(self, obs):
        """Puck measurement for the EKF: GT observation, or the camera sensor.

        A camera measurement is born in the WORLD frame, so it must travel W -> R
        (`AirHockeyBase._puck_2d_in_robot_frame`, the same helper the env's own
        observation path uses) -> V (`self._r2v`), yaw included -- converting the
        world yaw with `self._r2v.angle` alone would be off by the base rotation
        (+90 deg for agent 1); `test_yaw_frame_composition` pins this. The covariance
        rotates by the planar composite of the two rotations (a planar rotation
        leaves the yaw variance untouched).
        """
        if self._puck_obs_source == "gt":
            return super().get_puck_measurement(obs)
        m = self._sensor.measure()
        self._last_measurement = m   # eval tooling reads this instead of re-rendering
        # How stale this measurement is by the time the filter sees it: the camera's own capture
        # stamp against now, so it covers the frame period, the detector and the vision thread's
        # hand-off in one number (Fix 8c). Parked on the state for `update_observation`, which
        # the vendored draw_action calls on the very next line.
        age = None if (m is None or m.stamp is None) else max(0.0, time.perf_counter() - m.stamp)
        self._last_meas_age = age
        self.state._pending_meas_age = age
        if m is None:
            return None, None
        bf = self.env_info['robot']['base_frame'][self.agent_id - 1]
        pose_r = AirHockeyBase._puck_2d_in_robot_frame(
            np.array([m.pos_xy[0], m.pos_xy[1], 0.0 if m.yaw is None else m.yaw]), bf)
        pos_v = self._r2v.pos2(pose_r)
        G = self._r2v.rot2 @ bf[:2, :2].T
        R_v = G @ m.R_xy @ G.T
        if m.yaw is None:
            return pos_v, R_v                                      # 2-D measurement
        R3 = np.zeros((3, 3))
        R3[:2, :2] = R_v
        R3[2, 2] = m.R_yaw
        return np.array([*pos_v, self._r2v.angle(pose_r[2])]), R3  # 3-D measurement


def build_agent(env_info, **kwargs):
    if "hit" in env_info["env_name"]:
        return MiniBaselineAgent(env_info, **kwargs, agent_id=1)
    if "defend" in env_info["env_name"]:
        return MiniBaselineAgent(env_info, **kwargs, agent_id=1, only_tactic="defend")
    if "prepare" in env_info["env_name"]:
        return MiniBaselineAgent(env_info, **kwargs, agent_id=1, only_tactic="prepare")
    return MiniBaselineAgent(env_info, **kwargs)
