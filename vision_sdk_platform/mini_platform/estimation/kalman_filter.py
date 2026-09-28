# Vendored from the 2025 Robot Air Hockey Challenge (baseline/baseline_agent/kalman_filter.py).
# MIT License, Copyright (c) 2025 Atalay Donat, Puze Liu, Jonas Guenster, Davide Tateo.
# See ../../THIRD_PARTY.md. Modifications for the mini platform are marked MINI:.

import numpy as np
from scipy.spatial.transform import Rotation
# b_params = np.array([8.40683102e-01, 1.42380669e-01, 7.71445220e-04])
b_params = np.array([8.40683102e-01, 0, 7.71445220e-04])
n_params = np.array([0., -0.79, 0.])
theta_params = np.array([-6.48073315, 6.32545305, 0.8386719])
damping = np.array([0.2125, 0.2562])
LIN_COV = np.diag([8.90797655e-07, 5.49874493e-07, 2.54163138e-04, 3.80228296e-04, 7.19007035e-02, 1.58019149e+00])
COL_COV = \
    np.array([[0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0.],
              [0., 0., 2.09562546e-01, 3.46276805e-02, 0., -1.03489604e+00],
              [0., 0., 3.46276805e-02, 9.41218351e-02, 0., -1.67029496e+00],
              [0., 0., 0., 0., 0., 0.],
              [0., 0., -1.03489604e+00, -1.67029496e+00, 0., 1.78037877e+02]])

OBS_COV = np.diag([5.0650402e-07, 8.3995428e-07, 1.6572967e-03])


# MINI: --------------------------------------------------------- refit-constant load hook
#
# Everything above this line is a fit to the FULL-SIZE KUKA Air-Hockey Challenge table: a
# different rim material, a different air film, a different camera. On the mini table those
# constants bias `get_prediction`, and the bias compounds with travel time -- which is a
# direct contributor to "the arm is late when the puck is far". `EkfParams` lets a *measured*
# set (produced by `tools/fit_mini_table.py` from real rolls) be loaded at construction:
#
#     run_hw_match --ekf-params mini_table_ekf.npz
#
# The default is `DEFAULT_PARAMS`, which holds THE module-level arrays above (not copies), so
# every unflagged run -- sim, tests, the GT/bitwise regression -- executes the identical
# arithmetic on the identical objects. The refit is opt-in and reversible by dropping a flag.
class EkfParams:
    """The fitted puck-motion constants `SystemModel` / `PuckTracker` run on.

    Attributes mirror the module constants one-for-one:

    ``damping``       (2,)   per-axis linear velocity decay, 1/s (``F_linear[2, 2]``/``[3, 3]``)
    ``b_params``      (3,)   rim-local collision row for the TANGENTIAL velocity, vs [vt, vn, w]
    ``n_params``      (3,)   ... for the NORMAL velocity; ``n_params[1] = -restitution``
    ``theta_params``  (3,)   ... for the spin
    ``OBS_COV``       (3, 3) measurement noise for [x, y, yaw]
    ``LIN_COV``       (6, 6) free-flight process noise
    ``COL_COV``       (6, 6) collision process noise

    NB on the rim-local convention (it is easy to get backwards, and `local_rim_transform`
    is what settles it): rim 0 is the -y long rail and its transform is the identity, so in
    the rim-local frame index 2 is the **tangential** velocity and index 3 the **normal**
    one. `_jac_local_collision` row 2 is therefore the tangential row (`b_params`, retention
    ~0.84) and row 3 the normal row (`n_params`, whose [1] entry -0.79 is minus the
    coefficient of restitution). `tools/fit_mini_table.py` fits them under that convention.
    """

    #: npz keys read by :meth:`load` / written by :meth:`save`, in the order printed.
    KEYS = ("damping", "b_params", "n_params", "theta_params", "OBS_COV", "LIN_COV", "COL_COV")
    _SHAPES = {"damping": (2,), "b_params": (3,), "n_params": (3,), "theta_params": (3,),
               "OBS_COV": (3, 3), "LIN_COV": (6, 6), "COL_COV": (6, 6)}

    def __init__(self, damping=damping, b_params=b_params, n_params=n_params,
                 theta_params=theta_params, OBS_COV=OBS_COV, LIN_COV=LIN_COV, COL_COV=COL_COV,
                 source=None):
        self.damping = damping
        self.b_params = b_params
        self.n_params = n_params
        self.theta_params = theta_params
        self.OBS_COV = OBS_COV
        self.LIN_COV = LIN_COV
        self.COL_COV = COL_COV
        #: Where these came from (a path, or None for the vendored defaults) -- printed by
        #: the runtime so a log always says which constants a run used.
        self.source = source
        #: Metadata, not a filter constant: the centroid noise (px) the fitter measured on the
        #: same static frames. On hardware THAT is what sets the per-measurement R (the sensor
        #: builds it from sigma_px), so the runtime prints it as a reminder to pass --sigma-px.
        self.sigma_px = None

    @classmethod
    def load(cls, path):
        """Read an npz written by ``tools/fit_mini_table.py`` (or :meth:`save`).

        **Partial files are the normal case** and are the point of the design: the fitter
        writes only what the captured rolls could actually measure (e.g. no yaw marker ->
        no spin fit), and every absent key falls back to the vendored default above, so a
        partial refit is still a strictly better filter than none. Unknown keys (metadata
        the fitter records alongside) are ignored.
        """
        d = np.load(path, allow_pickle=False)
        kw = {}
        for k in cls.KEYS:
            if k not in d.files:
                continue
            a = np.asarray(d[k], dtype=float)
            if a.shape != cls._SHAPES[k]:
                raise ValueError(f"{path}: '{k}' has shape {a.shape}, expected {cls._SHAPES[k]}")
            if not np.all(np.isfinite(a)):
                # A NaN here would propagate silently into F_linear and poison every state the
                # filter ever produces -- on the ARM. Refuse the file instead.
                raise ValueError(f"{path}: '{k}' contains non-finite values ({a}) -- the fit "
                                 f"did not converge; re-run tools.fit_mini_table and read its "
                                 f"report before using this file")
            kw[k] = a
        if not kw:
            raise ValueError(f"{path}: none of {cls.KEYS} present -- not an EKF-params file")
        out = cls(source=str(path), **kw)
        if "sigma_px" in d.files:
            sp = float(np.asarray(d["sigma_px"]).reshape(-1)[0])
            out.sigma_px = sp if np.isfinite(sp) else None
        return out

    def save(self, path, **meta):
        """Write all seven constants (plus any metadata) to ``path``."""
        np.savez(path, **{k: np.asarray(getattr(self, k), dtype=float) for k in self.KEYS},
                 **meta)

    def diff_from_default(self, rtol=0.0, atol=0.0):
        """Names of the constants that differ from the vendored (module-level) values."""
        return [k for k in self.KEYS
                if not np.allclose(getattr(self, k), getattr(DEFAULT_PARAMS, k),
                                   rtol=rtol, atol=atol)]

    def describe(self):
        """One-line-per-changed-constant summary, for the runtime banner."""
        changed = self.diff_from_default()
        if not changed:
            return "EKF constants: vendored KUKA defaults (nothing overridden)"
        lines = [f"EKF constants from {self.source} -- overriding {', '.join(changed)}:"]
        for k in changed:
            new = np.asarray(getattr(self, k), dtype=float)
            old = np.asarray(getattr(DEFAULT_PARAMS, k), dtype=float)
            if new.ndim == 2:                       # covariances: compare the diagonals
                new, old = np.diag(new), np.diag(old)
                k = f"diag({k})"
            lines.append(f"    {k:16s} {np.array2string(old, precision=4)} -> "
                         f"{np.array2string(new, precision=4)}")
        return "\n".join(lines)


#: The vendored constants, wrapped. Holds the module arrays themselves, so the default path
#: is arithmetically and object-identically what it was before the hook existed.
DEFAULT_PARAMS = EkfParams()


class SystemModel:
    def __init__(self, env_info, agent_id, ekf_params=None):
        self.puck_radius = env_info['puck']['radius']
        self.mallet_radius = env_info['mallet']['radius']
        self.dt = env_info['dt']
        # MINI: fitted constants (see EkfParams); None -> the vendored module values.
        self.params = DEFAULT_PARAMS if ekf_params is None else ekf_params

        self.table = AirHockeyTable(env_info['table']['length'], env_info['table']['width'],
                                    env_info['table']['goal_width'], env_info['puck']['radius'],
                                    abs(env_info['robot']['base_frame'][agent_id - 1][0, 3]), env_info['dt'],
                                    ekf_params=self.params)
        self.F = np.eye(6)
        self.F_linear = np.eye(6)
        self.F_linear[0, 2] = self.F_linear[1, 3] = self.F_linear[4, 5] = self.dt
        self.F_linear[2, 2] = 1 - self.dt * self.params.damping[0]
        self.F_linear[3, 3] = 1 - self.dt * self.params.damping[1]
        self.Q_collision = np.zeros((6, 6))
        self.has_collision = False
        self.outside_boundary = False
        self.score = False

    def f(self, x):
        self.has_collision, self.outside_boundary, self.score, F, Q = self.table.check_collision(x)
        if self.has_collision:
            # Collision Dynamics
            self.F = F
            self.Q_collision = Q
        elif self.outside_boundary or self.score:
            # Stop Moving
            self.F = np.eye(6)
            self.Q_collision = np.zeros((6, 6))
        else:
            # Normal Prediction
            self.F = self.F_linear
        x = self.F @ x
        x[4] = (x[4] + np.pi) % (np.pi * 2) - np.pi
        return x


class AirHockeyTable:
    def __init__(self, length, width, goal_width, puck_radius, x_offset, dt, ekf_params=None):
        self.table_length = length
        self.table_width = width
        self.goal_width = goal_width
        self.puck_radius = puck_radius
        self.x_offset = x_offset
        self.dt = dt
        # MINI: fitted constants (see EkfParams); None -> the vendored module values.
        self.params = DEFAULT_PARAMS if ekf_params is None else ekf_params
        self.col_cov = self.params.COL_COV

        pos_offset = np.array([x_offset, 0])
        p1 = np.array([-length / 2 + puck_radius, -width / 2 + puck_radius]) + pos_offset
        p2 = np.array([length / 2 - puck_radius, -width / 2 + puck_radius]) + pos_offset
        p3 = np.array([length / 2 - puck_radius, width / 2 - puck_radius]) + pos_offset
        p4 = np.array([-length / 2 + puck_radius, width / 2 - puck_radius]) + pos_offset

        self.boundary = np.array([[p1, p2],
                                  [p2, p3],
                                  [p3, p4],
                                  [p4, p1]])

        self.local_rim_transform = np.zeros((4, 6, 6))
        self.local_rim_transform_inv = np.zeros((4, 6, 6))
        transform_tmp = np.eye(6)
        self.local_rim_transform[0] = transform_tmp.copy()
        self.local_rim_transform_inv[0] = transform_tmp.T.copy()

        transform_tmp = np.zeros((6, 6))
        transform_tmp[0, 1] = transform_tmp[2, 3] = transform_tmp[4, 4] = transform_tmp[5, 5] = 1
        transform_tmp[1, 0] = transform_tmp[3, 2] = -1
        self.local_rim_transform[1] = transform_tmp.copy()
        self.local_rim_transform_inv[1] = transform_tmp.T.copy()

        transform_tmp = np.eye(6)
        transform_tmp[0, 0] = transform_tmp[1, 1] = transform_tmp[2, 2] = transform_tmp[3, 3] = -1
        self.local_rim_transform[2] = transform_tmp.copy()
        self.local_rim_transform_inv[2] = transform_tmp.T.copy()

        transform_tmp = np.zeros((6, 6))
        transform_tmp[1, 0] = transform_tmp[3, 2] = transform_tmp[4, 4] = transform_tmp[5, 5] = 1
        transform_tmp[0, 1] = transform_tmp[2, 3] = -1
        self.local_rim_transform[3] = transform_tmp.copy()
        self.local_rim_transform_inv[3] = transform_tmp.T.copy()

        self._F_precollision = np.eye(6)
        self._F_postcollision = np.eye(6)
        self._jac_local_collision = np.eye(6)
        self._jac_local_collision[2, [2, 3, 5]] = self.params.b_params[0:3]
        self._jac_local_collision[3, [2, 3, 5]] = self.params.n_params[0:3]
        self._jac_local_collision[5, [2, 3, 5]] = self.params.theta_params[0:3]

    def check_collision(self, state):
        score = False
        outside_boundary = False
        collision = False

        u = state[2:4] * self.dt
        if np.abs(state[1]) < self.goal_width / 2:
            if state[0] + u[0] < -self.boundary[0, 0, 0] or state[0] + u[0] > self.boundary[0, 1, 0]:
                score = True
        elif np.any(state[:2] < self.boundary[0, 0]) or np.any(state[:2] > self.boundary[1, 1]):
            outside_boundary = True

        if not score and not outside_boundary:
            F, Q_collision, collision = self._check_collision_impl(state, u)

        else:
            F = np.eye(4)
            Q_collision = np.zeros((4, 6))
        return collision, outside_boundary, score, F, Q_collision

    def _cross_2d(self, u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    def _check_collision_impl(self, state, u):
        F = np.eye(4)
        Q_collision = np.zeros((4, 6))
        v = self.boundary[:, 1] - self.boundary[:, 0]
        w = self.boundary[:, 0] - state[:2]
        denominator = self._cross_2d(v, u)
        s = self._cross_2d(v, w) / (denominator + 1e-6)
        r = self._cross_2d(u, w) / (denominator + 1e-6)
        collide_idx = np.where(np.logical_and(np.logical_and(1e-6 < s, s < 1 - 1e-6), np.logical_and(1e-6 < r, r < 1 - 1e-6)))[0]
        collision = False

        if len(collide_idx) > 0:
            collision = True
            collide_rim_idx = collide_idx[0]
            s_i = s[collide_rim_idx]
            self._F_precollision[0][2] = self._F_precollision[1][3] = self._F_precollision[4][5] = s_i * self.dt
            self._F_postcollision[0][2] = self._F_postcollision[1][3] = self._F_postcollision[4][5] = (1 - s_i) * self.dt
            state_local = self.local_rim_transform[collide_rim_idx] @ state
            # Compute the slide direction
            slide_dir = 1 if state_local[2] + state_local[5] * self.puck_radius >= 0 else -1

            jac_local_collision = self._jac_local_collision.copy()
            jac_local_collision[2, 3] *= slide_dir
            jac_local_collision[5, 3] *= slide_dir

            F_collision = self.local_rim_transform_inv[collide_rim_idx] @ jac_local_collision @ self.local_rim_transform[collide_rim_idx]
            F = self._F_postcollision @ F_collision @ self._F_precollision
            Q_collision = self.local_rim_transform_inv[collide_rim_idx] @ self.col_cov @ self.local_rim_transform_inv[collide_rim_idx].T
        return F, Q_collision, collision


class PuckTracker:
    def __init__(self, env_info, agent_id=1, max_puck_speed=None, vel_lpf_hz=None,
                 ekf_params=None, meas_lag_comp=False):
        # MINI: `ekf_params` (an EkfParams, or None for the vendored KUKA fits) swaps in
        # constants measured on THIS table -- see EkfParams and tools/fit_mini_table.py.
        self.params = DEFAULT_PARAMS if ekf_params is None else ekf_params
        self.system = SystemModel(env_info, agent_id, ekf_params=self.params)
        self.Q = self.params.LIN_COV

        # MINI: optional physical cap (m/s) on the estimated puck speed, applied ONLY on the
        # gated (vision) update path. A detection outlier (e.g. the mallet momentarily detected
        # as the puck) makes the NIS relock set velocity = (meas-prev)/dt, which reads 40-60 m/s
        # on hardware and blows up the intercept. None (the sim/GT default) leaves the filter
        # untouched, so the R=None ground-truth path stays bitwise-identical.
        self._max_puck_speed = max_puck_speed

        # MINI: optional first-order low-pass (cutoff Hz) on the EKF OUTPUT velocity (state[2:4]),
        # applied once per step() after the update. get_prediction forward-integrates this
        # velocity into the intercept, so residual measurement noise / sign-flips here render as
        # intercept jitter -> joint jerk. A light LPF steadies it. None (default) => no filtering,
        # so sim / GT-regression stays bitwise-identical. Prefer a LIGHT cutoff (8-15 Hz): too low
        # lags the velocity and makes the arm react late.
        self._vel_lpf_hz = vel_lpf_hz
        self._vel_lpf = None            # filtered velocity state (None until first sample)

        self.R = self.params.OBS_COV
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 4] = 1

        self.state = None
        self.P = None
        # MINI: consecutive NIS-gate rejections (Part 3 vision work); at 5 the filter
        # declares divergence and re-locks on the raw measurement. _last_rejected
        # holds the previous step's rejected xy for the two-point consistency re-lock.
        self._reject_count = 0
        self._last_rejected = None
        # MINI: ticks elapsed since _last_rejected was recorded. The two-point re-lock differences
        # two REJECTED MEASUREMENTS, which are one CAMERA frame apart, not one control tick: the
        # vision thread coasts between frames, so at 11 detections/s they are ~4 ticks apart.
        # Dividing by system.dt regardless inflated the re-lock velocity by that factor (then
        # _clamp_speed saturated it at max_puck_speed), which reads as the arm over-leading.
        self._last_rejected_age = 0

        # MINI (Fix 8c): compensate the measurement's AGE on the gated (vision) path. A camera
        # measurement reaches the filter a camera period + detection + hand-off after the photons
        # landed -- detections arrive 40-60 ms apart on this rig -- and fusing it against the
        # CURRENT prediction asserts the puck is there NOW. At 2 m/s, 50 ms of age is 100 mm of
        # position error pulled straight into the estimate, and `get_prediction` then
        # forward-integrates that error into the intercept the arm is sent to.
        #
        # When enabled AND the caller supplies an age, the innovation is taken against the state
        # as it was when the frame was CAPTURED, and the correction is applied to the current
        # state. Off by default, so the sim and GT paths stay bitwise-identical.
        #
        # EXPERIMENTAL -- measured WORSE than leaving it off (tests/test_latency.py). Two faults,
        # both inherent to the approximation rather than to this code:
        #   * it is circular. The reference is this filter's own past estimate, so it is only the
        #     state "at capture time" if the filter already tracks NOW. Fed stale frames from a
        #     cold start the filter first settles onto the LAGGED track; its history is then
        #     stale by 2*lag, the innovation sits at a permanent +1 lag, and it never recovers.
        #   * a reference that straddles a rim bounce is on the wrong side of a discontinuity, so
        #     the innovation explodes, the NIS gate rejects, and the filter re-locks with a
        #     garbage velocity -- exactly at a bounce, where defence needs it most.
        # Measured on a 3-tick lag: 124 mm mean error against 50 mm with it off.
        # Prefer `meas_age_lead` (MiniSystemState), which leads the TARGET by the measured age
        # instead and goes through the existing bounce-aware prediction path.
        self._meas_lag_comp = bool(meas_lag_comp)
        self._history = []              # recent posterior states, newest last (see MAX_LAG_TICKS)

    def reset(self, puck_pos):
        self.P = np.eye(6)
        self.state = np.zeros(6)
        self.state[[0, 1, 4]] = puck_pos
        # MINI: a reset is a re-lock -- rejection history does not carry across it.
        self._reject_count = 0
        self._last_rejected = None
        self._last_rejected_age = 0
        # A re-lock is a velocity discontinuity; drop the LPF history so it does not drag the
        # fresh velocity back toward the pre-relock value.
        self._vel_lpf = None
        # ...and the lag history, for the same reason: states from before a re-lock describe a
        # different track, so comparing a new measurement against them is worse than not
        # compensating at all.
        self._history = []

    def _clamp_speed(self):
        """MINI: cap |velocity| to the physical max on the gated (vision) path (no-op if unset).
        Bounds the (meas-prev)/dt relock and any gain-driven spike so a detection outlier cannot
        explode the intercept."""
        if self._max_puck_speed is None or self.state is None:
            return
        sp = float(np.linalg.norm(self.state[2:4]))
        if sp > self._max_puck_speed:
            self.state[2:4] = self.state[2:4] * (self._max_puck_speed / sp)

    def predict(self, state, P):
        predicted_state = self.system.f(state)
        if self.system.has_collision:
            Q = self.system.Q_collision
        elif self.system.outside_boundary or self.system.score:
            Q = self.system.Q_collision
        else:
            Q = self.Q
        P = self.system.F @ P @ self.system.F.T + Q
        return predicted_state, P

    # MINI: variable-dimension innovation (Part 3): a 2-D measurement (vision saw the
    # puck but not the marker) updates xy only; the 3-D branch is the original
    # xy + wrapped-yaw computation moved verbatim.
    def _innovation(self, measurement, predicted_state):
        if len(measurement) == 2:
            return self.H[:2], measurement[:2] - predicted_state[:2]
        xy_innovation = measurement[:2] - predicted_state[:2]
        theta_innovation = (measurement[2] - predicted_state[4] + np.pi) % (np.pi * 2) - np.pi
        y = np.concatenate([xy_innovation, [theta_innovation]])
        return self.H, y

    # MINI: accepts 2-D measurements and an optional per-measurement R (the vision
    # sensor's pose-dependent covariance). The 3-D/R=None path is operand-for-operand
    # the original update -- np.linalg.inv stays -- which the bitwise GT-regression
    # pin in tests/test_kalman_variable_dim.py depends on.
    def update(self, measurement, predicted_state, P, R=None, ref_state=None):
        # MINI (8c): `ref_state` is the state the INNOVATION is measured against when it differs
        # from the one being corrected -- the puck as it was when the frame was captured. The
        # gain and the correction still apply to `predicted_state`. None (every vendored caller,
        # and the whole GT path) leaves this operand-for-operand the original update.
        R = self.R if R is None else R
        H, y = self._innovation(measurement,
                                predicted_state if ref_state is None else ref_state)
        if len(measurement) == 2:
            R = R[:2, :2]
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        state = predicted_state + K @ y
        P = (np.eye(6) - K @ H) @ P
        return state, P

    # MINI: how close (m) a measurement must be to a rim line for the NIS gate to be
    # bypassed, and how close two consecutively-rejected measurements must be to count
    # as one consistent new track. Both from the closed-loop error decomposition (20
    # eps x mini-hit/mini-defend): the >10 mm EKF-error tail splits roughly half
    # rim-adjacent (bounce mistimed by one step -> innovation ~2v*dt >> OBS_COV ->
    # NIS in the hundreds) and half mid-table mallet strikes; a one-step-mistimed
    # 1.6 m/s puck sits within ~32 mm of the rim. 0.08 m is the max plausible
    # per-step travel (~3.5 m/s * 20 ms) plus noise.
    RIM_MARGIN = 0.05
    RELOCK_DIST = 0.08
    # MINI: how long a rejected measurement stays eligible as the first half of a two-point
    # re-lock. It must span at least one CAMERA frame -- at 11 detections/s on the 50 Hz loop that
    # is ~4.5 ticks -- or the re-lock can never fire on the vision path. 10 ticks = 200 ms.
    RELOCK_MAX_AGE = 10
    # MINI: how far outside the puck-CENTRE box a measurement may still be taken seriously (m).
    # Sized for calibration error, not for physics: the detector reports a centre and the box is
    # the rim inset by a puck radius, so anything much past it is not a puck on this table.
    # 15 mm is several times the residual of the teleop calibration.
    OFF_TABLE_MARGIN = 0.015
    # MINI: how many ticks of state history the age compensation keeps. 10 ticks = 200 ms at the
    # 50 Hz loop, several times the worst frame age measured; an age beyond that is not
    # compensated at all, rather than compared against a state from the wrong time.
    MAX_LAG_TICKS = 10

    def _box(self):
        """(x_lo, x_hi, y_lo, y_hi) of the puck-CENTRE box (the rim inset by a puck radius)."""
        b = self.system.table.boundary
        return (float(b[0, 0, 0]), float(b[0, 1, 0]),
                float(b[0, 0, 1]), float(b[1, 1, 1]))

    def _outside_box(self, xy):
        """Outside the puck-centre box by more than OFF_TABLE_MARGIN.

        Used by :meth:`_near_rim` alone, and deliberately a STRICT box test with no goal-mouth
        exception: the rim bypass exists for a bounce the model mistimed by a step, and such a
        puck is ON the table. A point beyond the box has no claim on the bypass whether or not
        it happens to line up with a goal mouth -- and the phantom that motivated this sits
        within a couple of millimetres of the far mouth's edge, so a mouth-aware test here
        would hand it back the free pass it was skipping the gate with.

        The margin is sized for calibration error, not physics: the detector reports a centre,
        the box is the rim inset by a puck radius, so 15 mm is several times the residual of
        the teleop calibration.
        """
        x_lo, x_hi, y_lo, y_hi = self._box()
        m = self.OFF_TABLE_MARGIN
        return not (x_lo - m <= xy[0] <= x_hi + m and y_lo - m <= xy[1] <= y_hi + m)

    def _near_rim(self, xy):
        """Within RIM_MARGIN of one of the table's (puck-radius-inset) rim lines.

        ONE-SIDED (MINI): a point that is off the table does not get the bypass. The distance
        here is absolute, so before this an outlier 36 mm PAST the far rim measured as "near"
        it and skipped the NIS gate altogether -- which is how one frame of the object behind
        the far end moved the estimate 306 mm. The bypass exists for a bounce the model
        mistimed by a step, and such a puck is inside the box or within the tolerance of it.
        """
        if self._outside_box(xy):
            return False
        b = self.system.table.boundary
        x_lines = (b[0, 0, 0], b[0, 1, 0])
        y_lines = (b[0, 0, 1], b[1, 1, 1])
        return (min(abs(xy[0] - v) for v in x_lines) < self.RIM_MARGIN
                or min(abs(xy[1] - v) for v in y_lines) < self.RIM_MARGIN)

    # MINI: dropout handling + 99.9% chi-square NIS gate (Part 3/4).
    #
    # The gate applies ONLY to measurements that carry their own covariance -- i.e. a
    # fallible sensor. R=None is the trusted ground-truth channel and takes the
    # original unconditional update, so the GT path is byte-identical by construction.
    # This is a measured necessity, not caution: with the tiny fitted OBS_COV a GT
    # measurement one step after an unmodeled bounce carries NIS in the thousands --
    # instrumented at 89 rejections + 21 forced re-locks over 20 GT mini-defend
    # episodes -- so gating GT would (and did) change ground-truth behaviour, which
    # the Part 3 acceptance forbids. COL_COV inflates S only when the filter predicts
    # the bounce on the right step, which is exactly what it cannot do at contacts.
    #
    # For gated (vision) measurements, two escape hatches keep model surprises from
    # turning into multi-step predict-only drift (measured: they dominated the EKF
    # error tail, p99 24-74 mm):
    #   * near a rim, a huge innovation is bounce-timing model error, not a sensor
    #     fault -- accept unconditionally;
    #   * a SECOND consecutive rejection consistent with the first (within
    #     RELOCK_DIST) is a real new track (e.g. post-strike reversal), not a glitch
    #     -- glitches do not repeat consistently -- so re-lock on it immediately,
    #     with velocity differenced from the two frames (a plain reset zeroes the
    #     velocity and re-diverges). Five consecutive inconsistent rejections still
    #     force the plain re-lock as the last-resort fallback.
    def step(self, measurement, R=None, age=None):
        """Advance the filter one tick, then (optionally) low-pass the output velocity. The LPF
        wraps ALL exits of _step_impl (coast / relock / reject / normal) with one call, so it
        smooths every path and advances every tick (consistent cutoff regardless of fresh-rate).

        ``age`` is how old the measurement is, in seconds, when the sensor can say (Fix 8c). It
        is ignored unless the tracker was built with ``meas_lag_comp=True``, so passing it can
        never change a run on its own."""
        self._step_impl(measurement, R, age)
        self._apply_vel_lpf()

    def _apply_vel_lpf(self):
        """First-order LPF on the output velocity state[2:4] (no-op when disabled). alpha is
        derived from the cutoff and the tick dt; it feeds back into the next predict, so keep the
        cutoff light."""
        if self._vel_lpf_hz is None or self.state is None:
            return
        v = self.state[2:4].copy()
        if self._vel_lpf is None:
            self._vel_lpf = v
        else:
            tau = 1.0 / (2.0 * np.pi * self._vel_lpf_hz)
            alpha = self.system.dt / (tau + self.system.dt)
            self._vel_lpf = (1.0 - alpha) * self._vel_lpf + alpha * v
        self.state[2:4] = self._vel_lpf

    def _lag_reference(self, predicted_state, age):
        """Where the puck was when the frame was CAPTURED -- what the measurement describes.

        Falls back to `predicted_state` (the vendored behaviour) whenever the compensation is
        off, no age was supplied, or the age is under one tick: rounding a 5 ms age to a whole
        tick would invent a lag rather than remove one.
        """
        if not self._meas_lag_comp or age is None:
            return predicted_state
        d = int(round(float(age) / self.system.dt))
        # No state that old to compare against -- just after a reset or a re-lock, or an age
        # beyond MAX_LAG_TICKS. Clamping to the OLDEST snapshot instead (the obvious shortcut)
        # measures a stale measurement against a much more recent state: a large innovation
        # pointing backwards, taken at the near-unity gain a freshly reset P gives. That corrupts
        # the velocity, and a compensated filter then coasts on it -- its innovations are near
        # zero by construction, so nothing pulls it back. Declining to compensate degrades
        # exactly to the vendored behaviour, which is the right floor.
        if d < 1 or len(self._history) < d:
            return predicted_state
        return self._history[-d]

    def _step_impl(self, measurement, R=None, age=None):
        # One posterior per tick, appended BEFORE this tick's predict, so `_history[-d]` is the
        # estimate as of d ticks ago. Kept only when the compensation is on -- otherwise this is
        # a per-tick copy nothing reads.
        if self._meas_lag_comp and self.state is not None:
            self._history.append(np.asarray(self.state, dtype=float).copy())
            if len(self._history) > self.MAX_LAG_TICKS:
                del self._history[0]
        predicted_state, P = self.predict(self.state, self.P)
        if self._last_rejected is not None:
            self._last_rejected_age += 1
        if measurement is None:
            # A tick with no camera frame carries NO evidence about whether the last rejection was
            # a glitch, so it must not discard it. Clearing _last_rejected here (as this did)
            # made the two-point re-lock unreachable whenever detections are sparser than every
            # tick -- i.e. always, on the real 11-22 Hz vision path against a 50 Hz loop. The
            # filter could then only escape a bad lock via the 5-rejection plain reset, which
            # zeroes velocity; a struck puck crossed the whole table estimated at v = 0, which the
            # arm executes as aiming at a stationary puck. Age it out instead.
            self.state, self.P = predicted_state, P
            if self._last_rejected_age > self.RELOCK_MAX_AGE:
                self._last_rejected = None
                self._last_rejected_age = 0
            return
        if R is None:
            self.state, self.P = self.update(measurement, predicted_state, P)
            return
        # The gate asks the same question the update does, so it must ask it about the same
        # state: gating on the current prediction while fusing against the captured-time one
        # would reject exactly the fast-puck measurements the compensation exists to use.
        ref = self._lag_reference(predicted_state, age)
        H, y = self._innovation(measurement, ref)
        R_used = R[:2, :2] if len(y) == 2 else R
        S = H @ P @ H.T + R_used
        nis = y @ np.linalg.solve(S, y)
        if nis > (13.82 if len(y) == 2 else 16.27) and not self._near_rim(measurement[:2]):
            prev = self._last_rejected
            # RELOCK_DIST is a per-STEP travel bound, so it scales with how many ticks actually
            # elapsed between the two rejected measurements -- at one camera frame apart a fast
            # puck legitimately travels several times the single-tick bound.
            span = self.RELOCK_DIST * max(1, self._last_rejected_age)
            if prev is not None and np.linalg.norm(measurement[:2] - prev) < span:
                # Divide by the ACTUAL elapsed time between the two rejected measurements, not by
                # one control tick -- see _last_rejected_age. reset() clears the age, so read it
                # first.
                dt_elapsed = max(1, self._last_rejected_age) * self.system.dt
                self.reset(np.append(measurement[:2],
                                     measurement[2] if len(measurement) == 3 else 0.0))
                self.state[2:4] = (measurement[:2] - prev) / dt_elapsed
                self._clamp_speed()
                return
            self._reject_count += 1
            self._last_rejected = np.array(measurement[:2], dtype=float)
            self._last_rejected_age = 0
            if self._reject_count >= 5:
                self.reset(np.append(measurement[:2],
                                     measurement[2] if len(measurement) == 3 else 0.0))
            else:
                self.state, self.P = predicted_state, P
            return
        self._reject_count = 0
        self._last_rejected = None
        self._last_rejected_age = 0
        self.state, self.P = self.update(measurement, predicted_state, P, R, ref_state=ref)
        self._clamp_speed()

    def get_prediction(self, t, defend_line=0.):
        P_current = self.P.copy()
        state_current = self.state.copy()
        predict_time = 0

        for i in range(round(t / self.system.dt)):
            state_next, P_next = self.predict(state_current, P_current)
            if state_next[0] < defend_line:
                break
            if np.linalg.norm(state_current[2:4]) < 1e-2 and np.linalg.norm(state_next[2:4]) < 1e-2:
                predict_time = t
                break
            predict_time += self.system.dt
            state_current = state_next
            P_current = P_next
        return state_current, P_current, predict_time