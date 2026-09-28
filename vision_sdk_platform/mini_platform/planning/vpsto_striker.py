"""VPSTO strike planner (Cartesian, in-plane).

Plans a time-optimal, acceleration-bounded x-y table path from the current mallet state
(position + velocity) to the intercept point with a commanded terminal mallet velocity, then the
existing null-space QP-IK (planning/optimizer.py) turns that Cartesian path into joint commands
(pinning the play-plane z and enforcing the joint velocity limits). This replaces the vendored
cubic-Bezier x-y planner; VPSTO's advantages over it:
  * TIME-OPTIMAL within a Cartesian acceleration bound -> the mallet reaches a far intercept as
    fast as the bound allows, instead of over a fixed geometry-driven horizon (the "late when far"
    fix, together with the rendezvous in MiniTrajectoryGenerator that matches the intercept to the
    arm's feasible reach time);
  * a real acceleration limit -> a smooth, C2 path (the OBF basis is C2), so no replan-boundary
    acceleration spikes;
  * it is constrained to the current mallet VELOCITY, so a replan mid-swing is C1-continuous.

WHY CARTESIAN, NOT JOINT SPACE: a joint-space VPSTO swing does not keep the mallet in the play
plane -- the ee traces a joint-space geodesic that arcs tens of mm off the table (and >200 mm when
the start and hit configs are in different IK branches), blowing the +-20 mm ee_constr. Planning in
the plane keeps z pinned by construction; the QP handles the joint feasibility the joint-space
formulation was meant to provide. (Verified in tools/vpsto_bench.py.)

Dependencies: numpy + cma (via vpsto). The mean-path fallback is pure numpy (no cma).
"""
import numpy as np

from vpsto.vpsto import VPSTO, VPSTOOptions, VPSTOSolution

NDOF = 2   # planar x-y


class VpstoStriker:
    def __init__(self, env_info, vel_lim=1.5, acc_lim=8.0, smooth=0.0,
                 max_iter=25, pop_size=20, n_eval=20, n_via=4, use_cma=True):
        # Cartesian (table-plane) velocity / acceleration bounds the strike path is planned within.
        # Conservative bounds keep the downstream QP-IK feasible (a too-fast Cartesian path can
        # demand joint velocities the QP must clip near a singularity). Raise (Stage 4) to hit
        # harder/reach farther. Scalars broadcast to both axes.
        self.vel_lim = np.full(NDOF, float(vel_lim))
        self.acc_lim = np.full(NDOF, float(acc_lim))
        self.n_via = int(n_via)
        self.use_cma = bool(use_cma)
        # Smoothness weight on the objective. VPSTO's cost here is the trajectory DURATION, so
        # with smooth=0 the solver is a pure minimum-time planner: it saturates vel_lim/acc_lim
        # by construction and the swing always looks like a sprint. smooth>0 adds a normalised
        # acceleration-effort term, so a candidate that demands less acceleration can win despite
        # taking longer -- controlled motion instead of the fastest possible one. Dimensionless
        # (the penalty scales the duration), 0 = the original pure-min-time behaviour.
        self.smooth = max(0.0, float(smooth))

        self._opt = self._make_options(max_iter, pop_size, n_eval, n_via)
        self._vp = VPSTO(self._opt)
        self._p_prev = None
        self._prev_key = None

    def _make_options(self, max_iter, pop_size, n_eval, n_via):
        o = VPSTOOptions(ndof=NDOF)
        o.vel_lim = self.vel_lim.copy()
        o.acc_lim = self.acc_lim.copy()
        o.N_via, o.N_eval, o.pop_size = int(n_via), int(n_eval), int(pop_size)
        o.max_iter = int(max_iter)
        o.CMA_diagonal = True
        o.multithreading = False
        o.sigma_init = 0.5
        o.verbose = False
        o.log = False
        return o

    def _loss(self, c):
        """Population cost: duration, optionally scaled by normalised acceleration effort.

        ``c['acc']`` is (pop, N_eval, ndof) and ``c['T']`` is (pop,); the return must be (pop,).
        Normalising by ``acc_lim`` keeps ``smooth`` dimensionless and unit-independent, and
        multiplying (rather than adding) keeps the cost in units of time, so the weight means the
        same thing whether a strike takes 0.2 s or 1 s.
        """
        T = np.asarray(c['T'], float)
        if self.smooth <= 0.0:
            return T
        a = np.asarray(c['acc'], float)
        effort = np.mean(np.sum((a / self.acc_lim) ** 2, axis=-1), axis=-1)
        return T * (1.0 + self.smooth * effort)

    def clip_terminal_velocity(self, vT):
        """Scale a terminal cartesian velocity uniformly to fit vel_lim (keeps direction)."""
        vT = np.asarray(vT, float)
        m = np.max(np.abs(vT) / (self.vel_lim * 0.98)) if vT.size else 0.0
        return (vT / m, 1.0 / m) if m > 1.0 else (vT, 1.0)

    def _mean_via(self, x0, v0, xT, vT):
        vt = self._vp.vptraj
        n = NDOF
        ddPhi_b = np.concatenate((vt.ddPhi[:, :n], vt.ddPhi[:, -3 * n:]), axis=1)
        return - vt.ddPhi_p @ ddPhi_b @ np.concatenate((x0, xT, v0, vT))

    def min_feasible_T(self, x0, v0, xT, vT):
        """Minimum vel/acc-feasible cartesian duration for the mean path (rendezvous inner loop)."""
        x0, v0, xT, vT = (np.asarray(a, float) for a in (x0, v0, xT, vT))
        return float(self._vp.vptraj.get_min_duration(self._mean_via(x0, v0, xT, vT), x0, v0, xT, vT))

    def _mean_solution(self, x0, v0, xT, vT):
        mu_p = self._mean_via(x0, v0, xT, vT)
        T = float(self._vp.vptraj.get_min_duration(mu_p, x0, v0, xT, vT))
        sol = VPSTOSolution(self._opt)
        sol.T_best = max(T, 1e-3)
        sol.w_best = np.concatenate((x0, mu_p, xT, v0, vT))
        return sol

    def plan(self, x0, v0, xT, vT):
        """Return a VPSTOSolution (.T_best, .get_posvelacc) for the x-y strike path. Warm-starts
        across commits to the same (xT,vT); mean-path fallback on any solver failure."""
        x0, v0, xT, vT = (np.asarray(a, float) for a in (x0, v0, xT, vT))
        if not self.use_cma:
            return self._mean_solution(x0, v0, xT, vT)
        try:
            key = (np.round(xT, 3).tobytes(), np.round(vT, 3).tobytes())
            if self._p_prev is not None and key == self._prev_key:
                self._vp.opt.sigma_init = 0.1
                self._vp.set_initial_guess(self._p_prev)
            else:
                self._vp.opt.sigma_init = 0.5
            sol = self._vp.minimize(self._loss, x0, dq0=v0, qT=xT, dqT=vT)
            if sol.w_best is None or not np.all(np.isfinite(sol.w_best)) or sol.T_best is None:
                return self._mean_solution(x0, v0, xT, vT)
            self._p_prev, self._prev_key = sol.p_best, key
            return sol
        except Exception:
            return self._mean_solution(x0, v0, xT, vT)

    def sample_xy(self, sol, elapsed, k, dt):
        """Sample the x-y path on a dt grid starting one dt after `elapsed`, clamped to T.
        Returns (P, V, A) each (k, 2) -- ready to drop into a Cartesian trajectory row."""
        T = max(float(sol.T_best), 1e-3)
        ts = np.clip(elapsed + (1 + np.arange(int(k))) * dt, 0.0, T)
        P, V, A = (np.asarray(a, float) for a in sol.get_posvelacc(ts))
        return P[:, :NDOF], V[:, :NDOF], A[:, :NDOF]
