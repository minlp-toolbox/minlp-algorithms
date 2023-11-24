r"""
Benders equality to NLP LB solver.

This solver should find a point where the cut is equal to the LB. This can be
done using:


$$
    min_{x, \eta, \alpha} \sum(alpha \eta) - \sum(\alpha)

    s.t:
        \sum(\alpha) \geq 1
        \alpha \geq 0
        -(cut - LB) \leq \eta
        cut - LB \leq \eta
        \alpha_i \f_i(x) \geq \alpha_i f_j(x) \forall j \neq \i
$$
"""  # noqa: W605

import numpy as np
import casadi as ca
from benders_exp.solvers.utils import Constraints
from benders_exp.solvers import MiSolverClass, Stats, MinlpProblem, MinlpData, regularize_options
from benders_exp.defines import CASADI_VAR, Settings
from benders_exp.solvers.tighten import tighten_bounds_x


class BendersEquality(MiSolverClass):
    """Benders equality solver."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings):
        """Create benders equality problem."""
        super(BendersEquality, self).__init___(problem, stats, s)

        self.nlp_lb = None
        self.nlp_lb_x = None

        self.g_infeasible = Constraints(0, [], [], [])
        self.cuts_benders = []
        self.nr_cuts_benders = 0

        self.nlp_options = regularize_options(s.IPOPT_SETTINGS, {}, s)
        self.mip_options = regularize_options(s.MIP_SETTINGS, {}, s)
        self.ub = ca.inf
        self.nu_min = -ca.inf
        self._setup_func(problem, data)

    def _setup_func(self, problem, data):
        """Set up functions."""
        self.idx_x_bin = problem.idx_x_bin
        self.nr_x = problem.x.shape[0]
        self.nr_x_bin = len(problem.idx_x_bin)
        self._x_bin = ca.vcat(
            [CASADI_VAR.sym("x_benders", 1)
             for _ in range(len(problem.idx_x_bin))]
        )
        self._x_nbin = ca.vcat(
            [CASADI_VAR.sym("x", 1) for _ in range(self.nr_x - self.nr_x_bin)]
        )
        j, k = 0, 0
        self._x = []
        for i in range(self.nr_x):
            if i in problem.idx_x_bin:
                self._x.append(self._x_bin[j])
                j += 1
            else:
                self._x.append(self._x_nbin[k])
                k += 1

        self._x = ca.vcat(self._x)

        self.f = ca.Function(
            "f", [problem.x, problem.p], [problem.f],
            {"jit": self.settings.WITH_JIT}
        )
        self.g = ca.Function(
            "g", [problem.x, problem.p], [problem.g],
            {"jit": self.settings.WITH_JIT}
        )
        self.nr_g = problem.g.shape[0]
        self.jac_g_bin = ca.Function(
            "jac_g_bin", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)[:, problem.idx_x_bin]],
            {"jit": self.settings.WITH_JIT}
        )
        self.g_nlp = Constraints(self.nr_g, self.g(
            self._x, data.p), data.lbg, data.ubg)

    def _add_infeasible_cut(self, x_sol, lam_g_sol, nlpdata: MinlpData):
        """Create infeasibility cut."""
        x_sol = x_sol[:self.nr_x]
        h_k = self.g(x_sol, nlpdata.p)
        jac_h_k = self.jac_g_bin(x_sol, nlpdata.p)
        g_k = lam_g_sol.T @ (
            h_k + jac_h_k @ (self._x_bin - x_sol[self.idx_x_bin])
            - (lam_g_sol > 0) * np.where(np.isinf(nlpdata.ubg), 0, nlpdata.ubg)
            + (lam_g_sol < 0) * np.where(np.isinf(nlpdata.lbg), 0, nlpdata.lbg)
        )
        self.g_infeasible.add(-ca.inf, g_k, 0)

    def _add_benders_cut(self, x_sol, lam_x_sol, p):
        lambda_k = -lam_x_sol[self.idx_x_bin]
        f_k = self.f(x_sol, p)
        self.cuts_benders.append(
            f_k + lambda_k.T @ (self._x_bin - x_sol[self.idx_x_bin])
        )
        self.nr_cuts_benders += 1

    def tighten_bounds_x(self, data, idx=None):
        """Tighten bounds on x."""
        if idx is None:
            constraints = self.get_g_benders()
        else:
            constraints = Constraints(1, self.cuts_benders[idx], -ca.inf, self.ub)

        return tighten_bounds_x(data, constraints, self.idx_x_bin, self._x, self.nr_x)

    def check_lb_valid(self, data: MinlpData):
        """Check if the LB is valid."""
        if self.nlp_lb_x is None:
            return False

        if self.g_infeasible.nr > 0:
            g_ineq = ca.Function("g_ineq", [self._x_bin], [self.g_infeasible.eq[-1]])(
                self.nlp_lb_x[self.idx_x_bin]
            )
            if g_ineq > 0:
                return False

        if self.nr_cuts_benders > 0:
            g_benders = ca.Function("g_benders", [self._x_bin], [self.cuts_benders[-1]])(
                self.nlp_lb_x[self.idx_x_bin]
            )
            if g_benders > self.nlp_lb:
                return False

        if np.any(data.ubx < self.nlp_lb_x) or np.any(data.lbx > self.nlp_lb_x):
            return False

        return True

    def get_g_benders(self):
        g_benders = Constraints(0, [], [], [])
        for i in range(self.nr_cuts_benders):
            # cut < self.ub
            g_benders += Constraints(
                1, self.cuts_benders[i],
                -ca.inf, self.ub - 1e-4
            )
        return g_benders

    def solve_nlp_lb(self, nlpdata: MinlpData):
        """Solve for a new LB."""
        # Do we need to include benders cuts and how?
        g = self.g_nlp  # + self.g_infeasible  # TODO: Cuts of bender!
        if self.nr_cuts_benders > 0:
            g_benders = self.get_g_benders()
            g += g_benders

        nlp_solver = ca.nlpsol("nlpsol", "ipopt", {
            "f": self.f(self._x, nlpdata.p), "g": g.eq, "x": self._x,
        }, self.nlp_options)
        new_sol = nlp_solver(
            x0=nlpdata.x0,
            lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=g.lb, ubg=g.ub,
        )

        success, _ = self.collect_stats("NLP-LB", nlp_solver)
        if not success:
            raise Exception("Error LB calc")

        self.nlp_lb_x = new_sol['x']
        self.nlp_lb = float(new_sol['f'])
        print(f"NLP LB IS RECOMPUTED AND EQUAL TO {self.nlp_lb}")

    def solve_benders_eq_lb(self, nlpdata: MinlpData):
        """Solve benders equality to lb problem."""
        nu = CASADI_VAR.sym("slack", 1)
        f = nu + ca.norm_2(self._x_bin - self.nlp_lb_x[self.idx_x_bin])
        g = Constraints(0, [], [], [])
        for i in range(self.nr_cuts_benders):
            # cut < nu
            g += Constraints(
                1, self.cuts_benders[i] - nu,
                -ca.inf, 0
            )

        g += self.g_infeasible

        self.mip_options.update({
            # + 2 * self.nr_cuts_benders),
            "discrete": [1] * self.nr_x_bin + [0],
            "gurobi.NonConvex": 2
        })
        solver = ca.qpsol("benders_eq", self.settings.MIP_SOLVER, {
            "f": f, "g": g.eq, "x": ca.vertcat(
                self._x_bin, nu  # , slack, weight
            )
        }, self.mip_options)

        sol = solver(
            x0=ca.vertcat(
                # + 2 * self.nr_cuts_benders,))
                nlpdata.x_sol[self.idx_x_bin], np.zeros((1, ))
            ),
            lbx=ca.vertcat(
                nlpdata.lbx[self.idx_x_bin],
                -ca.inf * np.ones((1, )),
                # np.zeros((self.nr_cuts_benders,))
            ),
            ubx=ca.vertcat(
                nlpdata.ubx[self.idx_x_bin],
                10 * self.ub * np.ones((1, ))
                # + self.nr_cuts_benders, )),
                # np.ones((self.nr_cuts_benders, ))
            ),
            lbg=g.lb, ubg=g.ub,
        )

        self.nu_min = float(sol['x'][-1].full())
        x_full = nlpdata.x_sol.full()[:self.nr_x]
        x_full[self.idx_x_bin] = sol['x'][:self.nr_x_bin]
        sol['x'] = x_full
        sol['f'] = max(self.nlp_lb, float(sol['f']))

        nlpdata.prev_solution = sol
        nlpdata.solved, _ = self.collect_stats("MILP-LB", solver)
        return nlpdata

    def solve(self, nlpdata: MinlpData, prev_feasible=True, relaxed=False) -> MinlpData:
        """solve."""
        if relaxed:
            raise NotImplementedError()
        if prev_feasible:
            if self.ub > nlpdata.obj_val:
                self.ub = nlpdata.obj_val
                self.tighten_bounds_x(nlpdata)
            self._add_benders_cut(nlpdata.x_sol, nlpdata.lam_x_sol, nlpdata.p)
            self.tighten_bounds_x(nlpdata, -1)
        else:
            self._add_infeasible_cut(nlpdata.x_sol, nlpdata.lam_g_sol, nlpdata)

        if not self.check_lb_valid(nlpdata):
            self.solve_nlp_lb(nlpdata)

        # LB is now valid, solve for a new point
        return self.solve_benders_eq_lb(nlpdata)
