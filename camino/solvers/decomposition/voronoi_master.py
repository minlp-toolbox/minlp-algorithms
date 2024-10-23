# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Master problem of the sequential Voronoi-based MIQP algorithm."""

import logging
import casadi as ca
import numpy as np
from camino.solvers import SolverClass, Stats, MinlpProblem, MinlpData, \
    get_idx_linear_bounds, regularize_options, get_idx_inverse, extract_bounds
from camino.settings import GlobalSettings, Settings
from camino.utils import colored
from camino.utils.conversion import to_0d

logger = logging.getLogger(__name__)


class VoronoiTrustRegionMIQP(SolverClass):
    r"""
    Voronoi trust region problem.

    This implementation assumes the following input:
        min f(x)
        s.t. lb < g(x)

    It constructs the following problem:
        min f(x) + \nabla f(x) (x-x^i) + 0.5  (x-x^i)' (\nabla^2 f(x)) (x-x^i)
        s.t.
            lb \leq g(x) + \nabla  g(x) (x-x^i)
            NLPF constraints
            Voronoi trust region
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings):
        """Create sequential Voronoi master problem."""
        super(VoronoiTrustRegionMIQP, self).__init__(problem, stats, s)
        self.options = regularize_options(s.MIP_SETTINGS, {}, s)

        self.f = ca.Function(
            "f", [problem.x, problem.p], [problem.f],
            {"jit": s.WITH_JIT}
        )
        self.grad_f = ca.Function(
            "gradient_f_x",
            [problem.x, problem.p], [ca.gradient(
                problem.f, problem.x
            )], {"jit": s.WITH_JIT}
        )
        if problem.gn_hessian is not None:
            self.f_hess = ca.Function(
                "gn_hess_f_x", [problem.x, problem.p], [problem.gn_hessian])
        else:
            self.f_hess = ca.Function("hess_f_x", [problem.x, problem.p], [
                ca.hessian(problem.f, problem.x)[0]])

        self.g = ca.Function(
            "g", [problem.x, problem.p], [problem.g],
            {"jit": s.WITH_JIT}
        )
        self.jac_g_bin = ca.Function(
            "jac_g_bin", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)[:, problem.idx_x_integer]],
            {"jit": s.WITH_JIT}
        )
        self.idx_g_lin = get_idx_linear_bounds(problem)
        self.idx_g_nonlin = get_idx_inverse(self.idx_g_lin, problem.g.shape[0])
        self.g_lin = ca.Function(
            "g", [problem.x, problem.p], [problem.g[self.idx_g_lin]],
            {"jit": s.WITH_JIT}
        )
        self.g_nonlin = ca.Function(
            "g", [problem.x, problem.p], [problem.g[self.idx_g_nonlin]],
            {"jit": s.WITH_JIT}
        )
        self.jac_g_nonlin = ca.Function(
            "gradient_g_x",
            [problem.x, problem.p], [ca.jacobian(
                problem.g[self.idx_g_nonlin], problem.x
            )], {"jit": s.WITH_JIT}
        )

        self._x = GlobalSettings.CASADI_VAR.sym("x_voronoi", problem.x.numel())
        self.idx_x_integer = problem.idx_x_integer
        self.nr_g_orig = problem.g.shape[0]
        self.nr_x = problem.x.shape[0]
        # Copy the linear constraints in g
        self.nr_g, self._g, self._lbg, self._ubg = extract_bounds(
            problem, data, self.idx_g_lin, self._x, allow_fail=False
        )

        self.options["discrete"] = [
            1 if i in self.idx_x_integer else 0 for i in range(self.nr_x)]

        # Initialization for algorithm iterates
        self.ub = 1e15  # UB
        self.x_sol_list = []  # list of all the solution visited
        # Solution with best objective so far, we can safely assume the first
        # solution is the best even if infeasible since it is the only one
        # available yet.
        self.idx_best_x_sol = 0
        self.feasible_x_sol_list = []

    def add_solution(self, nlpdata, solved, solution, integers_relaxed=False):
        """Add a cut."""
        x_sol = solution['x']
        obj_val = float(solution['f'])
        if x_sol.shape[0] == 1:
            x_sol = to_0d(x_sol)[np.newaxis]
        else:
            x_sol = to_0d(x_sol)[:self.nr_x]

        self.x_sol_list.append(x_sol)
        self.feasible_x_sol_list.append(solved)
        if solved and (not integers_relaxed):
            if obj_val < self.ub:
                self.ub = obj_val
                # TODO: a surrogate for counting iterates, it's a bit clutter
                self.idx_best_x_sol = len(self.x_sol_list) - 1
                logger.info(
                    colored(f"New upperbound: {self.ub}", color='green'))
        else:
            g_k, lbg_k, ubg_k = self._generate_infeasible_cut(
                self._x, x_sol, solution['lam_g'], nlpdata.p)
            self._g = ca.vertcat(self._g, g_k)
            self._lbg = ca.vertcat(self._lbg, lbg_k)
            self._ubg = ca.vertcat(self._ubg, ubg_k)

    def solve(self, nlpdata: MinlpData, prev_feasible=True, is_qp=True, integers_relaxed=False) -> MinlpData:
        """Solve sequential Voronoi master problem (MIQP)."""
        # Update with the lowest upperbound and the corresponding best solution:
        for solved, solution in zip(nlpdata.solved_all, nlpdata.solutions_all):
            self.add_solution(nlpdata, solved, solution)

        x_sol_best = self.x_sol_list[self.idx_best_x_sol]

        # Create a new voronoi cut
        g_voronoi, lbg_voronoi, ubg_voronoi = self._generate_voronoi_tr(
            self._x[self.idx_x_integer], nlpdata.p)

        dx = self._x - x_sol_best

        f_k = self.f(x_sol_best, nlpdata.p)
        f_lin = self.grad_f(x_sol_best, nlpdata.p)
        if is_qp:
            f_hess = self.f_hess(x_sol_best, nlpdata.p)
            f = f_k + f_lin.T @ dx + 0.5 * dx.T @ f_hess @ dx
        else:
            f = f_k + f_lin.T @ dx

        g_lin = self.g_nonlin(x_sol_best, nlpdata.p)
        if g_lin.numel() > 0:
            jac_g = self.jac_g_nonlin(x_sol_best, nlpdata.p)
        else:
            g_lin = to_0d(g_lin)
            jac_g = np.zeros((0, self.nr_x))

        g = ca.vertcat(
            self._g,
            g_voronoi,
            g_lin + jac_g @ dx,
        )
        lbg = ca.vertcat(
            self._lbg,
            lbg_voronoi,
            nlpdata.lbg[self.idx_g_nonlin],
        )
        ubg = ca.vertcat(
            self._ubg,
            ubg_voronoi,
            nlpdata.ubg[self.idx_g_nonlin],
        )
        self.nr_g = ubg.numel()

        solver = ca.qpsol(
            f"voronoi_tr_milp_with_{self.nr_g}_constraints", self.settings.MIP_SOLVER, {
                "f": f, "g": g, "x": self._x,
            }, self.options)

        nlpdata.prev_solution = solver(
            x0=x_sol_best, lbx=nlpdata.lbx, ubx=nlpdata.ubx, lbg=lbg, ubg=ubg)

        nlpdata.solved, stats = self.collect_stats(
            "VTR-MIQP", solver, solution)
        return nlpdata

    def reset(self, nlpdata: MinlpData):  # TODO: to update, just copy paste from outer approx
        """Reset."""
        if self.idx_g_lin.numel() > 0:
            self.nr_g, self._g, self._lbg, self._ubg = extract_bounds(
                self.problem, nlpdata, self.idx_g_lin, self._x, self.problem.idx_x_integer
            )
        else:
            self.nr_g, self._g, self._lbg, self._ubg = 0, [], [], []

    def warmstart(self, nlpdata: MinlpData):
        """Warmstart algorithm."""
        relaxed = self.stats.relaxed_solution
        if relaxed:
            self.add_solution(nlpdata, True, relaxed.solutions_all[0], True)
        if not nlpdata.relaxed:
            for solved, solution in zip(nlpdata.solved_all, nlpdata.solutions_all):
                self.add_solution(nlpdata, solved, solution)

    def _generate_infeasible_cut(self, x, x_sol, lam_g, p):
        """Generate infeasibility cut."""
        h_k = self.g(x_sol, p)
        jac_h_k = self.jac_g_bin(x_sol, p)
        g_k = lam_g.T @ (h_k + jac_h_k @
                         (x[self.idx_x_integer] - x_sol[self.idx_x_integer]))
        return g_k, -ca.inf, 0.0

    def _generate_voronoi_tr(self, x_bin, p):
        r"""
        Generate Voronoi trust region based on the visited integer solutions and the best integer solution so far.

        :param p: parameters
        """
        g_k = []
        lbg_k = []
        ubg_k = []

        x_sol_bin_best = self.x_sol_list[self.idx_best_x_sol][self.idx_x_integer]
        x_sol_bin_best_norm2_squared = x_sol_bin_best.T @ x_sol_bin_best
        for x_sol, is_feas in zip(self.x_sol_list, self.feasible_x_sol_list):
            x_sol_bin = x_sol[self.idx_x_integer]
            if is_feas and not np.allclose(x_sol_bin, x_sol_bin_best):
                a = ca.DM(2 * (x_sol_bin - x_sol_bin_best))
                b = ca.DM(x_sol_bin.T @ x_sol_bin -
                          x_sol_bin_best_norm2_squared)
                g_k.append(a.T @ x_bin - b)
                lbg_k.append(-np.inf)
                ubg_k.append(0)

        return ca.vcat(g_k), lbg_k, ubg_k
