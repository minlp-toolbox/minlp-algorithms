# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Set of solvers based on outer approximation."""

import logging
import casadi as ca
import numpy as np
from camino.solvers import SolverClass, Stats, MinlpProblem, MinlpData, \
    get_idx_linear_bounds, regularize_options, get_idx_inverse, extract_bounds
from camino.settings import GlobalSettings, Settings

logger = logging.getLogger(__name__)


class OuterApproxMILP(SolverClass):
    r"""
    Outer approximation.

    This implementation assumes the following input:
        min f(x)
        s.t. lb < g(x)

    It constructs the following problem:
        min \alpha
        s.t.
            \alpha \geq f(x) + \nabla (x) (x-x^i)
            lb \leq g(x) + \nabla  g(x) (x-x^i)
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings, with_lin_bounds=True):
        """Create the outer approximation master problem."""
        super(OuterApproxMILP, self).__init__(problem, stats, s)
        self.setup_common(problem, s)
        # Last variable is alpha for epigraph reformulation
        self._x = GlobalSettings.CASADI_VAR.sym("x", self.nr_x)
        self._alpha = GlobalSettings.CASADI_VAR.sym("alpha")
        self.problem = problem
        OuterApproxMILP.reset(self, data)

    def setup_common(self, problem: MinlpProblem, s: Settings):
        """Set up common data"""
        self.settings = s
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
        self.g = ca.Function(
            "g", [problem.x, problem.p], [problem.g],
            {"jit": s.WITH_JIT}
        )
        self.jac_g = ca.Function(
            "gradient_g_x",
            [problem.x, problem.p], [ca.jacobian(
                problem.g, problem.x
            )], {"jit": s.WITH_JIT}
        )

        self.nr_x = problem.x.shape[0]
        self.nr_g_orig = problem.g.shape[0]
        self.idx_x_integer = problem.idx_x_integer
        self.nr_x_bin = len(problem.idx_x_integer)
        self.options["discrete"] = [
            1 if i in self.idx_x_integer else 0 for i in range(self.nr_x)] + [0]

    def add_solution(self, nlpdata, solved, solution, integers_relaxed=False):
        """Add a cut."""
        x_sol = solution['x'][:self.nr_x]
        if solved and (not integers_relaxed):
            self._g = ca.vertcat(
                self._g,
                self.f(x_sol, nlpdata.p) + self.grad_f(x_sol, nlpdata.p).T @
                (self._x - x_sol) - self._alpha
            )
            self._lbg.append(-ca.inf)
            self._ubg.append(0)

        g_lin = self.g(x_sol, nlpdata.p)
        jac_g = self.jac_g(x_sol, nlpdata.p)
        self._g = ca.vertcat(
            self._g,
            g_lin + jac_g @ (self._x - x_sol)
        )
        self._lbg.append(nlpdata.lbg)
        self._ubg.append(nlpdata.ubg)

    def solve(self, nlpdata: MinlpData, integers_relaxed=False) -> MinlpData:
        """Solve the outer approximation master problem (MILP)."""
        for solved, solution in zip(nlpdata.solved_all, nlpdata.solutions_all):
            self.add_solution(nlpdata, solved, solution, integers_relaxed)

        x_sol = nlpdata.x_sol[:self.nr_x]

        solver = ca.qpsol(f"oa_with_{self._g.shape[0]}_cut", self.settings.MIP_SOLVER, {
            "f": self._alpha, "g": self._g,
            "x": ca.vertcat(self._x, self._alpha),
        }, self.options)

        solution = solver(
            x0=ca.vertcat(x_sol, nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx, -1e8),
            ubx=ca.vertcat(nlpdata.ubx, ca.inf),
            lbg=ca.vertcat(*self._lbg),
            ubg=ca.vertcat(*self._ubg),
        )
        x_full = solution['x'].full()[:self.nr_x]
        solution['x'] = x_full
        nlpdata.prev_solution = solution
        nlpdata.solved, stats = self.collect_stats("OA-MILP", solver, solution)
        return nlpdata

    def reset(self, nlpdata: MinlpData):
        """Reset this problem."""
        self._g = np.array([])
        self._lbg = []
        self._ubg = []

    def warmstart(self, nlpdata: MinlpData):
        """Warmstart cuts."""
        relaxed = self.stats.relaxed_solution
        if self.stats.relaxed:
            self.add_solution(nlpdata, True, relaxed.solutions_all[0], True)

        if not nlpdata.relaxed:
            for solved, solution in zip(nlpdata.solved_all, nlpdata.solutions_all):
                self.add_solution(nlpdata, solved, solution)


class OuterApproxMIQP(OuterApproxMILP):
    "Implementation of quadratic outer approximation"

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings, hess: str = 'f'):
        """Create the quadratic outer approximation master problem."""
        super(OuterApproxMIQP, self).__init__(problem, data, stats, s)
        self.hessian_type = hess
        if self.hessian_type == 'f':
            self.hess = ca.Function("hess_f", [problem.x, problem.p], [ca.hessian(
                problem.f, problem.x)[0]], {"jit": self.settings.WITH_JIT})
        elif self.hessian_type == 'lag':
            lam = GlobalSettings.CASADI_VAR.sym("lam", self.nr_g_orig)
            lagrangian = problem.f + lam.T @ problem.g
            lagrangian_hess = ca.hessian(lagrangian, ca.vertcat(problem.x, lam))[
                0][:self.nr_x, :self.nr_x]
            self.hess = ca.Function("hess_lag", [problem.x, lam, problem.p], [lagrangian_hess], {
                "jit": self.settings.WITH_JIT})
        else:
            raise AttributeError(
                "Available Hessian types: 'f' for standard objective Hessian, 'lag' for Hessian of the Lagrangian")

    def solve(self, nlpdata: MinlpData, integers_relaxed=False) -> MinlpData:
        """Solve the quadratic outer approximation master problem (MIQP)."""
        for solved, solution in zip(nlpdata.solved_all, nlpdata.solutions_all):
            self.add_solution(nlpdata, solved, solution, integers_relaxed)

        x_sol = nlpdata.x_sol[:self.nr_x]
        if self.hessian_type == 'f':
            hess = self.hess(x_sol, nlpdata.p)
        elif self.hessian_type == 'lag':
            hess = self.hess(x_sol, nlpdata.lam_g_sol, nlpdata.p)
        f_quad = self._alpha + 0.5 * \
            (self._x - x_sol).T @ hess @ (self._x - x_sol)

        solver = ca.qpsol(f"oa_with_{self._g.shape[0]}_cut", self.settings.MIP_SOLVER, {
            "f": f_quad, "g": self._g,
            "x": ca.vertcat(self._x, self._alpha),
        }, self.options)

        solution = solver(
            x0=ca.vertcat(x_sol, nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx, -1e8),
            ubx=ca.vertcat(nlpdata.ubx, ca.inf),
            lbg=ca.vertcat(*self._lbg),
            ubg=ca.vertcat(*self._ubg),
        )
        x_full = solution['x'].full()[:self.nr_x]
        solution['x'] = x_full
        nlpdata.prev_solution = solution
        nlpdata.solved, stats = self.collect_stats("OA-MIQP", solver, solution)
        return nlpdata


class OuterApproxMILPImproved(OuterApproxMILP):
    """i
    Improved version of outer approximation.

    Cutting planes are added only for linear constraints to avoid the creation of infeasible master problems.
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings):
        """Create the improved outer approximation master problem."""
        super(OuterApproxMILPImproved, self).__init__(problem, data, stats, s)

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
        self.reset(data)

    def reset(self, data):
        """Reset internal data."""
        if self.idx_g_lin.shape[0] > 0:
            _, self._g, self._lbg, self._ubg = extract_bounds(
                self.problem, data, self.idx_g_lin, self._x, self.problem.idx_x_integer
            )
        else:
            _, self._g, self._lbg, self._ubg = 0, [], [], []

    def solve(self, nlpdata: MinlpData, integers_relaxed=False) -> MinlpData:
        """Solve the improved outer approximation master problem (MILP)."""
        for solved, solution in zip(nlpdata.solved_all, nlpdata.solutions_all):
            self.add_solution(nlpdata, solved, solution, integers_relaxed)

        x_sol = nlpdata.x_sol[:self.nr_x]

        solver = ca.qpsol(f"oa_with_{self._g.shape[0]}_cut", self.settings.MIP_SOLVER, {
            "f": self._alpha, "g": self._g,
            "x": ca.vertcat(self._x, self._alpha),
        }, self.options)

        solution = solver(
            x0=ca.vertcat(x_sol, nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx, -1e8),
            ubx=ca.vertcat(nlpdata.ubx, ca.inf),
            lbg=ca.vertcat(*self._lbg),
            ubg=ca.vertcat(*self._ubg),
        )
        x_full = solution['x'].full()[:self.nr_x]
        solution['x'] = x_full
        nlpdata.prev_solution = solution
        nlpdata.solved, stats = self.collect_stats(
            "OAI-MILP", solver, solution)
        return nlpdata


class OuterApproxMIQPImproved(OuterApproxMILP):
    """
    Improved version of quadratic outer approximation.

    Cutting planes are added only for linear constraints to avoid the creation of infeasible master problems.
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings, hess: str = 'f'):
        """Create improved version of the quadratic outer approximation master problem."""
        super(OuterApproxMIQPImproved, self).__init__(problem, data, stats, s)

        self.hessian_type = hess
        if self.hessian_type == 'f':
            self.hess = ca.Function("hess_f", [problem.x, problem.p], [ca.hessian(
                problem.f, problem.x)[0]], {"jit": self.settings.WITH_JIT})
        elif self.hessian_type == 'lag':
            lam = GlobalSettings.CASADI_VAR.sym("lam", self.nr_g_orig)
            lagrangian = problem.f + lam.T @ problem.g
            lagrangian_hess = ca.hessian(lagrangian, ca.vertcat(problem.x, lam))[
                0][:self.nr_x, :self.nr_x]
            self.hess = ca.Function("hess_lag", [problem.x, lam, problem.p], [lagrangian_hess], {
                "jit": self.settings.WITH_JIT})
        else:
            raise AttributeError(
                "Available Hessian types: 'f' for standard objective Hessian, 'lag' for Hessian of the Lagrangian")

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
        self.reset(data)

    def reset(self, data):
        """Reset internal data."""
        if self.idx_g_lin.shape[0] > 0:
            _, self._g, self._lbg, self._ubg = extract_bounds(
                self.problem, data, self.idx_g_lin, self._x, self.problem.idx_x_integer
            )
        else:
            _, self._g, self._lbg, self._ubg = 0, [], [], []

    def solve(self, nlpdata: MinlpData, integers_relaxed=False) -> MinlpData:
        """Solve the improved quadratic outer approximation master problem (MIQP)."""
        for solved, solution in zip(nlpdata.solved_all, nlpdata.solutions_all):
            self.add_solution(nlpdata, solved, solution, integers_relaxed)

        x_sol = nlpdata.x_sol[:self.nr_x]

        if self.hessian_type == 'f':
            hess = self.hess(x_sol, nlpdata.p)
        elif self.hessian_type == 'lag':
            hess = self.hess(x_sol, nlpdata.lam_g_sol, nlpdata.p)
        f_quad = self._alpha + 0.5 * \
            (self._x - x_sol).T @ hess @ (self._x - x_sol)

        solver = ca.qpsol(f"oa_with_{self._g.shape[0]}_cut", self.settings.MIP_SOLVER, {
            "f": f_quad, "g": self._g,
            "x": ca.vertcat(self._x, self._alpha),
        }, self.options)

        solution = solver(
            x0=ca.vertcat(x_sol, nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx, -1e8),
            ubx=ca.vertcat(nlpdata.ubx, ca.inf),
            lbg=ca.vertcat(*self._lbg),
            ubg=ca.vertcat(*self._ubg),
        )
        x_full = solution['x'].full()[:self.nr_x]
        solution['x'] = x_full
        nlpdata.prev_solution = solution
        nlpdata.solved, stats = self.collect_stats(
            "OAI-MIQP", solver, solution)
        return nlpdata
