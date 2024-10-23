# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Set of master solvers based on bender cuts.

Bender cuts are based on the principle of decomposing the problem into two
parts where the main part is only solving the integer variables.
"""

import casadi as ca
import numpy as np
from camino.solvers import SolverClass, Stats, MinlpProblem, MinlpData, \
    get_idx_linear_bounds, get_idx_linear_bounds_binary_x, regularize_options, \
    get_idx_inverse, extract_bounds
from camino.settings import GlobalSettings, Settings
import logging

logger = logging.getLogger(__name__)


class BendersMasterMILP(SolverClass):
    """Create benders master problem."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats,
                 s: Settings, with_lin_bounds=True):
        """Create benders master MILP."""
        super(BendersMasterMILP, self).__init__(problem, stats, s)
        self.setup_common(problem, s)
        self.jac_g_bin = ca.Function(
            "jac_g_bin", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)[:, problem.idx_x_integer]],
            {"jit": s.WITH_JIT}
        )
        self._x = GlobalSettings.CASADI_VAR.sym("x_bin", self.nr_x_bin)
        self.problem = problem

        if with_lin_bounds:
            self.idx_g_lin = get_idx_linear_bounds_binary_x(problem)
            self.nr_g, self._g, self._lbg, self._ubg = extract_bounds(
                problem, data, self.idx_g_lin, self._x, problem.idx_x_integer
            )
        else:
            self.idx_g_lin = np.array([])
            self.nr_g, self._g, self._lbg, self._ubg = 0, [], [], []

        self.cut_id = 0

    def reset(self, data):
        """Reset."""
        if self.idx_g_lin.numel() > 0:
            self.nr_g, self._g, self._lbg, self._ubg = extract_bounds(
                self.problem, data, self.idx_g_lin, self._x, self.problem.idx_x_integer
            )
        else:
            self.nr_g, self._g, self._lbg, self._ubg = 0, [], [], []

        self.cut_id = 0

    def warmstart(self, data):
        """Warmstart algorithm."""
        if not data.relaxed:
            for solution in data.best_solutions:
                self.add_solution(
                    data, solution['x'], solution['lam_g'],
                    solution['lam_x'], True
                )

    def setup_common(self, problem: MinlpProblem, s: Settings):
        """Set up common data."""
        self.settings = s
        self.options = regularize_options(s.MIP_SETTINGS, {}, s)

        self.f = ca.Function(
            "f", [problem.x, problem.p], [problem.f],
            {"jit": s.WITH_JIT}
        )
        self.g = ca.Function(
            "g", [problem.x, problem.p], [problem.g],
            {"jit": s.WITH_JIT}
        )

        self.idx_x_integer = problem.idx_x_integer
        self.nr_x_bin = len(problem.idx_x_integer)
        self._nu = GlobalSettings.CASADI_VAR.sym("nu", 1)
        self.nr_g_orig = problem.g.shape[0]
        self.nr_x_orig = problem.x.shape[0]
        self.options["discrete"] = np.ones_like(
            problem.idx_x_integer).tolist() + [0]

    def _generate_cut_equation(self, x, x_sol, x_sol_sub_set, lam_g, lam_x, p, lbg, ubg, prev_feasible):
        r"""
        Generate a cut.

        when feasible, this creates the cut:

        $$
         g_k = f_k \lambda_k^{\mathrm{T}} (x_{\mathrm{binary}} - x_{\mathrm{binary, prev\_sol}}) - \nu \leq 0 \\
        \lambda_k = \nabla f_k + J_{g_k}^{T} \lambda_{g_j}
        $$

        when infeasible, the cut is equal to
        $$
        g_k = \lambda_{g_k}^{\mathrm{T}} g_k + J_{g_k}(x_{\mathrm{binary}} - x_{\mathrm{binary, prev\_sol}})
        $$

        :param x: optimization variable
        :param x_sol: Complete x_solution
        :param x_sol_sub_set: Subset of the x solution to optimize the MILP to
            $x_{\mathrm{bin}}$
        :param lam_g: Lambda g solution
        :param p: parameters
        :param prev_feasible: If the previous solution was feasible
        :return: g_k the new cutting plane (should be > 0)
        """
        if prev_feasible:
            lambda_k = -lam_x[self.idx_x_integer]
            f_k = self.f(x_sol, p)
            g_k = (
                f_k + lambda_k.T @ (x - x_sol_sub_set)
                - self._nu
            )
        else:  # Not feasible solution
            h_k = self.g(x_sol, p)
            jac_h_k = self.jac_g_bin(x_sol, p)
            g_k = lam_g.T @ (
                h_k + jac_h_k @ (x - x_sol_sub_set)
                - (lam_g > 0) * np.where(np.isinf(ubg), 0, ubg)
                + (lam_g < 0) * np.where(np.isinf(lbg), 0, lbg)
            )

        return g_k

    def add_solution(self, nlpdata, x_sol, lam_g_sol, lam_x_sol, prev_feasible):
        """Create cut."""
        g_k = self._generate_cut_equation(
            self._x, x_sol[:self.nr_x_orig], x_sol[self.idx_x_integer],
            lam_g_sol, lam_x_sol, nlpdata.p,
            nlpdata.lbg, nlpdata.ubg, prev_feasible
        )
        self.cut_id += 1

        self._g = ca.vertcat(self._g, g_k)
        self._ubg.append(0)
        self._lbg.append(-ca.inf)
        self.nr_g += 1

    def solve(self, nlpdata: MinlpData, integers_relaxed=False) -> MinlpData:
        """solve."""
        x_bin_star = nlpdata.x_sol[self.idx_x_integer]
        if not integers_relaxed:
            self.add_solution(
                nlpdata, nlpdata.x_sol, nlpdata.lam_g_sol, nlpdata.lam_x_sol, nlpdata.solved
            )
        solver = ca.qpsol(f"benders_with_{self.nr_g}_cut", self.settings.MIP_SOLVER, {
            "f": self._nu, "g": self._g,
            "x": ca.vertcat(self._x, self._nu),
        }, self.options)

        # This solver solves only to the binary variables (_x)!
        solution = solver(
            x0=ca.vertcat(x_bin_star, nlpdata.obj_val),
            # NOTE harmonize lb with OA, here -1e5, in OA -1e8
            lbx=ca.vertcat(nlpdata.lbx[self.idx_x_integer], -1e5),
            ubx=ca.vertcat(nlpdata.ubx[self.idx_x_integer], ca.inf),
            lbg=ca.vertcat(*self._lbg), ubg=ca.vertcat(*self._ubg)
        )
        x_full = nlpdata.x_sol.full()[:self.nr_x_orig]
        x_full[self.idx_x_integer] = solution['x'][:-1]
        solution['x'] = x_full
        nlpdata.prev_solution = solution
        nlpdata.solved, stats = self.collect_stats(
            "BENDERS-MILP", solver, solution)
        return nlpdata


class BendersMasterMIQP(BendersMasterMILP):
    """MIQP implementation."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings):
        """Create the benders constraint MILP."""
        super(BendersMasterMIQP, self).__init__(problem, data, stats, s)
        self.f_hess_bin = ca.Function(
            "hess_f_x_bin",
            [problem.x, problem.p], [ca.hessian(
                problem.f, problem.x
            )[0][problem.idx_x_integer, :][:, problem.idx_x_integer]],
            {"jit": self.settings.WITH_JIT}
        )

    def solve(self, nlpdata: MinlpData, integers_relaxed=False) -> MinlpData:
        """solve."""
        x_bin_star = nlpdata.x_sol[self.idx_x_integer]
        if not integers_relaxed:
            g_k = self._generate_cut_equation(
                self._x, nlpdata.x_sol[:self.nr_x_orig], x_bin_star,
                nlpdata.lam_g_sol, nlpdata.lam_x_sol, nlpdata.p,
                nlpdata.lbg, nlpdata.ubg, nlpdata.solved
            )
            self.cut_id += 1
            self._g = ca.vertcat(self._g, g_k)
            self._ubg.append(0)
            self._lbg.append(-ca.inf)
            self.nr_g += 1

        f_hess = self.f_hess_bin(nlpdata.x_sol[:self.nr_x_orig], nlpdata.p)

        dx = self._x - x_bin_star
        solver = ca.qpsol(f"benders_qp{self.nr_g}", self.settings.MIP_SOLVER, {
            "f": self._nu + 0.5 * dx.T @ f_hess @ dx,
            "g": self._g,
            "x": ca.vertcat(self._x, self._nu),
        }, self.options)

        # This solver solves only to the binary variables (_x)!
        solution = solver(
            x0=ca.vertcat(x_bin_star, nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx[self.idx_x_integer], -1e5),
            ubx=ca.vertcat(nlpdata.ubx[self.idx_x_integer], ca.inf),
            lbg=ca.vertcat(*self._lbg), ubg=ca.vertcat(*self._ubg)
        )
        obj = solution['x'][-1].full()
        if obj > solution['f']:
            raise Exception("Possible thougth mistake!")
        solution['f'] = obj
        x_full = nlpdata.x_sol.full()[:self.nr_x_orig]
        x_full[self.idx_x_integer] = solution['x'][:-1]
        solution['x'] = x_full
        nlpdata.prev_solution = solution
        nlpdata.solved, stats = self.collect_stats(
            "BENDERS-MIQP", solver, solution)
        return nlpdata


class BendersTrustRegionMIP(BendersMasterMILP):
    """
    Create benders constraint MIP.

    By an idea of Moritz D. and Andrea G.
    Given the ordered sequence of integer solutions:
        Y := {y1, y2, ..., yN}
    such that J(y1) >= J(y2) >= ... >= J(yN) we define the
    benders polyhedral B := {y in R^n_y:
        J(y_i) + Nabla J(y_i)^T (y - y_i) <= J(y_N),
        forall i = 1,...,N-1
    }

    This MILP solves:
        min F(y, z | y_bar, z_bar)
        s.t ub >= H_L(y,z| y_bar, z_bar) >= lb
        with y in B

    For this implementation, since the original formulation implements:
        J(y_i) + Nabla J(yi) T (y - yi) <= nu,
        meaning: nu == J(y_N)
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings):
        """Create the benders constraint MILP."""
        super(BendersTrustRegionMIP, self).__init__(problem, data, stats, s)
        self.setup_common(problem, s)

        self.idx_g_lin = get_idx_linear_bounds(problem)
        self.idx_g_nonlin = get_idx_inverse(self.idx_g_lin, problem.g.shape[0])

        self.grad_f_x_sub = ca.Function(
            "gradient_f_x",
            [problem.x, problem.p], [ca.gradient(
                problem.f, problem.x
            )],
            {"jit": s.WITH_JIT}
        )
        self.jac_g_sub = ca.Function(
            "jac_g", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)],
            {"jit": s.WITH_JIT}
        )
        self.f_hess = ca.Function("hess_f_x", [problem.x, problem.p], [
                                  ca.hessian(problem.f, problem.x)[0]])

        self._x = GlobalSettings.CASADI_VAR.sym("x_benders", problem.x.numel())
        self.nr_g, self._g, self._lbg, self._ubg = extract_bounds(
            problem, data, self.idx_g_lin, self._x, allow_fail=False
        )

        self.options.update({
            "discrete": [1 if elm in problem.idx_x_integer else 0 for elm in range(self._x.shape[0])],
            "error_on_fail": False
        })
        self.y_N_val = 1e15  # Should be inf but can not at the moment ca.inf
        # We take a point
        self.best_data = None
        self.x_sol_best = data.x0

    def solve(self, nlpdata: MinlpData, relaxed=False) -> MinlpData:
        """Solve."""
        # Update with the lowest upperbound and the corresponding best solution:
        if relaxed:
            self.x_sol_best = nlpdata.x_sol[:self.nr_x_orig]
        elif nlpdata.obj_val < self.y_N_val and nlpdata.solved:
            self.y_N_val = nlpdata.obj_val
            self.x_sol_best = nlpdata.x_sol[:self.nr_x_orig]
            self.best_data = nlpdata._sol
            print(f"NEW BOUND {self.y_N_val}")

        # Create a new cut
        x_sol_prev = nlpdata.x_sol[:self.nr_x_orig]
        g_k = self._generate_cut_equation(
            self._x[self.idx_x_integer], x_sol_prev, x_sol_prev[self.idx_x_integer],
            nlpdata.lam_g_sol, nlpdata.lam_x_sol, nlpdata.p,
            nlpdata.lbg, nlpdata.ubg, nlpdata.solved
        )
        self.cut_id += 1

        self._g = ca.vertcat(self._g, g_k)
        self._lbg = ca.vertcat(self._lbg, -ca.inf)
        # Should be at least 1e-4 better and 1e-4 from the constraint bound!
        self._ubg = ca.vertcat(self._ubg, 0)  # -1e-4)
        self.nr_g += 1

        dx = self._x - self.x_sol_best

        g_lin = self.g(self.x_sol_best, nlpdata.p)[self.idx_g_nonlin]
        if g_lin.numel() > 0:
            jac_g = self.jac_g_sub(self.x_sol_best, nlpdata.p)[
                self.idx_g_nonlin, :]
            # Warning: Reversing these expressions give weird results!
            g = ca.vertcat(
                g_lin + jac_g @ dx,
                self._g,
            )
            lbg = ca.vertcat(
                nlpdata.lbg[self.idx_g_nonlin],
                self._lbg,
            )
            ubg = ca.vertcat(
                nlpdata.ubg[self.idx_g_nonlin],
                self._ubg,
            )
        else:
            g = self._g
            lbg = self._lbg
            ubg = self._ubg

        f_k = self.f(self.x_sol_best, nlpdata.p)
        f_lin = self.grad_f_x_sub(self.x_sol_best, nlpdata.p)
        f_hess = self.f_hess(self.x_sol_best, nlpdata.p)
        f = f_k + f_lin.T @ dx + 0.5 * dx.T @ f_hess @ dx

        solver = ca.qpsol(f"benders_constraint_{self.nr_g}", self.settings.MIP_SOLVER, {
            "f": f, "g": g, "x": self._x, "p": self._nu
        }, self.options)

        sol = solver(
            x0=self.x_sol_best,
            lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=lbg, ubg=ubg,
            p=[self.y_N_val - 1e-4]
        )

        nlpdata.solved, stats = self.collect_stats("BTR-MIP", solver, sol)
        if nlpdata.solved:
            nlpdata.prev_solution = sol
        else:
            nlpdata.prev_solution = self.best_data
            print("FINAL ITERATION")
            # Final iteration
            nlpdata.solved = True
        return nlpdata
