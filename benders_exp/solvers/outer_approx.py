"""Set of solvers based on outer approximation."""

import casadi as ca
import numpy as np
from benders_exp.solvers import MiSolverClass, Stats, MinlpProblem, MinlpData, \
    get_idx_linear_bounds, regularize_options, get_idx_inverse, extract_bounds
from benders_exp.defines import Settings, CASADI_VAR


class OuterApproxMILP(MiSolverClass):
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

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings):
        """Create benders master MILP."""
        super(OuterApproxMILP, self).__init___(problem, stats, s)
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
        # Last one is alpha
        self._x = CASADI_VAR.sym("x", self.nr_x + 1)
        self._g = np.array([])
        self._lbg = []
        self._ubg = []
        self._alpha = self._x[-1]

        discrete = [0] * (self.nr_x+1)
        for i in problem.idx_x_bin:
            discrete[i] = 1
        self.options.update({
            "discrete": discrete,
        })

    def solve(self, nlpdata: MinlpData, prev_feasible=True, relaxed=False) -> MinlpData:
        """Solve the outer approximation MILP."""
        x_sol = nlpdata.x_sol[:self.nr_x]
        if prev_feasible:
            self._g = ca.vertcat(
                self._g,
                self.f(x_sol, nlpdata.p) + self.grad_f(x_sol, nlpdata.p).T @
                (self._x[:self.nr_x] - x_sol) - self._alpha
            )
            self._lbg = ca.vertcat(self._lbg, -ca.inf)
            self._ubg = ca.vertcat(self._ubg, 0)

        g_lin = self.g(nlpdata.x_sol[:self.nr_x], nlpdata.p)
        jac_g = self.jac_g(nlpdata.x_sol[:self.nr_x], nlpdata.p)
        self._g = ca.vertcat(
            self._g,
            g_lin + jac_g @ (self._x[:self.nr_x] - x_sol)
        )
        self._lbg = ca.vertcat(self._lbg, nlpdata.lbg)
        self._ubg = ca.vertcat(self._ubg, nlpdata.ubg)

        self.solver = ca.qpsol(f"oa_with_{self._g.shape[0]}_cut", self.settings.MIP_SOLVER, {
            "f": self._alpha, "g": self._g,
            "x": self._x,
        }, self.options)

        sol = self.solver(
            x0=ca.vertcat(x_sol, nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx, -1e8),
            ubx=ca.vertcat(nlpdata.ubx, ca.inf),
            lbg=self._lbg,
            ubg=self._ubg,
        )
        sol['x'] = sol['x'][:self.nr_x]
        nlpdata.prev_solution = sol
        nlpdata.solved, stats = self.collect_stats("OA-MILP")
        return nlpdata


class OuterApproxMILPImproved(MiSolverClass):
    """Improvements over regular OA."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings):
        """Improved outer approximation."""
        super(OuterApproxMILPImproved, self).__init___(problem, stats, s)
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

        self.nr_x = problem.x.shape[0]
        # Last one is alpha
        self._x = CASADI_VAR.sym("x", self.nr_x + 1)
        _, self._g, self._lbg, self._ubg = extract_bounds(
            problem, data, self.idx_g_lin, self._x[:-1], allow_fail=False
        )
        self._alpha = self._x[-1]

        discrete = [0] * (self.nr_x+1)
        for i in problem.idx_x_bin:
            discrete[i] = 1
        self.options.update({
            "discrete": discrete,
        })

    def solve(self, nlpdata: MinlpData, prev_feasible=True, relaxed=False) -> MinlpData:
        """Solve the outer approximation MILP."""
        x_sol = nlpdata.x_sol[:self.nr_x]
        if prev_feasible:
            self._g = ca.vertcat(
                self._g,
                self.f(x_sol, nlpdata.p) + self.grad_f(x_sol, nlpdata.p).T @
                (self._x[:self.nr_x] - x_sol) - self._alpha
            )
            self._lbg = ca.vertcat(self._lbg, -ca.inf)
            self._ubg = ca.vertcat(self._ubg, 0)

        g_lin = self.g_nonlin(nlpdata.x_sol[:self.nr_x], nlpdata.p)
        jac_g = self.jac_g_nonlin(nlpdata.x_sol[:self.nr_x], nlpdata.p)
        self._g = ca.vertcat(
            self._g,
            g_lin + jac_g @ (self._x[:self.nr_x] - x_sol)
        )
        self._lbg = ca.vertcat(self._lbg, nlpdata.lbg[self.idx_g_nonlin])
        self._ubg = ca.vertcat(self._ubg, nlpdata.ubg[self.idx_g_nonlin])

        self.solver = ca.qpsol(f"oa_with_{self._g.shape[0]}_cut", self.settings.MIP_SOLVER, {
            "f": self._alpha, "g": self._g,
            "x": self._x,
        }, self.options)

        sol = self.solver(
            x0=ca.vertcat(x_sol, nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx, -1e8),
            ubx=ca.vertcat(nlpdata.ubx, ca.inf),
            lbg=self._lbg,
            ubg=self._ubg,
        )
        sol['x'] = sol['x'][:self.nr_x]
        nlpdata.prev_solution = sol
        nlpdata.solved, stats = self.collect_stats("OAI-MILP")
        return nlpdata
