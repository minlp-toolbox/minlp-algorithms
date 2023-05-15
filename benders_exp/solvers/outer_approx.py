"""Set of solvers based on outer approximation."""

import casadi as ca
import numpy as np
from benders_exp.solvers import SolverClass, Stats, MinlpProblem, MinlpData
from benders_exp.defines import GUROBI_SETTINGS, WITH_LOGGING, WITH_JIT, CASADI_VAR


class OuterApproxMILP(SolverClass):
    r"""
    Outer approximation.

    This implementation assumes the following input:
        min f(x)
        s.t. lb < g(x)

    It constructs the followign problem:
        min \alpha
        s.t.
            \alpha \geq f(x) + \nabla (x) (x-x^i)
            lb \leq g(x) + \nabla  g(x) (x-x^i)
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, options=None):
        """Create benders master MILP."""
        super(OuterApproxMILP, self).__init___(problem, stats)
        if options is None:
            if WITH_LOGGING:
                self.options = {}
            else:
                self.options = {
                    "verbose": False, "print_time": 0, "gurobi.output_flag": 0
                }
        else:
            self.options = options.copy()

        self.f = ca.Function(
            "f", [problem.x, problem.p], [problem.f],
            {"jit": WITH_JIT}
        )
        self.grad_f = ca.Function(
            "gradient_f_x",
            [problem.x, problem.p], [ca.gradient(
                problem.f, problem.x
            )], {"jit": WITH_JIT}
        )
        self.g = ca.Function(
            "g", [problem.x, problem.p], [problem.g],
            {"jit": WITH_JIT}
        )
        self.jac_g = ca.Function(
            "gradient_g_x",
            [problem.x, problem.p], [ca.jacobian(
                problem.g, problem.x
            )], {"jit": WITH_JIT}
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
        self.options.update(GUROBI_SETTINGS)
        self.options.update({
            "discrete": discrete,
        })

    def solve(self, nlpdata: MinlpData, prev_feasible=True) -> MinlpData:
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
        self._ubg = ca.vertcat(self._ubg, ca.inf * np.ones((self.nr_g_orig, 1)))

        self.solver = ca.qpsol(f"oa_with_{self._g.shape[0]}_cut", "gurobi", {
            "f": self._alpha, "g": self._g,
            "x": self._x,
        }, self.options)

        nlpdata.prev_solution = self.solver(
            x0=ca.vertcat(x_sol, nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx, -1e8),
            ubx=ca.vertcat(nlpdata.ubx, ca.inf),
            lbg=self._lbg,
            ubg=self._ubg,
        )
        nlpdata.prev_solution['x'] = nlpdata.prev_solution['x'][:self.nr_x]
        nlpdata.solved, stats = self.collect_stats()
        self.stats["milp_oa.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["milp_oa.iter"] += max(0, stats["iter_count"])
        return nlpdata
