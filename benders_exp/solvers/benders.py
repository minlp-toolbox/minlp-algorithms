"""
Set of solvers based on bender cuts.

Bender cuts are based on the principle of decomposing the problem into two
parts where the main part is only solving the integer variables.
"""

import matplotlib.pyplot as plt
import casadi as ca
import numpy as np
from benders_exp.solvers import SolverClass, Stats, MinlpProblem, MinlpData, \
        extract_linear_bounds_binary_x
from benders_exp.defines import WITH_LOGGING, WITH_JIT, CASADI_VAR, WITH_PLOT


class BendersMasterMILP(SolverClass):
    """Create benders master problem."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, options=None):
        """Create benders master MILP."""
        super(BendersMasterMILP, self).__init___(problem, stats)
        self.setup_common(problem, options)
        if WITH_PLOT:
            self.setup_plot()

        self.grad_f_x_sub_bin = ca.Function(
            "gradient_f_x_bin",
            [problem.x, problem.p], [ca.gradient(
                problem.f, problem.x
            )[problem.idx_x_bin]],
            {"jit": WITH_JIT}
        )
        self.jac_g_sub_bin = ca.Function(
            "jac_g_bin", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)[:, problem.idx_x_bin]],
            {"jit": WITH_JIT}
        )
        self._x = CASADI_VAR.sym("x_bin", self.nr_x_bin)
        self.cut_id = 0
        self.visualized_cuts = []

    def setup_common(self, problem: MinlpProblem, options):
        """Set up common data."""
        if options is None:
            if WITH_LOGGING:
                options = {}
            else:
                options = {"verbose": False,
                           "print_time": 0, "gurobi.output_flag": 0}

        self.f = ca.Function(
            "f", [problem.x, problem.p], [problem.f],
            {"jit": WITH_JIT}
        )
        self.g = ca.Function(
            "g", [problem.x, problem.p], [problem.g],
            {"jit": WITH_JIT}
        )

        self.idx_x_bin = problem.idx_x_bin
        self.nr_x_bin = len(problem.idx_x_bin)
        self._nu = CASADI_VAR.sym("nu", 1)
        self._g = np.array([])
        self.nr_g = 0
        self.options = options.copy()
        self.options["discrete"] = [1] * (self.nr_x_bin + 1)
        self.options["discrete"][-1] = 0
        self.options["gurobi.MIPGap"] = 0.05
        self.nr_g_orig = problem.g.shape[0]
        self.nr_x_orig = problem.x.shape[0]

    def _generate_cut_equation(self, x, x_sol, x_sol_sub_set, lam_g, p, prev_feasible):
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
            grad_f_k = self.grad_f_x_sub_bin(x_sol, p)
            jac_g_k = self.jac_g_sub_bin(x_sol, p)
            lambda_k = grad_f_k + jac_g_k.T @ lam_g
            f_k = self.f(x_sol, p)
            g_k = (
                f_k + lambda_k.T @ (x - x_sol_sub_set)
                - self._nu
            )
        else:  # Not feasible solution
            h_k = self.g(x_sol, p)
            jac_h_k = self.jac_g_sub_bin(x_sol, p)
            g_k = lam_g.T @ (h_k + jac_h_k @ (x - x_sol_sub_set))

        return g_k

    def solve(self, nlpdata: MinlpData, prev_feasible=True) -> MinlpData:
        """solve."""
        g_k = self._generate_cut_equation(
            self._x, nlpdata.x_sol[:self.nr_x_orig], nlpdata.x_sol[self.idx_x_bin],
            nlpdata.lam_g_sol, nlpdata.p, prev_feasible
        )

        if WITH_PLOT:
            self.visualize_cut(g_k, self._x, self._nu)

        self._g = ca.vertcat(self._g, g_k)
        self.nr_g += 1

        self.solver = ca.qpsol(f"benders_with_{self.nr_g}_cut", "gurobi", {
            "f": self._nu, "g": self._g,
            "x": ca.vertcat(self._x, self._nu),
        }, self.options)

        # This solver solves only to the binary variables (_x)!
        solution = self.solver(
            x0=ca.vertcat(nlpdata.x_sol[self.idx_x_bin], nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx[self.idx_x_bin], -1e5),
            ubx=ca.vertcat(nlpdata.ubx[self.idx_x_bin], ca.inf),
            lbg=-ca.inf * np.ones(self.nr_g),
            ubg=np.zeros(self.nr_g)
        )
        x_full = nlpdata.x_sol.full()[:self.nr_x_orig]
        x_full[self.idx_x_bin] = solution['x'][:-1]
        solution['x'] = x_full
        nlpdata.prev_solution = solution
        nlpdata.solved, stats = self.collect_stats()
        self.stats["milp_benders.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["milp_benders.iter"] += max(0, stats["iter_count"])
        return nlpdata

    def setup_plot(self):
        """Set up plot."""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def visualize_cut(self, g_k, x_bin, nu):
        """Visualize cut."""
        self.cut_id += 1
        xx, yy = np.meshgrid(range(10), range(10))
        cut = ca.Function("t", [x_bin, nu], [g_k])
        z = np.zeros(xx.shape)
        for i in range(10):
            for j in range(10):
                z[i, j] = cut(ca.vertcat(xx[i, j], yy[i, j]), 0).full()[0, 0]

        self.visualized_cuts.append(
            self.ax.plot_surface(
                xx, yy, z, alpha=0.2, label="Cut %d" % self.cut_id
            )
        )
        # HACK:
        for surf in self.visualized_cuts:
            surf._edgecolors2d = surf._edgecolor3d
            surf._facecolors2d = surf._facecolor3d
        # Legend
        self.ax.legend()

        plt.show(block=False)
        plt.pause(1)


class BendersMasterMIQP(BendersMasterMILP):
    """MIQP implementation."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, options=None):
        """Create the benders constraint MILP."""
        super(BendersMasterMIQP, self).__init__(problem, data, stats, options)
        self.f_hess_bin = ca.Function(
            "hess_f_x_bin",
            [problem.x, problem.p], [ca.hessian(
                problem.f, problem.x
            )[0][problem.idx_x_bin, :][:, problem.idx_x_bin]],
            {"jit": WITH_JIT}
        )

        # TODO: Strip all linear equations with binary variables only!
        # Format to g < 0 for simplicity
        # sparsity = ca.jacobian(problem.g, problem.x).sparsity
        g, lbg, ubg = extract_linear_bounds_binary_x(problem, data)

        self._g = ca.vertcat(*g)
        self._lbg = lbg
        self._ubg = ubg
        self.nr_g = len(g)

    def solve(self, nlpdata: MinlpData, prev_feasible=True) -> MinlpData:
        """solve."""
        x_bin_star = nlpdata.x_sol[self.idx_x_bin]
        g_k = self._generate_cut_equation(
            self._x, nlpdata.x_sol[:self.nr_x_orig], x_bin_star,
            nlpdata.lam_g_sol, nlpdata.p, prev_feasible
        )

        if WITH_PLOT:
            self.visualize_cut(g_k, self._x, self._nu)

        f_hess = self.f_hess_bin(nlpdata.x_sol[:self.nr_x_orig], nlpdata.p)
        self._g = ca.vertcat(self._g, g_k)
        self._ubg.append(0)
        self._lbg.append(-ca.inf)
        self.nr_g += 1

        dx = self._x - x_bin_star
        self.solver = ca.qpsol(f"benders_qp{self.nr_g}", "gurobi", {
            "f": self._nu + 0.5 * dx.T @ f_hess @ dx,
            "g": self._g,
            "x": ca.vertcat(self._x, self._nu),
        }, self.options)

        # This solver solves only to the binary variables (_x)!
        solution = self.solver(
            x0=ca.vertcat(nlpdata.x_sol[self.idx_x_bin], nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx[self.idx_x_bin], -1e5),
            ubx=ca.vertcat(nlpdata.ubx[self.idx_x_bin], ca.inf),
            lbg=self._lbg, ubg=self._ubg
        )
        obj = solution['x'][-1].full()
        if obj > solution['f']:
            raise Exception("Possible thougth mistake!")
        solution['f'] = obj

        x_full = nlpdata.x_sol.full()[:self.nr_x_orig]
        x_full[self.idx_x_bin] = solution['x'][:-1]
        solution['x'] = x_full
        nlpdata.prev_solution = solution
        nlpdata.solved, stats = self.collect_stats()
        self.stats["miqp_benders.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["miqp_benders.iter"] += max(0, stats["iter_count"])
        return nlpdata


class BendersConstraintMILP(BendersMasterMILP):
    """
    Create benders constraint MILP.

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

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create the benders constraint MILP."""
        super(BendersConstraintMILP, self).__init__(problem, stats, options)
        self.grad_f_x_sub = ca.Function(
            "gradient_f_x",
            [problem.x, problem.p], [ca.gradient(
                problem.f, problem.x
            )],
            {"jit": WITH_JIT}
        )
        self.jac_g_sub = ca.Function(
            "jac_g", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)],
            {"jit": WITH_JIT}
        )
        self.f_hess = ca.Function("hess_f_x", [problem.x, problem.p], [ca.hessian(problem.f, problem.x)[0]])

        self._x = CASADI_VAR.sym("x", self.nr_x_orig)
        self.y_N_val = 1e15  # Should be inf but can not at the moment ca.inf

    def solve(self, nlpdata: MinlpData, prev_feasible=True, integer=False) -> MinlpData:
        """Solve."""
        # Create a new cut
        x_sol = nlpdata.x_sol[:self.nr_x_orig]
        g_k = self._generate_cut_equation(
            self._x[self.idx_x_bin], x_sol, x_sol[self.idx_x_bin], nlpdata.lam_g_sol, nlpdata.p, prev_feasible
        )
        # If the upper bound improved, decrease it:
        if integer and prev_feasible:
            self.y_N_val = min(self.y_N_val, nlpdata.obj_val)
            print(f"NEW BOUND {self.y_N_val}")

        f_lin = self.grad_f_x_sub(nlpdata.x_sol[:self.nr_x_orig], nlpdata.p)
        g_lin = self.g(nlpdata.x_sol[:self.nr_x_orig], nlpdata.p)
        jac_g = self.jac_g_sub(nlpdata.x_sol[:self.nr_x_orig], nlpdata.p)
        # f_hess = self.f_hess(nlpdata.x_sol[:self.nr_x_orig], nlpdata.p)

        # TODO: When linearizing the bounds, remember they are two sided!
        # we need to take the other bounds into account as well
        self.solver = ca.qpsol(f"benders_constraint{self.nr_g}", "gurobi", {
            # "f": f_lin.T @ self._x + 0.5 * self._x.T @ f_hess @ self._x, #TODO: add a flag to solve the qp
            "f": f_lin.T @ self._x,
            "g": ca.vertcat(
                g_lin + jac_g @ self._x,  # TODO: Check sign error?
                self._g
            ),
            "x": self._x, "p": self._nu
        }, self.options)

        nlpdata.prev_solution = self.solver(
            x0=x_sol,
            lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=ca.vertcat(
                nlpdata.lbg,
                -ca.inf * np.ones(self.nr_g)
            ),
            ubg=ca.vertcat(
                # ca.inf * np.ones(self.nr_g_orig),
                # TODO: NEED TO TAKE INTO ACCOUNT: nlpdata.ubg,
                nlpdata.ubg,  # TODO: verify correctness
                np.zeros(self.nr_g)
            ),
            p=[self.y_N_val]
        )
        nlpdata.prev_solution['x'] = nlpdata.prev_solution['x'][:self.nr_x_orig]

        nlpdata.solved, stats = self.collect_stats()
        self.stats["milp_bconstraint.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["milp_bconstraint.iter"] += max(0, stats["iter_count"])
        self._g = ca.vertcat(self._g, g_k)
        self.nr_g += 1
        return nlpdata
