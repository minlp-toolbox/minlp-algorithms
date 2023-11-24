"""
Set of solvers based on bender cuts.

Bender cuts are based on the principle of decomposing the problem into two
parts where the main part is only solving the integer variables.
"""

from copy import deepcopy
import matplotlib.pyplot as plt
import casadi as ca
import numpy as np
from benders_exp.solvers import MiSolverClass, Stats, MinlpProblem, MinlpData, \
    get_idx_linear_bounds, get_idx_linear_bounds_binary_x, regularize_options, \
    get_idx_inverse, extract_bounds
from benders_exp.defines import Settings, CASADI_VAR
from benders_exp.utils import to_0d
import logging

logger = logging.getLogger(__name__)


class BendersMasterMILP(MiSolverClass):
    """Create benders master problem."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats,
                 s: Settings, with_lin_bounds=True):
        """Create benders master MILP."""
        super(BendersMasterMILP, self).__init___(problem, stats, s)
        self.setup_common(problem, s)
        if s.WITH_PLOT:
            self.setup_plot()

        self.jac_g_bin = ca.Function(
            "jac_g_bin", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)[:, problem.idx_x_bin]],
            {"jit": s.WITH_JIT}
        )
        self._x = CASADI_VAR.sym("x_bin", self.nr_x_bin)

        if with_lin_bounds:
            idx_g_lin = get_idx_linear_bounds_binary_x(problem)
            self.nr_g, self._g, self._lbg, self._ubg = extract_bounds(
                problem, data, idx_g_lin, self._x, problem.idx_x_bin
            )
        else:
            self.nr_g, self._g, self._lbg, self._ubg = 0, [], [], []

        self.cut_id = 0
        self.visualized_cuts = []

    def setup_common(self, problem: MinlpProblem, s):
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

        self.idx_x_bin = problem.idx_x_bin
        self.nr_x_bin = len(problem.idx_x_bin)
        self._nu = CASADI_VAR.sym("nu", 1)
        self.options["discrete"] = [1] * (self.nr_x_bin + 1)
        self.options["discrete"][-1] = 0
        self.nr_g_orig = problem.g.shape[0]
        self.nr_x_orig = problem.x.shape[0]

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
            # TODO: understand why need the minus!
            lambda_k = -lam_x[self.idx_x_bin]
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

    def solve(self, nlpdata: MinlpData, prev_feasible=True) -> MinlpData:
        """solve."""
        x_bin_star = nlpdata.x_sol[self.idx_x_bin]
        g_k = self._generate_cut_equation(
            self._x, nlpdata.x_sol[:self.nr_x_orig], x_bin_star,
            nlpdata.lam_g_sol, nlpdata.lam_x_sol, nlpdata.p,
            nlpdata.lbg, nlpdata.ubg, prev_feasible
        )
        self.cut_id += 1

        self._g = ca.vertcat(self._g, g_k)
        self._ubg.append(0)
        self._lbg.append(-ca.inf)
        self.nr_g += 1

        if self.settings.WITH_PLOT:
            self.visualize_trust_region(self._x[self.idx_x_bin], self._nu,
                                        x_sol_current=x_bin_star,
                                        x_sol_best=x_bin_star)

        self.solver = ca.qpsol(f"benders_with_{self.nr_g}_cut", self.settings.MIP_SOLVER, {
            "f": self._nu, "g": self._g,
            "x": ca.vertcat(self._x, self._nu),
        }, self.options)

        # This solver solves only to the binary variables (_x)!
        solution = self.solver(
            x0=ca.vertcat(x_bin_star, nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx[self.idx_x_bin], -1e5),
            ubx=ca.vertcat(nlpdata.ubx[self.idx_x_bin], ca.inf),
            lbg=ca.vertcat(*self._lbg), ubg=ca.vertcat(*self._ubg)
        )
        x_full = nlpdata.x_sol.full()[:self.nr_x_orig]
        x_full[self.idx_x_bin] = solution['x'][:-1]
        solution['x'] = x_full
        nlpdata.prev_solution = solution
        nlpdata.solved, stats = self.collect_stats("BENDERS-MILP")
        return nlpdata

    def setup_3d_cut_plot(self, problem: MinlpProblem, data: MinlpData):
        """Set up 3d plot."""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(data.lbx[problem.idx_x_bin]
                         [0], data.ubx[problem.idx_x_bin][0])
        self.ax.set_xticks(range(int(data.lbx[problem.idx_x_bin][0]), int(
            data.ubx[problem.idx_x_bin][0]) + 1))
        self.ax.set_ylim(data.lbx[problem.idx_x_bin]
                         [1], data.ubx[problem.idx_x_bin][1])
        self.ax.set_yticks(range(int(data.lbx[problem.idx_x_bin][1]), int(
            data.ubx[problem.idx_x_bin][1]) + 1))

    def setup_plot(self):
        """Set up plot."""
        self.fig = plt.figure()

    def visualize_cut(self, g_k, x_bin, nu, nu_val=0):
        """Visualize cut."""
        self.cut_id += 1
        xx, yy = np.meshgrid(range(10), range(10))
        cut = ca.Function("t", [x_bin, nu], [g_k])
        z = np.zeros(xx.shape)
        for i in range(10):
            for j in range(10):
                z[i, j] = cut(ca.vertcat(xx[i, j], yy[i, j]),
                              nu_val).full()[0, 0]

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
        self.ax.axes.set_zlim3d(bottom=-1e5, top=0)

        plt.show(block=False)
        plt.pause(1)

    def visualize_trust_region(self, x_bin, nu, nu_val=0, x_sol_current=None, x_sol_best=None):
        """Visualize trust region."""
        g_k = self._g[-self.cut_id:]
        if x_sol_current is not None:
            x_sol_current = deepcopy(x_sol_current)
            x_sol_current = to_0d(x_sol_current)
        if x_sol_best is not None:
            x_sol_best = deepcopy(x_sol_best)
            x_sol_best = to_0d(x_sol_best)

        if isinstance(g_k, CASADI_VAR):
            xlim = [0, 4]  # TODO parametric limits
            ylim = [0, 4]  # TODO parametric limits

            self.ax = self.fig.add_subplot(3, 4, self.cut_id)
            self.ax.grid(linestyle='--', linewidth=1)
            self.ax.set_xlim(xlim[0]-0.5, xlim[1]+0.5)
            self.ax.set_xticks(range(int(xlim[0]), int(xlim[1]) + 1))
            self.ax.set_ylim(ylim[0]-0.5, ylim[1]+0.5)
            self.ax.set_yticks(range(int(ylim[0]), int(ylim[1]) + 1))

            points = np.linspace(-1, 5, 100)
            xx, yy = np.meshgrid(points, points)
            feasible = np.ones(xx.shape, dtype=bool)
            for c in range(g_k.shape[0]):
                cut = ca.Function("t", [x_bin, nu], [g_k[c]])
                feasible_c = np.ones(xx.shape, dtype=bool)
                for i in range(points.shape[0]):
                    for j in range(points.shape[0]):
                        feasible_c[i, j] = cut(ca.vertcat(
                            xx[i, j], yy[i, j]), nu_val).full()[0, 0] < 0
                feasible = feasible & feasible_c
            self.ax.imshow(~feasible, cmap=plt.cm.binary, alpha=0.5,
                           origin="lower",
                           extent=[xlim[0]-1, xlim[1]+1, ylim[0]-1, ylim[1]+1]
                           )
            if x_sol_best is not None:
                self.ax.scatter(x_sol_best[0], x_sol_best[1], c='tab:orange')
                if not np.allclose(x_sol_best, x_sol_current):
                    self.ax.scatter(
                        x_sol_current[0], x_sol_current[1], c='tab:blue')

            plt.show(block=False)
            plt.pause(1)


class BendersMasterMIQP(BendersMasterMILP):
    """MIQP implementation."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings):
        """Create the benders constraint MILP."""
        super(BendersMasterMIQP, self).__init__(problem, data, stats, s)
        self.f_hess_bin = ca.Function(
            "hess_f_x_bin",
            [problem.x, problem.p], [ca.hessian(
                problem.f, problem.x
            )[0][problem.idx_x_bin, :][:, problem.idx_x_bin]],
            {"jit": self.settings.WITH_JIT}
        )

    def solve(self, nlpdata: MinlpData, prev_feasible=True) -> MinlpData:
        """solve."""
        x_bin_star = nlpdata.x_sol[self.idx_x_bin]
        g_k = self._generate_cut_equation(
            self._x, nlpdata.x_sol[:self.nr_x_orig], x_bin_star,
            nlpdata.lam_g_sol, nlpdata.lam_x_sol, nlpdata.p,
            nlpdata.lbg, nlpdata.ubg, prev_feasible
        )
        self.cut_id += 1

        f_hess = self.f_hess_bin(nlpdata.x_sol[:self.nr_x_orig], nlpdata.p)
        self._g = ca.vertcat(self._g, g_k)
        self._ubg.append(0)
        self._lbg.append(-ca.inf)
        self.nr_g += 1

        if self.settings.WITH_PLOT:
            self.visualize_trust_region(self._x[self.idx_x_bin], self._nu,
                                        x_sol_best=x_bin_star,
                                        x_sol_current=x_bin_star)

        dx = self._x - x_bin_star
        self.solver = ca.qpsol(f"benders_qp{self.nr_g}", self.settings.MIP_SOLVER, {
            "f": self._nu + 0.5 * dx.T @ f_hess @ dx,
            "g": self._g,
            "x": ca.vertcat(self._x, self._nu),
        }, self.options)

        # This solver solves only to the binary variables (_x)!
        solution = self.solver(
            x0=ca.vertcat(x_bin_star, nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx[self.idx_x_bin], -1e5),
            ubx=ca.vertcat(nlpdata.ubx[self.idx_x_bin], ca.inf),
            lbg=ca.vertcat(*self._lbg), ubg=ca.vertcat(*self._ubg)
        )
        obj = solution['x'][-1].full()
        if obj > solution['f']:
            raise Exception("Possible thougth mistake!")
        solution['f'] = obj

        x_full = nlpdata.x_sol.full()[:self.nr_x_orig]
        x_full[self.idx_x_bin] = solution['x'][:-1]
        solution['x'] = x_full
        nlpdata.prev_solution = solution
        nlpdata.solved, stats = self.collect_stats("BENDERS-MIQP")
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

        self._x = CASADI_VAR.sym("x_benders", problem.x.numel())
        self.nr_g, self._g, self._lbg, self._ubg = extract_bounds(
            problem, data, self.idx_g_lin, self._x, allow_fail=False
        )

        self.options.update({
            "discrete": [1 if elm in problem.idx_x_bin else 0 for elm in range(self._x.shape[0])],
            "error_on_fail": False
        })
        self.y_N_val = 1e15  # Should be inf but can not at the moment ca.inf
        # We take a point
        self.best_data = None
        self.x_sol_best = data.x0
        self.iter = 0

    def solve(self, nlpdata: MinlpData, prev_feasible=True, relaxed=False) -> MinlpData:
        """Solve."""
        # Update with the lowest upperbound and the corresponding best solution:
        if relaxed:
            self.x_sol_best = nlpdata.x_sol[:self.nr_x_orig]
        elif nlpdata.obj_val < self.y_N_val and prev_feasible:
            self.y_N_val = nlpdata.obj_val
            self.x_sol_best = nlpdata.x_sol[:self.nr_x_orig]
            self.best_data = nlpdata._sol
            print(f"NEW BOUND {self.y_N_val}")

        # Create a new cut
        x_sol_prev = nlpdata.x_sol[:self.nr_x_orig]
        g_k = self._generate_cut_equation(
            self._x[self.idx_x_bin], x_sol_prev, x_sol_prev[self.idx_x_bin],
            nlpdata.lam_g_sol, nlpdata.lam_x_sol, nlpdata.p,
            nlpdata.lbg, nlpdata.ubg, prev_feasible
        )
        self.cut_id += 1

        self._g = ca.vertcat(self._g, g_k)
        self._lbg = ca.vertcat(self._lbg, -ca.inf)
        # Should be at least 1e-4 better and 1e-4 from the constraint bound!
        self._ubg = ca.vertcat(self._ubg, 0)  # -1e-4)
        self.nr_g += 1

        if self.settings.WITH_PLOT:
            self.visualize_trust_region(self._x[self.idx_x_bin], self._nu, nu_val=self.y_N_val,
                                        x_sol_best=self.x_sol_best[self.idx_x_bin],
                                        x_sol_current=x_sol_prev[self.idx_x_bin])

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

        self.solver = ca.qpsol(f"benders_constraint_{self.nr_g}", self.settings.MIP_SOLVER, {
            "f": f, "g": g, "x": self._x, "p": self._nu
        }, self.options)

        sol = self.solver(
            x0=self.x_sol_best,
            lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=lbg, ubg=ubg,
            p=[self.y_N_val - 1e-4]
        )

        nlpdata.solved, stats = self.collect_stats("BTR-MIP")
        if nlpdata.solved:
            nlpdata.prev_solution = sol
        else:
            nlpdata.prev_solution = self.best_data
            print("FINAL ITERATION")
            # Final iteration
            nlpdata.solved = True

        return nlpdata
