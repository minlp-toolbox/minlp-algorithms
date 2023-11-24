"""Set of solvers based on Voronoi trust region."""

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from benders_exp.solvers import MiSolverClass, Stats, MinlpProblem, MinlpData, \
    get_idx_linear_bounds, regularize_options, get_idx_inverse, extract_bounds
from benders_exp.defines import MIP_SETTINGS, MIP_SOLVER, WITH_JIT, \
        CASADI_VAR, WITH_PLOT
from benders_exp.utils import to_0d


class VoronoiTrustRegionMILP(MiSolverClass):
    r"""
    Voronoi trust region problem.

    This implementation assumes the following input:
        min f(x)
        s.t. lb < g(x)

    It constructs the following problem:
        min f(x) + \nabla f(x) (x-x^i)
        s.t.
            lb \leq g(x) + \nabla  g(x) (x-x^i)
            NLPF constraints
            Voronoi trust region
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, options=None):
        """Improved outer approximation."""
        super(VoronoiTrustRegionMILP, self).__init___(problem, stats)
        if WITH_PLOT:
            self.setup_plot()
        self.options = regularize_options(options, MIP_SETTINGS)

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
        if problem.gn_hessian is not None:
            self.f_hess = ca.Function("gn_hess_f_x", [problem.x, problem.p], [
                                ca.hessian(problem.f, problem.x)[0]])
        else:
            self.f_hess = ca.Function("hess_f_x", [problem.x, problem.p], [
                                  ca.hessian(problem.f, problem.x)[0]])

        self.g = ca.Function(
            "g", [problem.x, problem.p], [problem.g],
            {"jit": WITH_JIT}
        )
        self.jac_g_bin = ca.Function(
            "jac_g_bin", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)[:, problem.idx_x_bin]],
            {"jit": WITH_JIT}
        )
        self.idx_g_lin = get_idx_linear_bounds(problem)
        self.idx_g_nonlin = get_idx_inverse(self.idx_g_lin, problem.g.shape[0])
        self.g_lin = ca.Function(
            "g", [problem.x, problem.p], [problem.g[self.idx_g_lin]],
            {"jit": WITH_JIT}
        )
        self.g_nonlin = ca.Function(
            "g", [problem.x, problem.p], [problem.g[self.idx_g_nonlin]],
            {"jit": WITH_JIT}
        )
        self.jac_g_nonlin = ca.Function(
            "gradient_g_x",
            [problem.x, problem.p], [ca.jacobian(
                problem.g[self.idx_g_nonlin], problem.x
            )], {"jit": WITH_JIT}
        )

        self._x = CASADI_VAR.sym("x_voronoi", problem.x.numel())
        self.idx_x_bin = problem.idx_x_bin
        self.nr_g_orig = problem.g.shape[0]
        self.nr_x_orig = problem.x.shape[0]
        # Copy the linear constraints in g
        self.nr_g, self._g, self._lbg, self._ubg = extract_bounds(
            problem, data, self.idx_g_lin, self._x, allow_fail=False
        )

        self.options["discrete"] = [1 if elm in problem.idx_x_bin else 0
                                    for elm in range(self._x.shape[0])]

        # Initialization for algorithm iterates
        self.ub = 1e15  # UB
        self.x_sol_list = []  # list of all the solution visited
        # Solution with best objective so far, we can safely assume the first
        # solution is the best even if infeasible since it is the only one
        # available yet.
        self.idx_best_x_sol = 0
        self.feasible_x_sol_list = []

        self.cut_id = 0

    def solve(self, nlpdata: MinlpData, prev_feasible=True, is_qp=False, relaxed=False) -> MinlpData:
        """Solve."""
        if relaxed:
            raise NotImplementedError()
        # Update with the lowest upperbound and the corresponding best solution:
        if nlpdata.x_sol.shape[0] == 1:
            x_sol = to_0d(nlpdata.x_sol)[np.newaxis]
        else:
            x_sol = to_0d(nlpdata.x_sol)[:self.nr_x_orig]
        self.x_sol_list.append(x_sol)
        self.feasible_x_sol_list.append(prev_feasible)
        if prev_feasible:
            if nlpdata.obj_val < self.ub:
                self.ub = nlpdata.obj_val
                # TODO: a surrogate for counting iterates, it's a bit clutter
                self.idx_best_x_sol = len(self.x_sol_list) - 1
                print(f"NEW BOUND {self.ub}")
        else:
            g_k, lbg_k, ubg_k = self._generate_infeasible_cut(self._x, x_sol, nlpdata.lam_g_sol, nlpdata.p)
            self._g = ca.vertcat(self._g, g_k)
            self._lbg = ca.vertcat(self._lbg, lbg_k)
            self._ubg = ca.vertcat(self._ubg, ubg_k)

        x_sol_best = self.x_sol_list[self.idx_best_x_sol]

        # Create a new voronoi cut
        g_voronoi, lbg_voronoi, ubg_voronoi = self._generate_voronoi_tr(self._x[self.idx_x_bin], nlpdata.p)

        if WITH_PLOT:
            self.visualize_trust_region(g_voronoi, self._x[self.idx_x_bin])

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
            jac_g = np.zeros((0, self.nr_x_orig))

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

        self.solver = ca.qpsol(
            f"voronoi_tr_milp_with_{self.nr_g}_constraints", MIP_SOLVER, {
            "f": f, "g": g, "x": self._x,
        }, self.options)

        nlpdata.prev_solution = self.solver(
            x0=x_sol_best,
            lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=lbg, ubg=ubg,
        )

        nlpdata.solved, stats = self.collect_stats("VTR-MILP")
        return nlpdata

    def _generate_infeasible_cut(self, x, x_sol, lam_g, p):
        """Generate infeasibility cut."""
        h_k = self.g(x_sol, p)
        jac_h_k = self.jac_g_bin(x_sol, p)
        g_k = lam_g.T @ (h_k + jac_h_k @ (x[self.idx_x_bin] - x_sol[self.idx_x_bin]))
        return g_k, -ca.inf, 0.0

    def _generate_voronoi_tr(self, x_bin, p):
        r"""
        Generate Voronoi trust region based on the visited integer solutions and the best integer solution so far.

        :param p: parameters
        """
        g_k = []
        lbg_k = []
        ubg_k = []

        x_sol_bin_best = self.x_sol_list[self.idx_best_x_sol][self.idx_x_bin]
        x_sol_bin_best_norm2_squared = x_sol_bin_best.T @ x_sol_bin_best
        for x_sol, is_feas in zip(self.x_sol_list, self.feasible_x_sol_list):
            x_sol_bin = x_sol[self.idx_x_bin]
            if is_feas and not np.allclose(x_sol_bin, x_sol_bin_best):
                a = ca.DM(2 * (x_sol_bin - x_sol_bin_best))
                b = ca.DM(x_sol_bin.T @ x_sol_bin - x_sol_bin_best_norm2_squared)
                g_k.append(a.T @ x_bin - b)
                lbg_k.append(-np.inf)
                ubg_k.append(0)

        return ca.vcat(g_k), lbg_k, ubg_k

    def setup_plot(self):
        """Set up plot."""
        self.fig = plt.figure()

    def visualize_trust_region(self, g_k, x_bin):
        """Visualize voronoi trust region in 2d."""
        if isinstance(g_k, CASADI_VAR):
            xlim = [0, 4]  # TODO parametric limits
            ylim = [0, 4]  # TODO parametric limits

            self.cut_id += 1
            self.ax = self.fig.add_subplot(3, 3, self.cut_id)
            self.ax.grid(linestyle='--', linewidth=1)
            self.ax.set_xlim(xlim[0]-0.5, xlim[1]+0.5)
            self.ax.set_xticks(range(int(xlim[0]), int(xlim[1]) + 1))
            self.ax.set_ylim(ylim[0]-0.5, ylim[1]+0.5)
            self.ax.set_yticks(range(int(ylim[0]), int(ylim[1]) + 1))

            points = np.linspace(-1, 5, 100)
            xx, yy = np.meshgrid(points, points)
            feasible = np.ones(xx.shape, dtype=bool)
            for c in range(g_k.shape[0]):
                cut = ca.Function("t", [x_bin], [g_k[c]])
                feasible_c = np.ones(xx.shape, dtype=bool)
                for i in range(points.shape[0]):
                    for j in range(points.shape[0]):
                        feasible_c[i, j] = cut(ca.vertcat(xx[i, j], yy[i, j])).full()[0, 0] < 0
                feasible = feasible & feasible_c
            self.ax.imshow(~feasible, cmap=plt.cm.binary, alpha=0.5,
                           origin="lower",
                           extent=[xlim[0]-1, xlim[1]+1, ylim[0]-1, ylim[1]+1])

            for i, x_sol in enumerate(self.x_sol_list):
                if i == self.idx_best_x_sol:
                    color = 'tab:orange'
                else:
                    color = 'tab:blue'
                self.ax.scatter(x_sol[0], x_sol[1], c=color)

            plt.show(block=False)
            plt.pause(1)
