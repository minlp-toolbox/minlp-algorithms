"""A mix of solvers."""
import numpy as np
import casadi as ca
from benders_exp.solvers import Stats, MinlpProblem, MinlpData, \
    get_idx_linear_bounds, get_idx_inverse, extract_bounds
from benders_exp.defines import WITH_JIT, CASADI_VAR, EPS
from benders_exp.solvers.benders import BendersMasterMILP
from enum import Enum
import logging

logger = logging.getLogger(__name__)
try:
    from colored import fg, stylize

    def colored(text, color="red"):
        """Color a text."""
        logger.info(stylize(text, fg(color)))
except Exception:
    def colored(text, color=None):
        """Color a text."""
        print(text)


class NonconvexStrategy(Enum):
    """Nonconvex strategy."""

    DISTANCE_BASED = 0
    GRADIENT_BASED = 1


class TrustRegionStrategy(Enum):
    """Trust region strategies."""

    DISTANCE_CORRECTION = 0
    TRUSTREGION_EXPANSION = 1
    GRADIENT_AMPLIFICATION = 2


class LowerApproximation:
    """Store info on benders bounds."""

    def __init__(self, x, nu):
        """Store info on benders bounds."""
        self.nr = 0
        self.g = []
        self.dg = []
        self.dg_corrected = []
        self.x_lin = []
        self.multipliers = []
        self.x = x
        self.nu = nu

    def add(self, point, value, gradient, gradient_corrected=None):
        """Add a benders bound cut."""
        self.nr += 1
        self.x_lin.append(point)
        self.g.append(value)
        self.dg.append(gradient)
        if gradient_corrected is None:
            self.dg_corrected.append(gradient)
        else:
            self.dg_corrected.append(gradient_corrected)
        if not self.multipliers:
            self.multipliers.append(1)
        else:
            self.multipliers.append(self.multipliers[0])  # the new constraint added should have the same multiplier of the others (cf. gradient-amplification strategy)

    def __call__(self, x_value, nu=0):
        """Evaluate the bounds."""
        return [
            gi + m * dgi.T @ (x_value - xi) - nu
            for gi, dgi, m, xi in zip(
                self.g, self.dg_corrected, self.multipliers, self.x_lin)
        ]

    def to_generic(self):
        """Create bounds."""
        return Constraints(
            self.nr,
            ca.vertcat(*self(self.x, self.nu)),
            -ca.inf * np.ones(self.nr),
            np.zeros(self.nr),
        )

    def __add__(self, other):
        """Add bounds."""
        return self.to_generic() + other


class Constraints:
    """Store bounds."""

    def __init__(self, nr, eq, lb, ub):
        """Store bounds."""
        self.nr = nr
        self.eq = eq
        self.lb = ca.DM(lb)
        self.ub = ca.DM(ub)

    def add(self, lb, eq, ub):
        """Add a bound."""
        self.nr += 1
        self.eq = ca.vertcat(self.eq, eq)
        self.lb = ca.vertcat(self.lb, lb)
        self.ub = ca.vertcat(self.ub, ub)

    def to_generic(self):
        """Convert to a generic class."""
        return self

    def __add__(self, other):
        """Add two bounds."""
        other = other.to_generic()
        return Constraints(
            self.nr + other.nr,
            ca.vertcat(self.eq, other.eq),
            ca.vertcat(self.lb, other.lb),
            ca.vertcat(self.ub, other.ub)
        )

    def __str__(self):
        """Represent."""
        out = f"Eq: {self.nr}\n\n"
        for i in range(self.nr):
            out += f"{self.lb[i]} <= {self.eq[i]} <= {self.ub[i]}\n"
        return out


def almost_equal(a, b):
    """Check if almost equal."""
    return a + EPS > b and a - EPS < b

def compute_gradient_correction(x_best, x_new, obj_best, obj_new, grad):
    """Compute gradient correction, based on L2 norm."""

    with_ipopt = False
    W = np.eye(x_best.shape[0]) # weighting matrix L2-norm
    W_inv = np.linalg.inv(W)

    if with_ipopt:
        nr_x = x_best.numel()
        grad_corr = CASADI_VAR.sym("gradient_correction", nr_x)
        obj = ca.norm_2(grad_corr)**2
        g = (obj_new - obj_best - EPS) + (grad + grad_corr).T @ (x_best - x_new)
        solver = ca.nlpsol("solver", "ipopt", {
            "f": obj, "g": g, "x": grad_corr}, {})
        sol = solver(x0=np.abs(x_new - x_best), lbx=-ca.inf * np.ones(nr_x),
                    ubx=ca.inf * np.ones(nr_x), lbg=-ca.inf, ubg=0)
        print(sol["x"])
        return sol["x"] + grad
    else:
        delta_x = x_best - x_new
        residual = obj_best - obj_new - grad.T @ delta_x
        return grad + (residual / (delta_x.T @ W_inv @ delta_x)) @ W_inv @ delta_x


class BendersTRandMaster(BendersMasterMILP):
    """Mixing the idea from Moritz with a slightly altered version of benders masters."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, options=None):
        """Create the benders constraint MILP."""
        super(BendersTRandMaster, self).__init__(problem, data, stats, options)
        # Settings
        self.nonconvex_strategy = NonconvexStrategy.GRADIENT_BASED
        self.nonconvex_strategy_alpha = 0.2
        self.trust_region_feasibility_strategy = TrustRegionStrategy.GRADIENT_AMPLIFICATION
        self.trust_region_feasibility_rho = 1.5

        # Setups
        self.setup_common(problem, options)
        self.idx_g_lin = get_idx_linear_bounds(problem)
        self.idx_g_nonlin = get_idx_inverse(self.idx_g_lin, problem.g.shape[0])

        self.grad_f_x_sub = ca.Function(
            "gradient_f_x",
            [problem.x, problem.p], [ca.gradient(
                problem.f, problem.x
            )],
            {"jit": WITH_JIT}
        )
        self.jac_g = ca.Function(
            "jac_g", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)],
            {"jit": WITH_JIT}
        )
        self.f_hess = ca.Function("hess_f_x", [problem.x, problem.p], [
                                  ca.hessian(problem.f, problem.x)[0]])

        self._x = CASADI_VAR.sym("x_benders", problem.x.numel())
        self._x_bin = self._x[problem.idx_x_bin]
        self.g_lin = Constraints(*extract_bounds(
            problem, data, self.idx_g_lin, self._x, allow_fail=False
        ))
        self.g_lowerapprox = LowerApproximation(self._x_bin, self._nu)
        self.g_infeasible = Constraints(0, [], [], [])
        self.g_other = Constraints(0, [], [], [])

        self.options.update({"discrete": [
                            1 if elm in problem.idx_x_bin else 0 for elm in range(self._x.shape[0])]})
        self.options_master = self.options.copy()
        self.options_master["discrete"] = self.options["discrete"] + [0]

        self.y_N_val = 1e15  # Should be inf but can not at the moment ca.inf
        self.x_sol_best = data.x0 # take a point as initialization
        self.qp_stagnates = False

    def _check_cut_valid(self, g, grad_g, x_best, x_sol, x_sol_obj):
        """Check if the cut is valid."""
        value = g + grad_g.T @ (x_best - x_sol)
        # print(f"Cut valid (lower bound)?: {value} vs real {x_sol_obj}")
        return (value - EPS <= x_sol_obj) # TODO: check sign EPS

    def _add_infeasible_cut(self, nlpdata: MinlpData):
        """Create infeasibility cut."""
        x_sol = nlpdata.x_sol[:self.nr_x_orig]
        h_k = self.g(x_sol, nlpdata.p)
        jac_h_k = self.jac_g_bin(x_sol, nlpdata.p)
        g_k = nlpdata.lam_g_sol.T @ (
            h_k + jac_h_k @ (self._x_bin - x_sol[self.idx_x_bin])
        )
        self.g_infeasible.add(-ca.inf, g_k, 0)

    def _gradient_amplification(self):
        if self.trust_region_feasibility_strategy == TrustRegionStrategy.GRADIENT_AMPLIFICATION:
        # Amplify the gradient of every new cut with the chosen rho.
            for i, m in enumerate(self.g_lowerapprox.multipliers):
                if m != self.trust_region_feasibility_rho:
                    self.g_lowerapprox.multipliers[i] = self.trust_region_feasibility_rho
        else:
            raise NotImplementedError()

    def _gradient_correction(self, nlpdata: MinlpData):
        # TODO: (to improve computation speed) if the best point does not change, check only the last point
        x_sol_best_bin = self.x_sol_best[self.idx_x_bin]

        # On the last integer point: check, correct (if needed) and add to g_lowerapprox
        lambda_k = -nlpdata.lam_x_sol[self.idx_x_bin]
        f_k = self.f(nlpdata.x_sol, nlpdata.p)
        x_bin_new = nlpdata.x_sol[self.idx_x_bin]

        if not self._check_cut_valid(f_k, lambda_k, x_sol_best_bin, x_bin_new, self.y_N_val):
            grad_corr = compute_gradient_correction(
                x_sol_best_bin, x_bin_new, self.y_N_val, f_k, lambda_k)
            self.g_lowerapprox.add(x_bin_new, f_k, lambda_k, grad_corr)
            logger.debug(f"Correcting gradient at {x_bin_new=}")
        else:
            self.g_lowerapprox.add(x_bin_new, f_k, lambda_k)

        # Check and correct - if necessary - all the points in memory
        for i in range(self.g_lowerapprox.nr - 1):
            self.g_lowerapprox.dg_corrected[i] = self.g_lowerapprox.dg[i] # Reset the corrected gradient to original
            if not self._check_cut_valid(
                self.g_lowerapprox.g[i], self.g_lowerapprox.dg[i],
                x_sol_best_bin, self.g_lowerapprox.x_lin[i], self.y_N_val
            ):
                self.g_lowerapprox.dg_corrected[i] = compute_gradient_correction(
                    x_sol_best_bin, self.g_lowerapprox.x_lin[i], self.y_N_val,
                    self.g_lowerapprox.g[i], self.g_lowerapprox.dg[i])
                logger.debug(f"Correcting gradient at {self.g_lowerapprox.x_lin[i]=}")

    def _trust_region_update(self, nlpdata: MinlpData):
        """
        Update TR via gradient correction and amplification
        """
        # if self.g_lowerapprox.nr == 0:  # There are no inequalities to generate cuts
        #     return

        self._gradient_correction(nlpdata)
        self._gradient_amplification()

    def _trust_region_is_empty(self):
        if self.g_lowerapprox.nr == 0:
            return False
        g_val = self.g_lowerapprox(self.x_sol_best[self.idx_x_bin])
        if (diff := np.max(np.array(g_val) - self.y_N_val)) > 0:
            return True

    def _get_g_linearized_nonlin(self, x, dx, nlpdata):
        g_lin = self.g(x, nlpdata.p)[self.idx_g_nonlin]
        if g_lin.numel() > 0:
            jac_g = self.jac_g(x, nlpdata.p)[self.idx_g_nonlin, :]

            return Constraints(
                g_lin.numel(),
                g_lin + jac_g @ dx,
                nlpdata.lbg[self.idx_g_nonlin],
                nlpdata.ubg[self.idx_g_nonlin],
            )
        else:
            return Constraints(0, [], [], [])

    def _solve_trust_region_problem(self, nlpdata: MinlpData, is_qp=True) -> MinlpData:
        """Solve QP problem."""
        dx = self._x - self.x_sol_best

        f_k = self.f(self.x_sol_best, nlpdata.p)
        f_lin = self.grad_f_x_sub(self.x_sol_best, nlpdata.p)
        if is_qp:
            f_hess = self.f_hess(self.x_sol_best, nlpdata.p)
            f = f_k + f_lin.T @ dx + 0.5 * dx.T @ f_hess @ dx
        else:
            f = f_k + f_lin.T @ dx

        g_cur_lin = self._get_g_linearized_nonlin(self.x_sol_best, dx, nlpdata)
        g_total = (
            g_cur_lin + self.g_lin + self.g_lowerapprox
            + self.g_infeasible + self.g_other
        )

        self.solver = ca.qpsol(f"benders_constraint_{self.g_lowerapprox.nr}", "gurobi", {
            "f": f, "g": g_total.eq, "x": self._x, "p": self._nu
        }, self.options)

        solution = self.solver(
            x0=self.x_sol_best,
            lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=g_total.lb, ubg=g_total.ub,
            p=[self.y_N_val + EPS]
        )
        logger.info("SOLVED TR-MIQP")
        return solution

    def _solve_benders_problem(self, nlpdata: MinlpData) -> MinlpData:
        """Solve benders master problem with one OA constraint."""
        dx = self._x - self.x_sol_best

        f_k = self.f(self.x_sol_best, nlpdata.p)
        f_lin = self.grad_f_x_sub(self.x_sol_best, nlpdata.p)
        f = f_k + f_lin.T @ dx

        # Adding the following linearization might not be the best idea since
        # They can lead to false results!
        # g_cur_lin = self._get_g_linearized_nonlin(self.x_sol_best, dx, nlpdata)
        g_total = (
            self.g_lin + self.g_lowerapprox + self.g_infeasible
        )
        # Add extra constraint (one step OA):
        g_total.add(-ca.inf, f - self._nu, 0)
        g, ubg, lbg = g_total.eq, g_total.ub, g_total.lb

        self.solver = ca.qpsol(f"benders_with_{self.g_lowerapprox.nr}_cut", "gurobi", {
            "f": self._nu, "g": g,
            "x": ca.vertcat(self._x, self._nu),
        }, self.options_master)

        solution = self.solver(
            x0=ca.vertcat(nlpdata.x_sol[:self.nr_x_orig], nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx, -1e5),
            ubx=ca.vertcat(nlpdata.ubx, ca.inf),
            lbg=lbg, ubg=ubg
        )

        solution['x'] = solution['x'][:-1]
        logger.info("SOLVED LB-MILP")
        return solution

    def solve(self, nlpdata: MinlpData, prev_feasible=True) -> MinlpData:
        """Solve."""
        require_benders = False
        x_sol = nlpdata.x_sol[:self.nr_x_orig]

        if self.qp_stagnates or almost_equal(nlpdata.obj_val, self.y_N_val):
            require_benders = True

        if prev_feasible and nlpdata.obj_val + EPS < self.y_N_val:  # check if new best solution found
            self.x_sol_best = x_sol[:self.nr_x_orig]
            self.y_N_val = nlpdata.obj_val  # update best objective
            self.qp_stagnates = False
            logger.info(f"NEW UPPER BOUND: {self.y_N_val}")

        if not prev_feasible:
            colored("Infeasibility Cut", "blue")
            self._add_infeasible_cut(nlpdata)
        else:
            self._trust_region_update(nlpdata)

        if not require_benders:
            nlpdata.prev_solution = self._solve_trust_region_problem(nlpdata)
            nlpdata.solved, _ = self.collect_stats("TR-MIQP")
            for x_best in nlpdata.best_solutions:
                    if np.allclose(nlpdata.x_sol[self.idx_x_bin], x_best[self.idx_x_bin], equal_nan=False, atol=EPS):
                        require_benders = True
                        self.qp_stagnates = True
                        break

        if require_benders:
            nlpdata.prev_solution = self._solve_benders_problem(nlpdata)
            nlpdata.solved, _ = self.collect_stats("LB-MILP")

        return nlpdata, require_benders
