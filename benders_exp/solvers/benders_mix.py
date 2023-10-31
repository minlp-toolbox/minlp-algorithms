"""A mix of solvers."""
import numpy as np
import casadi as ca
from benders_exp.solvers import Stats, MinlpProblem, MinlpData, \
    get_idx_linear_bounds, get_idx_inverse, extract_bounds
from benders_exp.utils import colored
from benders_exp.defines import WITH_JIT, CASADI_VAR, EPS, MIP_SOLVER, \
    WITH_DEBUG
from benders_exp.solvers.benders import BendersMasterMILP
from benders_exp.problems import check_integer_feasible, check_solution
from benders_exp.solvers.utils import Constraints, get_solutions_pool, any_equal
from benders_exp.solvers.tighten import tighten_bounds_x
from enum import Enum
import logging

logger = logging.getLogger(__name__)


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
            # the new constraint added should have the same multiplier
            # of the others (cf. gradient-amplification strategy)
            self.multipliers.append(self.multipliers[0])

    def __call__(self, x_value, nu=0):
        """Evaluate the bounds."""
        return [
            gi + m * dgi.T @ (x_value - xi) - nu
            for gi, dgi, m, xi in zip(
                self.g, self.dg_corrected, self.multipliers, self.x_lin)
        ]

    def to_generic(self, nu=None):
        """Create bounds."""
        if nu is None:
            nu = self.nu
        return Constraints(
            self.nr,
            ca.vertcat(*self(self.x, nu)),
            -ca.inf * np.ones(self.nr),
            np.zeros(self.nr),
        )

    def __add__(self, other):
        """Add bounds."""
        return self.to_generic() + other


def compute_gradient_correction(x_best, x_new, obj_best, obj_new, grad):
    """Compute gradient correction, based on L2 norm."""

    with_ipopt = False
    W = np.eye(x_best.shape[0])  # weighting matrix L2-norm
    W_inv = np.linalg.inv(W)

    if with_ipopt:
        nr_x = x_best.numel()
        grad_corr = CASADI_VAR.sym("gradient_correction", nr_x)
        obj = ca.norm_2(grad_corr) ** 2
        g = (obj_new - obj_best - EPS) + \
            (grad + grad_corr).T @ (x_best - x_new)
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

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, options=None, with_benders_master=True):
        """Create the benders constraint MILP."""
        super(BendersTRandMaster, self).__init__(problem, data, stats, options)
        # Settings
        self.nonconvex_strategy = NonconvexStrategy.GRADIENT_BASED
        self.nonconvex_strategy_alpha = 0.2
        self.trust_region_feasibility_strategy = TrustRegionStrategy.GRADIENT_AMPLIFICATION
        self.trust_region_feasibility_rho = 1.5

        if WITH_DEBUG:
            self.problem = problem

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
        if problem.gn_hessian is None:
            self.f_hess = ca.Function("hess_f_x", [problem.x, problem.p], [
                ca.hessian(problem.f, problem.x)[0]])
        else:
            self.f_hess = ca.Function("hess_f_x", [problem.x, problem.p], [
                problem.gn_hessian])

        self._x = CASADI_VAR.sym("x_benders", problem.x.numel())
        self._x_bin = self._x[problem.idx_x_bin]
        self.g_lin = Constraints(*extract_bounds(
            problem, data, self.idx_g_lin, self._x, allow_fail=False
        ))
        self.g_lowerapprox = LowerApproximation(self._x_bin, self._nu)
        self.g_infeasible = Constraints(0)

        self.options.update({"discrete": [
            1 if elm in problem.idx_x_bin else 0 for elm in range(self._x.shape[0])]})
        self.options_master = self.options.copy()
        self.options_master["discrete"] = self.options["discrete"] + [0]
        self.options['error_on_fail'] = False

        self.y_N_val = 1e15  # Should be inf but can not at the moment ca.inf
        self.internal_lb = -ca.inf
        self.x_sol_best = data.x0  # take a point as initialization
        self.sol_best = None
        self.qp_stagnates = False
        self.qp_not_improving = 0
        self.with_benders_master = with_benders_master
        self.hessian_not_psd = problem.hessian_not_psd

    def _check_cut_valid(self, g, grad_g, x_best, x_sol, x_sol_obj):
        """Check if the cut is valid."""
        value = g + grad_g.T @ (x_best - x_sol)
        # print(f"Cut valid (lower bound)?: {value} vs real {x_sol_obj}")
        return (value - EPS <= x_sol_obj)  # TODO: check sign EPS

    def _add_infeasible_cut(self, x_sol, lam_g_sol, nlpdata: MinlpData):
        """Create infeasibility cut."""
        x_sol = x_sol[:self.nr_x_orig]
        h_k = self.g(x_sol, nlpdata.p)
        jac_h_k = self.jac_g_bin(x_sol, nlpdata.p)
        g_k = lam_g_sol.T @ (
                h_k + jac_h_k @ (self._x_bin - x_sol[self.idx_x_bin])
                - (lam_g_sol > 0) * np.where(np.isinf(nlpdata.ubg), 0, nlpdata.ubg)
                + (lam_g_sol < 0) * np.where(np.isinf(nlpdata.lbg), 0, nlpdata.lbg)
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

    def _gradient_correction(self, x_sol, lam_x_sol, nlpdata: MinlpData):
        # TODO: (to improve computation speed) if the best point does not change, check only the last point
        x_sol_best_bin = self.x_sol_best[self.idx_x_bin]

        # On the last integer point: check, correct (if needed) and add to g_lowerapprox
        lambda_k = -lam_x_sol[self.idx_x_bin]
        f_k = self.f(x_sol, nlpdata.p)
        x_bin_new = x_sol[self.idx_x_bin]

        if not self._check_cut_valid(f_k, lambda_k, x_sol_best_bin, x_bin_new, self.y_N_val):
            grad_corr = compute_gradient_correction(
                x_sol_best_bin, x_bin_new, self.y_N_val, f_k, lambda_k)
            self.g_lowerapprox.add(x_bin_new, f_k, lambda_k, grad_corr)
            logger.debug(f"Correcting gradient at {x_bin_new=}")
        else:
            self.g_lowerapprox.add(x_bin_new, f_k, lambda_k)

        # Check and correct - if necessary - all the points in memory
        for i in range(self.g_lowerapprox.nr - 1):
            # Reset the corrected gradient to original
            self.g_lowerapprox.dg_corrected[i] = self.g_lowerapprox.dg[i]
            if not self._check_cut_valid(
                    self.g_lowerapprox.g[i], self.g_lowerapprox.dg[i],
                    x_sol_best_bin, self.g_lowerapprox.x_lin[i], self.y_N_val
            ):
                self.g_lowerapprox.dg_corrected[i] = compute_gradient_correction(
                    x_sol_best_bin, self.g_lowerapprox.x_lin[i], self.y_N_val,
                    self.g_lowerapprox.g[i], self.g_lowerapprox.dg[i])
                logger.debug(
                    f"Correcting gradient at {self.g_lowerapprox.x_lin[i]=}")

    def _trust_region_is_empty(self):
        if self.g_lowerapprox.nr == 0:
            return False
        g_val = self.g_lowerapprox(self.x_sol_best[self.idx_x_bin])
        if np.max(np.array(g_val) - self.y_N_val) > 0:
            return True

    def _get_g_linearized(self, x, dx, nlpdata):
        g_lin = self.g(x, nlpdata.p)
        jac_g = self.jac_g(x, nlpdata.p)

        return Constraints(
            g_lin.numel(),
            (g_lin + jac_g @ dx),
            nlpdata.lbg - EPS,
            nlpdata.ubg + EPS,
        )

    def _solve_trust_region_problem(self, nlpdata: MinlpData, constraint) -> MinlpData:
        """Solve QP problem."""
        dx = self._x - self.x_sol_best

        f_k = self.f(self.x_sol_best, nlpdata.p)
        f_lin = self.grad_f_x_sub(self.x_sol_best, nlpdata.p)
        f_hess = self.f_hess(self.x_sol_best, nlpdata.p)
        if self.hessian_not_psd:
            min_eigen_value = np.linalg.eigh(f_hess.full())[0][0]
            logger.info(f"Eigen value detected {min_eigen_value}")
            if min_eigen_value < 0:
                f_hess -= min_eigen_value * ca.DM.eye(self.nr_x_orig)
            else:
                self.hessian_not_psd = False

        f = f_k + f_lin.T @ dx + 0.5 * dx.T @ f_hess @ dx
        # Order seems to be important!
        g_total = self._get_g_linearized(
            self.x_sol_best, dx, nlpdata
        ) + self.g_lowerapprox + self.g_infeasible

        if WITH_DEBUG:
            check_integer_feasible(self.idx_x_bin, self.x_sol_best,
                                   eps=1e-3, throws=False)
            check_solution(self.problem, nlpdata, self.x_sol_best,
                           eps=1e-3, throws=False)

        self.solver = ca.qpsol(
            f"benders_constraint_{self.g_lowerapprox.nr}", MIP_SOLVER, {
                "f": f, "g": g_total.eq,
                "x": self._x, "p": self._nu
            }, self.options  # + {"error_on_fail": False}
        )

        solution = self.solver(
            x0=self.x_sol_best,
            lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=g_total.lb,
            ubg=g_total.ub,
            p=[constraint]
        )
        success, stats = self.collect_stats("TR-MILP")
        logger.info(f"SOLVED TR-MIQP with ub {constraint}")
        return solution, success, stats

    def _solve_benders_problem(self, nlpdata: MinlpData) -> MinlpData:
        """Solve benders master problem with one OA constraint."""
        dx = self._x - self.x_sol_best

        f_k = self.f(self.x_sol_best, nlpdata.p)
        f_lin = self.grad_f_x_sub(self.x_sol_best, nlpdata.p)
        f = f_k + f_lin.T @ dx

        # Adding the following linearization might not be the best idea since
        # They can lead to false results!
        # g_cur_lin = self._get_g_linearized_nonlin(self.x_sol_best, dx, nlpdata)
        g_total = self._get_g_linearized(
            self.x_sol_best, dx, nlpdata
        ) + self.g_lowerapprox + self.g_infeasible

        # Add extra constraint (one step OA):
        g_total.add(-ca.inf, f - self._nu, 0)
        g, ubg, lbg = g_total.eq, g_total.ub, g_total.lb

        self.solver = ca.qpsol(
            f"benders_with_{self.g_lowerapprox.nr}_cut", MIP_SOLVER, {
                "f": self._nu, "g": g,
                "x": ca.vertcat(self._x, self._nu),
            }, self.options_master
        )

        solution = self.solver(
            x0=ca.vertcat(self.x_sol_best, self.y_N_val),
            lbx=ca.vertcat(nlpdata.lbx, -1e5),
            ubx=ca.vertcat(nlpdata.ubx, ca.inf),
            lbg=lbg, ubg=ubg
        )
        success, stats = self.collect_stats("LB-MILP")

        solution['x'] = solution['x'][:-1]
        logger.info("SOLVED LB-MILP")
        return solution, success, stats

    def update_options(self, relaxed=False):
        """Update options."""
        if self.internal_lb > self.y_N_val:
            self.internal_lb = self.y_N_val

        if relaxed:
            self.options['gurobi.MIPGap'] = 1.0
        else:
            self.options['gurobi.MIPGap'] = 0.1
        logger.info(f"MIP Gap set to {self.options['gurobi.MIPGap']} - "
                    f"Expected Range {self.internal_lb} - {self.y_N_val}")

    def update_relaxed_solution(self, nlpdata: MinlpData):
        for prev_feasible, sol in zip(nlpdata.solved_all, nlpdata.prev_solutions):
            # check if new best solution found
            self.x_sol_best = sol['x'][:self.nr_x_orig]
            self.internal_lb = float(sol['f'])
            self._gradient_correction(sol['x'], sol['lam_x'], nlpdata)

    def _tighten(self, nlpdata: MinlpData):
        """Tighten bounds."""
        tighten_bounds_x(nlpdata, self.g_lowerapprox.to_generic(nu=self.y_N_val),
                         self.idx_x_bin, self._x, self.nr_x_orig)

    def _solve_mix(self, nlpdata: MinlpData):
        """Solve mix."""
        do_trust_region = not self.qp_stagnates
        if do_trust_region:
            constraint = (self.y_N_val + self.internal_lb) / 2
            solution, success, stats = self._solve_trust_region_problem(nlpdata, constraint)
            if success:
                self.qp_stagnates = any_equal(solution['x'], nlpdata.best_solutions, self.idx_x_bin)
                do_benders = self.qp_stagnates
            else:
                colored("Failed solving TR", "red")
                do_benders = True
        else:
            do_benders = True

        if do_benders:
            logger.info("QP stagnates, need benders problem!")
            solution, success, stats = self._solve_benders_problem(nlpdata)
            self.internal_lb = float(solution['f'])
            self.qp_not_improving = 0

        nlpdata = get_solutions_pool(nlpdata, success, stats, solution, self.idx_x_bin)
        return nlpdata, do_benders

    def _solve_tr_only(self, nlpdata: MinlpData):
        """Only solve trust regions."""
        MIPGap = 0.001
        constraint = self.y_N_val * (1 - MIPGap)
        self.options['gurobi.MIPGap'] = 0.1
        solution, success, stats = self._solve_trust_region_problem(nlpdata, constraint)
        if not success:
            solution = self.sol_best
            nlpdata.prev_solutions = [self.sol_best]
            nlpdata.solved_all = [True]
        else:
            nlpdata = get_solutions_pool(nlpdata, success, stats, solution, self.idx_x_bin)
        return nlpdata, False

    def solve(self, nlpdata: MinlpData, relaxed=False) -> MinlpData:
        """Solve."""
        if relaxed:
            self.update_relaxed_solution(nlpdata)
        else:
            self.qp_not_improving += 1
            needs_trust_region_update = False
            for prev_feasible, sol in zip(nlpdata.solved_all, nlpdata.prev_solutions):
                # check if new best solution found
                if prev_feasible:
                    self._gradient_correction(sol['x'], sol['lam_x'], nlpdata)
                    needs_trust_region_update = True
                    if float(sol['f']) + EPS < self.y_N_val:
                        self.x_sol_best = sol['x'][:self.nr_x_orig]
                        self.sol_best = sol
                        self.y_N_val = float(sol['f'])  # update best objective
                        self.qp_stagnates = False
                        self.qp_not_improving = 0
                        colored(f"NEW UPPER BOUND: {self.y_N_val}", "green")
                else:
                    colored("Infeasibility Cut", "blue")
                    self._add_infeasible_cut(sol['x'], sol['lam_g'], nlpdata)

            if needs_trust_region_update:
                self._gradient_amplification()
        self.update_options(relaxed)
        if self.with_benders_master:
            return self._solve_mix(nlpdata)
        else:
            return self._solve_tr_only(nlpdata)
