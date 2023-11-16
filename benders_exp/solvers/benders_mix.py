"""A mix of solvers."""
import numpy as np
import casadi as ca
from benders_exp.solvers import Stats, MinlpProblem, MinlpData, \
    extract_bounds
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
        if gradient_corrected is None:
            gradient_corrected = gradient

        self.nr += 1
        self.x_lin.append(point)
        self.g.append(value)
        self.dg.append(gradient)
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
        super(BendersTRandMaster, self).__init__(problem, data, stats, options, with_lin_bounds=False)
        # Settings
        self.nonconvex_strategy = NonconvexStrategy.GRADIENT_BASED
        self.nonconvex_strategy_alpha = 0.2
        self.trust_region_feasibility_strategy = TrustRegionStrategy.GRADIENT_AMPLIFICATION
        self.trust_region_feasibility_rho = 1.5

        if WITH_DEBUG:
            self.problem = problem

        # Setups
        self.setup_common(problem, options)

        self.grad_f_x = ca.Function(
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
        self.g_lowerapprox = LowerApproximation(self._x_bin, self._nu)
        self.g_lowerapprox_oa = LowerApproximation(self._x, self._nu)
        self.g_infeasible = LowerApproximation(self._x_bin, 0)
        self.g_infeasible_oa = LowerApproximation(self._x, 0)
        self.idx_g_conv = problem.idx_g_conv

        self.options.update({"discrete": [
            1 if elm in problem.idx_x_bin else 0 for elm in range(self._x.shape[0])]})
        self.options_master = self.options.copy()
        self.options_master["discrete"] = self.options["discrete"] + [0]
        self.options_master['error_on_fail'] = False
        self.options['error_on_fail'] = False

        self.y_N_val = 1e15  # Should be inf but can not at the moment ca.inf
        self.internal_lb = -ca.inf
        self.x_sol_best = data.x0  # take a point as initialization
        self.sol_best = None
        self.with_benders_master = with_benders_master
        self.hessian_not_psd = problem.hessian_not_psd

    def _check_cut_valid(self, g, grad_g, x_best, x_sol, x_sol_obj):
        """Check if the cut is valid."""
        value = g + grad_g.T @ (x_best - x_sol)
        # print(f"Cut valid (lower bound)?: {value} vs real {x_sol_obj}")
        return (value - EPS <= x_sol_obj)  # TODO: check sign EPS

    # Amplifications and corrections
    def clip_gradient(self, current_value, lower_bound, gradients):
        # TODO: Who knows the LB doesn't hold, might reclipping!
        max_gradient = float(current_value - lower_bound)
        return np.clip(gradients, - max_gradient, max_gradient)

    def _gradient_amplification(self):
        if self.trust_region_feasibility_strategy == TrustRegionStrategy.GRADIENT_AMPLIFICATION:
            # Amplify the gradient of every new cut with the chosen rho.
            for i, m in enumerate(self.g_lowerapprox.multipliers):
                if m != self.trust_region_feasibility_rho:
                    self.g_lowerapprox.multipliers[i] = self.trust_region_feasibility_rho
        else:
            raise NotImplementedError()

    def _gradient_corrections_old_cuts(self):
        x_sol_best_bin = self.x_sol_best[self.idx_x_bin]
        # Check and correct - if necessary - all the points in memory
        for i in range(self.g_lowerapprox.nr - 1):
            # New corrections based on previously corrected gradients
            if not self._check_cut_valid(
                    self.g_lowerapprox.g[i], self.g_lowerapprox.dg_corrected[i],
                    x_sol_best_bin, self.g_lowerapprox.x_lin[i], self.y_N_val
            ):
                self.g_lowerapprox.dg_corrected[i] = compute_gradient_correction(
                    x_sol_best_bin, self.g_lowerapprox.x_lin[i], self.y_N_val,
                    self.g_lowerapprox.g[i], self.g_lowerapprox.dg_corrected[i])
                logger.debug(f"Correcting gradient for lower approx {i}")

    # Infeasibility cuts
    def _add_infeasibility_cut(self, sol, nlpdata):
        """Add infeasibility cut."""
        if 'x_infeasible' in sol:
            self._add_infeasible_cut_closest_point(sol)
        else:
            self._add_infeasible_cut(sol['x'], sol['lam_g'], nlpdata)

    def _add_infeasible_cut_closest_point(self, sol):
        """Add infeasible cut closest point."""
        # Rotate currently around x, but preferrably around x_infeasible
        dx = sol['x_infeasible'] - sol['x']
        self.g_infeasible.add(sol['x'][self.idx_x_bin], 0, dx[self.idx_x_bin])
        colored("New cut type", "blue")

    def _add_infeasible_cut(self, x_sol, lam_g_sol, nlpdata: MinlpData):
        """Create infeasibility cut."""
        x_sol = x_sol[:self.nr_x_orig]
        g_k = self.g(x_sol, nlpdata.p)
        jac_g_k = self.jac_g_bin(x_sol, nlpdata.p)

        # g_k > ubg -> lam_g_sol > 0
        # Convert to g_lin < 0 -> g_k - ubg > 0
        g_bar_k = np.abs(lam_g_sol).T @ (
            (lam_g_sol > 0) * (g_k > nlpdata.ubg) * (g_k - np.where(np.isinf(nlpdata.ubg), 0, nlpdata.ubg))
            + (lam_g_sol < 0) * (g_k < nlpdata.lbg) * (g_k - np.where(np.isinf(nlpdata.lbg), 0, nlpdata.lbg))
        )
        assert g_bar_k > 0
        colored(f"Infeasibility cut of {g_bar_k}")
        # g_bar_k is positive by definition
        grad_g_bar_k = self.clip_gradient(g_bar_k + 10, 0, (lam_g_sol.T @ jac_g_k).T)

        x_sol_best_bin = self.x_sol_best[self.idx_x_bin]
        x_bin_new = x_sol[self.idx_x_bin]
        if not self._check_cut_valid(g_bar_k, grad_g_bar_k, x_sol_best_bin, x_bin_new, 0.0):
            # need gradient correction because we cut out best point
            grad_corr = compute_gradient_correction(
                x_sol_best_bin, x_bin_new, 0, g_bar_k, grad_g_bar_k)
            self.g_infeasible.add(x_bin_new, g_bar_k, grad_g_bar_k, grad_corr)
        else:
            self.g_infeasible.add(x_bin_new, g_bar_k, grad_g_bar_k)

    def _add_oa(self, x_sol, nlpdata: MinlpData):
        x_sol = x_sol[:self.nr_x_orig]
        g_k = self.g(x_sol, nlpdata.p)
        jac_g_k = self.jac_g(x_sol, nlpdata.p)
        if self.idx_g_conv is not None:
            for i in self.idx_g_conv:
                colored("Add OA cut.")
                if not np.isinf(nlpdata.ubg[i]):
                    self.g_infeasible_oa.add(x_sol, g_k[i] - nlpdata.ubg[i], jac_g_k[i, :].T)
                else:
                    self.g_infeasible_oa.add(x_sol, - g_k[i] + nlpdata.lbg[i], -jac_g_k[i, :].T)

    def _lowerapprox_oa(self, x, nlpdata):
        """Lower approximation."""
        f_k = self.f(x, nlpdata.p)
        f_grad = self.grad_f_x(x, nlpdata.p)

        if not self._check_cut_valid(f_k, f_grad, self.x_sol_best, x, self.y_N_val):
            grad_corr = compute_gradient_correction(
                self.x_sol_best, x, self.y_N_val, f_k, f_grad)
            self.g_lowerapprox_oa.add(x, f_k, f_grad, grad_corr)
        else:
            self.g_lowerapprox_oa.add(x, f_k, f_grad)

    def _gradient_correction(self, x_sol, lam_x_sol, nlpdata: MinlpData):
        # TODO: (to improve computation speed) if the best point does not change, check only the last point
        x_sol_best_bin = self.x_sol_best[self.idx_x_bin]

        # On the last integer point: check, correct (if needed) and add to g_lowerapprox
        f_k = self.f(x_sol, nlpdata.p)
        x_bin_new = x_sol[self.idx_x_bin]
        lambda_k = self.clip_gradient(f_k, self.internal_lb, -lam_x_sol[self.idx_x_bin])

        if not self._check_cut_valid(f_k, lambda_k, x_sol_best_bin, x_bin_new, self.y_N_val):
            grad_corr = compute_gradient_correction(
                x_sol_best_bin, x_bin_new, self.y_N_val, f_k, lambda_k)
            self.g_lowerapprox.add(x_bin_new, f_k, lambda_k, grad_corr)
            logger.debug("Correcting new gradient at current best point")
        else:
            self.g_lowerapprox.add(x_bin_new, f_k, lambda_k)

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
        f_lin = self.grad_f_x(self.x_sol_best, nlpdata.p)
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
        ) + self.g_lowerapprox + self.g_infeasible + self.g_lowerapprox_oa + self.g_infeasible_oa

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
        f_lin = self.grad_f_x(self.x_sol_best, nlpdata.p)
        f = f_k + f_lin.T @ dx

        # Adding the following linearization might not be the best idea since
        # They can lead to false results!
        g_cur_lin = self._get_g_linearized(self.x_sol_best, dx, nlpdata)
        g_total = g_cur_lin + self.g_lowerapprox + self.g_infeasible + self.g_lowerapprox_oa + self.g_infeasible_oa

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
            x0=ca.vertcat(self.x_sol_best, self.y_N_val + 1e-5),
            lbx=ca.vertcat(nlpdata.lbx, -ca.inf),
            ubx=ca.vertcat(nlpdata.ubx, self.y_N_val),
            lbg=lbg, ubg=ubg
        )
        success, stats = self.collect_stats("LB-MILP")
        if not success:
            if self.sol_best is None:
                raise Exception("Problem can not be solved - Feasible zone is empty")
            solution = self.sol_best
            nlpdata.prev_solutions = [self.sol_best]
            nlpdata.solved_all = [True]
            colored("Failed solving LB-MILP")
        else:
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
                    f"Expected Range lb={self.internal_lb}  ub={self.y_N_val}")

    def update_relaxed_solution(self, nlpdata: MinlpData):
        for prev_feasible, sol in zip(nlpdata.solved_all, nlpdata.prev_solutions):
            # check if new best solution found
            if np.isinf(self.internal_lb) or self.internal_lb > float(sol['f']):
                self.x_sol_best = sol['x'][:self.nr_x_orig]
                self.internal_lb = float(sol['f'])

        self._gradient_corrections_old_cuts()

    def _tighten(self, nlpdata: MinlpData):
        """Tighten bounds."""
        tighten_bounds_x(nlpdata, self.g_lowerapprox.to_generic(nu=self.y_N_val),
                         self.idx_x_bin, self._x, self.nr_x_orig)

    def _solve_mix(self, nlpdata: MinlpData):
        """Solve mix."""
        # We miss the LB, try to find one...
        do_benders = np.isinf(self.internal_lb)
        if not do_benders:
            constraint = (self.y_N_val + self.internal_lb) / 2
            solution, success, stats = self._solve_trust_region_problem(nlpdata, constraint)
            if self.g_lowerapprox.nr == 0:
                solution['f'] = self.internal_lb
            else:
                solution['f'] = self.compute_lb(solution['x'])

            if success:
                if any_equal(solution['x'], nlpdata.best_solutions, self.idx_x_bin):
                    colored("QP stagnates, need LB problem", "yellow")
                    do_benders = True
            else:
                colored("Failed solving TR", "red")
                do_benders = True

        if do_benders:
            solution, success, stats = self._solve_benders_problem(nlpdata)
            self.internal_lb = float(solution['f'])

        nlpdata = get_solutions_pool(nlpdata, success, stats, solution, self.idx_x_bin)
        return nlpdata, True

    def _solve_tr_only(self, nlpdata: MinlpData):
        """Only solve trust regions."""
        MIPGap = 0.001
        constraint = self.y_N_val * (1 - MIPGap)
        self.options['gurobi.MIPGap'] = 0.1
        solution, success, stats = self._solve_trust_region_problem(nlpdata, constraint)
        if not success:
            if self.sol_best is None:
                raise Exception("Problem can not be solved")
            solution = self.sol_best
            nlpdata.prev_solutions = [self.sol_best]
            nlpdata.solved_all = [True]
        else:
            nlpdata = get_solutions_pool(nlpdata, success, stats, solution, self.idx_x_bin)
        return nlpdata, False

    def compute_lb(self, x_sol):
        """Compute LB."""
        return np.max(self.g_lowerapprox(x_sol[self.idx_x_bin], 0))

    def solve(self, nlpdata: MinlpData, relaxed=False) -> MinlpData:
        """Solve."""
        if relaxed:
            self.update_relaxed_solution(nlpdata)
        else:
            needs_trust_region_update = False
            for prev_feasible, sol in zip(nlpdata.solved_all, nlpdata.prev_solutions):
                # check if new best solution found
                nonzero = np.count_nonzero((sol['x'][:self.nr_x_orig] - self.x_sol_best)[self.idx_x_bin])
                if prev_feasible:
                    self._gradient_correction(sol['x'], sol['lam_x'], nlpdata)
                    self._lowerapprox_oa(sol['x'], nlpdata)
                    needs_trust_region_update = True
                    if float(sol['f']) + EPS < self.y_N_val:
                        self.x_sol_best = sol['x'][:self.nr_x_orig]
                        self.sol_best = sol
                        self.y_N_val = float(sol['f'])  # update best objective
                        colored(f"New upper bound: {self.y_N_val}", "green")
                    colored(f"Regular Cut {float(sol['f']):.3f} - {nonzero}", "blue")
                else:
                    colored(f"Infeasibility Cut - distance {nonzero}", "blue")
                    self._add_infeasibility_cut(sol, nlpdata)
                self._add_oa(sol['x'], nlpdata)

            if needs_trust_region_update:
                self._gradient_amplification()
        self.update_options(relaxed)
        if self.with_benders_master:
            return self._solve_mix(nlpdata)
        else:
            return self._solve_tr_only(nlpdata)
