"""A mix of solvers."""
import numpy as np
import casadi as ca
from benders_exp.solvers import Stats, MinlpProblem, MinlpData
from benders_exp.utils import colored
from benders_exp.defines import Settings, CASADI_VAR
from benders_exp.solvers.benders import BendersMasterMILP
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


def compute_gradient_correction(x_best, x_new, obj_best, obj_new, grad, s: Settings):
    """Compute gradient correction, based on L2 norm."""

    with_ipopt = False
    W = np.eye(x_best.shape[0])  # weighting matrix L2-norm
    W_inv = np.linalg.inv(W)

    if with_ipopt:
        nr_x = x_best.numel()
        grad_corr = CASADI_VAR.sym("gradient_correction", nr_x)
        obj = ca.norm_2(grad_corr) ** 2
        g = (obj_new - obj_best - s.EPS) + \
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


def compute_infeasibility(g, lbg, ubg):
    """Compute infeasibility using benders-type infeasibility cuts."""
    g = g[:lbg.shape[0]]
    return np.sum(lbg[lbg > g] - g[lbg > g]) + np.sum(g[ubg < g] - ubg[ubg < g])


class BendersTRandMaster(BendersMasterMILP):
    """Mixing the idea from Moritz with a slightly altered version of benders masters."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings, with_benders_master=True, early_exit=False):
        """Create the benders constraint MILP."""
        super(BendersTRandMaster, self).__init__(
            problem, data, stats, s, with_lin_bounds=False)
        # Settings
        self.alpha_kronqvist = 0.5
        self.trust_region_feasibility_strategy = TrustRegionStrategy.GRADIENT_AMPLIFICATION
        self.trust_region_feasibility_rho = 1.5

        if s.WITH_DEBUG:
            self.problem = problem

        # Setups
        self.setup_common(problem, s)

        self.grad_f_x = ca.Function(
            "gradient_f_x",
            [problem.x, problem.p], [ca.gradient(
                problem.f, problem.x
            )],
            {"jit": s.WITH_JIT}
        )
        self.jac_g = ca.Function(
            "jac_g", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)],
            {"jit": s.WITH_JIT}
        )
        self.f_qp = None
        if problem.f_qp is not None:
            self.f_qp = problem.f_qp

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
        self.options_master['gurobi.MIPGap'] = s.MINLP_TOLERANCE
        self.options_master["discrete"] = self.options["discrete"] + [0]
        self.options_master['error_on_fail'] = False
        self.options['error_on_fail'] = False

        self.internal_lb = -ca.inf
        self.sol_best_feasible = False
        self.sol_infeasibility = ca.inf
        self.sol_best = data._sol  # take a point as initialization
        self.y_N_val = 1e15  # Should be inf but can not at the moment ca.inf
        self.with_benders_master = with_benders_master
        self.hessian_not_psd = problem.hessian_not_psd
        self.with_oa_conv_cuts = True
        self.trust_region_fails = False
        self.early_exit = early_exit
        self.early_benders = False

    def _check_cut_valid(self, g, grad_g, x_best, x_sol, x_sol_obj):
        """Check if the cut is valid."""
        value = g + grad_g.T @ (x_best - x_sol)
        # print(f"Cut valid (lower bound)?: {value} vs real {x_sol_obj}")
        return (value - self.settings.EPS <= x_sol_obj)  # TODO: check sign EPS

    # Amplifications and corrections
    def clip_gradient(self, current_value, lower_bound, gradients):
        # TODO: lower bound might be wrong when clipping Benders cut
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
        x_sol_best_bin = self.sol_best['x'][self.idx_x_bin]
        # Check and correct - if necessary - all the points in memory
        for i in range(self.g_lowerapprox.nr - 1):
            # New corrections based on previously corrected gradients
            if not self._check_cut_valid(
                    self.g_lowerapprox.g[i], self.g_lowerapprox.dg_corrected[i],
                    x_sol_best_bin, self.g_lowerapprox.x_lin[i], self.y_N_val
            ):
                self.g_lowerapprox.dg_corrected[i] = compute_gradient_correction(
                    x_sol_best_bin, self.g_lowerapprox.x_lin[i], self.y_N_val,
                    self.g_lowerapprox.g[i], self.g_lowerapprox.dg_corrected[i],
                    self.settings
                )
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
        dx = (sol['x_infeasible'] - sol['x'])[self.idx_x_bin]
        dx_min = abs(min(dx.full()))
        dx_max = abs(min(dx.full()))
        multiplier = 1 / max(dx_min, dx_max)
        if multiplier > 1000:
            sol['x'][self.idx_x_bin] -= 10 * dx
            multiplier = 1000

        if self.sol_best is not None:
            eps = max(
                float(ca.dot(multiplier * dx,
                      self.sol_best['x'][self.idx_x_bin])),
                0
            )
        else:
            eps = 0

        self.g_infeasible.add(sol['x'][self.idx_x_bin], eps, multiplier * dx)
        colored("New cut type", "blue")

    def _add_infeasible_cut(self, x_sol, lam_g_sol, nlpdata: MinlpData):
        """Add infeasibility cut using the default benders-inf cut method."""
        x_sol = x_sol[:self.nr_x_orig]
        g_k = self.g(x_sol, nlpdata.p)
        jac_g_k = self.jac_g_bin(x_sol, nlpdata.p)

        # g_k > ubg -> lam_g_sol > 0
        # Convert to g_lin < 0 -> g_k - ubg > 0
        g_bar_k = np.abs(lam_g_sol).T @ (
            (lam_g_sol > 0) * (g_k > nlpdata.ubg) *
            (g_k - np.where(np.isinf(nlpdata.ubg), 0, nlpdata.ubg))
            + (lam_g_sol < 0) * (g_k < nlpdata.lbg) *
            (g_k - np.where(np.isinf(nlpdata.lbg), 0, nlpdata.lbg))
        )
        # assert g_bar_k > 0
        if g_bar_k < self.settings.EPS:
            colored(f"Infeasibility cut of {g_bar_k}")
        else:
            colored(f"Infeasibility cut of {g_bar_k}", "blue")
        g_bar_k = max(g_bar_k, self.settings.EPS)
        # g_bar_k is positive by definition
        # + 10 is an extra tolerance.
        grad_g_bar_k = self.clip_gradient(
            g_bar_k + 10, 0, (lam_g_sol.T @ jac_g_k).T)

        x_sol_best_bin = self.sol_best['x'][self.idx_x_bin]
        x_bin_new = x_sol[self.idx_x_bin]
        if not self._check_cut_valid(g_bar_k, grad_g_bar_k, x_sol_best_bin, x_bin_new, 0.0):
            # need gradient correction because we cut out best point
            grad_corr = compute_gradient_correction(
                x_sol_best_bin, x_bin_new, 0, g_bar_k, grad_g_bar_k,
                self.settings
            )
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
                    self.g_infeasible_oa.add(
                        x_sol, g_k[i] - nlpdata.ubg[i], jac_g_k[i, :].T)
                else:
                    self.g_infeasible_oa.add(
                        x_sol, - g_k[i] + nlpdata.lbg[i], -jac_g_k[i, :].T)

    def _lowerapprox_oa(self, x, nlpdata):
        """Lower approximation."""
        f_k = self.f(x, nlpdata.p)
        f_grad = self.grad_f_x(x, nlpdata.p)

        if not self._check_cut_valid(f_k, f_grad, self.sol_best['x'], x, self.y_N_val):
            grad_corr = compute_gradient_correction(
                self.sol_best['x'], x, self.y_N_val, f_k, f_grad,
                self.settings
            )
            self.g_lowerapprox_oa.add(x, f_k, f_grad, grad_corr)
        else:
            self.g_lowerapprox_oa.add(x, f_k, f_grad)

    def _gradient_correction(self, x_sol, lam_x_sol, nlpdata: MinlpData):
        # TODO: (to improve computation speed) if the best point does not change, check only the last point
        x_sol_best_bin = self.sol_best['x'][self.idx_x_bin]

        # On the last integer point: check, correct (if needed) and add to g_lowerapprox
        f_k = self.f(x_sol, nlpdata.p)
        x_bin_new = x_sol[self.idx_x_bin]
        lambda_k = self.clip_gradient(
            f_k, self.internal_lb, -lam_x_sol[self.idx_x_bin])

        if not self._check_cut_valid(f_k, lambda_k, x_sol_best_bin, x_bin_new, self.y_N_val):
            grad_corr = compute_gradient_correction(
                x_sol_best_bin, x_bin_new, self.y_N_val, f_k, lambda_k,
                self.settings
            )
            self.g_lowerapprox.add(x_bin_new, f_k, lambda_k, grad_corr)
            logger.debug("Correcting new gradient at current best point")
        else:
            self.g_lowerapprox.add(x_bin_new, f_k, lambda_k)

    def _get_g_linearized(self, x, dx, nlpdata):
        if not self.sol_best_feasible and self.trust_region_fails:
            return Constraints()
        elif (self.g.size1_out("o0") == 0) & (self.g.size2_out("o0") == 0):
            if self.settings.WITH_DEBUG:
                breakpoint()
            return Constraints()
        else:
            g_lin = self.g(x, nlpdata.p)
            jac_g = self.jac_g(x, nlpdata.p)
            return Constraints(
                g_lin.numel(),
                (g_lin + jac_g @ dx),
                nlpdata.lbg - self.settings.EPS,
                nlpdata.ubg + self.settings.EPS,
            )

    def _solve_trust_region_problem(self, nlpdata: MinlpData, constraint) -> MinlpData:
        """Solve QP problem."""
        dx = self._x - self.sol_best['x']

        if self.f_qp is None:
            f_k = self.f(self.sol_best['x'], nlpdata.p)
            f_lin = self.grad_f_x(self.sol_best['x'], nlpdata.p)
            f_hess = self.f_hess(self.sol_best['x'], nlpdata.p)
            if self.hessian_not_psd:
                min_eigen_value = np.linalg.eigh(f_hess.full())[0][0]
                logger.info(f"Eigen value detected {min_eigen_value}")
                if min_eigen_value < 0:
                    f_hess -= min_eigen_value * ca.DM.eye(self.nr_x_orig)
                else:
                    self.hessian_not_psd = False

            f = f_k + f_lin.T @ dx + 0.5 * dx.T @ f_hess @ dx
        else:
            f = self.f_qp(self._x, self.sol_best['x'], nlpdata.p)
        # Order seems to be important!
        g_cur_lin = self._get_g_linearized(self.sol_best['x'], dx, nlpdata)

        g_total = g_cur_lin + self.g_lowerapprox + self.g_infeasible + \
            self.g_lowerapprox_oa + self.g_infeasible_oa

        self.solver = ca.qpsol(
            f"benders_constraint_{self.g_lowerapprox.nr}", self.settings.MIP_SOLVER, {
                "f": f, "g": g_total.eq,
                "x": self._x, "p": self._nu
            }, self.options  # + {"error_on_fail": False}
        )

        solution = self.solver(
            x0=self.sol_best['x'],
            lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=g_total.lb,
            ubg=g_total.ub,
            p=[constraint]
        )
        success, stats = self.collect_stats("TR-MIQP")
        if (stats['return_status'] == "TIME_LIMIT" and
                not np.any(np.isnan(solution['x'].full()))):
            success = True
        logger.info(f"SOLVED TR-MIQP with ub {constraint}")
        del self.solver
        return solution, success, stats

    def _solve_benders_problem(self, nlpdata: MinlpData) -> MinlpData:
        """Solve benders master problem with one OA constraint."""
        dx = self._x - self.sol_best['x']

        f_k = self.f(self.sol_best['x'], nlpdata.p)
        f_lin = self.grad_f_x(self.sol_best['x'], nlpdata.p)
        f = f_k + f_lin.T @ dx

        # Adding the following linearization might not be the best idea since
        # They can lead to false results!
        if not self.sol_best_feasible:
            g_cur_lin = Constraints()
        else:
            g_cur_lin = self._get_g_linearized(self.sol_best['x'], dx, nlpdata)
        g_total = g_cur_lin + self.g_lowerapprox + self.g_infeasible + \
            self.g_lowerapprox_oa + self.g_infeasible_oa

        # Add extra constraint (one step OA):
        g_total.add(-ca.inf, f - self._nu, 0)
        g, ubg, lbg = g_total.eq, g_total.ub, g_total.lb

        self.solver = ca.qpsol(
            f"benders_with_{self.g_lowerapprox.nr}_cut", self.settings.MIP_SOLVER, {
                "f": self._nu, "g": g,
                "x": ca.vertcat(self._x, self._nu),
            }, self.options_master
        )

        solution = self.solver(
            x0=ca.vertcat(self.sol_best['x'], self.y_N_val + 1e-5),
            lbx=ca.vertcat(nlpdata.lbx, -ca.inf),
            ubx=ca.vertcat(nlpdata.ubx, self.y_N_val),
            lbg=lbg, ubg=ubg
        )
        success, stats = self.collect_stats("LB-MILP")
        if not success:
            if not self.sol_best_feasible:
                raise Exception(
                    "Problem can not be solved - Feasible zone is empty")
            solution = self.sol_best
            success = True
            colored("Failed solving LB-MILP")
        else:
            solution['x'] = solution['x'][:-1]
            logger.info("SOLVED LB-MILP")

        del self.solver
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
                self.sol_best['x'] = sol['x'][:self.nr_x_orig]
                self.internal_lb = float(sol['f'])

        self._gradient_corrections_old_cuts()

    def _tighten(self, nlpdata: MinlpData):
        """Tighten bounds."""
        tighten_bounds_x(nlpdata, self.g_lowerapprox.to_generic(nu=self.y_N_val),
                         self.idx_x_bin, self._x, self.nr_x_orig)

    def _solve_mix(self, nlpdata: MinlpData):
        """Solve mix."""
        # We miss the LB, try to find one...
        do_benders = np.isinf(self.internal_lb) or self.early_benders
        if not do_benders:
            constraint = self.internal_lb + self.alpha_kronqvist * (self.y_N_val - self.internal_lb)  # Kronqvist's trick
            solution, success, stats = self._solve_trust_region_problem(
                nlpdata, constraint)
            if self.early_exit and solution['f'] > self.y_N_val:
                nlpdata = get_solutions_pool(
                    nlpdata, success, stats, self.settings,
                    solution, self.idx_x_bin
                )
                return nlpdata, True
            elif solution['f'] > self.y_N_val:
                self.early_benders = True

            if success:
                if any_equal(solution['x'], nlpdata.best_solutions, self.idx_x_bin):
                    colored("QP stagnates, need LB problem", "yellow")
                    do_benders = True
            else:
                self.trust_region_fails = True
                colored("Failed solving TR", "red")
                do_benders = True

        if do_benders:
            solution, success, stats = self._solve_benders_problem(nlpdata)
            self.internal_lb = float(solution['f'])

        nlpdata = get_solutions_pool(nlpdata, success, stats, self.settings,
                                     solution, self.idx_x_bin)
        return nlpdata, do_benders

    def _solve_tr_only(self, nlpdata: MinlpData):
        """Only solve trust regions."""
        MIPGap = 0.001
        constraint = self.y_N_val * (1 - MIPGap)
        self.options['gurobi.MIPGap'] = 0.1
        solution, success, stats = self._solve_trust_region_problem(
            nlpdata, constraint)
        if not success:
            if not self.sol_best_feasible:
                raise Exception("Problem can not be solved")
            solution = self.sol_best
            nlpdata.prev_solutions = [self.sol_best]
            nlpdata.solved_all = [True]
        else:
            nlpdata = get_solutions_pool(nlpdata, success, stats, self.settings,
                                         solution, self.idx_x_bin)
        return nlpdata, False

    def compute_lb(self, x_sol):
        """Compute LB."""
        return np.max(self.g_lowerapprox(x_sol[self.idx_x_bin], 0))

    def update_sol(self, x, sol, feasible, infeasibility=0):
        """Update solution."""
        if not feasible and self.sol_best_feasible:
            return
        self.sol_best_feasible = feasible
        self.sol_infeasibility = infeasibility
        sol['x'] = x
        self.sol_best = sol
        if feasible:
            self.y_N_val = float(sol['f'])
            self.early_benders = False
            colored(f"New upper bound: {self.y_N_val}", "green")
        else:
            colored(f"New upper bound: Inf {self.sol_infeasibility}", "red")

    def solve(self, nlpdata: MinlpData, relaxed=False) -> MinlpData:
        """Solve."""
        if relaxed:
            self.update_relaxed_solution(nlpdata)
        else:
            needs_trust_region_update = False
            for prev_feasible, sol in zip(nlpdata.solved_all, nlpdata.prev_solutions):
                # check if new best solution found
                nonzero = np.count_nonzero(
                    (sol['x'][:self.nr_x_orig] - self.sol_best['x'])[self.idx_x_bin])
                if prev_feasible:
                    self._gradient_correction(sol['x'], sol['lam_x'], nlpdata)
                    self._lowerapprox_oa(sol['x'], nlpdata)
                    needs_trust_region_update = True
                    if float(sol['f']) + self.settings.EPS < self.y_N_val:
                        self.update_sol(sol['x'][:self.nr_x_orig], sol, True)
                    colored(
                        f"Regular Cut {float(sol['f']):.3f} - {nonzero}", "blue")
                else:
                    if not self.sol_best_feasible:
                        if 'x_infeasible' in sol:
                            # Use distance
                            infeas = ca.norm_2(
                                sol['x_infeasible'][:self.nr_x_orig] - sol['x'][:self.nr_x_orig]
                            ) ** 2
                        else:
                            # Use regular benders
                            g = sol['g']
                            infeas = compute_infeasibility(
                                g.full(), nlpdata.lbg, nlpdata.ubg
                            )

                        if infeas < self.sol_infeasibility:
                            self.update_sol(sol.get('x_infeasible', sol['x'])[:self.nr_x_orig], sol, False)

                    colored(f"Infeasibility Cut - distance {nonzero}", "blue")
                    self._add_infeasibility_cut(sol, nlpdata)

                if self.with_oa_conv_cuts:
                    self._add_oa(sol['x'], nlpdata)

            if needs_trust_region_update:
                self._gradient_amplification()
        self.update_options(relaxed)
        if self.with_benders_master:
            return self._solve_mix(nlpdata)
        else:
            return self._solve_tr_only(nlpdata)
