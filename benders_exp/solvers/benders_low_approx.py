import numpy as np
import casadi as ca
from benders_exp.solvers.utils import get_solutions_pool
from benders_exp.solvers.benders_mix import BendersTRandMaster, LowerApproximation, \
        compute_gradient_correction
from benders_exp.solvers import Stats, MinlpProblem, MinlpData, regularize_options
from benders_exp.defines import EPS, WITH_DEBUG, MIP_SOLVER, CASADI_VAR, IPOPT_SETTINGS
from benders_exp.utils import colored
from benders_exp.problems import check_integer_feasible, check_solution
import logging

logger = logging.getLogger(__name__)


class BendersTRLB(BendersTRandMaster):
    """
    A benders trust region with corrections on the hessian.

    The main idea is to create a lower QP approximation by changing the hessian.
    This hessian is corrected s.t. the largest SVD < min(largest SVD up to now) / 2
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, options=None, with_benders_master=True):
        """Create the benders constraint MILP."""
        super(BendersTRLB, self).__init__(problem, data, stats, options)
        self.values = []
        self.hess_correction = 2.0
        self.prev_val_miqp = -ca.inf
        self.hess_trust_points_setting = 3

        self.g_lowerapprox_oa = LowerApproximation(self._x, self._nu)
        self.ipopt_settings = regularize_options(options, IPOPT_SETTINGS)

    def trust_hessian(self):
        """Trust hessian."""
        return (self.hess_trust_points_setting < len(self.values))

    def compute_hess_correction(self, nlpdata):
        correction = CASADI_VAR.sym("correction", 1)
        dx = self._x - self.x_sol_best
        f_k = self.f(self.x_sol_best, nlpdata.p)
        f_lin = self.grad_f_x(self.x_sol_best, nlpdata.p)
        f_hess = self.f_hess(self.x_sol_best, nlpdata.p)
        f = ca.Function("f", [self._x, correction], [
            f_k + f_lin.T @ dx + 0.5 * dx.T @ f_hess * correction @ dx])
        g, g_ub = [], []
        for f_val, x_val in self.values:
            if f(x_val, self.hess_correction) > f_val:  # Not a LB
                g.append(f(x_val, correction))
                g_ub.append(f_val)

        if len(g) > 0:
            solver = ca.nlpsol("correction", "ipopt", {
                "f": -correction,
                "g": ca.vcat(g),
                "x": correction
            }, self.ipopt_settings)
            sol = solver(lbx=0.0, ubx=self.hess_correction, ubg=g_ub)
            self.hess_correction = float(sol['x'])
            colored(f"Hessian correction to {self.hess_correction}")

    def solve(self, nlpdata: MinlpData, prev_feasible=False, relaxed=False) -> MinlpData:
        """Solve."""
        new_hessian_updates = 0
        for prev_feasible, sol in zip(nlpdata.solved_all, nlpdata.prev_solutions):
            if prev_feasible:
                new_hessian_updates += 1
                self.values.append(
                    [float(sol['f']), sol['x'].full()]
                )

        if relaxed:
            self.update_relaxed_solution(nlpdata)
        else:
            needs_trust_region_update = False
            for prev_feasible, sol in zip(nlpdata.solved_all, nlpdata.prev_solutions):
                # check if new best solution found
                try:
                    nonzero = np.count_nonzero((sol['x'][:self.nr_x_orig] - self.x_sol_best)[self.idx_x_bin])
                except TypeError:
                    colored(sol['x'])
                    nonzero = -1
                if nonzero == 0:
                    breakpoint()

                if prev_feasible:
                    self._gradient_correction(sol['x'], sol['lam_x'], nlpdata)
                    self._lowerapprox_oa(sol['x'], nlpdata)
                    needs_trust_region_update = True
                    if float(sol['f']) + EPS < self.y_N_val:
                        self.x_sol_best = sol['x'][:self.nr_x_orig]
                        self.sol_best = sol
                        self.y_N_val = float(sol['f'])  # update best objective
                        colored(f"New upper bound: {self.y_N_val}", "green")
                        # Correct hessian correction because new point!
                        self.hess_correction = 2.0

                    colored(f"Regular Cut {float(sol['f']):.3f} - {nonzero}", "blue")
                else:
                    colored(f"Infeasibility Cut nonzero {nonzero}", "blue")
                    self._add_infeasibility_cut(sol, nlpdata)

            if needs_trust_region_update:
                self._gradient_amplification()

        if new_hessian_updates > 0:
            self.compute_hess_correction(nlpdata)

        self.update_options(relaxed)
        nlpdata = self._solve_internal(nlpdata)
        self.prev_val_miqp = nlpdata.obj_val
        colored(f"Not trusted lb {self.prev_val_miqp} vs {self.internal_lb}")
        self.internal_lb = min(self.prev_val_miqp, self.internal_lb)
        for i in range(nlpdata.nr_sols):
            nlpdata.prev_solutions[i]['f'] = self.internal_lb
        return nlpdata

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

    def _solve_miqp(self, nlpdata: MinlpData, correction, constraint) -> MinlpData:
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

        f = f_k + f_lin.T @ dx + 0.5 * correction * dx.T @ f_hess @ dx
        # Order seems to be important!
        g_total = self._get_g_linearized(
            self.x_sol_best, dx, nlpdata
        ) + self.g_lowerapprox + self.g_infeasible + self.g_lowerapprox_oa

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
        logger.info(f"SOLVED TR-MIQP with ub {constraint} - {self.hess_correction=}")
        return solution, success, stats

    def _solve_internal(self, nlpdata: MinlpData):
        """Only solve trust regions."""
        if len(self.values) < 2:
            MIPGap = 0.1
        else:
            MIPGap = 0.01
        # constraint = self.y_N_val * (1 - MIPGap)
        constraint = (self.y_N_val + min(self.internal_lb, self.y_N_val - MIPGap)) / 2
        self.options['gurobi.MIPGap'] = MIPGap
        solution, success, stats = self._solve_miqp(nlpdata, 1.0, constraint)
        if not success or solution['f'] > self.y_N_val:
            solution, success, stats = self._solve_miqp(nlpdata, self.hess_correction, self.y_N_val)
            if self.trust_hessian():
                self.internal_lb = solution['f']
                colored(f"Trusted lb to {self.internal_lb}", "green")

        nlpdata = get_solutions_pool(nlpdata, success, stats, solution, self.idx_x_bin)
        return nlpdata
