import numpy as np
import casadi as ca
from minlp_algorithms.solvers.utils import get_solutions_pool
from minlp_algorithms.solvers.benders_mix import BendersTRandMaster, LowerApproximation
from minlp_algorithms.solvers import Stats, MinlpProblem, MinlpData, regularize_options
from minlp_algorithms.utils import colored
from minlp_algorithms.utils.validate import check_integer_feasible, check_solution
from minlp_algorithms.settings import Settings, GlobalSettings
from minlp_algorithms.utils.debugtools import CheckNoDuplicate
import logging

logger = logging.getLogger(__name__)


class BendersTRLB(BendersTRandMaster):
    """
    A benders trust region with corrections on the hessian.

    The main idea is to create a lower QP approximation by changing the hessian.
    This hessian is corrected s.t. the largest SVD < min(largest SVD up to now) / 2
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings, with_benders_master=True):
        """Create the benders constraint MILP."""
        super(BendersTRLB, self).__init__(problem, data, stats, s)
        self.values = []
        self.hess_correction = 2.0
        self.prev_val_miqp = -ca.inf
        self.hess_trust_points_setting = 3

        self.g_lowerapprox_oa = LowerApproximation(self._x, self._nu)
        self.ipopt_settings = regularize_options(s.IPOPT_SETTINGS, {}, s)
        self.check = CheckNoDuplicate(problem, s)

    def trust_hessian(self):
        """Trust hessian."""
        return (self.hess_trust_points_setting <= len(self.values))

    def compute_hess_correction(self, nlpdata):
        correction = GlobalSettings.CASADI_VAR.sym("correction", 1)
        dx = self._x - self.sol_best['x']
        f_k = self.f(self.sol_best['x'], nlpdata.p)
        f_lin = self.grad_f_x(self.sol_best['x'], nlpdata.p)
        f_hess = self.f_hess(self.sol_best['x'], nlpdata.p)
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
                    nonzero = np.count_nonzero((sol['x'][:self.nr_x_orig] - self.sol_best['x'])[self.idx_x_bin])
                except TypeError:
                    colored(sol['x'])
                    nonzero = -1

                if prev_feasible:
                    self._gradient_correction(sol['x'], sol['lam_x'], nlpdata)
                    self._lowerapprox_oa(sol['x'], nlpdata)
                    needs_trust_region_update = True
                    if float(sol['f']) + self.settings.EPS < self.y_N_val:
                        sol['x'] = sol['x'][:self.nr_x_orig]
                        self.sol_best = sol
                        self.sol_best_feasible = True
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

    def _solve_miqp(self, nlpdata: MinlpData, correction, constraint) -> MinlpData:
        """Solve QP problem."""
        dx = self._x - self.sol_best['x']
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

        f = f_k + f_lin.T @ dx + 0.5 * correction * dx.T @ f_hess @ dx
        # Order seems to be important!
        g_total = self._get_g_linearized(
            self.sol_best['x'], dx, nlpdata
        ) + self.g_lowerapprox + self.g_infeasible + self.g_lowerapprox_oa

        if self.settings.WITH_DEBUG and self.sol_best is not None:
            check_integer_feasible(self.idx_x_bin, self.sol_best['x'], self.settings, throws=False)
            check_solution(self.problem, self.sol_best, self.sol_best['x'], self.settings, throws=False)

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
        nlpdata.prev_solutions = [solution]
        success, stats = self.collect_stats("TR-MILP")
        logger.info(f"SOLVED TR-MIQP with ub {constraint} - {correction=} {success=}")
        del self.solver
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
            correction = self.hess_correction if self.trust_hessian() else 0
            solution, success, stats = self._solve_miqp(nlpdata, correction, self.y_N_val - MIPGap)
            if success:
                self.internal_lb = solution['f']
                colored(f"Trusted lb to {self.internal_lb}", "green")
            else:
                self.internal_lb = self.y_N_val

        if success:
            nlpdata = get_solutions_pool(nlpdata, success, stats, self.settings,
                                         solution, self.idx_x_bin)
        else:
            nlpdata.prev_solutions = [{"x": self.sol_best['x'], "f": self.y_N_val}]

        return nlpdata
