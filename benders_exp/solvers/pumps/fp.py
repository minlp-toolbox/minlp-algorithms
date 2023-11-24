"""An feasibility pump implementation."""

import numpy as np
import casadi as ca
from benders_exp.solvers import SolverClass, Stats, MinlpProblem, MinlpData, \
    regularize_options
from benders_exp.defines import CASADI_VAR, Settings
from benders_exp.utils import to_0d, logging

logger = logging.getLogger(__name__)


class LinearProjection(SolverClass):
    """
    Create NLP solver.

    This solver solves an NLP problem. This is either relaxed or
    the binaries are fixed.
    The NLP aims to satisfy original constraint in g and
    minimize the distance of relaxed x_bin from their rounded value in
    L1-norm. The rounding procedure follows a rule.
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, s: Settings):
        """Create NLP problem."""
        super(LinearProjection, self).__init___(problem, stats, s)
        options = regularize_options(s.IPOPT_SETTINGS, {"jit": s.WITH_JIT, "ipopt.max_iter": 5000}, s)

        self.idx_x_bin = problem.idx_x_bin
        x_bin_var = problem.x[self.idx_x_bin]
        self.nr_x_bin = x_bin_var.shape[0]
        rounded_value = CASADI_VAR.sym("rounded_value", self.nr_x_bin)
        slack_variables = CASADI_VAR.sym("slack", self.nr_x_bin)

        penalty_term = ca.sum1(slack_variables)
        g = ca.vertcat(
            problem.g,
            # x_bin_var - rounded < slack
            # - (x_bin_var - rounded) < slack
            slack_variables - (x_bin_var - rounded_value),
            slack_variables + (x_bin_var - rounded_value),
        )
        self.solver = ca.nlpsol("nlpsol", "ipopt", {
            "f": penalty_term,
            "g": g,
            "x": ca.vertcat(problem.x, slack_variables),
            "p": ca.vertcat(problem.p, rounded_value)
        }, options)

    def solve(self, nlpdata: MinlpData, **kwargs) -> MinlpData:
        """Solve NLP."""
        success_out = []
        sols_out = []
        for sol in nlpdata.solutions_all:
            lbx = nlpdata.lbx
            ubx = nlpdata.ubx

            x_bin_var = to_0d(sol['x'][self.idx_x_bin])
            new_sol = self.solver(
                x0=ca.vertcat(nlpdata.x0, np.zeros(self.nr_x_bin)),
                lbx=ca.vertcat(lbx, np.zeros(self.nr_x_bin)),
                ubx=ca.vertcat(ubx, ca.inf * np.ones(self.nr_x_bin)),
                lbg=ca.vertcat(nlpdata.lbg, np.zeros(2 * self.nr_x_bin)),
                ubg=ca.vertcat(nlpdata.ubg, ca.inf * np.ones(2 * self.nr_x_bin)),
                p=ca.vertcat(nlpdata.p, x_bin_var)
            )

            success, _ = self.collect_stats("FP")
            if not success:
                logger.debug("Infeasible solution!")
            success_out.append(success)
            sols_out.append(new_sol)

        nlpdata.prev_solutions = sols_out
        nlpdata.solved_all = success_out
        return nlpdata


class ObjectiveLinearProjection(SolverClass):
    """
    Create NLP solver.

    This solver solves an NLP problem. This is either relaxed or
    the binaries are fixed.
    The NLP aims to satisfy original constraint in g and
    minimize a linear combination of the original objective and the
    he distance of relaxed x_bin from their rounded value in
    L1-norm. The rounding procedure follows a rule and the weight of
    the L1-norm term in the objective is increased during iterations.
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, s: Settings):
        """Create NLP problem."""
        super(ObjectiveLinearProjection, self).__init___(problem, stats, s)
        options = regularize_options(s.IPOPT_SETTINGS, {}, s)

        self.idx_x_bin = problem.idx_x_bin
        x_bin_var = problem.x[self.idx_x_bin]
        self.nr_x_bin = x_bin_var.shape[0]
        rounded_value = CASADI_VAR.sym("rounded_value", self.nr_x_bin)
        slack_variables = CASADI_VAR.sym("slack", self.nr_x_bin)
        alpha = CASADI_VAR.sym("alpha", 1)
        int_error = CASADI_VAR.sym("int_error", 1)
        obj_val = CASADI_VAR.sym("obj_val", 1)
        self.alpha = 1.0
        self.alpha_reduction = 0.9

        penalty_term = ca.sum1(slack_variables)
        g = ca.vertcat(
            problem.g,
            # x_bin_var - rounded < slack
            # - (x_bin_var - rounded) < slack
            slack_variables - (x_bin_var - rounded_value),
            slack_variables + (x_bin_var - rounded_value),
        )
        options.update({"jit": s.WITH_JIT, "ipopt.max_iter": 5000})
        self.solver = ca.nlpsol("nlpsol", "ipopt", {
            "f": alpha / obj_val * problem.f + (1 - alpha) / int_error * penalty_term,
            "g": g,
            "x": ca.vertcat(problem.x, slack_variables),
            "p": ca.vertcat(problem.p, rounded_value, alpha, int_error, obj_val)
        }, options)

    def solve(self, nlpdata: MinlpData, int_error, obj_val) -> MinlpData:
        """Solve NLP."""
        success_out = []
        sols_out = []
        logger.info(f"Solving objective FP with alpha {self.alpha}")
        for sol in nlpdata.solutions_all:
            lbx = nlpdata.lbx
            ubx = nlpdata.ubx

            x_bin_var = to_0d(sol['x'][self.idx_x_bin])
            new_sol = self.solver(
                x0=ca.vertcat(nlpdata.x0, np.zeros(self.nr_x_bin)),
                lbx=ca.vertcat(lbx, np.zeros(self.nr_x_bin)),
                ubx=ca.vertcat(ubx, ca.inf * np.ones(self.nr_x_bin)),
                lbg=ca.vertcat(nlpdata.lbg, np.zeros(2 * self.nr_x_bin)),
                ubg=ca.vertcat(nlpdata.ubg, ca.inf * np.ones(2 * self.nr_x_bin)),
                p=ca.vertcat(nlpdata.p, x_bin_var, np.array(
                    [self.alpha, int_error, obj_val]
                ))
            )

            success, _ = self.collect_stats("OFP")
            if not success:
                logger.debug("Infeasible solution!")
            success_out.append(success)
            sols_out.append(new_sol)

        self.alpha *= self.alpha_reduction
        nlpdata.prev_solutions = sols_out
        nlpdata.solved_all = success_out
        return nlpdata
