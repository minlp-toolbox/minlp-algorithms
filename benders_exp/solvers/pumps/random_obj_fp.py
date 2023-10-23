"""An random direction search NLP solver."""

import casadi as ca
import numpy as np
from benders_exp.solvers import SolverClass, Stats, MinlpProblem, MinlpData, \
    regularize_options
from benders_exp.defines import WITH_JIT, IPOPT_SETTINGS, CASADI_VAR
from benders_exp.utils import to_0d, logging

logger = logging.getLogger(__name__)


class RandomDirectionNlpSolver(SolverClass):
    """
    Create NLP solver.

    This solver solves an NLP problem. This is either relaxed or
    the binaries are fixed.
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None, norm=2, penalty_scaling=0.5):
        """Create NLP problem."""
        super(RandomDirectionNlpSolver, self).__init___(problem, stats)
        options = regularize_options(options, IPOPT_SETTINGS)
        self.penalty_weight = 0.01
        self.penalty_scaling = penalty_scaling

        self.idx_x_bin = problem.idx_x_bin
        x_bin_var = problem.x[self.idx_x_bin]
        penalty = CASADI_VAR.sym("penalty", x_bin_var.shape[0])
        rounded_value = CASADI_VAR.sym("rounded_value", x_bin_var.shape[0])

        self.norm = norm
        self.max_rounding_error = x_bin_var.shape[0]
        if norm == 1:
            penalty_term = ca.sum1(
                ca.fabs(x_bin_var - rounded_value) * penalty)
        else:
            penalty_term = ca.norm_2(
                (x_bin_var - rounded_value) * penalty) ** 2

        options.update({"jit": WITH_JIT, "ipopt.max_iter": 5000})
        self.solver = ca.nlpsol("nlpsol", "ipopt", {
            "f": problem.f + penalty_term,
            "g": problem.g, "x": problem.x,
            "p": ca.vertcat(problem.p, rounded_value, penalty)
        }, options)

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve NLP."""
        success_out = []
        sols_out = []
        for sol in nlpdata.solutions_all:
            lbx = nlpdata.lbx
            ubx = nlpdata.ubx

            x_bin_var = to_0d(sol['x'][self.idx_x_bin])
            x_rounded = np.round(x_bin_var)
            if self.norm == 1:
                rounding_error = ca.norm_1((x_bin_var - x_rounded))
            else:
                rounding_error = ca.norm_2((x_bin_var - x_rounded)) ** 2

            penalty = self.penalty_weight * np.random.rand(len(self.idx_x_bin))
            self.penalty_weight += self.penalty_scaling * abs(nlpdata.obj_val) * (
                rounding_error / self.max_rounding_error + 0.01
            )
            logger.info(
                f"{nlpdata.obj_val=:.3} rounding error={float(rounding_error):.3f} "
                f"- weight {float(self.penalty_weight):.3e}"
            )

            new_sol = self.solver(
                x0=nlpdata.x0,
                lbx=lbx, ubx=ubx,
                lbg=nlpdata.lbg, ubg=nlpdata.ubg,
                p=ca.vertcat(nlpdata.p, x_rounded, penalty)
            )

            success, _ = self.collect_stats("RNLP")
            if not success:
                logger.debug("Infeasible solution!")
            success_out.append(success)
            sols_out.append(new_sol)

        nlpdata.prev_solutions = sols_out
        nlpdata.solved_all = success_out

        return nlpdata
