"""An random direction search NLP solver."""

import casadi as ca
import numpy as np
from typing import Tuple
from benders_exp.solvers.nlp import NlpSolver
from benders_exp.solvers import SolverClass, Stats, MinlpProblem, MinlpData, \
        regularize_options
from benders_exp.defines import WITH_JIT, IPOPT_SETTINGS, CASADI_VAR
from benders_exp.utils import to_0d, toc, logging

logger = logging.getLogger(__name__)


def integer_error(x_int):
    """Compute integer error."""
    ret = np.sum(np.abs(np.round(x_int) - x_int))
    logger.info(f"Integer error {ret}")
    return ret


def random_direction_rounding_algorithm(
    problem: MinlpProblem, data: MinlpData, stats: Stats
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """
    Random direction algorithm.

    parameters:
        - termination_type:
            - gradient: based on local linearization
            - equality: the binaries of the last solution coincides with the ones of the best solution
            - std: based on lower and upper bound
    """
    solver = RandomDirectionNlpSolver(problem, stats)
    nlp = NlpSolver(problem, stats)
    toc()
    data = nlp.solve(data)
    while integer_error(data.x_sol[problem.idx_x_bin]) > 1e-2:
        data = solver.solve(data)

    toc()
    return problem, data, data.x_sol


class RandomDirectionNlpSolver(SolverClass):
    """
    Create NLP solver.

    This solver solves an NLP problem. This is either relaxed or
    the binaries are fixed.
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create NLP problem."""
        super(RandomDirectionNlpSolver, self).__init___(problem, stats)
        options = regularize_options(options, IPOPT_SETTINGS)
        self.penalty_weight = 0.01
        self.penalty_scaling = 0.01

        self.idx_x_bin = problem.idx_x_bin
        x_bin_var = problem.x[self.idx_x_bin]
        penalty = CASADI_VAR.sym("penalty", x_bin_var.shape[0])
        rounded_value = CASADI_VAR.sym("rounded_value", x_bin_var.shape[0])

        penalty_term = ca.norm_2((x_bin_var - rounded_value) * penalty)

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
            rounding_error = ca.norm_2((x_bin_var - x_rounded))
            penalty = self.penalty_weight * np.random.rand(len(self.idx_x_bin))
            print(f"{nlpdata.obj_val=}")
            self.penalty_weight += self.penalty_scaling * rounding_error * abs(nlpdata.obj_val)
            print(f"{rounding_error=} - weight {self.penalty_weight=}")

            new_sol = self.solver(
                x0=nlpdata.x0,
                lbx=lbx, ubx=ubx,
                lbg=nlpdata.lbg, ubg=nlpdata.ubg,
                p=ca.vertcat(nlpdata.p, x_rounded, penalty)
            )

            success, _ = self.collect_stats("RNLP")
            if not success:
                raise Exception("Infeasible")
            success_out.append(success)
            sols_out.append(new_sol)

        nlpdata.prev_solutions = sols_out
        nlpdata.solved_all = success_out

        return nlpdata
