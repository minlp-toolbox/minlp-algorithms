"""Base algorithm for random feasibility pumps."""

from copy import deepcopy
import numpy as np
import casadi as ca
from minlp_algorithms.utils import colored
from minlp_algorithms.settings import Settings
from minlp_algorithms.solvers.subsolvers.nlp import NlpSolver
from minlp_algorithms.stats import Stats
from minlp_algorithms.data import MinlpData
from minlp_algorithms.problem import MinlpProblem
from minlp_algorithms.solvers.pumps.utils import integer_error, create_rounded_data
from minlp_algorithms.utils import logging
from minlp_algorithms.solvers import MiSolverClass

logger = logging.getLogger(__name__)


class PumpBaseRandom(MiSolverClass):
    """Random pump base algorithm."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, settings: Settings, pump, nlp=None):
        """Create a solver class."""
        super(PumpBaseRandom, self).__init__(problem, data, stats, settings)
        self.pump = pump
        self.idx_x_bin = problem.idx_x_bin
        if nlp is None:
            nlp = NlpSolver(problem, stats, settings)
        self.nlp = nlp

    def solve(self, nlpdata: MinlpData, relaxed: bool = False) -> MinlpData:
        """Solve the problem."""
        if self.stats.relaxed is None:
            if not relaxed:
                nlpdata = self.nlp.solve(nlpdata)
            self.stats.relaxed = nlpdata
        else:
            nlpdata = self.stats.relaxed

        logger.info("Solver initialized.")
        last_restart = 0
        self.stats["best_iter"] = -1
        done = False
        best_obj = ca.inf
        lb = nlpdata.obj_val
        prev_int_error = ca.inf

        while not done:
            logger.info(f"Starting iteration: {self.stats['iter_nr']}")
            nlpdata = self.pump.solve(nlpdata)
            random_obj_f = float(self.pump.f(nlpdata.x_sol, nlpdata.p))
            lb = min(random_obj_f, lb)

            colored(f"Current random NLP objective: {random_obj_f:.3e}", "blue")
            if random_obj_f < best_obj:
                datarounded = self.nlp.solve(create_rounded_data(nlpdata, self.idx_x_bin), set_x_bin=True)
                if datarounded.solved:
                    logger.debug(
                        f"NLP f={datarounded.obj_val:.3e} (iter {self.stats['iter_nr']}) "
                        f"vs old f={best_obj:.3e} (itr {self.stats['best_iter']})"
                    )
                    self.update_best_solutions(datarounded)
                else:
                    colored("Infeasible")
            else:
                colored("Not better than best found yet")

            int_error = integer_error(nlpdata.x_sol[self.idx_x_bin])

            self.stats["ub"] = best_obj
            self.stats["iter_nr"] += 1
            done = int_error < self.settings.CONSTRAINT_INT_TOL or (
                not np.isinf(best_obj)
                and (
                    (self.stats["iter_nr"] > self.settings.PUMP_MAX_STEP_IMPROVEMENTS and best_obj < random_obj_f)
                    or self.stats["iter_nr"] > self.settings.PUMP_MAX_STEP_IMPROVEMENTS + self.stats["best_iter"]
                )
            )
            retry = self.stats["iter_nr"] - last_restart > self.settings.PUMP_MAX_TRY and prev_int_error < int_error
            prev_int_error = int_error
            if not nlpdata.solved or retry:
                if len(self.best_solutions) > 0:
                    done = True
                else:
                    last_restart = self.stats["iter_nr"]
                    self.pump.alpha = 1.0
                    # If progress is frozen (unsolvable), try to fix it!
                    nlpdata = self.pump.solve(self.stats.relaxed)
                    logger.info(f"Current random NLP (restoration): f={random_obj_f:.3e}")
            if self.stats["iter_nr"] > self.settings.PUMP_MAX_ITER:
                if len(self.best_solutions) > 0:
                    done = True
                else:
                    return self.get_best_solutions(nlpdata)

        return self.get_best_solutions(nlpdata)

    def reset(self, nlpdata: MinlpData):
        """Reset problem data."""

    def warmstart(self, nlpdata: MinlpData):
        """Warmstart."""
        self.update_best_solutions(nlpdata)
