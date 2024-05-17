"""MIOCP specific solver based on the Combinatorial Integral Approximation with Pycombina."""

import casadi as ca
import logging
from typing import Tuple
from minlp_algorithms.data import MinlpData
from minlp_algorithms.problem import MinlpProblem
from minlp_algorithms.settings import Settings
from minlp_algorithms.solvers import MiSolverClass
from minlp_algorithms.solvers.approximation.cia import PycombinaSolver
from minlp_algorithms.solvers.subsolvers.nlp import NlpSolver
from minlp_algorithms.stats import Stats
from minlp_algorithms.utils import colored, toc
from minlp_algorithms.utils.conversion import to_0d

logger = logging.getLogger(__name__)


class CiaSolver(MiSolverClass):

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, settings: Settings):
        super(CiaSolver, self).__init__(problem, data, stats, settings)
        logger.info("Setup NLP solver and Pycombina...")
        self.nlp = NlpSolver(problem, self.stats, self.settings)
        self.combina_solver = PycombinaSolver(
            problem, self.stats, self.settings)
        logger.info("Solver initialized.")

    def solve(self, nlpdata: MinlpData) -> MinlpData:

        toc()
        # Solve relaxed NLP(y^k)
        nlpdata = self.nlp.solve(nlpdata, set_x_bin=False)
        self.stats["iter_nr"] = 0
        self.stats["best_iter"] = 0
        self.stats["nlp_obj"] = nlpdata.obj_val
        self.stats["lb"] = nlpdata.obj_val
        self.stats["x_sol"] = to_0d(nlpdata.x_sol)
        self.stats['nlp_rel_time'] = toc()
        if not nlpdata.solved:
            logger.error(colored("Relaxed NLP not solved."))
            raise RuntimeError()

        if self.settings.WITH_LOG_DATA:
            self.stats.save()

        # Solve CIA problem
        nlpdata = self.combina_solver.solve(nlpdata)

        # Solve NLP with fixed integers
        nlpdata = self.nlp.solve(nlpdata, set_x_bin=True)
        self.stats["iter_nr"] = 1
        self.stats["best_iter"] = 1
        self.stats["nlp_obj"] = nlpdata.obj_val
        self.stats["lb"] = nlpdata.obj_val
        self.stats["x_sol"] = to_0d(nlpdata.x_sol)
        self.stats['total_time_calc'] = toc(reset=True)
        if not nlpdata.solved:
            logger.error(colored("NLP with fixed binary not solved."))
            raise RuntimeError()
        self.update_best_solutions(nlpdata)
        return self.get_best_solutions(nlpdata)

    def reset(self, nlpdata: MinlpData):
        logger.warning(colored("Nothing to reset for CIA solver.", "yellow"))
        return nlpdata

    def warmstart(self, nlpdata: MinlpData):
        """Warmstart."""
        if not nlpdata.relaxed:
            self.update_best_solutions(nlpdata)
