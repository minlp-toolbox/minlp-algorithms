"""Class of decomposition solvers."""

import casadi as ca
import numpy as np
from minlp_algorithms.solvers import MiSolverClass, Stats, MinlpProblem, MinlpData, Settings
from minlp_algorithms.solvers.utils import get_termination_condition
from minlp_algorithms.solvers.subsolvers.nlp import NlpSolver
from minlp_algorithms.utils import colored, logging, toc
from minlp_algorithms.utils.conversion import to_0d

logger = logging.getLogger(__name__)


class GenericDecomposition(MiSolverClass):

    def __init__(
        self,
        problem: MinlpProblem, data: MinlpData,
        stats: Stats, settings: Settings,
        master: MiSolverClass, fnlp: NlpSolver, termination_type: str = "std",
        first_relaxed: bool = True
    ):
        """Generic decomposition algorithm."""
        super(GenericDecomposition, self).__init__(
            problem, data, stats, settings
        )
        self.termination_condition = get_termination_condition(
            termination_type, problem, data, settings
        )
        self.master = master
        self.nlp = NlpSolver(problem, stats, settings)
        self.fnlp = fnlp
        self.settings = settings
        self.stats = stats
        self.first_relaxed = first_relaxed

    def solve(self, data: MinlpData, *args, **kwargs) -> MinlpData:
        """Solve the problem."""
        logger.info("Solver initialized.")
        # Benders algorithm
        feasible = True
        x_star = np.nan * np.empty(data.x0.shape[0])
        x_hat = np.nan * np.empty(data.x0.shape[0])

        if self.first_relaxed:
            data = self.nlp.solve(data)
            self.stats['lb'] = data.obj_val
            data = self.master.solve(data, integers_relaxed=True)

        while (not self.termination_condition(
            self.stats, self.settings, self.stats['lb'], self.stats['ub'], x_star, x_hat
        )) and feasible:
            # Solve NLP(y^k)
            data = self.nlp.solve(data, set_x_bin=True)
            prev_feasible = data.solved

            # Is there a feasible success?
            x_star = self.update_best_solutions(data)

            # Is there any infeasible?
            if not np.all(data.solved_all):
                # Solve NLPF(y^k)
                data = self.fnlp.solve(data)
                logger.info(colored("Feasibility NLP solved.", "yellow"))

            # Solve master^k and set lower bound:
            data = self.master.solve(data, prev_feasible=prev_feasible)
            feasible = data.solved
            self.stats['lb'] = max(data.obj_val, self.stats['lb'])
            x_hat = data.x_sol
            logger.debug(f"x_hat = {to_0d(x_hat).tolist()}")
            logger.debug(f"{self.stats['ub']=}, {self.stats['lb']=}\n")
            self.stats['iter_nr'] += 1

        self.stats['total_time_calc'] = toc(reset=True)
        data.prev_solution = {'x': x_star, 'f': self.stats['ub']}
        return data

    def reset(self, nlpdata: MinlpData):
        """Reset Solvers."""
        self.master.reset()
