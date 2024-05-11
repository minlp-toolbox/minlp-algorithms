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
        super(GenericDecomposition, self).__init___(problem, stats, settings)
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
        lb = -ca.inf
        ub = ca.inf
        feasible = True
        best_iter = -1
        x_star = np.nan * np.empty(data.x0.shape[0])
        x_hat = np.nan * np.empty(data.x0.shape[0])

        if self.first_relaxed:
            data = self.nlp.solve(data)
            data = self.master.solve(data, integers_relaxed=True)
            breakpoint()

        while (not self.termination_condition(self.stats, self.settings, lb, ub, x_star, x_hat)) and feasible:
            # Solve NLP(y^k)
            data = self.nlp.solve(data, set_x_bin=True)
            prev_feasible = data.solved

            # Is there a feasible success?
            ub, x_star, best_iter = self.update_best_solutions(
                data, self.stats['iter_nr'], ub, x_star, best_iter, self.settings
            )

            # Is there any infeasible?
            if not np.all(data.solved_all):
                # Solve NLPF(y^k)
                data = self.fnlp.solve(data)
                logger.info(colored("Feasibility NLP solved.", "yellow"))

            # Solve master^k and set lower bound:
            data = self.master.solve(data, prev_feasible=prev_feasible)
            feasible = data.solved
            lb = data.obj_val
            x_hat = data.x_sol
            logger.debug(f"x_hat = {to_0d(x_hat).tolist()}")
            logger.debug(f"{ub=}, {lb=}\n")
            breakpoint()
            self.stats['iter_nr'] += 1

        self.stats['total_time_calc'] = toc(reset=True)
        data.prev_solution = {'x': x_star, 'f': ub}
        return data

    def reset(self, nlpdata: MinlpData):
        """Reset Solvers."""
        self.master.reset()
