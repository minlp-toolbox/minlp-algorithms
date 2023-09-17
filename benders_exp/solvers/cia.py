"""Implementation of the CIA algorithm."""

import casadi as ca
import numpy as np
from typing import Callable, Tuple
from benders_exp.solvers.nlp import NlpSolver
from benders_exp.solvers import SolverClass, Stats, MinlpProblem, MinlpData, \
        regularize_options
from benders_exp.defines import WITH_JIT, IPOPT_SETTINGS, CASADI_VAR
from benders_exp.utils import to_0d, toc, logging
from copy import deepcopy
from pycombina import BinApprox, CombinaBnB

logger = logging.getLogger(__name__)



def cia_decomposition_algorithm(problem: MinlpProblem, data: MinlpData,
                                stats: Stats) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Run the base strategy."""
    logger.info("Setup NLP solver and Pycombina...")
    nlp = NlpSolver(problem, stats)
    combina_solver = PycombinaSolver(problem, stats)
    logger.info("Solver initialized.")

    toc()
    # Solve relaxed NLP(y^k)
    data = nlp.solve(data, set_x_bin=False)
    # TODO add check if ipopt succeeded
    breakpoint()

    # Solve CIA problem
    data = combina_solver.solve(data)

    # Solve NLP with fixed integers
    data = nlp.solve(data, set_x_bin=True)

    stats['total_time_calc'] = toc(reset=True)
    breakpoint()
    return problem, data, data.x_sol


class PycombinaSolver(SolverClass):
    """
    Create NLP solver.

    This solver solves an NLP problem. This is either relaxed or
    the binaries are fixed.
    """

    def __init__(self, problem: MinlpProblem, stats: Stats):
        """Create NLP problem."""
        super(PycombinaSolver, self).__init___(problem, stats)
        self.idx_x_bin = problem.idx_x_bin
        self.meta = problem.meta
        self.dt = problem.meta.dt


    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve NLP."""

        for sol in nlpdata.solutions_all:

            b_rel = to_0d(sol['x'][self.idx_x_bin]).reshape(-1, self.meta.n_discrete_control)
            breakpoint()
            N = b_rel.shape[0] + 1
            t = np.arange(0, N*self.dt, self.dt)  # NOTE assuming uniform grid
            binapprox = BinApprox(t, b_rel)

            # TODO additional combinatorial constraints

            combina = CombinaBnB(binapprox)
            combina.solve()

            b_bin = binapprox.b_bin
            nlpdata.x_sol[self.idx_x_bin] = b_bin

        return nlpdata
