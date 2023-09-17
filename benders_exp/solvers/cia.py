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

    # Solve CIA problem
    data = combina_solver.solve(data)

    # Solve NLP with fixed integers
    if problem.meta.n_control > 0:
        data = nlp.solve(data, set_x_bin=True)
    else:
        f_fn = ca.Function('f', [problem.x, problem.p], [problem.f])
        data._sol['f'] = f_fn(data.x_sol, data.p)
        breakpoint()
        logger.warning("NEED A SIMULATOR FOR ROLLOUT!")

    stats['total_time_calc'] = toc(reset=True)
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

        b_rel = to_0d(nlpdata.x_sol[self.meta.idx_bin_control]).reshape(-1, self.meta.n_discrete_control)
        b_rel = np.hstack([np.asarray(b_rel), np.array(1-b_rel.sum(axis=1).reshape(-1, 1))]) # Make sos1 structure

        # Ensure values are not out of range due to numerical effects
        b_rel[b_rel < 0] = 0
        b_rel[b_rel > 1.0] = 1

        N = b_rel.shape[0] + 1
        t = np.arange(0, N*self.dt, self.dt)  # NOTE assuming uniform grid
        binapprox = BinApprox(t, b_rel)

        binapprox.set_min_down_times([self.dt * self.meta.min_uptime for _ in range(b_rel.shape[1])])
        binapprox.set_min_up_times([self.dt * self.meta.min_uptime for _ in range(b_rel.shape[1])])
        # binapprox.set_n_max_switches(...)
        # binapprox.set_max_up_times(...)

        combina = CombinaBnB(binapprox)
        combina.solve()

        b_bin = binapprox.b_bin[:-1, :].T.flatten()
        nlpdata.x_sol[self.meta.idx_bin_control] = b_bin

        return nlpdata
