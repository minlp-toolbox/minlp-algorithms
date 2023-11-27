"""Implementation of the CIA algorithm."""

import casadi as ca
import numpy as np
from typing import Tuple
from benders_exp.solvers.nlp import NlpSolver
from benders_exp.solvers import SolverClass, Stats, MinlpProblem, MinlpData
from benders_exp.defines import Settings
from benders_exp.utils import to_0d, toc, logging
try:
    from pycombina import BinApprox, CombinaBnB
except Exception:
    class BinApprox:
        pass

    class CombinaBnB:
        pass

logger = logging.getLogger(__name__)


def simulate(x0, u, f_dyn):
    N = u.shape[0]
    x = []
    for t in range(N):
        if t == 0:
            x.append(to_0d(f_dyn(x0, u[t, :])))
        else:
            x.append(to_0d(f_dyn(x[-1], u[t, :])))
    return np.array(x).flatten().tolist()


def cia_decomposition_algorithm(problem: MinlpProblem, data: MinlpData,
                                stats: Stats, s: Settings) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Run the base strategy."""
    logger.info("Setup NLP solver and Pycombina...")
    nlp = NlpSolver(problem, stats, s)
    combina_solver = PycombinaSolver(problem, stats, s)
    logger.info("Solver initialized.")

    toc()
    # Solve relaxed NLP(y^k)
    data = nlp.solve(data, set_x_bin=False)
    # TODO add check if ipopt succeeded

    stats["iter_nr"] = 0
    stats["best_iter"] = 0
    stats["nlp_obj"] = data.obj_val
    stats["lb"] = data.obj_val
    stats["x_sol"] = to_0d(data.x_sol)
    stats['nlp_rel_time'] = toc()
    # stats["success"] = ...

    if s.WITH_LOG_DATA:
        stats.save()

    # Solve CIA problem
    data = combina_solver.solve(data)

    # Solve NLP with fixed integers
    data = nlp.solve(data, set_x_bin=True)
    stats["iter_nr"] = 1
    stats["best_iter"] = 1
    stats["nlp_obj"] = data.obj_val
    stats["lb"] = data.obj_val
    stats["x_sol"] = to_0d(data.x_sol)
    stats['total_time_calc'] = toc(reset=True)

    return problem, data, data.x_sol


def to_list(dt, min_time, nr_b):
    """Create a min up or downtime list."""
    if isinstance(min_time, int):
        return [dt * min_time for _ in range(nr_b)]
    else:
        return [dt * min_time[i] for i in range(nr_b)]


class PycombinaSolver(SolverClass):
    """
    Create NLP solver.

    This solver solves an NLP problem. This is either relaxed or
    the binaries are fixed.
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, s: Settings):
        """Create NLP problem."""
        super(PycombinaSolver, self).__init___(problem, stats, s)
        self.idx_x_bin = problem.idx_x_bin
        self.meta = problem.meta

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve NLP."""
        b_rel = to_0d(nlpdata.x_sol[self.meta.idx_bin_control]
                      ).reshape(-1, self.meta.n_discrete_control)
        b_rel = np.hstack([np.asarray(b_rel), np.array(
            1-b_rel.sum(axis=1).reshape(-1, 1))])  # Make sos1 structure

        # Ensure values are not out of range due to numerical effects
        b_rel[b_rel < 0] = 0
        b_rel[b_rel > 1.0] = 1

        N = b_rel.shape[0] + 1
        # NOTE assuming uniform grid
        t = np.arange(0, N*self.meta.dt, self.meta.dt)
        binapprox = BinApprox(t, b_rel)

        value_set = False
        if self.meta.min_downtime is not None:
            value_set = True
            binapprox.set_min_down_times(
                to_list(self.meta.dt, self.meta.min_downtime, b_rel.shape[1]))
        if self.meta.min_uptime is not None:
            value_set = True
            binapprox.set_min_up_times(
                to_list(self.meta.dt, self.meta.min_uptime, b_rel.shape[1]))

        # binapprox.set_n_max_switches(...)
        # binapprox.set_max_up_times(...)

        if not value_set:
            raise Exception("Minimum uptime or downtime needs to be set!")

        combina = CombinaBnB(binapprox)
        combina.solve()

        b_bin = binapprox.b_bin[:-1, :].T.flatten()
        nlpdata.x_sol[self.meta.idx_bin_control] = b_bin

        return nlpdata
