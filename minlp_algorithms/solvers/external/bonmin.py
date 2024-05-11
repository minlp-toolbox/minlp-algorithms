# TODO
"""Bonmin solver."""

import casadi as ca
from minlp_algorithms.solvers import MiSolverClass, Stats, MinlpProblem, MinlpData, \
        regularize_options
from minlp_algorithms.settings import Settings


class BonminSolver(MiSolverClass):
    """Create MINLP solver (using bonmin)."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats,
                 s: Settings, algo_type="B-BB"):
        """Create MINLP problem.

        :param algo_type: Algorithm type, options: B-BB, B-OA, B-QG, or B-Hyb
        """
        super(BonminSolver, self).__init__(problem, stats, s)
        options = regularize_options(s.BONMIN_SETTINGS, {}, s)

        self.nr_x = problem.x.shape[0]
        discrete = [0] * self.nr_x
        for i in problem.idx_x_bin:
            discrete[i] = 1
        options.update({
            "discrete": discrete,
            "bonmin.algorithm": algo_type,
        })
        minlp = {
            "f": problem.f,
            "g": problem.g,
            "x": problem.x,
            "p": problem.p
        }
        self.solver = ca.nlpsol(
            "minlp", "bonmin", minlp, options
        )

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve MINLP."""
        nlpdata.prev_solution = self.solver(
            x0=nlpdata.x0,
            lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=nlpdata.lbg, ubg=nlpdata.ubg,
            p=nlpdata.p,
        )
        nlpdata.solved, stats = self.collect_stats("MINLP")
        return nlpdata

    def reset(self, nlpdata: MinlpData):
        """Reset problem data."""

    def warmstart(self, nlpdata: MinlpData):
        """Warmstart the algorithm."""
