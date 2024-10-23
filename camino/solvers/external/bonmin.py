# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bonmin solver."""

import casadi as ca
from camino.solvers import MiSolverClass, Stats, MinlpProblem, MinlpData, \
    regularize_options
from camino.settings import Settings


class BonminSolver(MiSolverClass):
    """Create MINLP solver (using bonmin)."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats,
                 s: Settings, algo_type="B-BB"):
        """Create MINLP problem.

        :param algo_type: Algorithm type, options: B-BB, B-OA, B-QG, or B-Hyb
        """
        super(BonminSolver, self).__init__(problem, data, stats, s)
        options = regularize_options(s.BONMIN_SETTINGS, {}, s)

        self.nr_x = problem.x.shape[0]
        discrete = [0] * self.nr_x
        for i in problem.idx_x_integer:
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
        nlpdata.solved, stats = self.collect_stats(
            "MINLP", sol=nlpdata.prev_solutions[-1])
        return nlpdata

    def reset(self, nlpdata: MinlpData):
        """Reset problem data."""

    def warmstart(self, nlpdata: MinlpData):
        """Warmstart the algorithm."""
