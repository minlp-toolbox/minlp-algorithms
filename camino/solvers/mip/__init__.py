# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""A MIP solver."""

import casadi as ca
import logging
from camino.solvers import MiSolverClass, Stats, MinlpProblem, MinlpData, \
    regularize_options
from camino.settings import Settings

logger = logging.getLogger(__name__)


class MipSolver(MiSolverClass):
    """
    Create MIP solver.

    This solver solves a mixed-integer problem (MIP).
    Constraints must be linear, for linear objective it solves a MILP, for quadratic
    cost it solves a MIQP.
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings):
        """Create NLP problem."""
        super(MipSolver, self).__init__(problem, data, stats, s)
        self.options = regularize_options(s.MIP_SETTINGS, {}, s)

        self.idx_x_integer = problem.idx_x_integer
        self.nr_x = problem.x.shape[0]

        self.options["discrete"] = [
            1 if i in self.idx_x_integer else 0 for i in range(self.nr_x)]

        self.solver = ca.qpsol(
            "mip_solver", self.settings.MIP_SOLVER, {
                "f": problem.f, "g": problem.g, "x": problem.x, "p": problem.p,
            }, self.options)

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve given MIP."""

        nlpdata.prev_solution = self.solver(
            x0=nlpdata.x0, lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=nlpdata.lbg, ubg=nlpdata.ubg, p=nlpdata.p)

        nlpdata.solved, stats = self.collect_stats(
            "MIP", sol=nlpdata.prev_solutions[-1])
        return nlpdata

    def reset(self, nlpdata: MinlpData):
        """Reset problem data."""
        logger.warning(
            "reset method not available for MipSolver class!")

    def warmstart(self, nlpdata: MinlpData):
        """Warmstart the algorithm."""
        logger.warning(
            "warmstart method not available for MipSolver class. To pass a linearization point use 'nlpdata.x0'.")
