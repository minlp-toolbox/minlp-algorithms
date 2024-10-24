# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Implementation of the CIA algorithm."""

import copy
import datetime
import casadi as ca
import numpy as np
from typing import Tuple
from camino.solvers.subsolvers.nlp import NlpSolver
from camino.solvers import SolverClass, Stats, MinlpProblem, MinlpData
from camino.settings import Settings
from camino.utils import toc, logging
from camino.utils.conversion import to_0d

logger = logging.getLogger(__name__)


def to_list(dt, min_time, nr_b):
    """Create a min up or downtime list."""
    if isinstance(min_time, int):
        return [dt * min_time for _ in range(nr_b)]
    else:
        return [dt * min_time[i] for i in range(nr_b)]


class SurSolver(SolverClass):
    """
    Create a solver to approximate relaxed integer sequences
    with the sum-up-rounding scheme.

    ASSUMPTION: uniform discretization grid!
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, s: Settings):
        """Create NLP problem."""
        super(SurSolver, self).__init__(problem, stats, s)
        self.idx_x_integer = problem.idx_x_integer
        self.dt = self.meta.dt
        self.meta = copy.deepcopy(problem.meta)

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve NLP."""
        b_rel = to_0d(nlpdata.x_sol)[self.idx_x_integer]
        if any(b_rel) < 0 or any(b_rel) > 1:
            raise ValueError("Spotted a relaxed binary out of range.")
        b_rel[b_rel < 0] = 0
        b_rel[b_rel > 1.0] = 1

        b_bin = np.zeros_like(b_rel)
        for i in range(b_bin.shape[1]):  # for each control
            marker = 0
            for t in range(b_bin.shape[0]):  # scam the horizon
                if np.sum(b_rel[i][-(t-marker):] * self.dt) - \
                        np.sum(b_bin[i][-(t-marker):-1] * self.dt) >= 0.5*self.dt:
                    b_bin[i][t] = 1
                else:
                    b_bin[i][t] = 0

        nlpdata.x_sol[self.idx_x_integer] = b_bin

        return nlpdata


if __name__ == "__main__":
    test_array = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                          [0.1, 0.1, 0.1, 0.5, 0.1, 0.1]])
    test_array = test_array.T

    # Add check for tall matrix shape
    dt = 1
    b_rel = test_array
    b_rel = np.atleast_2d(b_rel)
    if any(b_rel.flatten()) < -1e-3 or any(b_rel.flatten()) > 1 + 1e-3:
        raise ValueError("Spotted a relaxed binary out of range.")
    b_rel[b_rel < 0] = 0
    b_rel[b_rel > 1.0] = 1

    b_bin = np.ones_like(b_rel) * 99
    for b_bin_i, b_rel_i in zip(b_bin.T, b_rel.T):  # for each control
        marker = -1
        for t in range(b_bin_i.shape[0]):  # scam the horizon
            breakpoint()
            if t == 0:
                res = b_rel_i[0]
            else:
                res = np.sum(b_rel_i[marker:t+1] * dt) - \
                    np.sum(b_bin_i[marker:t] * dt)
            print(f"Res. {res}")
            if res >= 0.5*dt:
                b_bin_i[t] = 1
                marker = 0
            else:
                b_bin_i[t] = 0
                marker += 1
    print(f"{b_bin=}")
    breakpoint()
