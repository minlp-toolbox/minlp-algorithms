# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

# DISCLAIMER: this implementation is experimental and in general it does not work!

import casadi as ca
from camino.settings import Settings
from camino.solvers import SolverClass, Stats, MinlpProblem, MinlpData


class AmplSolver(SolverClass):
    """Create AMPL dump."""

    def __init__(self, problem: MinlpProblem, stats: Stats, s: Settings):
        """Create MINLP problem."""
        super(AmplSolver, self).__init__(problem, stats, s)
        options = s.AMPL_EXPORT_SETTINGS.copy()
        options.update({
            "solver": "python3 -m minlp_algorithms copy /tmp/out.nl"
        })

        self.nr_x = problem.x.shape[0]
        discrete = [0] * self.nr_x
        for i in problem.idx_x_integer:
            discrete[i] = 1
        # options.update({
        #     "discrete": discrete,
        # })
        minlp = {
            "f": problem.f,
            "g": problem.g,
            "x": ca.vertcat(problem.x, problem.p),
            # "p": problem.p
        }
        self.solver = ca.nlpsol(
            "ampl", "ampl", minlp, options
        )

    def solve(self, nlpdata: MinlpData, prev_feasible=True) -> MinlpData:
        """Solve MINLP."""
        try:
            nlpdata.prev_solution = self.solver(
                x0=nlpdata.x0,
                lbx=nlpdata.lbx, ubx=nlpdata.ubx,
                lbg=nlpdata.lbg, ubg=nlpdata.ubg,
                # p=nlpdata.p,
            )
        finally:
            print("File saved to /tmp/out.nl")

        raise Exception("See AMLP file!")
