# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Mixed-integer linearity in nonlinear optimization: a trust region approach

Alberto De Marchi 2023

Notes:
    - It seems that taking an L1-norm trust region doesn't work when handling
      problems with integer states. The algorithm is stuck in this case
    - Algorithm is not designed for nonlinear constraints yet. This might give
      problems.
    - The objective function should be linear in the integer variables
"""

from copy import deepcopy
import casadi as ca
import numpy as np
from typing import Tuple
from camino.solvers import SolverClass, Stats, MinlpProblem, MinlpData, regularize_options
from camino.settings import GlobalSettings, Settings
from camino.solvers import get_idx_inverse
from camino.solvers.subsolvers.nlp import NlpSolver
from camino.utils import colored
from camino.utils import toc, logging

logger = logging.getLogger(__name__)


class TrustRegionMILP(SolverClass):
    """Create trust region MILP master problem."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings):
        """Create benders master MILP."""
        super(TrustRegionMILP, self).__init__(problem, stats, s)
        self.settings = s
        self.options = regularize_options(s.MIP_SETTINGS, {}, s)

        self.f = ca.Function(
            "f", [problem.x, problem.p], [problem.f],
            {"jit": s.WITH_JIT}
        )
        self.g = ca.Function(
            "g", [problem.x, problem.p], [problem.g],
            {"jit": s.WITH_JIT}
        )

        self.jac_g = ca.Function(
            "jac_g", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)],
            {"jit": s.WITH_JIT}
        )
        self.jac_f = ca.Function(
            "jac_f", [problem.x, problem.p],
            [ca.jacobian(problem.f, problem.x)],
            {"jit": s.WITH_JIT}
        )

        self.nr_x = problem.x.numel()
        self.idx_x_c = get_idx_inverse(problem.idx_x_integer, self.nr_x)
        self.nr_x_c = len(self.idx_x_c)
        self.x = GlobalSettings.CASADI_VAR.sym("x", self.nr_x)
        self.options["discrete"] = [
            1 if i in problem.idx_x_integer else 0 for i in range(self.nr_x)]

    def solve(self, nlpdata: MinlpData, delta=1) -> MinlpData:
        """Solve MILP with TR."""
        p = nlpdata.p
        x_hat = nlpdata.x_sol
        dx = self.x - nlpdata.x_sol
        f_lin = self.jac_f(x_hat, p) @ dx
        g_lin = self.g(x_hat, p) + self.jac_g(x_hat, p) @ dx
        # 1 norm - only on the continuous variables...
        g_extra = dx[self.idx_x_c]
        g_extra_lb = -delta * np.ones((self.nr_x_c,))
        g_extra_ub = delta * np.ones((self.nr_x_c,))

        solver = ca.qpsol("milp_tr", self.settings.MIP_SOLVER, {
            "f": f_lin, "g": ca.vertcat(g_lin, g_extra), "x": self.x,
        }, self.options)
        solution = solver(
            x0=x_hat,
            ubx=nlpdata.ubx,
            lbx=nlpdata.lbx,
            ubg=ca.vertcat(nlpdata.ubg, g_extra_ub),
            lbg=ca.vertcat(nlpdata.lbg, g_extra_lb),
        )
        success, _ = self.collect_stats("TR-MILP", solver, solution)
        nlpdata.prev_solution = solution
        nlpdata.solved = success
        return nlpdata
