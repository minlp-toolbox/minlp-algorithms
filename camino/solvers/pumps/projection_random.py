# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""An random direction search NLP solver."""

import casadi as ca
import numpy as np
from camino.solvers import SolverClass, Stats, MinlpProblem, MinlpData, regularize_options
from camino.settings import GlobalSettings, Settings
from camino.utils import logging
from camino.utils.conversion import to_0d

logger = logging.getLogger(__name__)


class RandomDirectionProjection(SolverClass):
    """
    Create Random Projection Solver.

    This solver solves an NLP problem. This is either relaxed or the binaries are fixed. The NLP aims to satisfy
    original constraint in g and minimize a linear combination of the original objective and the he distance of
    relaxed x_bin from their rounded value in L1-norm. The rounding procedure to compute x_bin is deterministic
    (to closest integer) and the weight vector in the cost function is random.
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, s: Settings, norm=2, penalty_scaling=0.5):
        super(RandomDirectionProjection, self).__init__(problem, stats, s)
        options = regularize_options(
            s.IPOPT_SETTINGS, {"jit": s.WITH_JIT, "ipopt.max_iter": 5000}, s)
        self.penalty_scaling = penalty_scaling

        self.idx_x_integer = problem.idx_x_integer
        x_bin_var = problem.x[self.idx_x_integer]
        self.nr_x_bin = x_bin_var.shape[0]
        penalty = GlobalSettings.CASADI_VAR.sym("penalty", self.nr_x_bin)
        rounded_value = GlobalSettings.CASADI_VAR.sym(
            "rounded_value", x_bin_var.shape[0])
        alpha = GlobalSettings.CASADI_VAR.sym("alpha", 1)
        self.alpha = 1.0
        self.alpha_reduction = 0.9
        obj_val = GlobalSettings.CASADI_VAR.sym("obj_val", 1)
        int_error = GlobalSettings.CASADI_VAR.sym("int_error", 1)

        self.max_rounding_error = x_bin_var.shape[0]
        if norm == 1:
            penalty_term = ca.sum1(
                ca.fabs(x_bin_var - rounded_value) * penalty)
            self.int_error_calc = lambda x, y: float(ca.norm_1((x - y)))
        else:
            penalty_term = ca.norm_2(
                (x_bin_var - rounded_value) * penalty) ** 2
            self.int_error_calc = lambda x, y: float(ca.norm_2((x - y)) ** 2)

        self.solver = ca.nlpsol("nlpsol", "ipopt", {
            "f": alpha / obj_val * problem.f + (1 - alpha) / int_error * penalty_term,
            "g": problem.g,
            "x": problem.x,
            "p": ca.vertcat(problem.p, rounded_value, penalty, alpha, obj_val, int_error),
        }, options)
        self.f = ca.Function("f", [problem.x, problem.p], [problem.f])

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        success_out = []
        sols_out = []
        for sol in nlpdata.solutions_all:
            lbx = nlpdata.lbx
            ubx = nlpdata.ubx

            x_bin_var = to_0d(sol["x"][self.idx_x_integer])
            x_rounded = np.round(x_bin_var).flatten()
            int_error = self.int_error_calc(x_bin_var, x_rounded)

            penalty = np.random.rand(len(self.idx_x_integer))
            total = float(ca.sum1(penalty)) / self.nr_x_bin
            logger.info(
                f"{nlpdata.obj_val=:.3} rounding error={int_error:.3f} " f"- weight {float(self.alpha):.3e}")
            if self.alpha < 1e-4:
                self.alpha = 0

            new_sol = self.solver(
                x0=nlpdata.x0, lbx=lbx, ubx=ubx, lbg=nlpdata.lbg, ubg=nlpdata.ubg,
                p=ca.vertcat(nlpdata.p, x_rounded, penalty, np.array(
                    [self.alpha / total, int_error, nlpdata.obj_val])),
            )

            success, _ = self.collect_stats("RNLP", sol=new_sol)
            if not success:
                logger.debug("Infeasible solution!")
            success_out.append(success)
            sols_out.append(new_sol)

        self.alpha *= self.alpha_reduction
        nlpdata.prev_solutions = sols_out
        nlpdata.solved_all = success_out
        return nlpdata
