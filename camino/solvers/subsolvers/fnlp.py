# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""An NLP solver."""

import casadi as ca
import numpy as np
from camino.solvers import SolverClass, Stats, MinlpProblem, MinlpData, \
    regularize_options
from camino.settings import GlobalSettings, Settings
from camino.utils.conversion import to_0d


class FeasibilityNlpSolver(SolverClass):
    """Create benders master problem."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings):
        """Create benders master MILP."""
        super(FeasibilityNlpSolver, self).__init__(problem, stats, s)
        options = regularize_options(s.IPOPT_SETTINGS, {}, s)

        g = []
        self.g_idx = []
        beta = GlobalSettings.CASADI_VAR.sym("beta")
        for i in range(problem.g.shape[0]):
            if data.lbg[i] == data.ubg[i]:
                # when lbg == ubg we have an equality constraints, so we need to append it only once
                if abs(data.lbg[i]) == np.inf:
                    raise ValueError(
                        "lbg and ubg cannot be +-inf at the same time, "
                        "at instant {i}, you have {data.lbg[i]=} and {data.ubg[i]=}"
                    )
                g.append(-problem.g[i] + data.lbg[i] - beta)
                self.g_idx.append((i, 1))
            else:
                if data.lbg[i] != -np.inf:
                    g.append(-problem.g[i] + data.lbg[i] - beta)
                    self.g_idx.append((i, -1))
                if data.ubg[i] != np.inf:
                    g.append(problem.g[i] - data.ubg[i] - beta)
                    self.g_idx.append((i, 1))
        g = ca.vertcat(*g)
        self.nr_g = g.shape[0]

        self.lbg = -np.inf * np.ones((self.nr_g))
        self.ubg = np.zeros((self.nr_g))

        f = beta
        x = ca.vertcat(*[problem.x, beta])
        p = ca.vertcat(*[problem.p])
        self.idx_x_integer = problem.idx_x_integer
        options.update({"jit": s.WITH_JIT})
        self.solver = ca.nlpsol("nlpsol", "ipopt", {
            "f": f, "g": g, "x": x, "p": p
        }, options)

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """solve."""
        print("FEASIBILITY")
        success_out = []
        sols_out = []
        # add lower bound for the slacks
        lbx = ca.vcat([nlpdata.lbx, np.zeros(1)])
        # add upper bound for the slacks
        ubx = ca.vcat([nlpdata.ubx, np.inf * np.ones(1)])

        for success_prev, sol in zip(nlpdata.solved_all, nlpdata.solutions_all):
            if success_prev:
                success_out.append(success_prev)
                sols_out.append(sol)
            else:
                lbx[self.idx_x_integer] = to_0d(sol['x'][self.idx_x_integer])
                ubx[self.idx_x_integer] = to_0d(sol['x'][self.idx_x_integer])

                sol_new = self.solver(
                    x0=ca.vcat(
                        [nlpdata.x_sol[:nlpdata.x0.shape[0]],
                         np.zeros(1)]  # slacks initialization
                    ),
                    lbx=lbx,
                    ubx=ubx,
                    lbg=self.lbg,
                    ubg=self.ubg,
                    p=nlpdata.p
                )

                # Reconstruct lambda_g:
                lambda_g = to_0d(sol_new['lam_g'])
                lambda_g_req = np.zeros(nlpdata.lbg.shape[0])
                for lg, (idx, sgn) in zip(lambda_g, self.g_idx):
                    lambda_g_req[idx] = sgn * lg
                sol_new['lam_g'] = ca.DM(lambda_g_req)

                success, _ = self.collect_stats("F-NLP", sol=sol_new)
                if not success:
                    print("FNLP not solved")
                    raise Exception()

                # Maintain that it is caused due to infeasibility!!!
                success_out.append(False)
                sols_out.append(sol_new)

        nlpdata.prev_solutions = sols_out
        nlpdata.solved_all = success_out
        return nlpdata
