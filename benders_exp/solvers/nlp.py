"""An NLP solver."""

import casadi as ca
import numpy as np
from copy import deepcopy
from benders_exp.solvers import SolverClass, Stats, MinlpProblem, MinlpData, \
        regularize_options
from benders_exp.defines import WITH_JIT, IPOPT_SETTINGS, CASADI_VAR
from benders_exp.utils import to_0d


class NlpSolver(SolverClass):
    """
    Create NLP solver.

    This solver solves an NLP problem. This is either relaxed or
    the binaries are fixed.
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create NLP problem."""
        super(NlpSolver, self).__init___(problem, stats)
        options = regularize_options(options, {}, {"ipopt.print_level": 0})

        self.idx_x_bin = problem.idx_x_bin
        options.update(IPOPT_SETTINGS)
        options["calc_multipliers"] = True
        # self.callback = DebugCallBack(
        #     'mycallback', problem.x.shape[0],
        #     problem.g.shape[0], problem.p.shape[0]
        # )
        # self.callback.add_to_solver_opts(options, 50)

        if problem.precompiled_nlp is not None:
            self.solver = ca.nlpsol(
                "nlp", "ipopt", problem.precompiled_nlp, options
            )
        else:
            options.update({"jit": WITH_JIT})
            self.solver = ca.nlpsol("nlpsol", "ipopt", {
                "f": problem.f, "g": problem.g, "x": problem.x, "p": problem.p
            }, options)

    def solve(self, nlpdata: MinlpData, set_x_bin=False) -> MinlpData:
        """Solve NLP."""
        lbx = nlpdata.lbx
        ubx = nlpdata.ubx
        if set_x_bin:
            x_bin_var = to_0d(nlpdata.x_sol[self.idx_x_bin])
            lbx[self.idx_x_bin] = x_bin_var
            ubx[self.idx_x_bin] = x_bin_var

        new_sol = self.solver(
            p=nlpdata.p, x0=nlpdata.x0,
            lbx=lbx, ubx=ubx,
            lbg=nlpdata.lbg, ubg=nlpdata.ubg
        )

        nlpdata.solved, _ = self.collect_stats("nlp")
        if not nlpdata.solved:
            print("NLP not solved")
        else:
            nlpdata.prev_solution = new_sol
        return nlpdata


class FeasibilityNlpSolver(SolverClass):
    """Create benders master problem."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, options=None):
        """Create benders master MILP."""
        super(FeasibilityNlpSolver, self).__init___(problem, stats)
        options = regularize_options(options, {}, {"ipopt.print_level": 0})

        g = []
        self.g_idx = []
        beta = CASADI_VAR.sym("beta")
        for i in range(problem.g.shape[0]):
            if data.lbg[i] == data.ubg[i]:
                if abs(data.lbg[i]) == np.inf:  # when lbg == ubg we have an equality constraints, so we need to append it only once
                    raise ValueError(f"lbg and ubg cannot be +-inf at the same time, at instant {i}, you have {data.lbg[i]=} and {data.ubg[i]=}")
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
        self.idx_x_bin = problem.idx_x_bin
        options.update({"jit": WITH_JIT})
        options.update(IPOPT_SETTINGS)
        self.solver = ca.nlpsol("nlpsol", "ipopt", {
            "f": f, "g": g, "x": x, "p": p
        }, options)

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """solve."""
        print("FEASIBILITY")
        lbx = deepcopy(nlpdata.lbx)
        ubx = deepcopy(nlpdata.ubx)
        lbx[self.idx_x_bin] = to_0d(nlpdata.x_sol[self.idx_x_bin])
        ubx[self.idx_x_bin] = to_0d(nlpdata.x_sol[self.idx_x_bin])
        lbx = ca.vcat([lbx, np.zeros(1)])  # add lower bound for the slacks
        ubx = ca.vcat([ubx, np.inf * np.ones(1)])  # add upper bound for the slacks

        nlpdata.prev_solution = self.solver(
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
        lambda_g = to_0d(nlpdata.lam_g_sol)
        lambda_g_req = np.zeros(nlpdata.lbg.shape[0])
        for lg, (idx, sgn) in zip(lambda_g, self.g_idx):
            lambda_g_req[idx] = sgn * lg
        nlpdata.prev_solution['lam_g'] = ca.DM(lambda_g_req)

        nlpdata.solved, _ = self.collect_stats("fnlp")
        if not nlpdata.solved:
            print("FNLP not solved")
            raise Exception()
        return nlpdata
