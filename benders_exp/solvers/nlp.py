"""An NLP solver."""

import casadi as ca
import numpy as np
from benders_exp.solvers import SolverClass, Stats, MinlpProblem, MinlpData, \
    regularize_options
from benders_exp.defines import CASADI_VAR, Settings
from benders_exp.utils import to_0d


class NlpSolver(SolverClass):
    """
    Create NLP solver.

    This solver solves an NLP problem. This is either relaxed or
    the binaries are fixed.
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, s: Settings):
        """Create NLP problem."""
        super(NlpSolver, self).__init___(problem, stats, s)
        options = regularize_options(s.IPOPT_SETTINGS, {
            "calc_multipliers": True,
            "ipopt.expect_infeasible_problem": "yes",
            "error_on_fail": False
        }, s)

        self.idx_x_bin = problem.idx_x_bin
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
            options.update({
                "jit": s.WITH_JIT,
            })
            self.solver = ca.nlpsol("nlpsol", "ipopt", {
                "f": problem.f, "g": problem.g, "x": problem.x, "p": problem.p
            }, options)

    def solve(self, nlpdata: MinlpData, set_x_bin=False) -> MinlpData:
        """Solve NLP."""
        success_out = []
        sols_out = []
        # if set_x_bin:
        #     self.solver.set_option("ipopt.expect_infeasible_problem", "yes")
        for sol in nlpdata.solutions_all:
            lbx = nlpdata.lbx
            ubx = nlpdata.ubx
            if set_x_bin:
                # Remove integer errors
                x_bin_var = np.round(to_0d(sol['x'][self.idx_x_bin]))
                lbx[self.idx_x_bin] = x_bin_var
                ubx[self.idx_x_bin] = x_bin_var

            new_sol = self.solver(
                p=nlpdata.p, x0=nlpdata.x0,
                lbx=lbx, ubx=ubx,
                lbg=nlpdata.lbg, ubg=nlpdata.ubg
            )

            success, _ = self.collect_stats("NLP")
            if not success:
                print("NLP not solved")
            success_out.append(success)
            sols_out.append(new_sol)

        nlpdata.prev_solutions = sols_out
        nlpdata.solved_all = success_out

        return nlpdata


class FeasibilityNlpSolver(SolverClass):
    """Create benders master problem."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings):
        """Create benders master MILP."""
        super(FeasibilityNlpSolver, self).__init___(problem, stats, s)
        options = regularize_options(s.IPOPT_SETTINGS, {}, s)

        g = []
        self.g_idx = []
        beta = CASADI_VAR.sym("beta")
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
        self.idx_x_bin = problem.idx_x_bin
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
                lbx[self.idx_x_bin] = to_0d(sol['x'][self.idx_x_bin])
                ubx[self.idx_x_bin] = to_0d(sol['x'][self.idx_x_bin])

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

                success, _ = self.collect_stats("F-NLP")
                if not success:
                    print("FNLP not solved")
                    raise Exception()

                # Maintain that it is caused due to infeasibility!!!
                success_out.append(False)
                sols_out.append(sol_new)

        nlpdata.prev_solutions = sols_out
        nlpdata.solved_all = success_out
        return nlpdata


class FindClosestNlpSolver(SolverClass):
    """Find closest feasible."""

    def __init__(self, problem: MinlpProblem, stats: Stats, s: Settings):
        """Create NLP problem."""
        super(FindClosestNlpSolver, self).__init___(problem, stats, s)
        options = regularize_options(
            s.IPOPT_SETTINGS, {
                "ipopt.mu_strategy": "adaptive",
                "ipopt.mu_target": min(
                    1e-5,
                    s.IPOPT_SETTINGS.get("ipopt.mu_target", 1e-5),
                ),
                "calc_multipliers":  True,
                "jit": s.WITH_JIT
            }, s)

        self.idx_x_bin = problem.idx_x_bin
        x_hat = CASADI_VAR.sym("x_hat", len(self.idx_x_bin))

        f = ca.norm_2(problem.x[self.idx_x_bin] - x_hat)
        self.solver = ca.nlpsol("nlpsol", "ipopt", {
            "f": f, "g": problem.g, "x": problem.x,
            "p": ca.vertcat(problem.p, x_hat)
        }, options)

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve NLP."""
        success_out = []
        sols_out = []
        for success_prev, sol in zip(nlpdata.solved_all, nlpdata.solutions_all):
            if success_prev:
                success_out.append(success_prev)
                sols_out.append(sol)
            else:
                lbx = nlpdata.lbx
                ubx = nlpdata.ubx
                x_bin_var = to_0d(sol['x'][self.idx_x_bin])

                new_sol = self.solver(
                    x0=nlpdata.x0,
                    lbx=lbx, ubx=ubx,
                    lbg=nlpdata.lbg, ubg=nlpdata.ubg,
                    p=ca.vertcat(nlpdata.p, x_bin_var)
                )
                new_sol['x_infeasible'] = sol['x']

                success, _ = self.collect_stats("FC-NLP")
                if not success:
                    print("FC-NLP not solved")
                success_out.append(False)
                sols_out.append(new_sol)

        nlpdata.prev_solutions = sols_out
        nlpdata.solved_all = success_out
        return nlpdata
