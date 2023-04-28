"""An NLP solver."""

import casadi as ca
import numpy as np
from copy import deepcopy
from benders_exp.solvers import SolverClass, Stats, MinlpProblem, MinlpData
from benders_exp.defines import WITH_LOGGING, WITH_JIT, IPOPT_SETTINGS, \
        CASADI_VAR
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
        if options is None:
            if WITH_LOGGING:
                options = {}
            else:
                options = {"ipopt.print_level": 0,
                           "verbose": False, "print_time": 0}

        self.idx_x_bin = problem.idx_x_bin
        options.update(IPOPT_SETTINGS)
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
            lbx[self.idx_x_bin] = to_0d(nlpdata.x_sol[self.idx_x_bin])
            ubx[self.idx_x_bin] = to_0d(nlpdata.x_sol[self.idx_x_bin])

        new_sol = self.solver(
            p=nlpdata.p, x0=nlpdata.x0,
            lbx=lbx, ubx=ubx,
            lbg=nlpdata.lbg, ubg=nlpdata.ubg
        )

        nlpdata.solved, stats = self.collect_stats()
        self.stats["nlp.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["nlp.iter"] += max(0, stats["iter_count"])
        if not nlpdata.solved:
            print("NLP not solved")
        else:
            nlpdata.prev_solution = new_sol
        return nlpdata


class FeasibilityNlpSolver(SolverClass):
    """Create benders master problem."""

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create benders master MILP."""
        super(FeasibilityNlpSolver, self).__init___(problem, stats)
        if options is None:
            if WITH_LOGGING:
                options = {}
            else:
                options = {
                    "ipopt.print_level": 0, "verbose": False, "print_time": 0
                }

        self.nr_g = problem.g.shape[0]
        s_lbg = CASADI_VAR.sym("s_lbg", self.nr_g)
        lbg = CASADI_VAR.sym("lbg", self.nr_g)
        ubg = CASADI_VAR.sym("ubg", self.nr_g)

        g = ca.vertcat(
            problem.g - lbg + s_lbg,
            ubg + s_lbg - problem.g
        )
        self.lbg = np.zeros((self.nr_g * 2, 1))
        self.ubg = ca.inf * np.ones((self.nr_g * 2, 1))
        f = ca.sum1(s_lbg)
        x = ca.vertcat(problem.x, s_lbg)
        p = ca.vertcat(problem.p, lbg, ubg)

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

        nlpdata.prev_solution = self.solver(
            x0=ca.vertcat(
                nlpdata.x_sol[:nlpdata.x0.shape[0]
                              ], np.zeros((self.nr_g * 1, 1))
            ),
            lbx=ca.vertcat(lbx, np.zeros((self.nr_g * 1, 1))),
            ubx=ca.vertcat(ubx, ca.inf * np.ones((self.nr_g * 1, 1))),
            lbg=self.lbg,
            ubg=self.ubg,
            p=ca.vertcat(nlpdata.p, nlpdata.lbg, nlpdata.ubg)
        )
        nlpdata.solved, stats = self.collect_stats()
        self.stats["fnlp.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["fnlp.iter"] += max(0, stats["iter_count"])
        if not nlpdata.solved:
            print("MILP not solved")
            raise Exception()
        return nlpdata
