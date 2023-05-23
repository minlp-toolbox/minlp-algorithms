"""Bonmin solver."""

import casadi as ca
from benders_exp.solvers import SolverClass, Stats, MinlpProblem, MinlpData, \
        regularize_options


class BonminSolver(SolverClass):
    """Create MINLP solver (using bonmin)."""

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create MINLP problem."""
        super(BonminSolver, self).__init___(problem, stats)
        options = regularize_options(options, {}, {"ipopt.print_level": 0})

        self.nr_x = problem.x.shape[0]
        discrete = [0] * self.nr_x
        for i in problem.idx_x_bin:
            discrete[i] = 1
        options.update({
            "discrete": discrete,
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

    def solve(self, nlpdata: MinlpData, prev_feasible=True) -> MinlpData:
        """Solve MINLP."""
        nlpdata.prev_solution = self.solver(
            x0=nlpdata.x0,
            lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=nlpdata.lbg, ubg=nlpdata.ubg,
            p=nlpdata.p,
        )
        nlpdata.prev_solution['x'] = nlpdata.prev_solution['x']
        nlpdata.solved, stats = self.collect_stats("minlp")
        return nlpdata
