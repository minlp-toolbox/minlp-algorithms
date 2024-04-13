import casadi as ca
from minlp_algorithms.settings import Settings
from minlp_algorithms.solvers import SolverClass, Stats, MinlpProblem, MinlpData


class AmplSolver(SolverClass):
    """Create MINLP solver (using bonmin)."""

    def __init__(self, problem: MinlpProblem, stats: Stats, s: Settings, algo_type="B-BB"):
        """Create MINLP problem.

        :param algo_type: Algorithm type, options: B-BB, B-OA, B-QG, or B-Hyb
        """
        super(AmplSolver, self).__init___(problem, stats, s)
        options = s.AMPL_EXPORT_SETTINGS.copy()
        options.update({
            "solver": "python3 -m minlp_algorithms copy /tmp/out.nl"
        })

        self.nr_x = problem.x.shape[0]
        discrete = [0] * self.nr_x
        for i in problem.idx_x_bin:
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
