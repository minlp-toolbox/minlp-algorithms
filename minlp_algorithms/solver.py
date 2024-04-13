from minlp_algorithms.solvers import MiSolverClass
from minlp_algorithms.stats import Stats
from minlp_algorithms.settings import Settings
from minlp_algorithms.problem import MinlpProblem
from minlp_algorithms.data import MinlpData

SOLVER_MODES = {
}


class MinlpSolver(MiSolverClass):
    """Polysolver loading the required subsolver and solving the problem."""

    def __init__(
        self, name,
        problem: MinlpProblem, data: MinlpData, stats: Stats = None,
        settings: Settings = None, problem_name = "generic"
    ):
        if stats is None:
            stats = Stats(name, problem_name, "generic")
        if settings is None:
            settings = Settings()

        self.stats = stats
        self.settings = settings
        super(MinlpSolver, self).__init___(problem, stats, stats, settings)

        # Create actual solver
        if name in SOLVER_MODES:
            # TODO
            self._subsolver = None
            # Create solver class
            pass

    def solve(self, nlpdata: MinlpData, *args, **kwargs) -> MinlpData:
        """Solve the problem."""
        return self._subsolver.solve(nlpdata, *args, **kwargs)

    def reset(self, nlpdata: MinlpData):
        """Reset the results."""
        self._subsolver.reset()
        self.stats.data = {}

    def collect_stats(self):
        """Return the statistics."""
        return self.stats['success'], self.stats

    def get_settings(self):
        """Return settings."""
        return self.settings
