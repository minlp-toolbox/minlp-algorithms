from minlp_algorithms.solvers import MiSolverClass
from minlp_algorithms.stats import Stats
from minlp_algorithms.settings import Settings
from minlp_algorithms.problem import MinlpProblem
from minlp_algorithms.data import MinlpData
from minlp_algorithms.solvers.decomposition.benders import GeneralizedBenders, GeneralizedBendersQP
from minlp_algorithms.solvers.decomposition.outer_approximation import OuterApproximation, OuterApproximationQP, \
    OuterApproximationImproved, OuterApproximationQPImproved
from minlp_algorithms.solvers.decomposition.sequential_voronoi_based_miqp import SequentialVoronoiMIQP
from minlp_algorithms.solvers.decomposition.sequential_benders_based_miqp import SequentialBendersMIQP
from minlp_algorithms.solvers.pumps import FeasibilityPump, ObjectiveFeasibilityPump, RandomObjectiveFeasibilityPump

SOLVER_MODES = {
    "gbd": GeneralizedBenders,
    "gbd-qp": GeneralizedBendersQP,
    "oa": OuterApproximation,
    "oa-qp": OuterApproximationQP,
    "oa-i": OuterApproximationImproved,
    "oa-qp-i": OuterApproximationQPImproved,
    "s-v-miqp": SequentialVoronoiMIQP,
    "s-b-miqp": SequentialBendersMIQP,
    "fp": FeasibilityPump,
    "ofp": ObjectiveFeasibilityPump,
    "rofp": RandomObjectiveFeasibilityPump
}


class MinlpSolver(MiSolverClass):
    """Polysolver loading the required subsolver and solving the problem."""

    def __init__(
        self, name,
        problem: MinlpProblem, data: MinlpData, stats: Stats = None,
        settings: Settings = None, problem_name: str = "generic",
        *args
    ):
        if stats is None:
            stats = Stats(name, problem_name, "generic")
        if settings is None:
            settings = Settings()

        self.stats = stats
        self.settings = settings
        super(MinlpSolver, self).__init__(problem, data, stats, settings)

        # Create actual solver
        if name in SOLVER_MODES:
            self._subsolver = SOLVER_MODES[name](
                problem, data, stats, settings, *args
            )
        else:
            raise Exception(
                f"Solver mode {name} not implemented, options are:"
                + ", ".join(SOLVER_MODES.keys())
            )

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
