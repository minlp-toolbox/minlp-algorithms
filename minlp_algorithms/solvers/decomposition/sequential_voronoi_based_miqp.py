"""Sequential Voronoi-based MIQP solver."""

from minlp_algorithms.solvers import Stats, MinlpProblem, MinlpData, Settings
from minlp_algorithms.solvers.subsolvers.fnlp import FeasibilityNlpSolver
from minlp_algorithms.solvers.decomposition import GenericDecomposition
from minlp_algorithms.solvers.decomposition.voronoi_master import VoronoiTrustRegionMIQP
from minlp_algorithms.utils import toc


class SequentialVoronoiMIQP(GenericDecomposition):
    """Generalized Benders."""

    def __init__(
        self, problem: MinlpProblem, data: MinlpData, stats: Stats,
        settings: Settings, termination_type="equality",
    ):
        """Generic decomposition algorithm."""
        master = VoronoiTrustRegionMIQP(problem, data, stats, settings)
        fnlp = FeasibilityNlpSolver(problem, data, stats, settings)
        GenericDecomposition.__init__(
            self, problem, data, stats, settings,
            master, fnlp, termination_type,
            first_relaxed=False
        )
        stats['total_time_loading'] = toc(reset=True)
