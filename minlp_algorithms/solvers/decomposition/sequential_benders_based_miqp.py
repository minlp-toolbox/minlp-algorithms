"""Sequential Benders-based MIQP solver."""

from minlp_algorithms.solvers import Stats, MinlpProblem, MinlpData, Settings
from minlp_algorithms.solvers.subsolvers.fnlp_closest import FindClosestNlpSolver
from minlp_algorithms.solvers.decomposition import GenericDecomposition
from minlp_algorithms.solvers.decomposition.sequential_benders_master import BendersRegionMasters
from minlp_algorithms.utils import toc


class SequentialBendersMIQP(GenericDecomposition):
    """Generalized Benders."""

    def __init__(
        self, problem: MinlpProblem, data: MinlpData, stats: Stats,
        settings: Settings, termination_type="std",
    ):
        """Generic decomposition algorithm."""
        master = BendersRegionMasters(problem, data, stats, settings)
        fnlp = FindClosestNlpSolver(problem, stats, settings)
        GenericDecomposition.__init__(
            self, problem, data, stats, settings,
            master, fnlp, termination_type,
            first_relaxed=True
        )
        stats['total_time_loading'] = toc(reset=True)
