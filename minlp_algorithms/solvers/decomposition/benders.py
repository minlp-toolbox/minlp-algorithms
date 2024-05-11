"""Set of benders-based solvers."""

from minlp_algorithms.solvers import Stats, MinlpProblem, MinlpData, Settings
from minlp_algorithms.solvers.subsolvers.fnlp import FeasibilityNlpSolver
from minlp_algorithms.solvers.decomposition import GenericDecomposition
from minlp_algorithms.solvers.decomposition.benders_master import BendersMasterMILP, BendersMasterMIQP
from minlp_algorithms.utils import toc


class GeneralizedBenders(GenericDecomposition):
    """Generalized Benders."""

    def __init__(
        self, problem: MinlpProblem, data: MinlpData, stats: Stats,
        settings: Settings, termination_type="std",
    ):
        """Generic decomposition algorithm."""
        master = BendersMasterMILP(problem, data, stats, settings)
        fnlp = FeasibilityNlpSolver(problem, data, stats, settings)
        GenericDecomposition.__init__(
            self, problem, data, stats, settings,
            master, fnlp, termination_type,
            first_relaxed=False
        )
        stats['total_time_loading'] = toc(reset=True)


class GeneralizedBendersQP(GenericDecomposition):
    """Generalized Benders with an additional hessian in the cost function."""

    def __init__(
        self,
        problem: MinlpProblem, data: MinlpData, stats: Stats,
        settings: Settings = None,
        termination_type="std",
    ):
        """Generic decomposition algorithm."""
        master = BendersMasterMIQP(problem, data, stats, settings)
        fnlp = FeasibilityNlpSolver(problem, data, stats, settings)
        GenericDecomposition.__init__(
            self, problem, data, stats, settings,
            master, fnlp, termination_type,
            first_relaxed=True
        )
        stats['total_time_loading'] = toc(reset=True)
