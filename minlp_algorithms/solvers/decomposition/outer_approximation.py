"""Set of outer approximation-based solvers."""

from minlp_algorithms.solvers import Stats, MinlpProblem, MinlpData, Settings
from minlp_algorithms.solvers.subsolvers.fnlp import FeasibilityNlpSolver
from minlp_algorithms.solvers.decomposition import GenericDecomposition
from minlp_algorithms.solvers.decomposition.oa_master import OuterApproxMILP, OuterApproxMIQP, OuterApproxMILPImproved, OuterApproxMIQPImproved
from minlp_algorithms.utils import toc


class OuterApproximation(GenericDecomposition):
    """Linear Outer Approximation with standard feasibility restoration"""

    def __init__(
        self,
        problem: MinlpProblem, data: MinlpData, stats: Stats,
        settings: Settings = None,
        termination_type="std",
    ):
        """Generic decomposition algorithm."""
        master = OuterApproxMILP(problem, data, stats, settings)
        fnlp = FeasibilityNlpSolver(problem, data, stats, settings)
        GenericDecomposition.__init__(
            self, problem, data, stats, settings,
            master, fnlp, termination_type,
            first_relaxed=False
        )
        stats['total_time_loading'] = toc(reset=True)


class OuterApproximationQP(GenericDecomposition):
    """Quadratic Outer Approximation with standard feasibility restoration"""

    def __init__(
        self,
        problem: MinlpProblem, data: MinlpData, stats: Stats,
        settings: Settings = None,
        termination_type="std",
    ):
        """Generic decomposition algorithm."""
        master = OuterApproxMIQP(problem, data, stats, settings)
        fnlp = FeasibilityNlpSolver(problem, data, stats, settings)
        GenericDecomposition.__init__(
            self, problem, data, stats, settings,
            master, fnlp, termination_type,
            first_relaxed=False
        )
        stats['total_time_loading'] = toc(reset=True)


class OuterApproximationImproved(GenericDecomposition):
    """Improve Outer Approximation by adding cuts only on linear constraint to avoid infeasibility"""

    def __init__(
        self,
        problem: MinlpProblem, data: MinlpData, stats: Stats,
        settings: Settings = None,
        termination_type="std",
    ):
        """Generic decomposition algorithm."""
        master = OuterApproxMILPImproved(problem, data, stats, settings)
        fnlp = FeasibilityNlpSolver(problem, data, stats, settings)
        GenericDecomposition.__init__(
            self, problem, data, stats, settings,
            master, fnlp, termination_type,
            first_relaxed=False
        )
        stats['total_time_loading'] = toc(reset=True)


class OuterApproximationQPImproved(GenericDecomposition):
    """Improve quadratic Outer Approximation by adding cuts only on linear constraint to avoid infeasibility"""

    def __init__(
        self,
        problem: MinlpProblem, data: MinlpData, stats: Stats,
        settings: Settings = None,
        termination_type="std",
    ):
        """Generic decomposition algorithm."""
        master = OuterApproxMIQPImproved(problem, data, stats, settings)
        fnlp = FeasibilityNlpSolver(problem, data, stats, settings)
        GenericDecomposition.__init__(
            self, problem, data, stats, settings,
            master, fnlp, termination_type,
            first_relaxed=False
        )
        stats['total_time_loading'] = toc(reset=True)
