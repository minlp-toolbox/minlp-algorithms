"""
Collection of Pumps.

This folder contains a collection of solvers based on the 'feasibility pump' idea.
They include:
    - Feasibility Pump
    - Objective Feasibility Pump
    - Random Objective Feasibility Pump
"""

from minlp_algorithms.settings import Settings
from minlp_algorithms.stats import Stats
from minlp_algorithms.data import MinlpData
from minlp_algorithms.problem import MinlpProblem
from minlp_algorithms.solvers.pumps.projections import (
    LinearProjection,
    ObjectiveLinearProjection,
)
from minlp_algorithms.solvers.pumps.projection_random import RandomDirectionProjection
from minlp_algorithms.solvers.pumps.base import PumpBase
from minlp_algorithms.solvers.pumps.base_random import PumpBaseRandom


class FeasibilityPump(PumpBase):
    """
    Feasibility Pump

    According to:
        Bertacco, L., Fischetti, M., & Lodi, A. (2007). A feasibility pump heuristic for general mixed-integer problems.
        Discrete Optimization, 4(1), 63-76.
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, settings: Settings, nlp=None):
        """Create a solver class."""
        pump = LinearProjection(problem, stats, settings)
        super(FeasibilityPump, self).__init__(problem, data, stats, settings, pump, nlp=nlp)


class ObjectiveFeasibilityPump(PumpBase):
    """
    Objective Feasibility Pump

        Sharma, S., Knudsen, B. R., & Grimstad, B. (2016). Towards an objective feasibility pump for convex MINLPs.
        Computational Optimization and Applications, 63, 737-753.
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, settings: Settings, nlp=None):
        """Create a solver class."""
        pump = ObjectiveLinearProjection(problem, stats, settings)
        super(ObjectiveFeasibilityPump, self).__init__(problem, data, stats, settings, pump, nlp=nlp)


class RandomObjectiveFeasibilityPump(PumpBaseRandom):
    """
    Random objective pump

    Alternative version of the objective pump where the weights are randomly chosen. This allows to remove some other
    random recovery mechanisms.
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, settings: Settings, nlp=None,
                 norm=2, penalty_scaling=0.5):
        """Create a solver class."""
        pump = RandomDirectionProjection(
            problem, stats, settings, norm=norm, penalty_scaling=penalty_scaling)
        super(RandomObjectiveFeasibilityPump, self).__init__(problem, data, stats, settings, pump, nlp=nlp)
