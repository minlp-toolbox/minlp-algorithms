# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Collection of Pumps.

This folder contains a collection of solvers based on the 'feasibility pump' idea.
They include:
    - Feasibility Pump
    - Objective Feasibility Pump
    - Random Objective Feasibility Pump
"""

from camino.settings import Settings
from camino.stats import Stats
from camino.data import MinlpData
from camino.problem import MinlpProblem
from camino.solvers.pumps.projections import (
    LinearProjection,
    ObjectiveLinearProjection,
)
from camino.solvers.pumps.projection_random import RandomDirectionProjection
from camino.solvers.pumps.base import PumpBase
from camino.solvers.pumps.base_random import PumpBaseRandom


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
        super(FeasibilityPump, self).__init__(
            problem, data, stats, settings, pump, nlp=nlp)


class ObjectiveFeasibilityPump(PumpBase):
    """
    Objective Feasibility Pump

        Sharma, S., Knudsen, B. R., & Grimstad, B. (2016). Towards an objective feasibility pump for convex MINLPs.
        Computational Optimization and Applications, 63, 737-753.
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, settings: Settings, nlp=None):
        """Create a solver class."""
        pump = ObjectiveLinearProjection(problem, stats, settings)
        super(ObjectiveFeasibilityPump, self).__init__(
            problem, data, stats, settings, pump, nlp=nlp)


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
        super(RandomObjectiveFeasibilityPump, self).__init__(
            problem, data, stats, settings, pump, nlp=nlp)
