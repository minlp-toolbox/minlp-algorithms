# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Set of benders-based solvers."""

from camino.utils import toc
from camino.solvers import Stats, MinlpProblem, MinlpData, Settings
from camino.solvers.subsolvers.fnlp import FeasibilityNlpSolver
from camino.solvers.subsolvers.fnlp_closest import FindClosestNlpSolver
from camino.solvers.decomposition.base import GenericDecomposition
from camino.solvers.decomposition.benders_master import BendersMasterMILP, BendersMasterMIQP
from camino.solvers.decomposition.oa_master import OuterApproxMILP, OuterApproxMIQP, \
    OuterApproxMILPImproved, OuterApproxMIQPImproved
from camino.solvers.decomposition.voronoi_master import VoronoiTrustRegionMIQP
from camino.solvers.decomposition.sequential_benders_master import BendersRegionMasters


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
