# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

from camino.solvers import MiSolverClass
from camino.stats import Stats
from camino.settings import Settings
from camino.problem import MinlpProblem
from camino.data import MinlpData
from camino.solvers.decomposition import GeneralizedBenders, GeneralizedBendersQP, \
    OuterApproximation, OuterApproximationQP, OuterApproximationImproved, OuterApproximationQPImproved, \
    SequentialVoronoiMIQP, SequentialBendersMIQP
from camino.solvers.sequential import SequentialTrustRegionMILP
from camino.solvers.external.bonmin import BonminSolver
from camino.solvers.pumps import FeasibilityPump, ObjectiveFeasibilityPump, RandomObjectiveFeasibilityPump
from camino.solvers.approximation import CiaSolver
from camino.solvers.relaxation import NlpSolver
from camino.solvers.mip import MipSolver

SOLVER_MODES = {
    "gbd": GeneralizedBenders,
    "gbd-qp": GeneralizedBendersQP,
    "oa": OuterApproximation,
    "oa-qp": OuterApproximationQP,
    "oa-i": OuterApproximationImproved,
    "oa-qp-i": OuterApproximationQPImproved,
    "s-v-miqp": SequentialVoronoiMIQP,
    "s-b-miqp": SequentialBendersMIQP,
    "s-tr-milp": SequentialTrustRegionMILP,
    "fp": FeasibilityPump,
    "ofp": ObjectiveFeasibilityPump,
    "rofp": RandomObjectiveFeasibilityPump,
    "bonmin": lambda *args, **kwargs: BonminSolver(*args, **kwargs, algo_type="B-BB"),
    "bonmin-bb": lambda *args, **kwargs: BonminSolver(*args, **kwargs, algo_type="B-BB"),
    "bonmin-oa": lambda *args, **kwargs: BonminSolver(*args, **kwargs, algo_type="B-OA"),
    "bonmin-qg": lambda *args, **kwargs: BonminSolver(*args, **kwargs, algo_type="B-QG"),
    "bonmin-hyb": lambda *args, **kwargs: BonminSolver(*args, **kwargs, algo_type="B-Hyb"),
    "bonmin-ifp": lambda *args, **kwargs: BonminSolver(*args, **kwargs, algo_type="B-iFP"),
    "cia": CiaSolver,
    "nlp": lambda *args, **kwargs: NlpSolver(*args, **kwargs, set_bin=False),
    "nlp-fxd": lambda *args, **kwargs: NlpSolver(*args, **kwargs, set_bin=True),
    "mip": MipSolver,
}


def as_list(item, nr):
    """Make list if it is not a list yet."""
    if isinstance(item, list):
        return item
    else:
        return [item] * nr


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
            self._subsolvers = [SOLVER_MODES[name](
                problem, data, stats, settings, *args
            )]
            return
        elif "+" in name:
            names = name.split("+")
            for name in names:
                if name not in SOLVER_MODES:
                    raise Exception(f"Subsolver {name} does not exists")
            self._subsolvers = [
                SOLVER_MODES[subname](
                    problem, data, stats, setting, *args
                ) for subname, setting in zip(names, as_list(settings, len(names)))
            ]
        else:
            raise Exception(
                f"Solver mode {name} not implemented, options are:"
                + ", ".join(SOLVER_MODES.keys())
            )

    def solve(self, nlpdata: MinlpData, *args, **kwargs) -> MinlpData:
        """Solve the problem."""
        nlpdata = self._subsolvers[0].solve(nlpdata, *args, **kwargs)
        for subsolver in self._subsolvers[1:]:
            subsolver.warmstart(nlpdata)
            nlpdata = subsolver.solve(nlpdata, *args, **kwargs)

        return nlpdata

    def reset(self, nlpdata: MinlpData):
        """Reset the results."""
        [subsolver.reset() for subsolver in self._subsolvers]
        self.stats.reset()

    def warmstart(self, nlpdata: MinlpData):
        """Warmstart."""
        self._subsolver[0].warmstart(nlpdata)

    def collect_stats(self):
        """Return the statistics."""
        return self.stats['success'], self.stats

    def get_settings(self):
        """Return settings."""
        return self.settings
