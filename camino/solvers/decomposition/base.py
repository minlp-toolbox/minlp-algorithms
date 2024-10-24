# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Class of decomposition solvers."""

import casadi as ca
import numpy as np
from camino.solvers import MiSolverClass, Stats, MinlpProblem, MinlpData, Settings
from camino.solvers.utils import get_termination_condition
from camino.solvers.subsolvers.nlp import NlpSolver
from camino.utils import colored, logging, toc
from camino.utils.conversion import to_0d

logger = logging.getLogger(__name__)


class GenericDecomposition(MiSolverClass):

    def __init__(
        self,
        problem: MinlpProblem, data: MinlpData,
        stats: Stats, settings: Settings,
        master: MiSolverClass, fnlp: NlpSolver, termination_type: str = "std",
        first_relaxed: bool = True
    ):
        """Generic decomposition algorithm."""
        super(GenericDecomposition, self).__init__(
            problem, data, stats, settings
        )
        self.termination_condition = get_termination_condition(
            termination_type, problem, data, settings
        )
        self.master = master
        self.nlp = NlpSolver(problem, stats, settings)
        self.fnlp = fnlp
        self.settings = settings
        self.stats = stats
        self.first_relaxed = first_relaxed
        self._x_nan = np.nan * np.empty(data.x0.shape[0])

    def solve(self, data: MinlpData, *args, **kwargs) -> MinlpData:
        """Solve the problem."""
        logger.info("Solver initialized.")
        # Benders algorithm
        feasible = True
        x_hat = np.nan * np.empty(data.x0.shape[0])

        if self.first_relaxed:
            data = self.nlp.solve(data)
            self.stats['lb'] = data.obj_val
            data = self.master.solve(data, integers_relaxed=True)

        while (not self.termination_condition(
                self.stats, self.settings, self.stats['lb'], self.stats['ub'],
                self._get_x_star(), x_hat)) and feasible:
            # Solve NLP(y^k)
            data = self.nlp.solve(data, set_x_bin=True)

            # Is there a feasible success?
            self.update_best_solutions(data)

            # Is there any infeasible?
            if not np.all(data.solved_all):
                # Solve NLPF(y^k)
                data = self.fnlp.solve(data)
                logger.info(colored("Feasibility NLP solved.", "yellow"))

            # Solve master^k and set lower bound:
            data = self.master.solve(data)
            feasible = data.solved
            self.stats['lb'] = max(data.obj_val, self.stats['lb'])
            x_hat = data.x_sol
            logger.debug(f"x_hat = {to_0d(x_hat).tolist()}")
            logger.debug(f"{self.stats['ub']=}, {self.stats['lb']=}\n")
            self.stats['iter_nr'] += 1

        self.stats['total_time_calc'] = toc(reset=True)
        return self.get_best_solutions(data)

    def _get_x_star(self):
        if len(self.best_solutions) == 0:
            return self._x_nan
        else:
            return self.best_solutions[-1]['x']

    def reset(self, nlpdata: MinlpData):
        """Reset Solvers."""
        self.master.reset(nlpdata)
        nlpdata.best_solutions = []

    def warmstart(self, nlpdata: MinlpData):
        """Warmstart procedure."""
        if not nlpdata.relaxed:
            self.update_best_solutions(nlpdata)
            self.master.warmstart(nlpdata)
