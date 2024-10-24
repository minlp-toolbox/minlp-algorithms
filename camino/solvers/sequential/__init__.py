# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from copy import deepcopy
from camino.data import MinlpData
from camino.solvers import MiSolverClass
from camino.solvers.sequential.milp_tr import TrustRegionMILP
from camino.solvers.subsolvers.nlp import NlpSolver
from camino.utils import colored, toc

logger = logging.getLogger(__name__)


class SequentialTrustRegionMILP(MiSolverClass):

    def __init__(self, problem, data, stats, settings):
        super(SequentialTrustRegionMILP, self).__init__(
            problem, data, stats, settings)
        # Required subsolvers
        self.tr_milp = TrustRegionMILP(problem, data, stats, settings)
        self.nlp = NlpSolver(problem, stats, settings)
        # Algorithm parameters
        self.delta = 1.0  # Initial radius
        self.delta_max = 64  # Max delta
        self.eps = 1e-4  # Tolerance
        # Sufficient decrease, and update rhos:
        self.rho = [0.1, 0.1, 0.2]
        self.kappa = 0.5  # Reduction factor
        self.p = 0.5  # Monotonicity parameter

    def solve(self, nlpdata: MinlpData, with_nlp_improvement: bool = False):
        self.reset()
        nlpdata = self.nlp.solve(nlpdata, set_x_bin=True)
        phi = nlpdata.obj_val
        logger.info(f"Initial start {phi=}")
        while True:
            data_p = self.tr_milp.solve(deepcopy(nlpdata), self.delta)
            logger.info(colored(f"TR-MILP {data_p.x_sol=}"))
            # Possible gain:
            psi_k = -data_p.obj_val
            # Make feasible if possible:
            if with_nlp_improvement:
                # This is a step, not described by De Marchi:
                data_p = self.nlp.solve(data_p, set_x_bin=True)
                f = data_p.obj_val
            else:
                f = float(self.tr_milp.f(data_p.x_sol, data_p.p))
            if data_p.solved:
                logger.info(f"MILP-TR result {psi_k=}, {f=} < {phi=}")
                # Merit decrease
                a_k = phi - f
                logger.info(f"{a_k=} | {psi_k=} <? {self.eps=}")
                if psi_k < self.eps:
                    self.stats['total_time_calc'] = toc(reset=True)
                    colored("Problem solved", "green")
                    data_p.prev_solutions[0]['f'] = self.tr_milp.f(
                        data_p.x_sol, data_p.p)
                    return data_p
                if a_k < self.rho[0] * psi_k:
                    self.delta = self.kappa * self.delta
                    colored(f"Trust region decreases to {self.delta}")
                else:
                    # Accept step
                    nlpdata = data_p
                    # Update merit function
                    phi = (1 - self.p) * phi + self.p * f
                    # Update step size
                    rho_k = a_k / psi_k
                    if rho_k < self.rho[1]:
                        self.delta = self.kappa * self.delta
                    elif rho_k < self.rho[2]:
                        self.delta = self.delta
                    else:
                        self.delta = self.delta / self.kappa
                    self.delta = min(self.delta, self.delta_max)
                    colored(
                        f"Step accepted, trust radius {self.delta}", "green")
            else:
                raise NotImplementedError(
                    "The original paper doesn't handle nonlinear constraints, "
                    "this implementation can not make it feasible!"
                )

    def reset(self):
        """Reset."""
        self.delta = 1.0

    def warmstart(self, nlpdata: MinlpData):
        """Warmstart."""
        logger.info(colored(
            "s-tr-milp warmstarting is automatic, new linearization point inherited from previous solver.", "yellow"))
