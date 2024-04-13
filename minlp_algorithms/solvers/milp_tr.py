"""
MI linearity in Nonlinear optimization: A Trust Region approach

Alberto De Marchi 2023

Notes:
    - It seems that taking an L1-norm trust region doesn't work when handling
      problems with integer states. The algorithm is stuck in this case
    - Algorithm is not designed for nonlinear constraints yet. This might give
      problems.
    - The objective function should be linear in the integer variables
"""

from copy import deepcopy
import casadi as ca
import numpy as np
from typing import Tuple
from minlp_algorithms.solvers import SolverClass, Stats, MinlpProblem, MinlpData, regularize_options
from minlp_algorithms.defines import CASADI_VAR, Settings
from minlp_algorithms.solvers import get_idx_inverse
from minlp_algorithms.solvers.nlp import NlpSolver
from minlp_algorithms.utils import colored
from minlp_algorithms.utils import toc, logging

logger = logging.getLogger(__name__)


def milp_tr(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings,
    with_nlp_improvement=False
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    logger.info("START")
    tr = MILPTrustRegion(problem, data, stats, s)
    nlp = NlpSolver(problem, stats, s)

    delta = 1.0  # Initial radius
    delta_max = 64  # Max delta
    eps = 1e-4  # Tolerance
    # Sufficient decrease, and update rhos:
    rho = [0.1, 0.1, 0.2]
    kappa = 0.5  # Reduction factor
    p = 0.5  # Monotonicity parameter
    data = nlp.solve(data, set_x_bin=True)
    phi = data.obj_val
    logger.info(f"Initial start {phi=}")
    while True:
        data_p = tr.solve(deepcopy(data), delta)
        # Possible gain:
        psi_k = -data_p.obj_val
        # Make feasible if possible:
        if with_nlp_improvement:
            # This is a step, not described by De Marchi:
            data_p = nlp.solve(data_p, set_x_bin=True)
            f = data_p.obj_val
        else:
            f = float(tr.f(data_p.x_sol, data_p.p))
        if data_p.solved:
            logger.info(f"MILP-TR result {psi_k=}, {f=} < {phi=}")
            # Merit decrease
            a_k = phi - f
            logger.info(f"{a_k=} {psi_k=} <? {eps=}")
            if psi_k < eps:
                stats['total_time_calc'] = toc(reset=True)
                colored("Problem solved", "green")
                data_p.prev_solutions[0]['f'] = tr.f(data_p.x_sol, data_p.p)
                return problem, data_p, data_p.x_sol
            if a_k < rho[0] * psi_k:
                delta = kappa * delta
                colored(f"Trust region decreases to {delta}")
            else:
                # Accept step
                data = data_p
                # Update merit function
                phi = (1 - p) * phi + p * f
                # Update step size
                rho_k = a_k / psi_k
                if rho_k < rho[1]:
                    delta = kappa * delta
                elif rho_k < rho[2]:
                    delta = delta
                else:
                    delta = delta / kappa
                delta = min(delta, delta_max)
                colored(f"Step accepted, trust radius {delta}", "green")
        else:
            raise NotImplementedError(
                "The original paper doesn't handle nonlinear constraints, "
                "this implementation can not make it feasible!"
            )


class MILPTrustRegion(SolverClass):
    """Create benders master problem."""

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings,
                 options=None, with_lin_bounds=True):
        """Create benders master MILP."""
        super(MILPTrustRegion, self).__init___(problem, stats, s)

        self.f = ca.Function(
            "f", [problem.x, problem.p], [problem.f],
            {"jit": s.WITH_JIT}
        )
        self.g = ca.Function(
            "g", [problem.x, problem.p], [problem.g],
            {"jit": s.WITH_JIT}
        )

        self.jac_g = ca.Function(
            "jac_g", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)],
            {"jit": s.WITH_JIT}
        )
        self.jac_f = ca.Function(
            "jac_f", [problem.x, problem.p],
            [ca.jacobian(problem.f, problem.x)],
            {"jit": s.WITH_JIT}
        )

        self.nr_x = problem.x.numel()
        self.idx_x_c = get_idx_inverse(problem.idx_x_bin, self.nr_x)
        self.nr_x_c = len(self.idx_x_c)
        self.x = CASADI_VAR.sym("x", self.nr_x)
        self.options = regularize_options(options, s.MIP_SETTINGS, s)
        self.settings = s
        discrete = [0] * (self.nr_x)
        for i in problem.idx_x_bin:
            discrete[i] = 1
        self.options.update({
            "discrete": discrete,
        })

    def solve(self, nlpdata: MinlpData, delta=1) -> MinlpData:
        """Solve MILP with TR."""
        p = nlpdata.p
        x_hat = nlpdata.x_sol
        dx = self.x - nlpdata.x_sol
        f_lin = self.jac_f(x_hat, p) @ dx
        g_lin = self.g(x_hat, p) + self.jac_g(x_hat, p) @ dx
        # 1 norm - only on the continuous variables...
        g_extra = dx[self.idx_x_c]
        g_extra_lb = -delta * np.ones((self.nr_x_c,))
        g_extra_ub = delta * np.ones((self.nr_x_c,))

        self.solver = ca.qpsol("milp_tr", self.settings.MIP_SOLVER, {
            "f": f_lin, "g": ca.vertcat(g_lin, g_extra), "x": self.x,
        }, self.options)
        solution = self.solver(
            x0=x_hat,
            ubx=nlpdata.ubx,
            lbx=nlpdata.lbx,
            ubg=ca.vertcat(nlpdata.ubg, g_extra_ub),
            lbg=ca.vertcat(nlpdata.lbg, g_extra_lb),
        )
        success, _ = self.collect_stats("OFP")
        nlpdata.prev_solution = solution
        nlpdata.solved = success
        del self.solver
        return nlpdata
