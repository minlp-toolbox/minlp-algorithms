# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import casadi as ca
from camino.settings import Settings
from camino.data import MinlpData
from camino.problem import MinlpProblem
from camino.stats import Stats
from camino.utils import toc, logging
from camino.utils.conversion import to_0d


def almost_equal(a, b, EPS=1e-5):
    """Check if almost equal."""
    return a + EPS > b and a - EPS < b


class Constraints:
    """Store bounds."""

    def __init__(self, nr=0, eq=None, lb=None, ub=None):
        """Store bounds."""
        if nr == 0:
            eq, lb, ub = [], [], []

        self.nr = nr
        self.eq = eq
        self.lb = lb
        self.ub = ub
        self.a = None
        self.b = None

    def add(self, lb, eq, ub):
        """Add a bound."""
        if self.nr == 0:
            self.nr += 1
            self.eq = eq
            self.lb = ca.DM(lb)
            self.ub = ca.DM(ub)
        else:
            self.nr += 1
            self.eq = ca.vertcat(self.eq, eq)
            self.lb = ca.vertcat(self.lb, lb)
            self.ub = ca.vertcat(self.ub, ub)

    def to_generic(self):
        """Convert to a generic class."""
        return self

    def __add__(self, other):
        """Add two bounds."""
        other = other.to_generic()
        if other.nr == 0:
            return self
        if self.nr == 0:
            return other

        return Constraints(
            self.nr + other.nr,
            ca.vertcat(self.eq, other.eq),
            ca.vertcat(self.lb, other.lb),
            ca.vertcat(self.ub, other.ub)
        )

    def get_a(self, x, nr_x):
        """Get A matrix."""
        if self.a is None:
            self.a = ca.Function("geq", [x], [ca.jacobian(self.eq, x)])(
                np.zeros((nr_x,))
            )
        return self.a

    def get_b(self, x, nr_x):
        """Get B matrix."""
        if self.b is None:
            self.b = ca.Function("eq", [x], [self.eq])(np.zeros((nr_x,)))
        return self.b

    def __str__(self):
        """Represent."""
        out = f"Eq: {self.nr}\n\n"
        for i in range(self.nr):
            out += f"{self.lb[i]} <= {self.eq[i]} <= {self.ub[i]}\n"
        return out


def bin_equal(sol1, sol2, idx_x_integer):
    """Binary variables equal."""
    return np.allclose(sol1[idx_x_integer], sol2[idx_x_integer], equal_nan=False, atol=1e-2)


def any_equal(sol, refs, idx_x_integer):
    """Check if any is equal."""
    for ref in refs:
        if bin_equal(sol, ref['x'], idx_x_integer):
            return True
    return False


def get_solutions_pool(nlpdata, success, stats, s: Settings, solution, idx_x_integer):
    """Get pool of solutions if exists."""
    if s.USE_SOLUTION_POOL and stats and "pool_sol_nr" in stats:
        sols = [solution]
        x_sols = [solution['x']]

        for i in range(1, stats["pool_sol_nr"]):
            x = ca.DM(stats['pool_solutions'][i])
            if not any_equal(x, x_sols, idx_x_integer):
                sols.append({"f": stats["pool_obj_val"][i], "x": x})
                x_sols.append(x)
        nlpdata.prev_solutions = sols
        nlpdata.solved_all = [
            success for i in sols
        ]
    else:
        nlpdata.prev_solutions = [solution]
        nlpdata.solved_all = [success]

    return nlpdata


def get_termination_condition(termination_type, problem: MinlpProblem, data: MinlpData, s: Settings):
    """
    Get termination condition.

    :param termination_type: String of the termination type (gradient, std or equality)
    :param problem: problem
    :param data: data
    :return: callable that returns true if the termination condition holds
    """
    def max_time(ret, s, stats):
        done = False
        if s.TIME_LIMIT_SOLVER_ONLY:
            done = (stats["t_solver_total"] >
                    s.TIME_LIMIT or toc() > s.TIME_LIMIT * 3)
        else:
            done = (toc() > s.TIME_LIMIT)

        if done:
            logging.info("Terminated - TIME LIMIT")
            return True
        return ret

    if termination_type == 'gradient':
        idx_x_integer = problem.idx_x_integer
        f_fn = ca.Function("f", [problem.x, problem.p], [
                           problem.f], {"jit": s.WITH_JIT})
        grad_f_fn = ca.Function("gradient_f_x", [problem.x, problem.p], [ca.gradient(problem.f, problem.x)],
                                {"jit": s.WITH_JIT})

        def func(stats: Stats, s: Settings, lb=None, ub=None, x_best=None, x_current=None):
            ret = to_0d(
                f_fn(x_current, data.p)
                + grad_f_fn(x_current, data.p)[idx_x_integer].T @ (
                    x_current[idx_x_integer] - x_best[idx_x_integer])
                - f_fn(x_best, data.p)
            ) >= 0
            if ret:
                logging.info("Terminated - gradient ok")
            return max_time(ret, s, stats)
    elif termination_type == 'equality':
        idx_x_integer = problem.idx_x_integer

        def func(stats: Stats, s: Settings, lb=None, ub=None, x_best=None, x_current=None):
            if isinstance(x_best, list):
                for x in x_best:
                    if np.allclose(x[idx_x_integer], x_current[idx_x_integer], equal_nan=False, atol=s.EPS):
                        logging.info(f"Terminated - all close within {s.EPS}")
                        return True
                return max_time(False, s, stats)
            else:
                ret = np.allclose(
                    x_best[idx_x_integer], x_current[idx_x_integer], equal_nan=False, atol=s.EPS)
                if ret:
                    logging.info(f"Terminated - all close within {s.EPS}")
                return max_time(ret, s, stats)

    elif termination_type == 'std':
        def func(stats: Stats, s: Settings, lb=None, ub=None, x_best=None, x_current=None):
            tol_abs = max((abs(lb) + abs(ub)) *
                          s.MINLP_TOLERANCE / 2, s.MINLP_TOLERANCE_ABS)
            ret = (lb + tol_abs - ub) >= 0
            if ret:
                logging.info(
                    f"Terminated: {lb} >= {ub} - {tol_abs} ({tol_abs})")
            else:
                logging.info(
                    f"Not Terminated: {lb} <= {ub} - {tol_abs} ({tol_abs})")
            return max_time(ret, s, stats)
    else:
        raise AttributeError(
            f"Invalid type of termination condition, you set '{termination_type}' but the only option is 'std'!")
    return func
