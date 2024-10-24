# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""A generic simple solver."""

from abc import ABC, abstractmethod
from typing import List, Optional
from camino.problems import MinlpProblem, MinlpData
from camino.settings import Settings
from camino.utils import colored
from camino.stats import Stats
import casadi as ca
import numpy as np
import logging

from camino.utils.conversion import to_0d

logger = logging.getLogger(__name__)


class SolverClass(ABC):
    """Create solver class."""

    def __init__(self, problem: MinlpProblem, stats: Stats, s: Settings):
        """Create a solver class."""
        self.settings = s
        self.stats = stats
        self.solver = None

    @abstractmethod
    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve the problem."""

    def collect_stats(self, algo_name, solver=None, sol=None):
        """Collect statistics."""
        logger.info(f"Solved {algo_name}")
        if solver is None:
            stats = self.solver.stats()
        else:
            stats = solver.stats()

        if "t_proc_my_solver" in stats:
            t_proc = stats["t_proc_my_solver"]
        elif "t_proc_total" in stats:
            t_proc = stats["t_proc_total"]
        else:
            logger.info(colored(f"t_proc_total not found for {algo_name}"))
            t_proc = sum(
                [v for k, v in stats.items() if "t_proc" in k]
            )

        if "t_wall_my_solver" in stats:
            t_wall = stats["t_proc_my_solver"]
        elif "t_wall_total" in stats:
            t_wall = stats["t_wall_total"]
        else:
            logger.info(
                colored(f"t_wall_total not found for {algo_name}", "red"))
            t_wall = sum(
                [v for k, v in stats.items() if "t_wall" in k]
            )
        self.stats[f"{algo_name}.time"] += t_proc
        self.stats[f"{algo_name}.time_wall"] += t_wall
        self.stats[f"{algo_name}.iter"] += max(
            stats.get("n_call_solver", 0), stats["iter_count"]
        )
        self.stats[f"{algo_name}.runs"] += 1
        self.stats["t_solver_total"] += max(t_wall, t_proc)
        self.stats["success"] = stats["success"]
        self.stats["iter_type"] = algo_name
        if sol is not None:
            self.stats["sol_x"] = to_0d(sol["x"])
            self.stats["sol_obj"] = float(sol["f"])
        if self.settings.WITH_LOG_DATA:
            self.stats.save()
        return stats["success"], stats


class MiSolverClass(SolverClass):

    def __init__(self, problem, data, stats, settings):
        """Create a generic MiSolverClass."""
        SolverClass.__init__(self, problem, stats, settings)
        self.stats['lb'] = -ca.inf
        self.stats['ub'] = ca.inf
        self.stats['iter_nr'] = 0
        self.stats['best_iter'] = -1
        self.best_solutions = []

    def update_best_solutions(self, data: MinlpData):
        """Update best solutions,"""
        if np.any(data.solved_all):
            for i, success in enumerate(data.solved_all):
                obj_val = float(data.prev_solutions[i]['f'])
                if success:
                    if obj_val + self.settings.EPS < self.stats['ub']:
                        logger.info(
                            f"Decreased UB from {self.stats['ub']} to {obj_val}")
                        self.stats['ub'] = obj_val
                        self.best_solutions.append(data.prev_solutions[i])
                        self.stats['best_iter'] = self.stats['iter_nr']
                    elif obj_val - self.settings.EPS < self.stats['ub']:
                        self.best_solutions.append(data.prev_solutions[i])

        if len(self.best_solutions) > 0:
            return self.best_solutions[-1]
        else:
            return None

    def get_best_solutions(self, data: MinlpData):
        """Get best solutions."""
        data.best_solutions = self.best_solutions
        if len(data.best_solutions) > 0:
            data.prev_solutions = [data.best_solutions[-1]]
            data.solved_all = [True]
        else:
            data.solved_all = [False] * len(data.solutions_all)
        return data

    @abstractmethod
    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve the problem."""

    @abstractmethod
    def reset(self, nlpdata: MinlpData):
        """Reset problem data."""

    @abstractmethod
    def warmstart(self, nlpdata: MinlpData):
        """Warmstart the algorithm."""


def regularize_options(options, default, s: Settings):
    """Regularize options."""
    ret = {} if options is None else options.copy()

    if not s.WITH_DEBUG:
        ret.update({"verbose": False, "print_time": 0})

    ret.update(default)

    return ret


def inspect_problem(problem: MinlpProblem, data: MinlpData):
    """Inspect problem for linear and convex bounds."""
    x_cont_idx = [
        i for i in range(problem.x.shape[0]) if i not in problem.idx_x_integer
    ]
    g_expr = ca.Function("g_func", [problem.x, problem.p], [problem.g])
    sp = np.array(g_expr.sparsity_jac(0, 0))

    g_lin = []
    g_lin_bin = []
    g_other = []
    g_conv = []
    for i in range(problem.g.shape[0]):
        hess = ca.hessian(problem.g[i], problem.x)[0]
        if hess.nnz() == 0:
            g_lin.append(i)
            if sum(sp[i, x_cont_idx]) == 0:
                g_lin_bin.append(i)
        elif not np.isinf(data.ubg[i]) and not np.isinf(data.lbg[i]):
            g_other.append(i)
        else:
            # NOTE: strong assumption! Works only for problems coming from the MINLPlib
            # A possible implementation is:
            # ccs = hess.sparsity().get_ccs()
            # hess[0, ccs[0]].is_numeric()
            g_conv.append(i)

    return g_lin, g_lin_bin, g_other, g_conv


def set_constraint_types(problem, g_lin, g_lin_bin, g_other, g_conv):
    """Set problem indices."""
    problem.idx_g_lin = g_lin
    problem.idx_g_lin_bin = g_lin_bin
    problem.idx_g_other = g_other
    problem.idx_g_conv = g_conv


def get_idx_linear_bounds_binary_x(problem: MinlpProblem):
    """Get the indices for the linear bounds that are purely on the binary x."""
    if problem.idx_g_lin_bin is None:
        x_cont_idx = [
            i for i in range(problem.x.shape[0]) if i not in problem.idx_x_integer
        ]
        g_expr = ca.Function("g_func", [problem.x, problem.p], [problem.g])
        sp = np.array(g_expr.sparsity_jac(0, 0))

        iterator = get_idx_linear_bounds(problem)
        problem.idx_g_lin_bin = np.array(list(
            filter(lambda i: (sum(sp[i, problem.idx_x_integer]) > 0
                              and sum(sp[i, x_cont_idx]) == 0),
                   iterator)
        ))

    return problem.idx_g_lin_bin


def get_lin_bounds(problem: MinlpProblem):
    get_idx_linear_bounds_binary_x(problem)
    return problem.idx_g_lin, problem.idx_g_lin_bin


def get_idx_linear_bounds(problem: MinlpProblem):
    """Get the indices of the linear bounds."""
    if problem.idx_g_lin is None:
        nr_g = problem.g.shape[0]
        problem.idx_g_lin = np.array(list(
            filter(lambda i: ca.hessian(problem.g[i], problem.x)[0].nnz() == 0,
                   range(nr_g)))
        )

    return problem.idx_g_lin


def get_idx_inverse(indices, nr):
    """Get an inver list of indices."""
    full_indices = np.arange(0, nr)
    return list(set(full_indices) - set(indices))


def extract_bounds(problem: MinlpProblem, data: MinlpData,
                   idx_g: List[int], new_x,
                   idx_x: Optional[List[int]] = None, allow_fail=True):
    """Extract bounds."""
    empty = False
    nr_g = len(idx_g)
    if nr_g == 0:
        empty = True
    else:
        try:
            if idx_x is None:
                _x = problem.x
                g = ca.Function("g_lin", [_x, problem.p], [
                                problem.g[idx_g]])(new_x, data.p)
            else:
                vec = []
                j = 0
                for i in range(problem.x.shape[0]):
                    if i in idx_x:
                        vec.append(new_x[j])
                        j += 1
                    else:
                        vec.append(0)
                vec = ca.vertcat(*vec)
                vec_fn = ca.Function("v", [new_x], [vec])
                g = ca.Function("g_lin", [problem.x, problem.p], [
                                problem.g[idx_g]])(vec_fn(new_x), data.p)

            lbg = data.lbg[idx_g].flatten().tolist()
            ubg = data.ubg[idx_g].flatten().tolist()
        except Exception as e:
            if allow_fail:
                logger.warning(str(e))
                empty = True
            else:
                raise e

    if empty:
        return 0, [], [], []
    else:
        return nr_g, g, lbg, ubg
