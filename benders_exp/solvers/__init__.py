"""A generic simple solver."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict
from benders_exp.problems import MinlpProblem, MinlpData
import casadi as ca
import numpy as np
from benders_exp.defines import CASADI_VAR


@dataclass
class Stats:
    """Collect stats."""

    data: Dict[str, float]

    def __getitem__(self, key):
        """Get attribute."""
        if key not in self.data:
            return 0
        return self.data[key]

    def __setitem__(self, key, value):
        """Set item."""
        self.data[key] = value

    def print(self):
        """Print statistics."""
        print("Statistics")
        for k, v in self.data.items():
            print(f"\t{k}: {v}")


class SolverClass(ABC):
    """Create solver class."""

    def __init___(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create a solver class."""
        self.stats = stats
        self.solver = None

    @abstractmethod
    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve the problem."""

    def collect_stats(self):
        """Collect statistics."""
        stats = self.solver.stats()
        return stats["success"], stats


def extract_linear_bounds_binary_x(problem: MinlpProblem, data: MinlpData):
    """Extract the linear bounds on g."""
    x_cont_idx = [
        i for i in range(problem.x.shape[0]) if i not in problem.idx_x_bin
    ]
    g_out = []
    lbg_out = []
    ubg_out = []
    g_expr = ca.Function("g_func", [problem.x, problem.p], [problem.g])
    sp = np.array(g_expr.sparsity_jac(0, 0))

    nr_g = problem.g.shape[0]
    for i in range(nr_g):
        if (sum(sp[i, problem.idx_x_bin]) > 0 and sum(sp[i, x_cont_idx]) == 0
                and ca.hessian(problem.g[i], problem.x)[0].nnz() == 0):
            g_out.append(problem.g[i])
            lbg_out.append(data.lbg[i])
            ubg_out.append(data.ubg[i])

    return g_out, lbg_out, ubg_out


def extract_linear_bounds(problem: MinlpProblem, data: MinlpData):
    """Extract the linear bounds on g."""
    g_out = []
    lbg_out = []
    ubg_out = []

    nr_g = problem.g.shape[0]
    for i in range(nr_g):
        if ca.hessian(problem.g[i], problem.x)[0].nnz() == 0:
            g_out.append(problem.g[i])
            lbg_out.append(data.lbg[i])
            ubg_out.append(data.ubg[i])

    return g_out, lbg_out, ubg_out
