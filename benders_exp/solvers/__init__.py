"""A generic simple solver."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List
from benders_exp.problems import MinlpProblem, MinlpData
from benders_exp.defines import WITH_LOGGING
import casadi as ca
import numpy as np


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
        for k, v in sorted(self.data.items()):
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

    def collect_stats(self, algo_name):
        """Collect statistics."""
        stats = self.solver.stats()

        self.stats[f"{algo_name}.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats[f"{algo_name}.iter"] += max(
            stats.get("n_call_solver", 0), stats["iter_count"]
        )
        return stats["success"], stats


def regularize_options(options, log_opts, nolog_opts):
    """Regularize options."""
    if options is None:
        if WITH_LOGGING:
            return log_opts
        else:
            nolog_opts.update({"verbose": False, "print_time": 0})
            return nolog_opts
    else:
        return options.copy()


def get_idx_linear_bounds_binary_x(problem: MinlpProblem):
    """Get the indices for the linear bounds that are purely on the binary x."""
    x_cont_idx = [
        i for i in range(problem.x.shape[0]) if i not in problem.idx_x_bin
    ]
    g_expr = ca.Function("g_func", [problem.x, problem.p], [problem.g])
    sp = np.array(g_expr.sparsity_jac(0, 0))

    nr_g = problem.g.shape[0]
    return np.array(list(
        filter(lambda i: (sum(sp[i, problem.idx_x_bin]) > 0
                          and sum(sp[i, x_cont_idx]) == 0
                          and ca.hessian(problem.g[i], problem.x)[0].nnz() == 0),
               range(nr_g))
    ))


def get_idx_linear_bounds(problem: MinlpProblem):
    """Get the indices of the linear bounds."""
    nr_g = problem.g.shape[0]
    return np.array(list(
        filter(lambda i: ca.hessian(problem.g[i], problem.x)[0].nnz() == 0,
               range(nr_g)))
    )


def get_idx_inverse(indices, nr):
    """Get an inver list of indices."""
    full_indices = np.arange(0, nr)
    return list(set(full_indices) - set(indices))


def extract_bounds(problem: MinlpProblem, data: MinlpData, idx: List[int]):
    """Extract bounds."""
    return problem.g[idx], data.lbg[idx], data.ubg[idx]
