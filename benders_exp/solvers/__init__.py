"""A generic simple solver."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Dict, List, Optional
from benders_exp.problems import MinlpProblem, MinlpData
from benders_exp.defines import _DATA_FOLDER, WITH_DEBUG
import casadi as ca
import numpy as np
import logging
import pickle

logger = logging.getLogger(__name__)


@dataclass
class Stats:
    """Collect stats."""

    mode: str
    problem_name: str
    datetime: str
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

    @staticmethod
    def create_iter_dict(iter_nr, best_iter, prev_feasible, ub, nlp_obj, last_benders, lb, x_sol):
        return {"iter_nr": iter_nr,
                "best_iter": best_iter,
                "prev_feasible": prev_feasible,
                "ub": ub,
                "nlp_obj": nlp_obj,
                "last_benders": last_benders,
                "lb": lb,
                "x_sol": x_sol}

    def save(self, x_star):
        with open(os.path.join(_DATA_FOLDER, f'{self.datetime}_{self.mode}_{self.problem_name}.pkl'), 'wb') as handle:
            pickle.dump(self.data, handle)


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


def regularize_options(options, default):
    """Regularize options."""
    ret = {} if options is None else options.copy()

    if WITH_DEBUG:
        ret.update({"verbose": False, "print_time": 0})

    ret.update(default)

    return ret


def get_idx_linear_bounds_binary_x(problem: MinlpProblem):
    """Get the indices for the linear bounds that are purely on the binary x."""
    if problem.idx_g_lin_bin is None:
        x_cont_idx = [
            i for i in range(problem.x.shape[0]) if i not in problem.idx_x_bin
        ]
        g_expr = ca.Function("g_func", [problem.x, problem.p], [problem.g])
        sp = np.array(g_expr.sparsity_jac(0, 0))

        iterator = get_idx_linear_bounds(problem)
        problem.idx_g_lin_bin = np.array(list(
            filter(lambda i: (sum(sp[i, problem.idx_x_bin]) > 0
                              and sum(sp[i, x_cont_idx]) == 0),
                   iterator)
        ))

    return problem.idx_g_lin_bin


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
                g = ca.Function("g_lin", [_x, problem.p], [problem.g[idx_g]])(new_x, data.p)
            else:
                vec = []
                j=0
                for i in range(problem.x.shape[0]):
                    if i in idx_x:
                        vec.append(new_x[j])
                        j += 1
                    else:
                        vec.append(0)
                vec = ca.vertcat(*vec)
                vec_fn = ca.Function("v", [new_x], [vec])
                g = ca.Function("g_lin", [problem.x, problem.p], [problem.g[idx_g]])(vec_fn(new_x), data.p)

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
