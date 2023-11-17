"""A generic simple solver."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Dict, List, Optional
from benders_exp.problems import MinlpProblem, MinlpData
from benders_exp.defines import OUT_DIR, WITH_DEBUG, EPS
from benders_exp.utils import toc
import casadi as ca
import numpy as np
import logging
from benders_exp.utils.data import save_pickle

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
            if k not in ["iterate_data"]:
                print(f"\t{k}: {v}")

    @staticmethod
    def create_iter_dict(iter_nr, best_iter, prev_feasible, ub, nlp_obj, last_benders, lb, x_sol):
        """Create dictionary with useful data of this iteration."""
        return {
            "iter_nr": iter_nr,
            "best_iter": best_iter,
            "prev_feasible": prev_feasible,
            "ub": ub,
            "nlp_obj": nlp_obj,
            "last_benders": last_benders,
            "lb": lb,
            "x_sol": x_sol,
            "time": toc()
        }

    def save(self, dest=None):
        """Save statistics."""
        if dest is None:
            dest = os.path.join(
                OUT_DIR, f'{self.datetime}_{self.mode}_{self.problem_name}.pkl')
        print(f"Saving to {dest}")
        save_pickle(self.data, dest)


class SolverClass(ABC):
    """Create solver class."""

    def __init___(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create a solver class."""
        self.stats = stats
        self.solver = None

    @abstractmethod
    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve the problem."""

    def collect_stats(self, algo_name, solver=None):
        """Collect statistics."""
        logger.info(f"Solved {algo_name}")
        if solver is None:
            stats = self.solver.stats()
        else:
            stats = solver.stats()

        self.stats[f"{algo_name}.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats[f"{algo_name}.time_wall"] += sum(
            [v for k, v in stats.items() if "t_wall" in k]
        )
        self.stats[f"{algo_name}.iter"] += max(
            stats.get("n_call_solver", 0), stats["iter_count"]
        )
        self.stats[f"{algo_name}.runs"] += 1
        return stats["success"], stats


def regularize_options(options, default):
    """Regularize options."""
    ret = {} if options is None else options.copy()

    if not WITH_DEBUG:
        ret.update({"verbose": False, "print_time": 0})

    ret.update(default)

    return ret


def inspect_problem(problem: MinlpProblem, data: MinlpData):
    """Inspect problem for linear and convex bounds."""
    x_cont_idx = [
        i for i in range(problem.x.shape[0]) if i not in problem.idx_x_bin
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
            # Assume for now...
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
