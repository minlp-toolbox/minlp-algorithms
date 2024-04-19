"""Utilities for generating plots."""

import numpy as np
from minlp_algorithms.data import MinlpData
from minlp_algorithms.problem import MinlpProblem
from minlp_algorithms.utils.conversion import to_0d


def get_control_vector(problem: MinlpProblem, data: MinlpData):
    out = []
    if problem.meta.n_continuous_control > 0:
        out.append(to_0d(data.x_sol)[problem.meta.idx_control].reshape(-1, problem.meta.n_continuous_control))
    if problem.meta.n_discrete_control > 0:
        out.append(to_0d(data.x_sol)[problem.meta.idx_bin_control].reshape(-1, problem.meta.n_discrete_control))
    if len(out) > 1:
        out = np.hstack(out)
    return out