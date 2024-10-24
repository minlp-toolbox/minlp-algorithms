# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utilities for pump algorithms."""

import numpy as np
from random import randint, random
from camino.data import MinlpData
from camino.utils import logging
from camino.utils.conversion import to_0d
from copy import deepcopy

logger = logging.getLogger(__name__)


def integer_error(x_int, norm=1):
    """Compute integer error."""
    if norm == 1:
        ret = np.sum(np.abs(np.round(x_int) - x_int))
    else:
        ret = np.linalg.norm(np.round(x_int) - x_int)

    logger.info(f"Integer error {ret:.3f} / {x_int.shape[0]:.3f}")
    return ret


def create_rounded_data(data: MinlpData, idx_x_integer):
    for i in range(data.nr_sols):
        # Round the continuous solution
        x_var = to_0d(data.prev_solutions[i]["x"])
        x_var[idx_x_integer] = np.round(x_var[idx_x_integer])
        datarounded = deepcopy(data)
        datarounded.prev_solutions[i]["x"] = x_var
    return datarounded


def perturbe_x(x_current, idx_x_integer):
    """
    Perturbe x as described in:

        Bertacco, L., Fischetti, M., & Lodi, A. (2007). A feasibility pump heuristic for general mixed-integer
        problems. Discrete Optimization, 4(1), 63-76.

    For TT integer variables with largest integer difference, round in the other direction if the fractional difference
    is large than 0.02.
    """
    N = len(idx_x_integer)
    T = N / 10  # TunedParameter
    TT = min(randint(T // 2, (T * 3) // 2), N)
    x_bin = x_current[idx_x_integer]
    x_rounded = np.round(x_bin)
    x_diff = np.abs(x_bin - x_rounded)
    idx_largest = np.argpartition(x_diff, -TT)[-TT:]

    for i in idx_largest:
        if x_rounded[i] < x_bin[i]:
            x_rounded[i] += 1
        else:
            x_rounded[i] -= 1

    x_current[idx_x_integer] = x_rounded
    return x_current


def randrange(xmin, xmax):
    """Rand range."""
    return random() * (xmax - xmin) + xmin


def random_perturbe_x(x_current, idx_x_integer):
    """
    Random perturbation of x as described in:

        Bertacco, L., Fischetti, M., & Lodi, A. (2007). A feasibility pump heuristic for general mixed-integer problems.
        Discrete Optimization, 4(1), 63-76.
    """
    N = len(idx_x_integer)
    random_range = [-0.3, 0.7]
    x_bin = x_current[idx_x_integer]
    x_rounded = np.round(x_bin)
    x_diff = np.abs(x_bin - x_rounded)

    for i in range(N):
        if x_diff[i] + max(randrange(*random_range), 0) > 0.5:
            if x_rounded[i] < x_bin[i]:
                x_rounded[i] += 1
            else:
                x_rounded[i] -= 1

    x_current[idx_x_integer] = x_rounded
    return x_current
