import numpy as np
from random import randint, random
from minlp_algorithms.data import MinlpData
from minlp_algorithms.utils import to_0d, logging
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


def create_rounded_data(data: MinlpData, idx_x_bin):
    for i in range(data.nr_sols):
        # Round the continuous solution
        x_var = to_0d(data.prev_solutions[i]['x'])
        x_var[idx_x_bin] = np.round(x_var[idx_x_bin])
        datarounded = deepcopy(data)
        datarounded.prev_solutions[i]['x'] = x_var
    return datarounded


def any_equal(x_current, x_best, idx_x_bin):
    for x in x_best:
        if np.allclose(x[idx_x_bin], x_current[idx_x_bin], equal_nan=False, atol=1e-2):
            logging.info("Terminated - all close within 1e-2")
            return True
    return False


def perturbe_x(x_current, idx_x_bin):
    """
    Perturbe x as described in:

        Bertacco, L., Fischetti, M., & Lodi, A. (2007). A feasibility pump
        heuristic for general mixed-integer problems. Discrete Optimization,
        4(1), 63-76.

    For TT integer variables with largest integer difference, round in the other
    direction if the fractional difference is large than 0.02.
    """
    N = len(idx_x_bin)
    T = N / 10  # TunedParameter
    TT = min(randint(T // 2, (T * 3) // 2), N)
    x_bin = x_current[idx_x_bin]
    x_rounded = np.round(x_bin)
    x_diff = np.abs(x_bin - x_rounded)
    idx_largest = np.argpartition(x_diff, -TT)[-TT:]

    for i in idx_largest:
        if x_rounded[i] < x_bin[i]:
            x_rounded[i] += 1
        else:
            x_rounded[i] -= 1

    x_current[idx_x_bin] = x_rounded
    return x_current


def randrange(xmin, xmax):
    """Rand range."""
    return random() * (xmax - xmin) + xmin


def random_perturbe_x(x_current, idx_x_bin):
    """
    Random perturbation of x as described in:

        Bertacco, L., Fischetti, M., & Lodi, A. (2007). A feasibility pump
        heuristic for general mixed-integer problems. Discrete Optimization,
        4(1), 63-76.
    """
    N = len(idx_x_bin)
    random_range = [-0.3, 0.7]
    x_bin = x_current[idx_x_bin]
    x_rounded = np.round(x_bin)
    x_diff = np.abs(x_bin - x_rounded)

    for i in range(N):
        if x_diff[i] + max(randrange(*random_range), 0) > 0.5:
            if x_rounded[i] < x_bin[i]:
                x_rounded[i] += 1
            else:
                x_rounded[i] -= 1

    x_current[idx_x_bin] = x_rounded
    return x_current
