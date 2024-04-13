import numpy as np
from minlp_algorithms.utils import to_0d
from minlp_algorithms.solvers.utils import Constraints
from minlp_algorithms.problems import MinlpData
import logging

logger = logging.getLogger(__name__)


def _dyn_prog_knapsack(bound, a, min_contrib, lb, ub, eps=1e-3):
    """
    Knapsack program to compute best bound.
    Works only when there are few x variables.

    a x <= bound

    Compute the maximum bound that can be achieved for these binary variables.
    Given min_contrib, the minimum contribution in total, lb and ub
    the bounds for x.

    E.g. 2.1 * x1 + 2 * x2 <= 7.5 with x1, x2 >= 0 and x1 <= 1
         2.1 + 2 * 2 = 6.1 <= 7.5
         So the tightest bound is 6.1

    Only works for small problems...
    """
    remainder = 0
    new_bound = min_contrib
    sol = [min_contrib]
    nr_i = lb.shape[0]
    if nr_i > 10:
        return bound
    for i in range(nr_i):
        if abs(a[i]) < eps:
            remainder += (ub[i] - lb[i]) * abs(a[i])
        else:
            for xi in range(int(ub[i] - lb[i]+1)):
                diff = xi * abs(a[i])
                len_sol = len(sol)
                for other in sol[:len_sol]:
                    new_val = other + diff
                    if new_val <= bound:
                        if new_val > new_bound:
                            new_bound = new_val
                        if new_val not in sol:
                            sol.append(new_val)
                    else:
                        break
        sol.sort()
    if remainder > new_bound:
        return bound  # Can not be tightened using few steps
    else:
        return new_bound


def tighten_bounds_x(data: MinlpData, constraints: Constraints, idx_x_bin, x, nr_x):
    """Tighten bounds on x."""
    reduced = 0
    a = constraints.get_a(x, nr_x).full()
    b = - to_0d(constraints.get_b(x, nr_x).full()) + constraints.ub

    lbx = to_0d(data.lbx)
    lbx[np.isneginf(lbx)] = -1e16
    ubx = to_0d(data.ubx)
    ubx[np.isposinf(ubx)] = 1e16

    for j in range(a.shape[0]):
        # Find max b:
        lba = lbx * a[j, :]
        uba = ubx * a[j, :]
        min_eq = np.sum(lba * (lba < uba)) + np.sum(uba * (uba < lba))
        diff = abs(uba - lba)
        for i in idx_x_bin:
            if min_eq + diff[i] > b[j]:
                reduced += 1
                if a[j, i] < 0:
                    lbx[i] -= np.floor(
                        ((min_eq + diff[i]) - b[j]) / a[j, i]
                    )
                    lba[i] = lbx[i] * a[j, i]
                else:
                    ubx[i] -= np.ceil(
                        ((min_eq + diff[i]) - b[j]) / a[j, i]
                    )
                    uba[i] = ubx[i] * a[j, i]

    if reduced > 0:
        data._lbx = lbx
        data._ubx = ubx
        logger.info(f"Reduced {reduced} bounds")

    return reduced
