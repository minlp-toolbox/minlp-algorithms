"""Overview of all problems."""

from benders_exp.problems import MinlpProblem, CASADI_VAR, MinlpData
import casadi as ca
import numpy as np
from benders_exp.solarsys import extract as extract_solarsys


def create_dummy_problem(p_val=[1000, 3]):
    """
    Create a dummy problem.

    This problem corresponds to the tutorial example in the GN-Voronoi paper.
    """
    x = CASADI_VAR.sym("x", 3)
    x0 = np.array([0, 4, 100])
    idx_x_bin = [0, 1]
    p = CASADI_VAR.sym("p", 2)
    f = (x[0] - 4.1)**2 + (x[1] - 4.0)**2 + x[2] * p[0]
    g = ca.vertcat(
        x[2],
        -(x[0]**2 + x[1]**2 - x[2] - p[1]**2)
    )
    ubg = np.array([ca.inf, ca.inf])
    lbg = np.array([0, 0])
    lbx = -1e3 * np.ones((3,))
    ubx = 1e3 * np.ones((3,))
    # ubx = np.array([ca.inf, ca.inf, ca.inf])

    problem = MinlpProblem(x=x, f=f, g=g, p=p, idx_x_bin=idx_x_bin)
    data = MinlpData(x0=x0, _ubx=ubx, _lbx=lbx,
                     _ubg=ubg, _lbg=lbg, p=p_val, solved=True)
    return problem, data


def create_dummy_problem_2():
    """Create a dummy problem."""
    x = CASADI_VAR.sym("x", 2)
    x0 = np.array([0, 4])
    idx_x_bin = [0]
    p = CASADI_VAR.sym("p", 1)
    f = x[0]**2 + x[1]
    g = ca.vertcat(
        x[1],
        -(x[0]**2 + x[1] - p[0]**2)
    )
    ubg = np.array([ca.inf, ca.inf])
    lbg = np.array([0, 0])
    lbx = -1e3 * np.ones((2,))
    ubx = np.array([ca.inf, ca.inf])

    problem = MinlpProblem(x=x, f=f, g=g, p=p, idx_x_bin=idx_x_bin)
    data = MinlpData(x0=x0, _ubx=ubx, _lbx=lbx,
                     _ubg=ubg, _lbg=lbg, p=[3], solved=True)
    return problem, data


PROBLEMS = {
    "dummy": create_dummy_problem,
    "dummy2": create_dummy_problem_2,
    "orig": extract_solarsys
}
