# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Validation tools.

Several validation tools to check if a solution is feasible and valid.
"""

from camino.problem import MinlpProblem
from camino.data import MinlpData
import casadi as ca
from camino.settings import Settings
import numpy as np
from camino.utils.conversion import to_float


def check_solution(problem: MinlpProblem, data: MinlpData, x_star, s: Settings, throws=True, check_objval=True):
    """Check a solution."""
    f = ca.Function("f", [problem.x, problem.p], [problem.f])
    g = ca.Function("g", [problem.x, problem.p], [problem.g])
    f_val = to_float(f(x_star, data.p).full())
    g_val = g(x_star, data.p).full().squeeze()
    lbg, ubg = data.lbg.squeeze(), data.ubg.squeeze()
    print(f"Objective value {f_val} (real) vs {data.obj_val}")
    msg = []
    if check_objval and abs(to_float(data.obj_val) - f_val) > s.OBJECTIVE_TOL:
        msg.append("Objective value wrong!")
    if np.any(data.lbx > x_star + s.CONSTRAINT_TOL):
        msg.append(f"Lbx > x* for indices:\n{np.nonzero(data.lbx > x_star).T}")
    if np.any(data.ubx < x_star - s.CONSTRAINT_TOL):
        msg.append(f"Ubx > x* for indices:\n{np.nonzero(data.ubx < x_star).T}")
    if np.any(lbg > g_val + s.CONSTRAINT_TOL):
        msg.append(f"{g_val=}  {lbg=}")
        msg.append("Lbg > g(x*,p) for indices:\n"
                   f"{np.nonzero(lbg > g_val)}")
    if np.any(ubg < g_val - s.CONSTRAINT_TOL):
        msg.append(f"{g_val=}  {ubg=}")
        msg.append("Ubg < g(x*,p) for indices:\n"
                   f"{np.nonzero(ubg < g_val)}")

    check_integer_feasible(problem.idx_x_integer, x_star, s, throws=throws)
    if msg:
        msg = "\n".join(msg)
        if throws:
            raise Exception(msg)
        else:
            print(msg)


def check_integer_feasible(idx_x_integer, x_star, s: Settings, throws=True):
    """Check if the solution is integer feasible."""
    x_bin = np.array(x_star)[idx_x_integer].squeeze()
    x_bin_rounded = np.round(x_bin)
    if np.any(np.abs(x_bin_rounded - x_bin) > s.CONSTRAINT_TOL):
        idx = np.nonzero(np.abs(x_bin_rounded - x_bin) > s.CONSTRAINT_INT_TOL)
        msg = f"Integer infeasible: {x_bin[idx]} {idx}"
        if throws:
            raise Exception(msg)
        else:
            print(throws)
