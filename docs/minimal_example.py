# This file is part of minlp-algorithms
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

from minlp_algorithms.solver import MinlpSolver, MinlpProblem, MinlpData, Settings, Stats
from minlp_algorithms.settings import GlobalSettings

# A minimal example to show how effortless is to call the algorithms in this toolbox if you already have a CasADi description.
# Illustrations of this minimal example are available in the paper (Section 2.7) https://arxiv.org/pdf/2404.11786

import casadi as ca
import numpy as np

x0 = np.array([0, 4, 100])
lbx = np.array([0, 0, 0])
ubx = np.array([4, 4, np.inf])
# The first two variables are integer, the last one is continuous
x = ca.SX.sym("x", 3)
p = ca.SX.sym("p", 2)
p_val = np.array([1000, 3])
f = (x[0] - 4.1)**2 + (x[1] - 4.0)**2 + x[2] * p[0]
ubg = np.array([ca.inf, ca.inf])
lbg = np.array([0, 0])
g = ca.vertcat(x[2], -(x[0]**2 + x[1]**2 - x[2] - p[1]**2))

# ------------------- Solving MINLP within CasADi -------------------
# To solve the above MINLP in CasADi, you can use the solver Bonmin interfaced via `nlpsol` as follows.

# The way to declare integer variables for CasADi nlpsol
is_integer = [1, 1, 0]
myminlp = ca.nlpsol("myminlp", "bonmin", {
                    "f": f, "g": g, "x": x, "p": p},
                    {"discrete": is_integer, "bonmin.algorithm": "B-OA"})
solution = myminlp(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val)

print(f"Bonmin solution: x={solution['x']}, objective value={solution['f']}")
# NOTE: We used Bonmin outer approximation (B-OA) because this problem cannot be solved
# with the default Bonmin nonlinear branch-and-bound solver.

# ------------------- Solving MINLP within CAMINO -------------------
# The following is required to use the MINLP algorithms implemented in our package.
# Notice that we require the exact information needed for CasADi nlpsol.

problem = MinlpProblem(f, g, x, p, idx_x_integer=is_integer)
data = MinlpData(p_val, x0, _lbg=lbg, _ubg=ubg, _lbx=lbx, _ubx=ubx)
settings = Settings()
settings.MIP_SOLVER = 'highs'  # gurobi
stats = Stats("oa", "example-problem")

solver = MinlpSolver('oa', problem, data, stats, settings)
result = solver.solve(data)
solver.stats.print()
