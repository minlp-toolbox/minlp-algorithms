from minlp_algorithms.solver import MinlpSolver, MinlpProblem, MinlpData, Settings, Stats
from minlp_algorithms.settings import GlobalSettings

# An example
import casadi as ca
import numpy as np

x0 = np.array([0, 4, 100])
lbx = np.array([0, 0, 0])
ubx = np.array([4, 4, np.inf])
x = GlobalSettings.CASADI_VAR.sym("x", 3)
p = GlobalSettings.CASADI_VAR.sym("p", 2)
p_val = np.array([1000, 3])
f = (x[0] - 4.1)**2 + (x[1] - 4.0)**2 + x[2] * p[0]
ubg = np.array([ca.inf, ca.inf])
lbg = np.array([0, 0])
g = ca.vertcat(x[2], -(x[0]**2 + x[1]**2 - x[2] - p[1]**2))

problem = MinlpProblem(f, g, x, p, idx_x_bin=[0, 1])
data = MinlpData(p_val, x0, _lbg=lbg, _ubg=ubg, _lbx=lbx, _ubx=ubx)
settings = Settings()
stats = Stats("s-b-miqp", "example-problem")

solver = MinlpSolver('s-b-miqp', problem, data, stats, settings)
result = solver.solve(data)
solver.stats.print()

# A template to fill-in
problem = MinlpProblem()
data = MinlpData()
settings = Settings()
stats = Stats()

solver = MinlpSolver('s-b-miqp', problem, data, stats, settings)
result = solver.solve(data)
solver.stats.print()
