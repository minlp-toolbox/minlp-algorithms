# This file is part of minlp-algorithms
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

from minlp_algorithms.solver import MinlpSolver, MinlpProblem, MinlpData, Settings, Stats


# A template to fill-in
problem = MinlpProblem(...)
data = MinlpData(...)
settings = Settings(...)
stats = Stats(...)

solver = MinlpSolver('s-b-miqp', problem, data, stats, settings)
result = solver.solve(data)
solver.stats.print()
