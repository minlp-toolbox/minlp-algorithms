# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test if all combinations work."""

import unittest
from parameterized import parameterized
from camino.problems.overview import PROBLEMS
from camino.solver import SOLVER_MODES
from camino.settings import Settings
from camino.runner import runner
import timeout_decorator

options = [("cia", "doubletank2")] + [
    (solver, "dummy")
    for solver in SOLVER_MODES.keys()
    if solver not in (
        "cia", "s-tr-milp", "ampl",  # Almost all solvers
    )
] + [
    (solver, problem)
    for solver in
    [
        "s-b-miqp",
    ]
    for problem in PROBLEMS.keys()
    if problem not in [
        # Exclude duplicates
        "dummy2",
        # Exclude difficult problems with long runtimes
        "alan", "stcs", "to_car", "doubletank2", "doubletank",
        # Exclude some errors:
        "unstable_ocp", "particle", "from_nlpsol_dsc",
        # Interfaces:
        "nl_file", "from_sto", "nosnoc"
    ]
]

obj_val = {
    "dummy": 8.41,
    "doublepipe": 1,
    "dummy2": -3,
    "doubletank": 18.6826,
    "sign_check": 9,
    "nonconvex": 0.0567471,
    "gearbox": 6550.833,
    "particle": 1.3797,
    "unstable_ocp": -0.05129210,
    "gearbox_complx": 234.027,
    "gearbox_int": 14408.4375,
}

# Number of digits after the komma to be accurate:
obj_tolerance_default = 3
obj_tolerance = {
    "dummy": 2,
    "doubletank": 1,
    "nonconvex": 0,
    "particle": 2,
}
obj_tolerance_heuristic = {
    "nlp": -3,
    # Heuristics
    "rofp": -3,
    "ofp": -3,
    "fp": -3,
    # Disable:
    "cia": -100,
    "bonmin-qg": -100,
    "bonmin-hyb": -100,
    "nlp-fxd": -100,
}
sol_not_valid = ["nlp", "cia"]


class TestSolver(unittest.TestCase):
    """Test."""

    @parameterized.expand(options)
    @timeout_decorator.timeout(20.0)
    def test_solver(self, mode, problem_name):
        """Test runner."""
        s = Settings(TIME_LIMIT=10.0)
        s.MIP_SETTINGS_ALL["gurobi"]["gurobi.TimeLimit"] = 5.0
        stats, data = runner(mode, problem_name, None, None)
        desired_obj = obj_val.get(problem_name, -1)
        desired_tol = obj_tolerance.get(problem_name, obj_tolerance_default)
        desired_tol += obj_tolerance_heuristic.get(mode, 0)
        if desired_tol > -10:
            self.assertAlmostEqual(data.obj_val / desired_obj, 1, desired_tol,
                                   msg=f"Failed for {mode} & {problem_name}")


if __name__ == "__main__":
    unittest.main()
