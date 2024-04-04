"""Test if all combinations work."""

import unittest
from parameterized import parameterized
from benders_exp.problems.overview import PROBLEMS
from benders_exp.quick_and_dirty import run_problem, Stats, SOLVER_MODES
from benders_exp.problems import check_solution
from benders_exp.defines import Settings
import timeout_decorator

options = [("cia", "doubletank2")] + [
    (solver, "dummy")
    for solver in SOLVER_MODES.keys()
    if solver not in (
        "cia", "milp_tr", "ampl", "benderseq"  # Almost all solvers
    )
] + [
    (solver, problem)
    for solver in
    [
        "benders_trmi",
    ]
    for problem in PROBLEMS.keys()
    if problem not in [
        # Exclude duplicates
        "dummy",
        # Exclude difficult problems with long runtimes
        "alan", "orig", "stcs", "to_car", "doubletank2",
        # Exclude some errors:
        "unstable_ocp",
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
    "relaxed": -3,
    # Heuristics
    "rofp": -3,
    "ofp": -3,
    "fp": -3,
    # Disable:
    "cia": -100,
    "test": -100,
    "bonmin-qg": -100,
    "bonmin-hyb": -100,
}
sol_not_valid = ["test", "relaxed", "cia"]


class TestSolver(unittest.TestCase):
    """Test."""

    @parameterized.expand(options)
    @timeout_decorator.timeout(20.0)
    def test_solver(self, mode, problem_name):
        """Test runner."""
        s = Settings(TIME_LIMIT=10.0)
        s.MIP_SETTINGS_ALL["gurobi"]["gurobi.TimeLimit"] = 5.0
        stats = Stats(mode, problem_name, "test", {})
        (problem, data, x_star), s = run_problem(mode, problem_name, stats, [], s)
        desired_obj = obj_val.get(problem_name, -1)
        desired_tol = obj_tolerance.get(problem_name, obj_tolerance_default)
        desired_tol += obj_tolerance_heuristic.get(mode, 0)
        if desired_tol > -10:
            self.assertAlmostEqual(data.obj_val / desired_obj, 1, desired_tol,
                                   msg=f"Failed for {mode} & {problem_name}")
        if mode not in sol_not_valid:
            check_solution(problem, data, x_star, s)


if __name__ == "__main__":
    unittest.main()
