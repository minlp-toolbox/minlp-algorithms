"""Test if all combinations work."""

import unittest
from parameterized import parameterized
from benders_exp.problems.overview import PROBLEMS
from benders_exp.quick_and_dirty import run_problem, Stats
from benders_exp.problems import check_solution

options = [
    (solver, problem)
    for solver in ["benders", "bendersqp", "bonmin"]
    for problem in PROBLEMS.keys()
    if (
        problem not in ["orig", "doubletank2"]
        and (problem, solver) not in [("doubletank", "bonmin")]
    )
]

obj_val = {
    "dummy": 8.41,
    "doublepipe": 1,
    "dummy2": 0,
    "doubletank": 18.6826,
    "sign_check": 9,
}

# Number of digits after the komma to be accurate:
obj_tolerance_default = 3
obj_tolerance = {
    "doubletank": 1
}


class TestSolver(unittest.TestCase):
    """Test."""

    @parameterized.expand(options)
    def test_solver(self, mode, problem_name):
        """Test runner."""
        stats = Stats({})
        problem, data, x_star = run_problem(mode, problem_name, stats)
        desired_obj = obj_val.get(problem_name, -1)
        desired_tol = obj_tolerance.get(problem.name, obj_tolerance_default)
        self.assertAlmostEqual(data.obj_val, desired_obj, desired_tol,
                               msg=f"Failed for {mode} & {problem_name}")
        check_solution(problem, data, x_star)


if __name__ == "__main__":
    unittest.main()
