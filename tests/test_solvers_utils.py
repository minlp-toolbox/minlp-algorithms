"""Test solver utilities."""

import unittest
import casadi as ca
from benders_exp.defines import CASADI_VAR
from benders_exp.solvers import MinlpProblem, get_idx_linear_bounds_binary_x, \
    get_idx_linear_bounds


class TestSolverUtils(unittest.TestCase):
    """Test utilities for the solvers."""

    def test_get_idx_linear_bounds(self):
        """Get indices for linear bounds."""
        x = CASADI_VAR.sym("x", 4)
        f = x[0] + x[1] + x[2] + x[3]
        g = [
            x[0] + x[2],  # Linear on binary variables
            x[0]**2 + x[2],  # Non-linear on binary
            x[1]**2 + x[3],  # Non-linear
            x[0] - x[2],  # Linear on binary
            x[1] + x[3],  # Linear
        ]
        p = CASADI_VAR.sym("p", 0)

        problem = MinlpProblem(
            f=f, g=ca.vcat(g), x=x, p=p, idx_x_bin=[0, 2]
        )
        idx_lin_bin = get_idx_linear_bounds_binary_x(problem)
        self.assertEqual(idx_lin_bin.tolist(), [0, 3])
        idx_lin = get_idx_linear_bounds(problem)
        self.assertEqual(idx_lin.tolist(), [0, 3, 4])


if __name__ == "__main__":
    unittest.main()