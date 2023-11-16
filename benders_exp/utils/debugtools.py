import numpy as np
from benders_exp.solvers import MinlpProblem, MinlpData
from benders_exp.defines import EPS


class CheckNoDuplicate:
    """Check duplicates."""

    def __init__(self, problem: MinlpProblem):
        """Check if no duplicates passes through."""
        self.idx_x_bin = problem.idx_x_bin
        self.old = []

    def __call__(self, nlpdata: MinlpData):
        """Check if no old solutions pass through."""
        for sol in nlpdata.prev_solutions:
            new = sol['x'][self.idx_x_bin]
            for el in self.old:
                if np.allclose(el, new, equal_nan=False, atol=EPS):
                    print("Duplicate!")
                    breakpoint()
            self.old.append(new)
