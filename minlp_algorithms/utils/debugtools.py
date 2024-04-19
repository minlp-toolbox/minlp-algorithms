import numpy as np
from minlp_algorithms.solvers import MinlpProblem, MinlpData
from minlp_algorithms.settings import Settings


class CheckNoDuplicate:
    """Check duplicates."""

    def __init__(self, problem: MinlpProblem, s: Settings):
        """Check if no duplicates pass through."""
        self.idx_x_bin = problem.idx_x_bin
        self.s = s
        self.old = []
        self.count = 0

    def __call__(self, nlpdata: MinlpData):
        """Check if no old solutions pass through."""
        if nlpdata.prev_solutions is None:
            return
        for sol in nlpdata.prev_solutions:
            new = sol['x'][self.idx_x_bin]
            for el in self.old:
                if np.allclose(el, new, equal_nan=False, atol=self.s.EPS):
                    print("Duplicate!")
                    self.count += 1
            self.old.append(new)

        if self.count > 10:
            raise Exception()
