import numpy as np
import casadi as ca
from benders_exp.defines import Settings


def almost_equal(a, b, EPS=1e-5):
    """Check if almost equal."""
    return a + EPS > b and a - EPS < b


class Constraints:
    """Store bounds."""

    def __init__(self, nr=0, eq=None, lb=None, ub=None):
        """Store bounds."""
        if nr == 0:
            eq, lb, ub = [], [], []

        self.nr = nr
        self.eq = eq
        self.lb = lb
        self.ub = ub
        self.a = None
        self.b = None

    def add(self, lb, eq, ub):
        """Add a bound."""
        if self.nr == 0:
            self.nr += 1
            self.eq = eq
            self.lb = ca.DM(lb)
            self.ub = ca.DM(ub)
        else:
            self.nr += 1
            self.eq = ca.vertcat(self.eq, eq)
            self.lb = ca.vertcat(self.lb, lb)
            self.ub = ca.vertcat(self.ub, ub)

    def to_generic(self):
        """Convert to a generic class."""
        return self

    def __add__(self, other):
        """Add two bounds."""
        other = other.to_generic()
        if other.nr == 0:
            return self
        if self.nr == 0:
            return other

        return Constraints(
            self.nr + other.nr,
            ca.vertcat(self.eq, other.eq),
            ca.vertcat(self.lb, other.lb),
            ca.vertcat(self.ub, other.ub)
        )

    def get_a(self, x, nr_x):
        """Get A matrix."""
        if self.a is None:
            self.a = ca.Function("geq", [x], [ca.jacobian(self.eq, x)])(
                np.zeros((nr_x,))
            )
        return self.a

    def get_b(self, x, nr_x):
        """Get B matrix."""
        if self.b is None:
            self.b = ca.Function("eq", [x], [self.eq])(np.zeros((nr_x,)))
        return self.b

    def __str__(self):
        """Represent."""
        out = f"Eq: {self.nr}\n\n"
        for i in range(self.nr):
            out += f"{self.lb[i]} <= {self.eq[i]} <= {self.ub[i]}\n"
        return out


def bin_equal(sol1, sol2, idx_x_bin):
    """Binary variables equal."""
    return np.allclose(sol1[idx_x_bin], sol2[idx_x_bin], equal_nan=False, atol=1e-2)


def any_equal(sol, refs, idx_x_bin):
    """Check if any is equal."""
    for ref in refs:
        if bin_equal(sol, ref, idx_x_bin):
            return True
    return False


def get_solutions_pool(nlpdata, success, stats, s: Settings, solution, idx_x_bin):
    """Get pool of solutions if exists."""
    if s.USE_SOLUTION_POOL and stats and "pool_sol_nr" in stats:
        sols = [solution]
        x_sols = [solution['x']]

        for i in range(1, stats["pool_sol_nr"]):
            x = ca.DM(stats['pool_solutions'][i])
            if not any_equal(x, x_sols, idx_x_bin):
                sols.append({"f": stats["pool_obj_val"][i], "x": x})
                x_sols.append(x)
        nlpdata.prev_solutions = sols
        nlpdata.solved_all = [
            success for i in sols
        ]
    else:
        nlpdata.prev_solutions = [solution]
        nlpdata.solved_all = [success]

    return nlpdata
