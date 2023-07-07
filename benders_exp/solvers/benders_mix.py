"""A mix of solvers."""

from copy import deepcopy
import numpy as np
import casadi as ca
from benders_exp.solvers import Stats, MinlpProblem, MinlpData, \
    get_idx_linear_bounds, get_idx_inverse, extract_bounds
from benders_exp.defines import WITH_JIT, CASADI_VAR, EPS
from benders_exp.solvers.benders import BendersMasterMILP
try:
    from colored import fg, stylize

    def colored(text, color="red"):
        """Color a text."""
        print(stylize(text, fg(color)))
except Exception:
    def colored(text, color=None):
        """Color a text."""
        print(text)


class EqBounds:
    """Store bounds."""

    def __init__(self, nr, eq, lb, ub):
        """Store bounds."""
        self.nr = nr
        self.eq = eq
        self.lb = ca.DM(lb)
        self.ub = ca.DM(ub)

    def add(self, lb, eq, ub):
        """Add a bound."""
        self.nr += 1
        self.eq = ca.vertcat(self.eq, eq)
        self.lb = ca.vertcat(self.lb, lb)
        self.ub = ca.vertcat(self.ub, ub)

    def __add__(self, other):
        """Add two bounds."""
        return EqBounds(
            self.nr + other.nr,
            ca.vertcat(self.eq, other.eq),
            ca.vertcat(self.lb, other.lb),
            ca.vertcat(self.ub, other.ub)
        )

    def __str__(self):
        """Represent."""
        out = f"Eq: {self.nr}nn"
        for i in range(self.nr):
            out += f"{self.lb[i]} <= {self.eq[i]} <= {self.ub[i]}\n"
        return out


def almost_equal(a, b):
    """Check if almost equal."""
    return a + EPS > b and a - EPS < b


class BendersTRandMaster(BendersMasterMILP):
    """
    Create benders constraint and benders master.

    By an idea of Moritz D. and Andrea G.
    Given the ordered sequence of integer solutions:
        Y := {y1, y2, ..., yN}
    such that J(y1) >= J(y2) >= ... >= J(yN) we define the
    benders polyhedral B := {y in R^n_y:
        J(y_i) + Nabla J(y_i)^T (y - y_i) <= J(y_N),
        forall i = 1,...,N-1
    }

    This MILP solves:
        min F(y, z | y_bar, z_bar)
        s.t ub >= H_L(y,z| y_bar, z_bar) >= lb
        with y in B

    For this implementation, since the original formulation implements:
        J(y_i) + Nabla J(yi) T (y - yi) <= nu,
        meaning: nu == J(y_N)
    """

    def __init__(self, problem: MinlpProblem, data: MinlpData, stats: Stats, options=None):
        """Create the benders constraint MILP."""
        super(BendersTRandMaster, self).__init__(
            problem, data, stats, options)
        # Settings
        self.nonconvex_strategy = "distance-based"
        self.nonconvex_strategy_alpha = 0.2
        self.trust_region_feasibility_strategy = "distance-correction"

        # Setups
        self.setup_common(problem, options)
        self.idx_g_lin = get_idx_linear_bounds(problem)
        self.idx_g_nonlin = get_idx_inverse(self.idx_g_lin, problem.g.shape[0])

        self.grad_f_x_sub = ca.Function(
            "gradient_f_x",
            [problem.x, problem.p], [ca.gradient(
                problem.f, problem.x
            )],
            {"jit": WITH_JIT}
        )
        self.jac_g = ca.Function(
            "jac_g", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)],
            {"jit": WITH_JIT}
        )
        self.f_hess = ca.Function("hess_f_x", [problem.x, problem.p], [
                                  ca.hessian(problem.f, problem.x)[0]])

        self._x = CASADI_VAR.sym("x_benders", problem.x.numel())
        self._x_bin = self._x[problem.idx_x_bin]
        self.g_lin = EqBounds(*extract_bounds(
            problem, data, self.idx_g_lin, self._x, allow_fail=False
        ))
        self.g_benders = EqBounds(0, [], [], [])
        self.g_benders_inf = EqBounds(0, [], [], [])
        self.g_nonconvex_cut = EqBounds(0, [], [], [])

        self.options.update({"discrete": [
                            1 if elm in problem.idx_x_bin else 0 for elm in range(self._x.shape[0])]})
        self.options_master = self.options.copy()
        self.options_master["discrete"] = self.options["discrete"] + [0]

        self.y_N_val = 1e15  # Should be inf but can not at the moment ca.inf
        self.x_valid = False
        # We take a point
        self.x_sol_best = data.x0
        self.qp_makes_progression = True

    def _get_g_linearized_nonlin(self, x, dx, nlpdata):
        g_lin = self.g(x, nlpdata.p)[self.idx_g_nonlin]
        if g_lin.numel() > 0:
            jac_g = self.jac_g(x, nlpdata.p)[self.idx_g_nonlin, :]

            return EqBounds(
                g_lin.numel(),
                g_lin + jac_g @ dx,
                nlpdata.lbg[self.idx_g_nonlin],
                nlpdata.ubg[self.idx_g_nonlin],
            )
        else:
            return EqBounds(0, [], [], [])

    def _solve_trust_region(self, nlpdata: MinlpData, is_qp=True) -> MinlpData:
        """Solve QP problem."""
        dx = self._x - self.x_sol_best

        f_k = self.f(self.x_sol_best, nlpdata.p)
        f_lin = self.grad_f_x_sub(self.x_sol_best, nlpdata.p)
        if is_qp:
            f_hess = self.f_hess(self.x_sol_best, nlpdata.p)
            f = f_k + f_lin.T @ dx + 0.5 * dx.T @ f_hess @ dx
        else:
            f = f_k + f_lin.T @ dx

        g_cur_lin = self._get_g_linearized_nonlin(self.x_sol_best, dx, nlpdata)
        g_total = (
            g_cur_lin + self.g_lin + self.g_benders
            + self.g_benders_inf + self.g_nonconvex_cut
        )
        g, ubg, lbg = g_total.eq, g_total.ub, g_total.lb

        self.solver = ca.qpsol(f"benders_constraint_{self.g_benders.nr}", "gurobi", {
                "f": f, "g": g, "x": self._x, "p": self._nu
            }, self.options)

        colored("NORMAL ITERATION", "blue")
        return self.solver(
            x0=self.x_sol_best,
            lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=lbg, ubg=ubg,
            p=[self.y_N_val]
        )

    def _solve_benders(self, nlpdata: MinlpData) -> MinlpData:
        """Solve benders master problem with one OA constraint."""
        dx = self._x - self.x_sol_best

        f_k = self.f(self.x_sol_best, nlpdata.p)
        f_lin = self.grad_f_x_sub(self.x_sol_best, nlpdata.p)
        f = f_k + f_lin.T @ dx

        # Adding the following linearization might not be the best idea since
        # They can lead to false results!
        # g_cur_lin = self._get_g_linearized_nonlin(self.x_sol_best, dx, nlpdata)

        g_total = (
            self.g_lin + self.g_benders + self.g_benders_inf
        )
        # Add extra constraint (one step OA):
        g_total.add(-ca.inf, f - self._nu, 0)
        g, ubg, lbg = g_total.eq, g_total.ub, g_total.lb

        self.solver = ca.qpsol(f"benders_with_{self.g_benders.nr}_cut", "gurobi", {
            "f": self._nu, "g": g,
            "x": ca.vertcat(self._x, self._nu),
        }, self.options_master)

        solution = self.solver(
            x0=ca.vertcat(nlpdata.x_sol[:self.nr_x_orig], nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx, -1e5),
            ubx=ca.vertcat(nlpdata.ubx, ca.inf),
            lbg=lbg, ubg=ubg
        )

        solution['x'] = solution['x'][:-1]
        colored("SOLVED BENDERS")
        return solution

    def _check_cut_valid(self, g_k, x_sol, x_sol_obj):
        """Check if the cut is valid."""
        g = ca.Function("g", [self._x, self._nu], [g_k])
        value = g(x_sol, 0)
        print(f"Cut valid (lower bound)?: {value} vs real {x_sol_obj}")
        return (value - EPS <= x_sol_obj)

    def _create_nonconvex_cut(self, nlpdata: MinlpData):
        """Create nonconvex cut."""
        if self.nonconvex_strategy == "distance-based":
            x_sol = nlpdata.x_sol[self.idx_x_bin]
            x_sol_best = self.x_sol_best[self.idx_x_bin]
            lambda_k = -nlpdata.lam_x_sol[self.idx_x_bin]

            f_prev = self.f(nlpdata.x_sol[:self.nr_x_orig], nlpdata.p)
            f_best = self.f(self.x_sol_best, nlpdata.p)
            print(f"Nonconvex cut direction from new {f_prev} to {f_best}")
            g_k = lambda_k.T @ (self._x_bin - x_sol)
            g_min = self.nonconvex_strategy_alpha * lambda_k.T @ (x_sol_best - x_sol)
            return g_min, g_k, ca.inf
        elif self.nonconvex_strategy == "gradient-based":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def _update_tr_bound(self, x_sol, obj_val):
        """Update bound."""
        self.x_sol_best = x_sol
        self.x_valid = True
        self.qp_makes_progression = True
        # Update trust region bounds
        self.y_N_val = obj_val
        # Make best point feasible
        if self.trust_region_feasibility_strategy == "distance-correction":
            for i in range(self.g_benders.nr):
                # Eval and move down
                pass

            pass
        elif self.trust_region_feasibility_strategy == "gradient-correction":
            pass
        else:
            raise NotImplementedError()

    def solve(self, nlpdata: MinlpData, prev_feasible=True, require_benders=False) -> MinlpData:
        """Solve."""
        # Update with the lowest upperbound and the corresponding best solution:
        x_sol = nlpdata.x_sol[:self.nr_x_orig]
        if almost_equal(nlpdata.obj_val, self.y_N_val):
            require_benders = True

        if prev_feasible and nlpdata.obj_val < self.y_N_val:
            # New upper bound found + benders cut might need update!
            self._update_tr_bound(x_sol, nlpdata.obj_val)
            print(f"NEW BOUND {self.y_N_val}")

        # Create a new cut
        g_k = self._generate_cut_equation(
            self._x_bin, x_sol, x_sol[self.idx_x_bin],
            nlpdata.lam_g_sol, nlpdata.lam_x_sol, nlpdata.p, prev_feasible
        )

        if not prev_feasible:
            # Benders infeasibility cut is always valid
            self.g_benders_inf.add(-ca.inf, g_k, 0)
        elif self.x_valid and self._check_cut_valid(g_k, self.x_sol_best, self.y_N_val):
            # Add normal benders cut
            self.g_benders.add(-ca.inf, g_k, 0)
        else:
            # Make special cuts for nonconvex case!
            self.g_nonconvex_cut.add(*self._create_nonconvex_cut(nlpdata))

        if self.qp_makes_progression and not require_benders:
            nlpdata.prev_solution = self._solve_trust_region(nlpdata)
            if np.allclose(nlpdata.x_sol[self.idx_x_bin], self.x_sol_best[self.idx_x_bin]):
                self.qp_makes_progression = False
                nlpdata.prev_solution = self._solve_benders(nlpdata)
        else:
            nlpdata.prev_solution = self._solve_benders(nlpdata)

        nlpdata.solved, _ = self.collect_stats("milp_bconstraint")
        return nlpdata, require_benders
