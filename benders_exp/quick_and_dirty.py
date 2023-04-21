"""Quick and dirty implementation."""

import matplotlib.pyplot as plt
from sys import argv
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict
import numpy as np
from abc import ABC, abstractmethod

import casadi as ca
from benders_exp.utils import tic, toc  # , DebugCallBack
from benders_exp.defines import WITH_JIT, WITH_LOGGING, WITH_PLOT, CASADI_VAR, IPOPT_SETTINGS
from benders_exp.problems.overview import PROBLEMS
from benders_exp.problems import MinlpProblem, MinlpData

if WITH_PLOT:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


def to_0d(array):
    """To zero dimensions."""
    if isinstance(array, np.ndarray):
        return array.squeeze()
    else:
        return array.full().squeeze()


@dataclass
class Stats:
    """Collect stats."""

    data: Dict[str, float]

    def __getitem__(self, key):
        """Get attribute."""
        if key not in self.data:
            return 0
        return self.data[key]

    def __setitem__(self, key, value):
        """Set item."""
        self.data[key] = value

    def print(self):
        """Print statistics."""
        print("Statistics")
        for k, v in self.data.items():
            print(f"\t{k}: {v}")


def visualize_cut(g_k, x_bin, nu):
    """Visualize cut."""
    xx, yy = np.meshgrid(range(10), range(10))
    cut = ca.Function("t", [x_bin, nu], [g_k])
    z = np.zeros(xx.shape)
    for i in range(10):
        for j in range(10):
            z[i, j] = cut(ca.vertcat(xx[i, j], yy[i, j]), 0).full()[0, 0]

    ax.plot_surface(xx, yy, z, alpha=0.2)
    plt.show(block=False)
    plt.pause(1)


def make_bounded(problem: MinlpProblem, new_inf=1e5):
    """Make bounded."""
    problem.lbx[problem.lbx < -new_inf] = -new_inf
    problem.ubx[problem.ubx > new_inf] = new_inf
    problem.lbg[problem.lbg < -1e9] = -1e9
    problem.ubg[problem.ubg > 1e9] = 1e9


class SolverClass(ABC):
    """Create solver class."""

    def __init___(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create a solver class."""
        self.stats = stats
        self.solver = None

    @abstractmethod
    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve the problem."""

    def collect_stats(self):
        """Collect statistics."""
        stats = self.solver.stats()
        return stats["success"], stats


class NlpSolver(SolverClass):
    """
    Create NLP solver.

    This solver solves the NLP problem. This is either relaxed or
    the binaries are fixed.
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create NLP problem."""
        super(NlpSolver, self).__init___(problem, stats)
        if options is None:
            if WITH_LOGGING:
                options = {}
            else:
                options = {"ipopt.print_level": 0,
                           "verbose": False, "print_time": 0}

        self.idx_x_bin = problem.idx_x_bin
        options.update(IPOPT_SETTINGS)
        # self.callback = DebugCallBack(
        #     'mycallback', problem.x.shape[0],
        #     problem.g.shape[0], problem.p.shape[0]
        # )
        # self.callback.add_to_solver_opts(options, 50)

        if problem.precompiled_nlp is not None:
            # TODO: Clutter!
            self.solver = ca.nlpsol(
                "nlp", "ipopt", problem.precompiled_nlp, options
            )
        else:
            options.update({"jit": WITH_JIT})
            self.solver = ca.nlpsol("nlpsol", "ipopt", {
                "f": problem.f, "g": problem.g, "x": problem.x, "p": problem.p
            }, options)

    def solve(self, nlpdata: MinlpData, set_x_bin=False) -> MinlpData:
        """Solve NLP."""
        lbx = nlpdata.lbx
        ubx = nlpdata.ubx
        if set_x_bin:
            lbx[self.idx_x_bin] = to_0d(nlpdata.x_sol[self.idx_x_bin])
            ubx[self.idx_x_bin] = to_0d(nlpdata.x_sol[self.idx_x_bin])

        new_sol = self.solver(
            p=nlpdata.p, x0=nlpdata.x0,  # _sol[:nlpdata.x0.shape[0]],
            lbx=lbx, ubx=ubx,
            lbg=nlpdata.lbg, ubg=nlpdata.ubg
        )
        # self.callback.save(new_sol["x"])

        nlpdata.solved, stats = self.collect_stats()
        self.stats["nlp.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["nlp.iter"] += max(0, stats["iter_count"])
        if not nlpdata.solved:
            print("NLP not solved")
        else:
            nlpdata.prev_solution = new_sol
        return nlpdata


class BendersMasterMILP(SolverClass):
    """Create benders master problem."""

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create benders master MILP."""
        super(BendersMasterMILP, self).__init___(problem, stats)
        self.setup_common(problem, options)

        self.grad_f_x_sub_bin = ca.Function(
            "gradient_f_x_bin",
            [problem.x, problem.p], [ca.gradient(
                problem.f, problem.x
            )[problem.idx_x_bin]],
            {"jit": WITH_JIT}
        )
        self.jac_g_sub_bin = ca.Function(
            "jac_g_bin", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)[:, problem.idx_x_bin]],
            {"jit": WITH_JIT}
        )
        self._x = CASADI_VAR.sym("x_bin", self.nr_x_bin)

    def setup_common(self, problem: MinlpProblem, options):
        """Set up common data."""
        if options is None:
            if WITH_LOGGING:
                options = {}
            else:
                options = {"verbose": False,
                           "print_time": 0, "gurobi.output_flag": 0}

        self.f = ca.Function(
            "f", [problem.x, problem.p], [problem.f],
            {"jit": WITH_JIT}
        )
        self.g = ca.Function(
            "g", [problem.x, problem.p], [problem.g],
            {"jit": WITH_JIT}
        )

        self.idx_x_bin = problem.idx_x_bin
        self.nr_x_bin = len(problem.idx_x_bin)
        self._nu = CASADI_VAR.sym("nu", 1)
        self._g = np.array([])
        self.nr_g = 0
        self.options = options.copy()
        self.options["discrete"] = [1] * (self.nr_x_bin + 1)
        self.options["discrete"][-1] = 0
        self.options["gurobi.MIPGap"] = 0.05
        self.nr_g_orig = problem.g.shape[0]
        self.nr_x_orig = problem.x.shape[0]

    def _generate_cut_equation(self, x, x_sol, x_sol_sub_set, lam_g, p, prev_feasible):
        """
        Generate a cut.

        :param x: optimization variable
        :param x_sol: Complete x_solution
        :param x_sol_sub_set: Subset of the x solution to optimize the MILP to
        :param lam_g: Lambda g solution
        :param p: parameters
        :param prev_feasible: If the previous solution was feasible
        :return: g_k the new cutting plane (should be > 0)
        """
        if prev_feasible:
            grad_f_k = self.grad_f_x_sub_bin(x_sol, p)
            jac_g_k = self.jac_g_sub_bin(x_sol, p)
            lambda_k = grad_f_k - jac_g_k.T @ - lam_g
            f_k = self.f(x_sol, p)
            g_k = (
                f_k + lambda_k.T @ (x - x_sol_sub_set)
                - self._nu
            )
        else:  # Not feasible solution
            h_k = self.g(x_sol, p)
            jac_h_k = self.jac_g_sub_bin(x_sol, p)
            lam_g = lam_g[:self.nr_g_orig] - lam_g[self.nr_g_orig:]
            g_k = lam_g.T @ (h_k + jac_h_k @ (x - x_sol_sub_set))

        return g_k

    def solve(self, nlpdata: MinlpData, prev_feasible=True) -> MinlpData:
        """solve."""
        g_k = self._generate_cut_equation(
            self._x, nlpdata.x_sol[:self.nr_x_orig], nlpdata.x_sol[self.idx_x_bin],
            nlpdata.lam_g_sol, nlpdata.p, prev_feasible
        )

        if WITH_PLOT:
            visualize_cut(g_k, self._x, self._nu)

        self._g = ca.vertcat(self._g, g_k)
        self.nr_g += 1

        self.solver = ca.qpsol(f"benders{self.nr_g}", "gurobi", {
            "f": self._nu, "g": self._g,
            "x": ca.vertcat(self._x, self._nu),
        }, self.options)

        # This solver solves only to the binary variables (_x)!
        solution = self.solver(
            x0=ca.vertcat(nlpdata.x_sol[self.idx_x_bin], nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx[self.idx_x_bin], -1e5),
            ubx=ca.vertcat(nlpdata.ubx[self.idx_x_bin], ca.inf),
            lbg=-ca.inf * np.ones(self.nr_g),
            ubg=np.zeros(self.nr_g)
        )
        x_full = nlpdata.x_sol.full()[:self.nr_x_orig]
        x_full[self.idx_x_bin] = solution['x'][:-1]
        solution['x'] = x_full
        nlpdata.prev_solution = solution
        nlpdata.solved, stats = self.collect_stats()
        self.stats["milp_benders.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["milp_benders.iter"] += max(0, stats["iter_count"])
        return nlpdata


class BendersConstraintMILP(BendersMasterMILP):
    """
    Create benders constraint MILP.

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

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create the benders constraint MILP."""
        super(BendersConstraintMILP, self).__init__(problem, stats, options)
        self.setup_common(problem, options)

        self.grad_f_x_sub = ca.Function(
            "gradient_f_x",
            [problem.x, problem.p], [ca.gradient(
                problem.f, problem.x
            )],
            {"jit": WITH_JIT}
        )
        self.jac_g_sub = ca.Function(
            "jac_g", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)],
            {"jit": WITH_JIT}
        )
        self.f_hess = ca.Function("hess_f_x", [problem.x, problem.p], [ca.hessian(problem.f, problem.x)[0]])

        self._x = CASADI_VAR.sym("x", self.nr_x_orig)
        self.y_N_val = 1e15  # Should be inf but can not at the moment ca.inf

    def solve(self, nlpdata: MinlpData, prev_feasible=True, integer=False) -> MinlpData:
        """Solve."""
        # Create a new cut
        x_sol = nlpdata.x_sol[:self.nr_x_orig]
        g_k = self._generate_cut_equation(
            self._x[self.idx_x_bin], x_sol, x_sol[self.idx_x_bin], nlpdata.lam_g_sol, nlpdata.p, prev_feasible
        )
        # If the upper bound improved, decrease it:
        if integer and prev_feasible:
            self.y_N_val = min(self.y_N_val, nlpdata.obj_val)
            print(f"NEW BOUND {self.y_N_val}")

        f_lin = self.grad_f_x_sub(nlpdata.x_sol[:self.nr_x_orig], nlpdata.p)
        g_lin = self.g(nlpdata.x_sol[:self.nr_x_orig], nlpdata.p)
        jac_g = self.jac_g_sub(nlpdata.x_sol[:self.nr_x_orig], nlpdata.p)
        # f_hess = self.f_hess(nlpdata.x_sol[:self.nr_x_orig], nlpdata.p)

        # TODO: When linearizing the bounds, remember they are two sided!
        # we need to take the other bounds into account as well
        self.solver = ca.qpsol(f"benders_constraint{self.nr_g}", "gurobi", {
            # "f": f_lin.T @ self._x + 0.5 * self._x.T @ f_hess @ self._x, #TODO: add a flag to solve the qp
            "f": f_lin.T @ self._x,
            "g": ca.vertcat(
                g_lin + jac_g @ self._x,  # TODO: Check sign error?
                self._g
            ),
            "x": self._x, "p": self._nu
        }, self.options)

        nlpdata.prev_solution = self.solver(
            x0=x_sol,
            lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=ca.vertcat(
                nlpdata.lbg,
                -ca.inf * np.ones(self.nr_g)
            ),
            ubg=ca.vertcat(
                # ca.inf * np.ones(self.nr_g_orig),
                # TODO: NEED TO TAKE INTO ACCOUNT: nlpdata.ubg,
                nlpdata.ubg,  # TODO: verify correctness
                np.zeros(self.nr_g)
            ),
            p=[self.y_N_val]
        )
        nlpdata.prev_solution['x'] = nlpdata.prev_solution['x'][:self.nr_x_orig]

        nlpdata.solved, stats = self.collect_stats()
        self.stats["milp_bconstraint.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["milp_bconstraint.iter"] += max(0, stats["iter_count"])
        self._g = ca.vertcat(self._g, g_k)
        self.nr_g += 1
        return nlpdata


class FeasibilityNLP(SolverClass):
    """Create benders master problem."""

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create benders master MILP."""
        super(FeasibilityNLP, self).__init___(problem, stats)
        if options is None:
            if WITH_LOGGING:
                options = {}
            else:
                options = {
                    "ipopt.print_level": 0, "verbose": False, "print_time": 0
                }

        self.nr_g = problem.g.shape[0]
        s_lbg = CASADI_VAR.sym("s_lbg", self.nr_g)
        lbg = CASADI_VAR.sym("lbg", self.nr_g)
        ubg = CASADI_VAR.sym("ubg", self.nr_g)

        g = ca.vertcat(
            problem.g - lbg + s_lbg,
            ubg + s_lbg - problem.g
        )
        self.lbg = np.zeros((self.nr_g * 2, 1))
        self.ubg = ca.inf * np.ones((self.nr_g * 2, 1))
        f = ca.sum1(s_lbg)
        x = ca.vertcat(problem.x, s_lbg)
        p = ca.vertcat(problem.p, lbg, ubg)

        self.idx_x_bin = problem.idx_x_bin
        options.update({"jit": WITH_JIT})
        options.update(IPOPT_SETTINGS)
        self.solver = ca.nlpsol("nlpsol", "ipopt", {
            "f": f, "g": g, "x": x, "p": p
        }, options)

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """solve."""
        print("FEASIBILITY")
        lbx = deepcopy(nlpdata.lbx)
        ubx = deepcopy(nlpdata.ubx)
        lbx[self.idx_x_bin] = to_0d(nlpdata.x_sol[self.idx_x_bin])
        ubx[self.idx_x_bin] = to_0d(nlpdata.x_sol[self.idx_x_bin])

        nlpdata.prev_solution = self.solver(
            x0=ca.vertcat(
                nlpdata.x_sol[:nlpdata.x0.shape[0]
                              ], np.zeros((self.nr_g * 1, 1))
            ),
            lbx=ca.vertcat(lbx, np.zeros((self.nr_g * 1, 1))),
            ubx=ca.vertcat(ubx, ca.inf * np.ones((self.nr_g * 1, 1))),
            lbg=self.lbg,
            ubg=self.ubg,
            p=ca.vertcat(nlpdata.p, nlpdata.lbg, nlpdata.ubg)
        )
        nlpdata.solved, stats = self.collect_stats()
        self.stats["fnlp.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["fnlp.iter"] += max(0, stats["iter_count"])
        if not nlpdata.solved:
            print("MILP not solved")
        return nlpdata


def benders_algorithm(problem, data, stats, ):
    """Create benders algorithm."""
    tic()
    toc()
    print("Setup NLP solver...")
    nlp = NlpSolver(problem, stats, )
    toc()
    print("Setup FNLP solver...")
    fnlp = FeasibilityNLP(problem, stats)
    toc()
    print("Setup MILP solver...")
    benders_milp = BendersMasterMILP(problem, stats)
    t_load = toc()

    print("Solver initialized.")
    # Benders algorithm
    lb = -ca.inf
    ub = ca.inf
    tolerance = 0.04
    feasible = True
    data = nlp.solve(data)
    x_bar = data.x_sol
    x_star = x_bar
    prev_feasible = True
    while lb + tolerance < ub and feasible:
        toc()
        # Solve MILP-BENDERS and set lower bound:
        data = benders_milp.solve(data, prev_feasible=prev_feasible)
        feasible = data.solved
        lb = data.obj_val
        # x_hat = data.x_sol

        # Obtain new linearization point for NLP:
        data = nlp.solve(data, set_x_bin=True)
        x_bar = data.x_sol
        prev_feasible = data.solved
        if not prev_feasible:
            data = fnlp.solve(data)
            x_bar = data.x_sol
            print("Infeasible")
        elif data.obj_val < ub:
            ub = data.obj_val
            x_star = x_bar
            print("Feasible")

        print(f"{ub=} {lb=}")
        print(f"{x_bar=}")

    t_total = toc()
    print(f"{t_total=} of with calc: {t_total - t_load}")
    return data, x_star


def idea_algorithm(problem, data, stats):
    """Create benders algorithm."""
    tic()
    toc()
    print("Setup NLP solver...")
    nlp = NlpSolver(problem, stats)
    toc()
    print("Setup FNLP solver...")
    fnlp = FeasibilityNLP(problem, stats)
    toc()
    print("Setup MILP solver...")
    benders_milp = BendersConstraintMILP(problem, stats)
    toc(reset=True)

    print("Solver initialized.")
    # Benders algorithm
    lb = -ca.inf
    ub = ca.inf
    tolerance = 0.04
    feasible = True
    # TODO: setting x_bin to start the algorithm with a integer solution,
    # no guarantees about its feasibility!
    data = nlp.solve(data, set_x_bin=True)
    x_bar = data.x_sol
    x_star = x_bar
    prev_feasible = True  # TODO: check feasibility of nlp.solve(...)
    is_integer = True
    while lb + tolerance < ub and feasible:
        toc()
        # Solve MILP-BENDERS and set lower bound:
        data = benders_milp.solve(data, prev_feasible=prev_feasible, integer=is_integer)
        feasible = data.solved
        lb = data.obj_val
        # x_hat = data.x_sol

        # Obtain new linearization point for NLP:
        data = nlp.solve(data, set_x_bin=True)
        x_bar = data.x_sol
        prev_feasible = data.solved
        if not prev_feasible:
            data = fnlp.solve(data)
            x_bar = data.x_sol
            print("Infeasible")
        elif data.obj_val < ub:
            ub = data.obj_val
            x_star = x_bar
            print("Feasible")

        is_integer = True
        print(f"{ub=} {lb=}")
        print(f"{x_bar=}")

    return data, x_star


if __name__ == "__main__":
    if len(argv) == 1:
        print("Usage: mode problem")
        print("Available modes are: benders, idea, ...")
        print("Available problems are: dummy, dummy2, orig, ...")

    if len(argv) > 1:
        mode = argv[1]
    else:
        mode = "benders"

    if len(argv) > 2:
        problem = argv[2]
    else:
        problem = "orig"

    new_inf = 1e3
    print(problem)
    if problem in PROBLEMS:
        problem, data = PROBLEMS[problem]()
        if problem == "orig":
            new_inf = 1e5
    else:
        raise Exception(f"No {problem=}")

    # make_bounded(data, new_inf=new_inf)
    print("Problem loaded")
    stats = Stats({})
    if mode == "benders":
        data, x_star = benders_algorithm(
            problem, data, stats
        )
    elif mode == "idea":
        data, x_star = idea_algorithm(problem, data,  stats)

    stats.print()
    print(x_star)
    if WITH_PLOT:
        plt.show()
