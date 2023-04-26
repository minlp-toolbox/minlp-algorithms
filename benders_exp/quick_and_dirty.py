"""Quick and dirty implementation."""

import matplotlib.pyplot as plt
from sys import argv

import casadi as ca
import numpy as np
from benders_exp.utils import tic, toc  # , DebugCallBack
from benders_exp.defines import WITH_PLOT
from benders_exp.problems.overview import PROBLEMS
from benders_exp.problems import MinlpData
from benders_exp.solvers import Stats
from benders_exp.solvers.nlp import NlpSolver, FeasibilityNlpSolver
from benders_exp.solvers.benders import BendersMasterMILP, BendersConstraintMILP, BendersMasterMIQP


def make_bounded(data: MinlpData, new_inf=1e5):
    """Make bounded."""
    if any(data.lbx < -new_inf):
        new_lbx = [-new_inf if elm else data.lbx[i] for i, elm in enumerate(data.lbx < -new_inf)]
        data.lbx = np.array(new_lbx)
    if any(data.ubx > new_inf):
        new_ubx = [new_inf if elm else data.ubx[i] for i, elm in enumerate(data.ubx > new_inf)]
        data.ubx = np.array(new_ubx)
    if any(data.lbg < -1e9):
        new_lbg = [-1e9 if elm else data.lbg[i] for i, elm in enumerate(data.lbg < -1e9)]
        data.lbg = np.array(new_lbg)
    if any(data.ubg > 1e9):
        new_ubg = [1e9 if elm else data.ubg[i] for i, elm in enumerate(data.ubg > 1e9)]
        data.ubg = np.array(new_ubg)



def benders_algorithm(problem, data, stats, with_qp=False):
    """Create benders algorithm."""
    tic()
    toc()
    print("Setup NLP solver...")
    nlp = NlpSolver(problem, stats, )
    toc()
    print("Setup FNLP solver...")
    fnlp = FeasibilityNlpSolver(problem, stats)
    toc()
    print("Setup MILP solver...")
    if with_qp:
        # This class provides an improved benders implementation
        # where the hessian of the original function is used as stabelizer
        # it is motivated by the approximation that benders make on the
        # original function. Of course the lower bound can no longer
        # be guaranteed, so care should be taken with this approach as it
        # might only work in some circumstances!
        benders_master = BendersMasterMIQP(problem, stats)
    else:
        # This class implements the original benders decomposition
        benders_master = BendersMasterMILP(problem, stats)

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
        data = benders_master.solve(data, prev_feasible=prev_feasible)
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
    fnlp = FeasibilityNlpSolver(problem, stats)
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
        print("Available problems are: dummy, dummy2, orig, doublepipe")

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

    make_bounded(data, new_inf=new_inf)
    print("Problem loaded")
    stats = Stats({})
    if mode == "benders":
        data, x_star = benders_algorithm(
            problem, data, stats
        )
    elif mode == "bendersqp":
        data, x_star = benders_algorithm(
            problem, data, stats, with_qp=True
        )
    elif mode == "idea":
        data, x_star = idea_algorithm(problem, data,  stats)

    stats.print()
    print(x_star)
    if WITH_PLOT:
        plt.show()
