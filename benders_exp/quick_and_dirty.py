"""Quick and dirty implementation."""

import matplotlib.pyplot as plt
from sys import argv

from typing import Tuple
import casadi as ca
import numpy as np
from benders_exp.utils import plot_trajectory, tic, to_0d, toc  # , DebugCallBack
from benders_exp.defines import IMG_DIR, WITH_PLOT
from benders_exp.problems.overview import PROBLEMS
from benders_exp.problems import MinlpData, MinlpProblem, MetaDataOcp
from benders_exp.solvers import Stats
from benders_exp.solvers.nlp import NlpSolver, FeasibilityNlpSolver, SolverClass
from benders_exp.solvers.benders import BendersMasterMILP, BendersConstraintMILP, BendersMasterMIQP
from benders_exp.solvers.outer_approx import OuterApproxMILP
from benders_exp.utils import make_bounded


def base_strategy(problem: MinlpProblem, data: MinlpData, stats: Stats,
                  master_problem: SolverClass) -> Tuple[MinlpData, ca.DM]:
    """Run the base strategy."""
    print("Setup NLP solver...")
    nlp = NlpSolver(problem, stats)
    toc()
    print("Setup FNLP solver...")
    fnlp = FeasibilityNlpSolver(problem, data, stats)
    t_load = toc()
    print("Solver initialized.")
    # Benders algorithm
    lb = -ca.inf
    ub = ca.inf
    tolerance = 0.04
    feasible = True
    while lb + tolerance < ub and feasible:
        toc()
        # Solve NLP(y^k)
        data = nlp.solve(data, set_x_bin=True)
        prev_feasible = data.solved
        x_bar = data.x_sol
        print(f"{x_bar=}")

        if not prev_feasible:
            # Solve NLPF(y^k)
            data = fnlp.solve(data)
            x_bar = data.x_sol
            print("Infeasible")
        elif data.obj_val < ub:
            ub = data.obj_val
            x_star = x_bar
            print("Feasible")

        # Solve master^k and set lower bound:
        data = master_problem.solve(data, prev_feasible=prev_feasible)
        feasible = data.solved
        lb = data.obj_val
        x_hat = data.x_sol
        print(f"\n\n\n{x_hat=}")
        print(f"{ub=}\n{lb=}\n\n\n")
        stats['iter'] += 1

    t_total = toc()
    print(f"{t_total=} of with calc: {t_total - t_load}")
    return data, x_star


def benders_algorithm(problem: MinlpProblem, data: MinlpData, stats: Stats,
                      with_qp: bool = False) -> Tuple[MinlpData, ca.DM]:
    """Create and run benders algorithm."""
    print("Setup MILP solver...")
    if with_qp:
        # This class provides an improved benders implementation
        # where the hessian of the original function is used as stabelizer
        # it is motivated by the approximation that benders make on the
        # original function. Of course the lower bound can no longer
        # be guaranteed, so care should be taken with this approach as it
        # might only work in some circumstances!
        benders_master = BendersMasterMIQP(problem, data, stats)
    else:
        # This class implements the original benders decomposition
        benders_master = BendersMasterMILP(problem, data, stats)

    toc()
    return base_strategy(problem, data, stats, benders_master)


def outer_approx_algorithm(
    problem: MinlpProblem, data: MinlpData, stats: Stats
) -> Tuple[MinlpData, ca.DM]:
    """Create and run outer approximation."""
    print("Setup MILP solver...")
    outer_approx = OuterApproxMILP(problem, data, stats)
    toc()
    return base_strategy(problem, data, stats, outer_approx)


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
        data = benders_milp.solve(
            data, prev_feasible=prev_feasible, integer=is_integer)
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
        print("Available problems are: %s" % ", ".join(PROBLEMS.keys()))

    if len(argv) > 1:
        mode = argv[1]
    else:
        mode = "bendersqp"

    if len(argv) > 2:
        problem_name = argv[2]
    else:
        problem_name = "dummy"

    tic()
    new_inf = 1e3
    print(problem_name)
    if problem_name in PROBLEMS:
        problem, data = PROBLEMS[problem_name]()
        if problem == "orig":
            new_inf = 1e5
    else:
        raise Exception(f"No {problem_name=}, available: {PROBLEMS.keys()}")

    make_bounded(problem, data, new_inf=new_inf)
    print("Problem loaded")
    toc()
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
    elif mode == "outerapprox":
        data, x_star = outer_approx_algorithm(problem, data, stats)

    stats.print()
    print(x_star)
    if isinstance(problem.meta, MetaDataOcp):
        meta = problem.meta
        state = to_0d(x_star)[meta.idx_state].reshape(-1, meta.n_state)
        state = np.vstack([meta.initial_state, state])
        control = to_0d(x_star)[meta.idx_control].reshape(-1, meta.n_control)
        fig, axs = plot_trajectory(state, control, meta, title=problem_name)
        fig.savefig(f"{IMG_DIR}/ocp_trajectory.pdf", bbox_inches='tight')

        plt.show()

    if WITH_PLOT:
        plt.show()
