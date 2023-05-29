"""Quick and dirty implementation."""

import matplotlib.pyplot as plt
from sys import argv

from typing import Tuple, Union
import casadi as ca
import numpy as np
from benders_exp.utils import plot_trajectory, tic, to_0d, toc  # , DebugCallBack
from benders_exp.defines import IMG_DIR, WITH_PLOT
from benders_exp.problems.overview import PROBLEMS
from benders_exp.problems import MinlpData, MinlpProblem, MetaDataOcp, check_solution
from benders_exp.solvers import Stats
from benders_exp.solvers.nlp import NlpSolver, FeasibilityNlpSolver, SolverClass
from benders_exp.solvers.benders import BendersMasterMILP, BendersConstraintMILP, BendersMasterMIQP
from benders_exp.solvers.outer_approx import OuterApproxMILP, OuterApproxMILPImproved
from benders_exp.solvers.bonmin import BonminSolver
from benders_exp.utils import make_bounded


def base_strategy(problem: MinlpProblem, data: MinlpData, stats: Stats,
                  master_problem: SolverClass) -> Tuple[MinlpData, ca.DM]:
    """Run the base strategy."""
    print("Setup NLP solver...")
    nlp = NlpSolver(problem, stats)
    toc()
    print("Setup FNLP solver...")
    fnlp = FeasibilityNlpSolver(problem, data, stats)
    stats['total_time_loading'] = toc(reset=True)
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

    stats['total_time_calc'] = toc(reset=True)
    return problem, data, x_star


def benders_algorithm(problem: MinlpProblem, data: MinlpData, stats: Stats,
                      with_qp: bool = False) -> Tuple[MinlpData, ca.DM]:
    """Create and run benders algorithm."""
    if with_qp:
        # This class provides an improved benders implementation
        # where the hessian of the original function is used as stabelizer
        # it is motivated by the approximation that benders make on the
        # original function. Of course the lower bound can no longer
        # be guaranteed, so care should be taken with this approach as it
        # might only work in some circumstances!
        print("Setup Benders MIQP solver...")
        benders_master = BendersMasterMIQP(problem, data, stats)
    else:
        print("Setup Benders MILP solver...")
        # This class implements the original benders decomposition
        benders_master = BendersMasterMILP(problem, data, stats)

    toc()
    return base_strategy(problem, data, stats, benders_master)


def outer_approx_algorithm(
    problem: MinlpProblem, data: MinlpData, stats: Stats
) -> Tuple[MinlpData, ca.DM]:
    """Create and run outer approximation."""
    print("Setup OA MILP solver...")
    outer_approx = OuterApproxMILP(problem, data, stats)
    toc()
    return base_strategy(problem, data, stats, outer_approx)


def outer_approx_algorithm_improved(
    problem: MinlpProblem, data: MinlpData, stats: Stats
) -> Tuple[MinlpData, ca.DM]:
    """Create and run improved outer approximation."""
    print("Setup OAI MILP solver...")
    outer_approx = OuterApproxMILPImproved(problem, data, stats)
    toc()
    return base_strategy(problem, data, stats, outer_approx)


def benders_constrained_milp(
    problem: MinlpProblem, data: MinlpData, stats: Stats
) -> Tuple[MinlpData, ca.DM]:
    """Create and run benders constrained milp algorithm."""
    print("Setup Idea MIQP solver...")
    outer_approx = BendersConstraintMILP(problem, data, stats)
    toc()
    return base_strategy(problem, data, stats, outer_approx)


def bonmin(problem, data, stats):
    """Create benders algorithm."""
    tic()
    toc()
    print("Create bonmin.")
    minlp = BonminSolver(problem, stats)
    stats['total_time_loading'] = toc(reset=True)
    data = minlp.solve(data)
    stats['total_time_calc'] = toc(reset=True)
    return problem, data, data.x_sol


def run_problem(mode_name, problem_name, stats) -> Union[MinlpProblem, MinlpData, ca.DM]:
    """Run a problem and return the results."""
    if problem_name in PROBLEMS:
        problem, data = PROBLEMS[problem_name]()
        if problem == "orig":
            new_inf = 1e5
        else:
            new_inf = 1e3
        make_bounded(problem, data, new_inf=new_inf)
        print(f"Problem {problem_name} loaded")
    else:
        raise Exception(f"No {problem_name=}, available: {PROBLEMS.keys()}")

    MODES = {
        "benders": benders_algorithm,
        "bendersqp": lambda p, d, s: benders_algorithm(p, d, s, with_qp=True),
        "idea": benders_constrained_milp,
        "outerapprox": outer_approx_algorithm,
        "oa": outer_approx_algorithm,
        "oai": outer_approx_algorithm_improved,
        "bonmin": bonmin,
    }

    if mode_name in MODES:
        print(f"Start mode {mode_name}")
        return MODES[mode_name](problem, data, stats)
    else:
        raise Exception(f"No mode {mode_name=}, available {MODES.keys()}")


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
    stats = Stats({})
    toc()

    problem, data, x_star = run_problem(mode, problem_name, stats)
    stats.print()
    print(f"Objective value: {data.obj_val}")

    print(x_star)
    check_solution(problem, data, x_star)
    if isinstance(problem.meta, MetaDataOcp):
        meta = problem.meta
        state = to_0d(x_star)[meta.idx_state].reshape(-1, meta.n_state)
        state = np.vstack([meta.initial_state, state])
        control = to_0d(x_star)[meta.idx_control].reshape(-1, meta.n_control)
        fig, axs = plot_trajectory(state, control, meta, title=problem_name)

        # TODO the next is only a patch for plotting the demand for the double tank problem
        time_array = np.linspace(0, meta.dt * state.shape[0], state.shape[0] + 1)
        demand = np.array([2 + 0.5 * np.sin(x) for x in time_array])
        axs[1].plot(time_array, demand, "r--", alpha=0.5)

        uptime = problem.meta.min_uptime
        fig.savefig(f"{IMG_DIR}/ocp_trajectory_{mode}_uptime_{uptime}.pdf", bbox_inches='tight')
        plt.show()

    if WITH_PLOT:
        plt.show()
