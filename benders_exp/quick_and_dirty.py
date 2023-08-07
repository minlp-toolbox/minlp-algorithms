"""Quick and dirty implementation."""

import matplotlib.pyplot as plt
from sys import argv

from typing import Callable, Tuple, Union
import casadi as ca
import numpy as np
from benders_exp.utils import plot_trajectory, tic, to_0d, toc, \
        make_bounded, setup_logger, logging
from benders_exp.defines import EPS, IMG_DIR, WITH_JIT, WITH_PLOT
from benders_exp.problems.overview import PROBLEMS
from benders_exp.problems import MinlpData, MinlpProblem, MetaDataOcp, check_solution, MetaDataMpc
from benders_exp.solvers import Stats
from benders_exp.solvers.nlp import NlpSolver, FeasibilityNlpSolver, SolverClass
from benders_exp.solvers.benders import BendersMasterMILP, BendersTrustRegionMIP, BendersMasterMIQP
from benders_exp.solvers.benders_mix import BendersTRandMaster
from benders_exp.solvers.outer_approx import OuterApproxMILP, OuterApproxMILPImproved
from benders_exp.solvers.bonmin import BonminSolver
from benders_exp.solvers.voronoi import VoronoiTrustRegionMILP

logger = logging.getLogger(__name__)


def base_strategy(problem: MinlpProblem, data: MinlpData, stats: Stats,
                  master_problem: SolverClass, termination_condition: Callable[..., bool]
                  ) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Run the base strategy."""
    logger.info("Setup NLP solver...")
    nlp = NlpSolver(problem, stats)
    toc()
    logger.info("Setup FNLP solver...")
    fnlp = FeasibilityNlpSolver(problem, data, stats)
    stats['total_time_loading'] = toc(reset=True)
    logger.info("Solver initialized.")
    # Benders algorithm
    lb = -ca.inf
    ub = ca.inf
    tolerance = 0.04
    feasible = True
    x_star = np.nan * np.empty(problem.x.shape[0])
    x_hat = -np.nan * np.empty(problem.x.shape[0])

    while (not termination_condition(lb, ub, tolerance, x_star, x_hat)) and feasible:
        toc()
        # Solve NLP(y^k)
        data = nlp.solve(data, set_x_bin=True)
        prev_feasible = data.solved
        x_bar = data.x_sol

        if not prev_feasible:
            # Solve NLPF(y^k)
            data = fnlp.solve(data)
            x_bar = data.x_sol
            logger.debug("Infeasible")
        elif data.obj_val < ub:
            ub = data.obj_val
            x_star = x_bar
            logger.debug("Feasible")

        # Solve master^k and set lower bound:
        data = master_problem.solve(data, prev_feasible=prev_feasible)
        feasible = data.solved
        lb = data.obj_val
        x_hat = data.x_sol
        logger.debug(f"{ub=}, {lb=}\n")
        stats['iter'] += 1

    stats['total_time_calc'] = toc(reset=True)
    return problem, data, x_star


def get_termination_condition(termination_type, problem: MinlpProblem,  data: MinlpData):
    """
    Get termination condition.

    :param termination_type: String of the termination type (gradient, std or equality)
    :param problem: problem
    :param data: data
    :return: callable that returns true if the termination condition holds
    """
    if termination_type == 'gradient':
        idx_x_bin = problem.idx_x_bin
        f_fn = ca.Function("f", [problem.x, problem.p], [
                           problem.f], {"jit": WITH_JIT})
        grad_f_fn = ca.Function("gradient_f_x", [problem.x, problem.p], [ca.gradient(problem.f, problem.x)],
                                {"jit": WITH_JIT})

        def func(lb=None, ub=None, tol=None, x_best=None, x_current=None):
            return to_0d(
                f_fn(x_current, data.p)
                + grad_f_fn(x_current, data.p)[idx_x_bin].T @ (
                    x_current[idx_x_bin] - x_best[idx_x_bin])
                - f_fn(x_best, data.p)
            ) >= 0
    elif termination_type == 'equality':
        idx_x_bin = problem.idx_x_bin

        def func(lb=None, ub=None, tol=None, x_best=None, x_current=None):
            if isinstance(x_best, list):
                for x in x_best:
                    if np.allclose(x[idx_x_bin], x_current[idx_x_bin], equal_nan=False, atol=EPS):
                        return True
            else:
                return np.allclose(x_best[idx_x_bin], x_current[idx_x_bin], equal_nan=False, atol=EPS)

    elif termination_type == 'std':
        def func(lb=None, ub=None, tol=None, x_best=None, x_current=None):
            return (lb + tol - ub) >= 0
    else:
        raise AttributeError(
            f"Invalid type of termination condition, you set '{termination_type}' but the only option is 'std'!")
    return func


def benders_algorithm(problem: MinlpProblem, data: MinlpData, stats: Stats,
                      with_qp: bool = False, termination_type: str = 'std') -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """
    Create and run benders algorithm.

    parameters:
        - termination_type:
            - std: based on lower and upper bound
    """
    if with_qp:
        # This class provides an improved benders implementation
        # where the hessian of the original function is used as stabilizer
        # it is motivated by the approximation that benders make on the
        # original function. Of course the lower bound can no longer
        # be guaranteed, so care should be taken with this approach as it
        # might only work in some circumstances!
        logger.info("Setup Benders MIQP solver...")
        benders_master = BendersMasterMIQP(problem, data, stats)
    else:
        logger.info("Setup Benders MILP solver...")
        # This class implements the original benders decomposition
        benders_master = BendersMasterMILP(problem, data, stats)

    termination_condition = get_termination_condition(
        termination_type, problem, data)
    toc()
    return base_strategy(problem, data, stats, benders_master, termination_condition)


def outer_approx_algorithm(
    problem: MinlpProblem, data: MinlpData, stats: Stats, termination_type: str = 'std'
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """
    Create and run outer approximation.

    parameters:
        - termination_type:
            - std: based on lower and upper bound
    """
    logger.info("Setup OA MILP solver...")
    outer_approx = OuterApproxMILP(problem, data, stats)
    termination_condition = get_termination_condition(
        termination_type, problem, data)
    toc()
    return base_strategy(problem, data, stats, outer_approx, termination_condition)


def outer_approx_algorithm_improved(
    problem: MinlpProblem, data: MinlpData, stats: Stats, termination_type: str = 'std'
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """
    Create and run improved outer approximation.

    parameters:
        - termination_type:
            - std: based on lower and upper bound
    """
    logger.info("Setup OAI MILP solver...")
    outer_approx = OuterApproxMILPImproved(problem, data, stats)
    termination_condition = get_termination_condition(
        termination_type, problem, data)
    toc()
    return base_strategy(problem, data, stats, outer_approx, termination_condition)


def benders_constrained_milp(
    problem: MinlpProblem, data: MinlpData, stats: Stats, termination_type: str = 'std'
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """
    Create and run benders constrained milp algorithm.

    parameters:
        - termination_type:
            - gradient: based on local linearization
            - equality: the binaries of the last solution coincides with the ones of the best solution
            - std: based on lower and upper bound
    """
    logger.info("Setup Idea MIQP solver...")
    benders_tr_master = BendersTrustRegionMIP(problem, data, stats)
    termination_condition = get_termination_condition(
        termination_type, problem, data)
    toc()
    return base_strategy(problem, data, stats, benders_tr_master, termination_condition)


def relaxed(
    problem: MinlpProblem, data: MinlpData, stats: Stats
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Create and the relaxed problem."""
    nlp = NlpSolver(problem, stats)
    data = nlp.solve(data)
    return problem, data, data.x_sol


def bonmin(
    problem: MinlpProblem, data: MinlpData, stats: Stats,
    algo_type="B-BB"
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Create benders algorithm."""
    tic()
    toc()
    logger.info("Create bonmin.")
    minlp = BonminSolver(problem, stats, algo_type=algo_type)
    stats['total_time_loading'] = toc(reset=True)
    data = minlp.solve(data)
    stats['total_time_calc'] = toc(reset=True)
    return problem, data, data.x_sol


def voronoi_tr_algorithm(
    problem: MinlpProblem, data: MinlpData, stats: Stats, termination_type: str = 'gradient'
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """
    Create and run voronoi trust region milp algorithm.

    parameters:
        - termination_type:
            - gradient: based on local linearization
            - equality: the binaries of the last solution coincides with the ones of the best solution
            - std: based on lower and upper bound
    """
    logger.info("Setup Voronoi trust region MILP solver...")
    voronoi_tr_master = VoronoiTrustRegionMILP(problem, data, stats)
    termination_condition = get_termination_condition(
        termination_type, problem, data)
    toc()
    return base_strategy(problem, data, stats, voronoi_tr_master, termination_condition)


def benders_tr_master(
    problem: MinlpProblem, data: MinlpData, stats: Stats, termination_type: str = 'equality'
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Run the base strategy."""
    termination_condition = get_termination_condition(termination_type, problem, data)
    logger.info("Setup Mixed Benders TR/Master")
    master_problem = BendersTRandMaster(problem, data, stats)
    toc()
    logger.info("Setup NLP solver...")
    nlp = NlpSolver(problem, stats)
    toc()
    logger.info("Setup FNLP solver...")
    fnlp = FeasibilityNlpSolver(problem, data, stats)
    stats['total_time_loading'] = toc(reset=True)
    logger.info("Solver initialized.")
    # Benders algorithm
    lb = -ca.inf
    ub = ca.inf
    tolerance = 0.04
    feasible = True
    data.best_solutions = []
    x_star = np.nan * np.empty(problem.x.shape[0])
    x_hat = -np.nan * np.empty(problem.x.shape[0])
    last_benders = True # only for doing at least one iteration of the while-loop
    termination_met = False

    while feasible and not (last_benders and termination_met):
        toc()
        # Solve NLP(y^k)
        data = nlp.solve(data, set_x_bin=True)
        prev_feasible = data.solved
        x_bar = data.x_sol
        if not prev_feasible:
            # Solve NLPF(y^k)
            data = fnlp.solve(data)
            x_bar = data.x_sol
            logger.debug("Infeasible")
        elif data.obj_val + EPS < ub:
            ub = data.obj_val
            data.best_solutions = []
            data.best_solutions.append(x_bar)
            x_star = data.best_solutions[-1]
            logger.info("Feasible")
        elif np.allclose(data.obj_val, ub, atol=EPS):
            data.best_solutions.append(x_bar)

        # Solve master^k and set lower bound:
        data, last_benders = master_problem.solve(
            data, prev_feasible=prev_feasible, require_benders=termination_met
        )
        lb = data.obj_val
        x_hat = data.x_sol
        logger.debug(f"{x_bar=}")
        logger.debug(f"{ub=}, {lb=}")
        stats['iter'] += 1

        feasible = data.solved
        termination_met = termination_condition(lb, ub, tolerance, data.best_solutions, x_hat)

    stats['total_time_calc'] = toc(reset=True)
    return problem, data, x_star


def run_problem(mode_name, problem_name, stats, args) -> Union[MinlpProblem, MinlpData, ca.DM]:
    """Run a problem and return the results."""
    if problem_name in PROBLEMS:
        problem, data = PROBLEMS[problem_name](*args)
        if problem == "orig":
            new_inf = 1e5
        else:
            new_inf = 1e3
        make_bounded(problem, data, new_inf=new_inf)
        logger.info(f"Problem {problem_name} loaded")
    else:
        raise Exception(f"No {problem_name=}, available: {PROBLEMS.keys()}")

    MODES = {
        "benders": benders_algorithm,
        "bendersqp": lambda p, d, s: benders_algorithm(p, d, s, with_qp=True),
        "benders_tr": lambda p, d, s: benders_constrained_milp(p, d, s, termination_type='std'),
        "benders_trm": benders_tr_master,
        "oa": outer_approx_algorithm,
        "oai": outer_approx_algorithm_improved,
        "bonmin": bonmin,
        # B-BB is a NLP-based branch-and-bound algorithm
        "bonmin-bb": lambda p, d, s: bonmin(p, d, s, "B-BB"),
        # B-Hyb is a hybrid outer-approximation based branch-and-cut algorithm
        "bonmin-hyb": lambda p, d, s: bonmin(p, d, s, "B-Hyb"),
        # B-OA is an outer-approximation decomposition algorithm
        "bonmin-oa": lambda p, d, s: bonmin(p, d, s, "B-OA"),
        # B-QG is an implementation of Quesada and Grossmann's branch-and-cut algorithm
        "bonmin-qg": lambda p, d, s: bonmin(p, d, s, "B-QG"),
        # B-iFP: an iterated feasibility pump algorithm
        "bonmin-ifp": lambda p, d, s: bonmin(p, d, s, "B-iFP"),
        "voronoi_tr": lambda p, d, s: voronoi_tr_algorithm(p, d, s, termination_type='equality'),
        "relaxed": relaxed,
    }

    if mode_name in MODES:
        logger.info(f"Start mode {mode_name}")
        return MODES[mode_name](problem, data, stats)
    else:
        raise Exception(f"No mode {mode_name=}, available {MODES.keys()}")


if __name__ == "__main__":
    setup_logger(logging.DEBUG)
    if len(argv) == 1:
        logger.info("Usage: mode problem")
        logger.info("Available modes are: benders, idea, ...")
        logger.info("Available problems are: %s" % ", ".join(PROBLEMS.keys()))

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

    problem, data, x_star = run_problem(mode, problem_name, stats, argv[3:])
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
        time_array = np.linspace(
            0, meta.dt * state.shape[0], state.shape[0] + 1)
        demand = np.array([2 + 0.5 * np.sin(x) for x in time_array])
        axs[1].plot(time_array, demand, "r--", alpha=0.5)

        uptime = problem.meta.min_uptime
        fig.savefig(
            f"{IMG_DIR}/ocp_trajectory_{mode}_uptime_{uptime}.pdf", bbox_inches='tight')
        plt.show()
    elif isinstance(problem.meta, MetaDataMpc):
        problem.meta.plot(data, x_star)

    if WITH_PLOT:
        plt.show()
