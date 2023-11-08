"""Quick and dirty implementation."""

from os import path
from datetime import datetime
import matplotlib.pyplot as plt
from sys import argv
from typing import Callable, Tuple, Union
import casadi as ca
import numpy as np
from benders_exp.utils import get_control_vector, plot_trajectory, tic, to_0d, toc, \
    make_bounded, setup_logger, logging, colored
from benders_exp.json_tools import write_json
from benders_exp.defines import EPS, IMG_DIR, WITH_JIT, WITH_PLOT, WITH_LOG_DATA, WITH_DEBUG
from benders_exp.problems.overview import PROBLEMS
from benders_exp.problems import MinlpData, MinlpProblem, MetaDataOcp, check_solution, MetaDataMpc
from benders_exp.solvers import Stats
from benders_exp.solvers.nlp import NlpSolver, FeasibilityNlpSolver, SolverClass
from benders_exp.solvers.benders import BendersMasterMILP, BendersTrustRegionMIP, BendersMasterMIQP
from benders_exp.solvers.benders_mix import BendersTRandMaster
from benders_exp.solvers.outer_approx import OuterApproxMILP, OuterApproxMILPImproved
from benders_exp.solvers.bonmin import BonminSolver
from benders_exp.solvers.voronoi import VoronoiTrustRegionMILP
from benders_exp.solvers.pumps import random_direction_rounding_algorithm, random_objective_feasibility_pump, \
        feasibility_pump, objective_feasibility_pump
from benders_exp.solvers.cia import cia_decomposition_algorithm
from benders_exp.solvers.benders_equal_lb import BendersEquality
from benders_exp.solvers.milp_tr import milp_tr

logger = logging.getLogger(__name__)


def update_best_solutions(data, itr, ub, x_star, best_iter):
    """Update best solutions,"""
    if np.any(data.solved_all):
        for i, success in enumerate(data.solved_all):
            obj_val = float(data.prev_solutions[i]['f'])
            if success:
                if obj_val + EPS < ub:
                    ub = obj_val
                    x_star = data.prev_solutions[i]['x']
                    logger.info(f"\n{x_star=}")
                    logger.debug(f"Decreased UB to {ub}")
                    data.best_solutions.append(x_star)
                    best_iter = itr
                elif obj_val - EPS < ub:
                    data.best_solutions.append(
                        data.prev_solutions[i]['x']
                    )
    return ub, x_star, best_iter


def base_strategy(problem: MinlpProblem, data: MinlpData, stats: Stats,
                  master_problem: SolverClass, termination_condition: Callable[..., bool],
                  first_relaxed=False) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
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
    tolerance = 1e-5
    feasible = True
    best_iter = -1
    x_star = np.nan * np.empty(problem.x.shape[0])
    x_hat = -np.nan * np.empty(problem.x.shape[0])

    if first_relaxed:
        data = nlp.solve(data)
        data = master_problem.solve(data, relaxed=True)

    while (not termination_condition(lb, ub, tolerance, x_star, x_hat)) and feasible:
        toc()
        # Solve NLP(y^k)
        data = nlp.solve(data, set_x_bin=True)
        prev_feasible = data.solved

        # Is there a feasible success?
        ub, x_star, best_iter = update_best_solutions(
            data, stats['iter'], ub, x_star, best_iter
        )

        # Is there any infeasible?
        if not np.all(data.solved_all):
            # Solve NLPF(y^k)
            data = fnlp.solve(data)
            logger.debug("Infeasibility problem solved")

        # Solve master^k and set lower bound:
        data = master_problem.solve(data, prev_feasible=prev_feasible)
        feasible = data.solved
        lb = data.obj_val
        x_hat = data.x_sol
        logger.debug(f"\n{x_hat=}")
        logger.debug(f"{ub=}, {lb=}\n")
        stats['iter'] += 1

    stats['total_time_calc'] = toc(reset=True)
    data.prev_solution = {'x': x_star, 'f': ub}
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
            ret = to_0d(
                f_fn(x_current, data.p)
                + grad_f_fn(x_current, data.p)[idx_x_bin].T @ (
                    x_current[idx_x_bin] - x_best[idx_x_bin])
                - f_fn(x_best, data.p)
            ) >= 0
            if ret:
                logging.info("Terminated - gradient ok")
            return ret
    elif termination_type == 'equality':
        idx_x_bin = problem.idx_x_bin

        def func(lb=None, ub=None, tol=None, x_best=None, x_current=None):
            if isinstance(x_best, list):
                for x in x_best:
                    if np.allclose(x[idx_x_bin], x_current[idx_x_bin], equal_nan=False, atol=EPS):
                        logging.info(f"Terminated - all close within {EPS}")
                        return True
                return False
            else:
                ret = np.allclose(
                    x_best[idx_x_bin], x_current[idx_x_bin], equal_nan=False, atol=EPS)
                if ret:
                    logging.info(f"Terminated - all close within {EPS}")
                return ret

    elif termination_type == 'std':
        def func(lb=None, ub=None, tol=None, x_best=None, x_current=None):
            tol_abs = (abs(lb) + abs(ub)) * tol / 2
            ret = (lb + tol_abs - ub) >= 0
            if ret:
                logging.info(f"Terminated: {lb} >= {ub} - {tol_abs} ({tol*100}%)")
            return ret
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


def export_ampl(problem: MinlpProblem, data: MinlpData, stats: Stats) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Export AMPL."""
    from benders_exp.solvers.ampl import AmplSolver
    AmplSolver(problem, stats).solve(data)
    raise Exception("DONE")


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
    problem: MinlpProblem, data: MinlpData, stats: Stats, termination_type: str = 'std',
    first_relaxed=False
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
    return base_strategy(problem, data, stats, benders_tr_master, termination_condition, first_relaxed=first_relaxed)


def benders_equality(
    problem: MinlpProblem, data: MinlpData, stats: Stats, termination_type: str = 'equality'
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """
    Create and run benders equality algorithm.

    parameters:
        - termination_type:
            - gradient: based on local linearization
            - equality: the binaries of the last solution coincides with the ones of the best solution
            - std: based on lower and upper bound
    """
    logger.info("Setup Idea MIQP solver...")
    benders_tr_master = BendersEquality(problem, data, stats)
    termination_condition = get_termination_condition(
        termination_type, problem, data)
    toc()
    return base_strategy(problem, data, stats, benders_tr_master, termination_condition)


def relaxed(
    problem: MinlpProblem, data: MinlpData, stats: Stats
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Create and the relaxed problem."""
    nlp = NlpSolver(problem, stats)
    stats['total_time_loading'] = toc(reset=True)
    data = nlp.solve(data)
    stats['total_time_calc'] = toc(reset=True)
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
    problem: MinlpProblem, data: MinlpData, stats: Stats, termination_type: str = 'std',
    first_relaxed=True, use_feasibility_pump=True, with_benders_master=True
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Run the base strategy."""
    termination_condition = get_termination_condition(
        termination_type, problem, data)
    logger.info("Setup Mixed Benders TR/Master")
    master_problem = BendersTRandMaster(
        problem, data, stats, with_benders_master=with_benders_master)
    toc()
    logger.info("Setup NLP solver...")
    nlp = NlpSolver(problem, stats)
    toc()
    logger.info("Setup FNLP solver...")
    fnlp = FeasibilityNlpSolver(problem, data, stats)
    lb = -ca.inf
    ub = ca.inf
    tolerance = 1e-5
    feasible = True
    x_star = np.nan * np.empty(problem.x.shape[0])
    x_hat = -np.nan * np.empty(problem.x.shape[0])
    last_benders = True  # only for doing at least one iteration of the while-loop
    termination_met = False
    stats['iterate_data'] = []
    best_iter = None
    old_sol = x_star[problem.idx_x_bin]

    if use_feasibility_pump:
        data = nlp.solve(data)
        master_problem.update_relaxed_solution(data)
        lb = data.obj_val
        problem, data, _, is_relaxed = random_objective_feasibility_pump(
            problem, data, stats, data, nlp)
        if not is_relaxed:
            ub, x_star, best_iter = update_best_solutions(
                data, 0, ub, x_star, best_iter
            )

        data, last_benders = master_problem.solve(data, relaxed=is_relaxed)
    elif first_relaxed:
        stats['total_time_loading'] = toc(reset=True)
        logger.info("Solver initialized.")
        data = nlp.solve(data)
        lb = data.obj_val
        data, last_benders = master_problem.solve(data, relaxed=True)
    else:
        stats['total_time_loading'] = toc(reset=True)
        logger.info("Solver initialized.")

    try:
        while feasible and not termination_met:
            toc()
            # Solve NLP(y^k)
            data = nlp.solve(data, set_x_bin=True)
            logger.info("SOLVED NLP")

            ub, x_star, best_iter = update_best_solutions(
                data, stats['iter_nr'], ub, x_star, best_iter
            )

            if not np.all(data.solved_all):
                # Solve NLPF(y^k)
                data = fnlp.solve(data)
                logger.debug("SOLVED FEASIBILITY NLP")

            if WITH_DEBUG:
                if np.allclose(old_sol, to_0d(data.x_sol)[[problem.idx_x_bin]]):
                    colored("Possible error!")
                else:
                    colored("All ok", "green")

            old_sol = to_0d(data.x_sol)[problem.idx_x_bin]
            logger.debug(f"Adding {data.nr_sols} solutions")
            logger.debug(f"NLP {data.obj_val=}, {ub=}, {lb=}")
            stats['iterate_data'].append((stats.create_iter_dict(
                stats['iter_nr'], best_iter, data.solved,
                ub, data.obj_val, last_benders, lb, to_0d(data.x_sol))
            ))
            # Solve master^k and set lower bound:
            data, last_benders = master_problem.solve(data)
            if last_benders:
                lb = max(data.obj_val, lb)
            logger.debug(f"MIP {data.obj_val=}, {ub=}, {lb=}")

            x_hat = data.x_sol
            stats['iter_nr'] += 1
            if WITH_LOG_DATA:
                stats.save()

            feasible = data.solved
            termination_met = termination_condition(
                lb, ub, tolerance, data.best_solutions, x_hat)

        stats['total_time_calc'] = toc(reset=True)

    except KeyboardInterrupt:
        exit(0)

    data.prev_solution = {'x': x_star, 'f': ub}
    return problem, data, x_star


def run_problem(mode_name, problem_name, stats, args) -> Union[MinlpProblem, MinlpData, ca.DM]:
    """Run a problem and return the results."""
    MODES = {
        "benders": benders_algorithm,
        "bendersqp": lambda p, d, s: benders_algorithm(p, d, s, with_qp=True),
        "benders_old_tr": lambda p, d, s: benders_constrained_milp(p, d, s, termination_type='std',
                                                                   first_relaxed=False),
        "benders_old_tr_rel": lambda p, d, s: benders_constrained_milp(p, d, s, termination_type='std',
                                                                       first_relaxed=True),
        "benders_tr": lambda p, d, s: benders_tr_master(p, d, s, termination_type='equality',
                                                        use_feasibility_pump=False, with_benders_master=False),
        "benders_tr_fp": lambda p, d, s: benders_tr_master(p, d, s, termination_type='equality',
                                                           use_feasibility_pump=True, with_benders_master=False),
        "benders_trm": lambda p, d, s: benders_tr_master(p, d, s, use_feasibility_pump=False, with_benders_master=True),
        "benders_trm_fp": lambda p, d, s: benders_tr_master(p, d, s, use_feasibility_pump=True,
                                                            with_benders_master=True),
        "oa": outer_approx_algorithm,
        "oai": outer_approx_algorithm_improved,
        "bonmin": bonmin,
        "benderseq": benders_equality,
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
        "ampl": export_ampl,
        "rofp": random_direction_rounding_algorithm,
        "fp": feasibility_pump,
        "ofp": objective_feasibility_pump,
        "cia": cia_decomposition_algorithm,
        "milp_tr": milp_tr
    }

    if mode_name not in MODES:
        raise Exception(f"No mode {mode_name=}, available {MODES.keys()}")
    if problem_name not in PROBLEMS:
        raise Exception(f"No {problem_name=}, available: {PROBLEMS.keys()}")

    logger.info(f"Load problem {problem_name}")
    problem, data = PROBLEMS[problem_name](*args)
    if problem == "orig":
        new_inf = 1e5
    else:
        new_inf = 1e3
    make_bounded(problem, data, new_inf=new_inf)

    if len(problem.idx_x_bin) == 0:
        mode_name = "relaxed"

    logger.info(f"Start mode {mode_name}")
    return MODES[mode_name](problem, data, stats)


def batch_nl_runner(mode_name, target, nl_files):
    """Run a batch of problems."""
    from os import makedirs
    from time import time
    overview_target = path.join(target, "overview.json")
    if path.exists(overview_target):
        raise Exception(f"Overview is already existing: {overview_target}")

    makedirs(target, exist_ok=True)
    total_stats = [["id", "path", "obj",
                    "load_time", "calctime", "iter", "nr_int"]]
    start = time()
    total_to_compute = len(nl_files)
    for i, nl_file in enumerate(nl_files):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            tic()
            stats = Stats(mode_name, "nl_file", timestamp, {})
            toc()
            problem, data, x_star = run_problem(
                mode_name, "nl_file", stats, [nl_file]
            )
            stats["x_star"] = x_star
            stats["f_star"] = data.obj_val
            total_stats.append(
                [i, nl_file, data.obj_val, stats["total_time_calc"],
                    stats["iter"], len(problem.idx_x_bin)]
            )
        except Exception as e:
            print(f"{e}")
            total_stats.append(
                [i, nl_file, -ca.inf, "FAILED", f"{e}"]
            )
        stats.print()
        stats.save(path.join(target, f"stats_{i}.pkl"))
        time_now = time() - start
        total_time = time_now / (i + 1) * total_to_compute
        write_json({
            "time": time_now,
            "total": total_to_compute,
            "done": (i+1),
            "progress": (i+1) / total_to_compute,
            "time_remaining_est": total_time - time_now,
            "time_total_est": total_time,
            "data": total_stats
        }, overview_target)


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

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tic()
    stats = Stats(mode, problem_name, timestamp, {})
    toc()

    extra_args = argv[3:]
    target_file = None
    if len(extra_args) >= 2 and extra_args[-2] == "--save":
        target_file = extra_args[-1]
        extra_args = extra_args[:-2]

    problem, data, x_star = run_problem(mode, problem_name, stats, extra_args)
    stats.print()
    if WITH_LOG_DATA:
        stats.save()

    if target_file is not None:
        write_json({"w0": x_star}, target_file)

    print(f"Objective value: {data.obj_val}")

    print(x_star)
    check_solution(problem, data, x_star)
    if isinstance(problem.meta, MetaDataOcp):
        meta = problem.meta
        state = to_0d(x_star)[meta.idx_state].reshape(-1, meta.n_state)
        state = np.vstack([meta.initial_state, state])
        control = get_control_vector(problem, data)
        fig, axs = plot_trajectory(state, control, meta, title=problem_name)

        # TODO the next is only a patch for plotting the demand for the double tank problem
        if problem_name == 'doubletank2':
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
