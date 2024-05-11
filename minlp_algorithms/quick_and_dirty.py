"""Quick and dirty implementation."""

from os import path
from datetime import datetime
import matplotlib.pyplot as plt
from sys import argv
from typing import Callable, Tuple, Union
import casadi as ca
import numpy as np
from minlp_algorithms.utils import get_control_vector, plot_trajectory, tic, toc, \
    make_bounded, setup_logger, logging, colored
from minlp_algorithms.utils.conversion import to_0d
from minlp_algorithms.utils.data import write_json, read_json
from minlp_algorithms.settings import Settings, GlobalSettings
from minlp_algorithms.problems.overview import PROBLEMS
from minlp_algorithms.problems import MinlpData, MinlpProblem, MetaDataOcp, MetaDataMpc
from minlp_algorithms.solvers import MiSolverClass, Stats
from minlp_algorithms.solvers.subsolvers.nlp import NlpSolver
from minlp_algorithms.solvers.subsolvers.fnlp import FeasibilityNlpSolver
from minlp_algorithms.solvers.subsolvers.fnlp_closest import FindClosestNlpSolver
from minlp_algorithms.solvers.decomposition.benders_master import BendersMasterMILP, BendersTrustRegionMIP, \
        BendersMasterMIQP
from minlp_algorithms.solvers.decomposition.sequential_benders_trustregion_master import BendersTRandMaster
from minlp_algorithms.solvers.decomposition.oa_master import OuterApproxMILP, OuterApproxMILPImproved
from minlp_algorithms.solvers.external.bonmin import BonminSolver
from minlp_algorithms.solvers.decomposition.voronoi_master import VoronoiTrustRegionMILP
from minlp_algorithms.solvers.pumps import random_direction_rounding_algorithm, random_objective_feasibility_pump, \
    feasibility_pump, objective_feasibility_pump
from minlp_algorithms.solvers.approximation.cia import cia_decomposition_algorithm
from minlp_algorithms.solvers.milp_tr import milp_tr
from minlp_algorithms.solvers.decomposition.benders_lbqp_master import BendersTRLB
from minlp_algorithms.utils.debugtools import CheckNoDuplicate
from minlp_algorithms.utils.validate import check_solution
from minlp_algorithms.solvers.utils import get_termination_condition

logger = logging.getLogger(__name__)


# TODO: Already moved
def update_best_solutions(data, itr, ub, x_star, best_iter, s: Settings):
    """Update best solutions,"""
    if np.any(data.solved_all):
        for i, success in enumerate(data.solved_all):
            obj_val = float(data.prev_solutions[i]['f'])
            if success:
                if obj_val + s.EPS < ub:
                    logger.info(f"Decreased UB from {ub} to {obj_val}")
                    ub = obj_val
                    x_star = data.prev_solutions[i]['x']
                    data.best_solutions.append(x_star)
                    best_iter = itr
                elif obj_val - s.EPS < ub:
                    data.best_solutions.append(
                        data.prev_solutions[i]['x']
                    )
    return ub, x_star, best_iter


# TODO: Already moved
def base_strategy(problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings,
                  master_problem: MiSolverClass, termination_condition: Callable[..., bool],
                  first_relaxed=False) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Run the base strategy."""
    logger.info("Setup NLP solver...")
    nlp = NlpSolver(problem, stats, s)
    toc()
    logger.info("Setup FNLP solver...")
    fnlp = FeasibilityNlpSolver(problem, data, stats, s)
    stats['total_time_loading'] = toc(reset=True)
    logger.info("Solver initialized.")
    # Benders algorithm
    lb = -ca.inf
    ub = ca.inf
    feasible = True
    best_iter = -1
    x_star = np.nan * np.empty(problem.x.shape[0])
    x_hat = -np.nan * np.empty(problem.x.shape[0])
    check = CheckNoDuplicate(problem, s)

    if first_relaxed:
        data = nlp.solve(data)
        data = master_problem.solve(data, relaxed=True)

    while (not termination_condition(stats, s, lb, ub, x_star, x_hat)) and feasible:
        check(data)
        # Solve NLP(y^k)
        data = nlp.solve(data, set_x_bin=True)
        prev_feasible = data.solved

        # Is there a feasible success?
        ub, x_star, best_iter = update_best_solutions(
            data, stats['iter_nr'], ub, x_star, best_iter, s
        )

        # Is there any infeasible?
        if not np.all(data.solved_all):
            # Solve NLPF(y^k)
            data = fnlp.solve(data)
            logger.info("Infeasibility problem solved")

        # Solve master^k and set lower bound:
        data = master_problem.solve(data, prev_feasible=prev_feasible)
        feasible = data.solved
        lb = data.obj_val
        x_hat = data.x_sol
        logger.debug(f"\n{x_hat=}")
        logger.debug(f"{ub=}, {lb=}\n")
        stats['iter_nr'] += 1

    stats['total_time_calc'] = toc(reset=True)
    data.prev_solution = {'x': x_star, 'f': ub}
    return problem, data, x_star


# TODO: Already moved
def benders_algorithm(problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings,
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
        benders_master = BendersMasterMIQP(problem, data, stats, s)
    else:
        logger.info("Setup Benders MILP solver...")
        # This class implements the original benders decomposition
        benders_master = BendersMasterMILP(problem, data, stats, s)

    termination_condition = get_termination_condition(
        termination_type, problem, data, s
    )
    toc()
    return base_strategy(problem, data, stats, s, benders_master, termination_condition)


def export_ampl(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Export AMPL."""
    from minlp_algorithms.solvers.ampl import AmplSolver
    AmplSolver(problem, stats, s).solve(data)
    return problem, data, data.x0


def outer_approx_algorithm(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings,
    termination_type: str = 'std'
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """
    Create and run outer approximation.

    parameters:
        - termination_type:
            - std: based on lower and upper bound
    """
    logger.info("Setup OA MILP solver...")
    outer_approx = OuterApproxMILP(problem, data, stats, s)
    termination_condition = get_termination_condition(
        termination_type, problem, data, s
    )
    toc()
    return base_strategy(problem, data, stats, s, outer_approx, termination_condition, first_relaxed=True)


def outer_approx_algorithm_improved(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings,
    termination_type: str = 'std'
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """
    Create and run improved outer approximation.

    parameters:
        - termination_type:
            - std: based on lower and upper bound
    """
    logger.info("Setup OAI MILP solver...")
    outer_approx = OuterApproxMILPImproved(problem, data, stats, s)
    termination_condition = get_termination_condition(
        termination_type, problem, data, s
    )
    toc()
    return base_strategy(problem, data, stats, s, outer_approx, termination_condition)


def benders_constrained_milp(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings, termination_type: str = 'std',
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
    benders_tr_master = BendersTrustRegionMIP(problem, data, stats, s)
    termination_condition = get_termination_condition(
        termination_type, problem, data, s
    )
    toc()
    return base_strategy(problem, data, stats, s, benders_tr_master, termination_condition, first_relaxed=first_relaxed)


def benders_trm_lp(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings,
    termination_type: str = 'std'
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """
    Create and run benders TRM as LB.

    parameters:
        - termination_type:
            - gradient: based on local linearization
            - equality: the binaries of the last solution coincides with the ones of the best solution
            - std: based on lower and upper bound
    """
    logger.info("Setup Idea MIQP solver...")
    benders_tr_master = BendersTRLB(problem, data, stats, s)
    termination_condition = get_termination_condition(
        termination_type, problem, data, s
    )
    toc()
    return base_strategy(problem, data, stats, s, benders_tr_master, termination_condition, first_relaxed=True)


def relaxed(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Create and the relaxed problem."""
    nlp = NlpSolver(problem, stats, s)
    stats['total_time_loading'] = toc(reset=True)
    data = nlp.solve(data)
    stats["x_sol"] = to_0d(data.x_sol)
    stats["obj_sol"] = float(data.obj_val)
    stats['total_time_calc'] = toc(reset=True)
    return problem, data, data.x_sol


def bonmin(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings,
    algo_type="B-BB"
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Create benders algorithm."""
    logger.info("Create bonmin.")
    minlp = BonminSolver(problem, stats, s, algo_type=algo_type)
    stats['total_time_loading'] = toc(reset=True)
    data = minlp.solve(data)
    stats['total_time_calc'] = toc(reset=True)
    stats['x_sol'] = to_0d(data.x_sol)
    stats["obj_sol"] = float(data.obj_val)
    return problem, data, data.x_sol


def voronoi_tr_algorithm(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings,
    termination_type: str = 'gradient'
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
    voronoi_tr_master = VoronoiTrustRegionMILP(problem, data, stats, s)
    termination_condition = get_termination_condition(
        termination_type, problem, data, s
    )
    toc()
    return base_strategy(problem, data, stats, s, voronoi_tr_master, termination_condition)


def benders_tr_master(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings, termination_type: str = 'std',
    first_relaxed=True, use_feasibility_pump=True, with_benders_master=True,
    with_new_inf=False, early_exit=False
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """Run the base strategy."""
    check = CheckNoDuplicate(problem, s)
    termination_condition = get_termination_condition(
        termination_type, problem, data, s
    )
    logger.info("Setup Mixed Benders TR/Master")
    master_problem = BendersTRandMaster(
        problem, data, stats, s,
        with_benders_master=with_benders_master, early_exit=early_exit
    )
    toc()
    logger.info("Setup NLP solver...")
    nlp = NlpSolver(problem, stats, s)
    toc()
    logger.info("Setup FNLP solver...")
    if with_new_inf:
        fnlp = FindClosestNlpSolver(problem, stats, s)
    else:
        fnlp = FeasibilityNlpSolver(problem, data, stats, s)
    stats['iter_nr'] = 0
    stats["lb"] = -ca.inf
    stats["ub"] = ca.inf
    feasible = True
    x_star = np.nan * np.empty(problem.x.shape[0])
    x_hat = -np.nan * np.empty(problem.x.shape[0])
    stats["last_benders"] = True  # only for doing at least one iteration of the while-loop
    termination_met = False
    old_sol = x_star[problem.idx_x_bin]

    if use_feasibility_pump:
        data = nlp.solve(data)
        master_problem.update_relaxed_solution(data)
        problem, data, _, is_relaxed, stats["lb"] = random_objective_feasibility_pump(
            problem, data, stats, s, data, nlp)
        if not is_relaxed:
            stats["ub"], x_star, stats["best_iter"] = update_best_solutions(
                data, 0, stats["ub"], x_star, stats["best_iter"], s
            )

        data, stats["last_benders"] = master_problem.solve(data, relaxed=is_relaxed)
    elif first_relaxed:
        stats['total_time_loading'] = toc(reset=True)
        logger.info("Solver initialized.")
        data = nlp.solve(data)
        stats["lb"] = data.obj_val
        stats["solutions_all"] = data.solutions_all
        stats["solved_all"] = data.solved_all
        data, stats["last_benders"] = master_problem.solve(data, relaxed=True)
        stats["mip_solutions_all"] = data.solutions_all
        stats["mip_solved_all"] = data.solved_all
        if s.WITH_LOG_DATA:
            stats.save()
    else:
        stats['total_time_loading'] = toc(reset=True)
        logger.info("Solver initialized.")

    try:
        while feasible and not termination_met:
            check(data)
            # Solve NLP(y^k)
            data = nlp.solve(data, set_x_bin=True)
            logger.info("SOLVED NLP")

            if not np.all(data.solved_all):
                # Solve NLPF(y^k)
                data = fnlp.solve(data)
                logger.info("SOLVED FEASIBILITY NLP")

            stats["ub"], x_star, stats["best_iter"] = update_best_solutions(
                data, stats['iter_nr'], stats["ub"], x_star, stats["best_iter"], s
            )

            if s.WITH_DEBUG:
                if np.allclose(old_sol, to_0d(data.x_sol)[[problem.idx_x_bin]]):
                    colored("Possible error!")
                else:
                    colored("All ok", "green")

            old_sol = to_0d(data.x_sol)[problem.idx_x_bin]
            logger.info(f"Adding {data.nr_sols} solutions")
            logger.debug(f"NLP {data.obj_val=}, {stats['ub']=}, {stats['lb']=}")
            stats["solutions_all"] = data.solutions_all
            stats["solved_all"] = data.solved_all
            if s.WITH_LOG_DATA:
                stats.save()

            # Solve master^k and set lower bound:
            data, stats["last_benders"] = master_problem.solve(data)
            stats["mip_solutions_all"] = data.solutions_all
            stats["mip_solved_all"] = data.solved_all
            if stats["last_benders"]:
                if stats["ub"] < stats["lb"]:
                    # Problems!
                    stats["lb"] = data.obj_val
                else:
                    stats["lb"] = max(data.obj_val, stats["lb"])
            logger.debug(f"MIP {data.obj_val=}, {stats['ub']=}, {stats['lb']=}")

            x_hat = data.x_sol
            stats['iter_nr'] += 1

            feasible = data.solved
            termination_met = termination_condition(
                stats, s, stats["lb"], stats["ub"], data.best_solutions, x_hat
            )

        stats['total_time_calc'] = toc(reset=True)

    except KeyboardInterrupt:
        exit(0)

    data.prev_solution = {'x': x_star, 'f': stats["ub"]}
    return problem, data, x_star


def test(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    nlp = NlpSolver(problem, stats, s)
    x_sol = nlp.solve(data).x_sol
    f = ca.Function("f", [problem.x, problem.p], [problem.f])
    df = ca.Function("df", [problem.x, problem.p], [ca.gradient(
        problem.f, problem.x
    )])(x_sol, data.p)
    results = []
    for i in problem.idx_x_bin:
        if np.floor(x_sol[i]) != np.ceil(x_sol[i]):
            x = x_sol.full()
            x[i] = np.floor(x_sol[i])
            lb = float(f(x, data.p))
            x[i] = np.ceil(x_sol[i])
            ub = float(f(x, data.p))
            results.append([lb, ub])
            print(f"{i}: {lb} - {ub} ({x_sol[i]}) - df {df[i]}")
    return problem, data, data.x_sol


SOLVER_MODES = {
    "benders": benders_algorithm,
    "bendersqp": lambda p, d, st, s: benders_algorithm(
        p, d, st, s, with_qp=True
    ),
    "benders_old_tr": lambda p, d, st, s: benders_constrained_milp(
        p, d, st, s, termination_type='std', first_relaxed=False
    ),
    "benders_old_tr_rel": lambda p, d, st, s: benders_constrained_milp(
        p, d, st, s, termination_type='std', first_relaxed=True
    ),
    "benders_tr": lambda p, d, st, s: benders_tr_master(
        p, d, st, s, termination_type='equality',
        use_feasibility_pump=False, with_benders_master=False
    ),
    "benders_tri": lambda p, d, st, s: benders_tr_master(
        p, d, st, s, termination_type='equality',
        use_feasibility_pump=False, with_benders_master=False, with_new_inf=True
    ),
    "benders_tr_fp": lambda p, d, st, s: benders_tr_master(
        p, d, st, s, termination_type='equality',
        use_feasibility_pump=True, with_benders_master=False
    ),
    "benders_trm": lambda p, d, st, s: benders_tr_master(
        p, d, st, s, use_feasibility_pump=False, with_benders_master=True
    ),
    "benders_trmi": lambda p, d, st, s: benders_tr_master(
        p, d, st, s, use_feasibility_pump=False, with_benders_master=True, with_new_inf=True
    ),
    "benders_trmi+": lambda p, d, st, s: benders_tr_master(
        p, d, st, s, use_feasibility_pump=False, with_benders_master=True, with_new_inf=True,
        early_exit=True
    ),
    "benders_trmi_fp": lambda p, d, st, s: benders_tr_master(
        p, d, st, s, use_feasibility_pump=True, with_benders_master=True, with_new_inf=True
    ),
    "benders_trm_fp": lambda p, d, st, s: benders_tr_master(p, d, st, s, use_feasibility_pump=True,
                                                            with_benders_master=True),
    "benders_trl": benders_trm_lp,
    "oa": outer_approx_algorithm,
    "oai": outer_approx_algorithm_improved,
    "bonmin": bonmin,
    # B-BB is a NLP-based branch-and-bound algorithm
    "bonmin-bb": lambda p, d, st, s: bonmin(p, d, st, s, "B-BB"),
    # B-Hyb is a hybrid outer-approximation based branch-and-cut algorithm
    "bonmin-hyb": lambda p, d, st, s: bonmin(p, d, st, s, "B-Hyb"),
    # B-OA is an outer-approximation decomposition algorithm
    "bonmin-oa": lambda p, d, st, s: bonmin(p, d, st, s, "B-OA"),
    # B-QG is an implementation of Quesada and Grossmann's branch-and-cut algorithm
    "bonmin-qg": lambda p, d, st, s: bonmin(p, d, st, s, "B-QG"),
    # B-iFP: an iterated feasibility pump algorithm
    "bonmin-ifp": lambda p, d, st, s: bonmin(p, d, st, s, "B-iFP"),
    "voronoi_tr": lambda p, d, st, s: voronoi_tr_algorithm(p, d, st, s, termination_type='equality'),
    "relaxed": relaxed,
    "ampl": export_ampl,
    "rofp": random_direction_rounding_algorithm,
    "fp": feasibility_pump,
    "ofp": objective_feasibility_pump,
    "cia": cia_decomposition_algorithm,
    "milp_tr": milp_tr,
    "test": test
}


# TODO: Already moved
def run_problem(mode_name, problem_name, stats, args, s=None) -> Union[MinlpProblem, MinlpData, ca.DM]:
    """Run a problem and return the results."""
    if mode_name not in SOLVER_MODES:
        raise Exception(
            f"No mode {mode_name=}, available {SOLVER_MODES.keys()}")
    if problem_name not in PROBLEMS:
        raise Exception(f"No {problem_name=}, available: {PROBLEMS.keys()}")

    logger.info(f"Load problem {problem_name}")
    output = PROBLEMS[problem_name](*args)
    if len(output) == 2:
        problem, data = output
        if s is None:
            s = Settings()
        logger.info("Using default settings")
    else:
        logger.info("Using custom settings")
        problem, data, s = output

    if problem == "orig":
        new_inf = 1e5
    else:
        new_inf = 1e3
    make_bounded(problem, data, new_inf=new_inf)

    if len(problem.idx_x_bin) == 0:
        mode_name = "relaxed"

    logger.info(f"Start mode {mode_name}")
    return SOLVER_MODES[mode_name](problem, data, stats, s), s


# TODO: Already moved
def batch_nl_runner(mode_name, target, nl_files):
    """Run a batch of problems."""
    from os import makedirs
    from time import time

    def do_write(overview_target, start, i, mode_name, total_stats):
        time_now = time() - start
        total_time = time_now / (i + 1) * total_to_compute
        write_json({
            "time": time_now,
            "total": total_to_compute,
            "done": (i+1),
            "progress": (i+1) / total_to_compute,
            "time_remaining_est": total_time - time_now,
            "time_total_est": total_time,
            "mode": mode_name,
            "data": total_stats
        }, overview_target)

    overview_target = path.join(target, "overview.json")
    start = time()
    total_to_compute = len(nl_files)
    if path.exists(overview_target):
        data = read_json(overview_target)
        total_stats = data['data']
        mode_name = data["mode"]
        i_start = data['done']
        start -= data["time"]
        if total_stats[-1][0] != i_start:
            total_stats.append([
                i_start,
                nl_files[i_start],
                -ca.inf, "FAILED", "CRASH"
            ])
        do_write(overview_target, start, i_start, mode_name, total_stats)
        i_start += 1
    else:
        makedirs(target, exist_ok=True)
        total_stats = [["id", "path", "obj",
                        "load_time", "calctime",
                        "solvertime", "iter_nr", "nr_int"]]
        i_start = 0

    for i in range(i_start, len(nl_files)):
        nl_file = nl_files[i]
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            tic()
            stats = Stats(mode_name, "nl_file", timestamp, {})
            toc()
            (problem, data, x_star), s = run_problem(
                mode_name, "nl_file", stats, [nl_file]
            )
            stats["x_star"] = x_star
            stats["f_star"] = data.obj_val
            total_stats.append([
                i, nl_file, data.obj_val,
                stats["total_time_loading"], stats["total_time_calc"],
                stats['t_solver_total'], stats["iter_nr"], len(problem.idx_x_bin)
            ])
        except Exception as e:
            print(f"{e}")
            total_stats.append(
                [i, nl_file, -ca.inf, "FAILED", f"{e}"]
            )
        stats.print()
        stats.save(path.join(target, f"stats_{i}.pkl"))
        do_write(overview_target, start, i, mode_name, total_stats)


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

    (problem, data, x_star), s = run_problem(
        mode, problem_name, stats, extra_args)
    stats.print()
    if s.WITH_LOG_DATA:
        stats.save()

    if target_file is not None:
        if isinstance(problem.meta, MetaDataOcp) and problem.meta.dump_solution is not None:
            output_data = problem.meta.dump_solution(x_star)
        else:
            output_data = x_star

        write_json({"w0": np.array(output_data).tolist()}, target_file)

    print(f"Objective value: {data.obj_val}")

    # TODO: Plot not yet moved!
    print(x_star)
    if isinstance(problem.meta, MetaDataOcp):
        meta = problem.meta
        state = to_0d(x_star)[meta.idx_state].reshape(-1, meta.n_state)
        state = np.vstack([meta.initial_state, state])
        control = get_control_vector(problem, data)
        fig, axs = plot_trajectory(
            to_0d(x_star), state, control, meta, title=problem_name)

        # TODO the next is only a patch for plotting the demand for the double tank problem
        if problem_name == 'doubletank2':
            time_array = np.linspace(
                0, meta.dt * state.shape[0], state.shape[0] + 1)
            demand = np.array([2 + 0.5 * np.sin(x) for x in time_array])
            axs[1].plot(time_array, demand, "r--", alpha=0.5)

        uptime = problem.meta.min_uptime
        fig.savefig(
            f"{GlobalSettings.IMG_DIR}/ocp_trajectory_{mode}_uptime_{uptime}.pdf", bbox_inches='tight')
        plt.show()
    elif isinstance(problem.meta, MetaDataMpc):
        problem.meta.plot(data, x_star)

    check_solution(problem, data, x_star, s)
    if s.WITH_PLOT:
        plt.show()