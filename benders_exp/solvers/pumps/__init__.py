"""
Collection of Pumps.

This folder contains a collection of solvers based on the 'pump' idea. They
include:
    - Feasibility Pump
    - Objective Feasibility Pump
    - Random Objective Feasibility Pump
"""

import numpy as np
import casadi as ca
from typing import Tuple
from copy import deepcopy
from benders_exp.utils import colored
from benders_exp.defines import Settings
from benders_exp.solvers.nlp import NlpSolver
from benders_exp.solvers import Stats, MinlpProblem, MinlpData
from benders_exp.solvers.pumps.fp import LinearProjection, ObjectiveLinearProjection
from benders_exp.solvers.pumps.utils import integer_error, create_rounded_data, \
    perturbe_x, any_equal, random_perturbe_x
from benders_exp.solvers.pumps.random_obj_fp import RandomDirectionNlpSolver
from benders_exp.utils import to_0d, toc, logging


logger = logging.getLogger(__name__)


def feasibility_pump(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings, nlp=None,
    fp=None
) -> Tuple[MinlpProblem, MinlpData, ca.DM, bool]:
    """
    Feasibility Pump

    According to:
        Bertacco, L., Fischetti, M., & Lodi, A. (2007). A feasibility pump
        heuristic for general mixed-integer problems. Discrete Optimization,
        4(1), 63-76.

    returns all feasible integer solutions
    returns problem, data, best_solution, is_last_relaxed
    """
    is_relaxed = nlp is not None
    if not is_relaxed:
        stats['total_time_calc'] += toc(reset=True)
        nlp = NlpSolver(problem, stats, s)

    if fp is None:
        fp = LinearProjection(problem, stats, s)

    stats['total_time_loading'] = toc(reset=True)
    if not is_relaxed:
        data = nlp.solve(data)

    relaxed_value = data.obj_val
    ATTEMPT_TOLERANCE = 0.1
    TOLERANCE = 0.01
    MAX_ITER = 1000
    KK = 5  # Need an integer improvement in 5 steps
    prev_x = []
    distances = [integer_error(data.x_sol[problem.idx_x_bin])]
    while distances[-1] > TOLERANCE and stats['iter'] < MAX_ITER:
        datarounded = create_rounded_data(data, problem.idx_x_bin)
        require_restart = False
        for i, sol in enumerate(datarounded.solutions_all):
            new_x = to_0d(sol['x'])
            perturbe_remaining = 5
            while any_equal(new_x, prev_x, problem.idx_x_bin) and perturbe_remaining > 0:
                new_x = perturbe_x(
                    to_0d(data.solutions_all[i]['x']), problem.idx_x_bin
                )
                perturbe_remaining -= 1

            datarounded.prev_solutions[i]['x'] = new_x
            if perturbe_remaining == 0:
                require_restart = True

            prev_x.append(new_x)

        if not require_restart:
            data = fp.solve(
                datarounded, int_error=distances[-1], obj_val=relaxed_value)
            distances.append(integer_error(data.x_sol[problem.idx_x_bin]))

        if (len(distances) > KK and distances[-KK - 1] < distances[-1]) or require_restart:
            data.prev_solutions[0]['x'] = random_perturbe_x(
                data.x_sol, problem.idx_x_bin)
            data = fp.solve(
                data, int_error=distances[-1], obj_val=relaxed_value)
            distances.append(integer_error(data.x_sol[problem.idx_x_bin]))

        # Added heuristic, not present in the original implementation
        if distances[-1] < ATTEMPT_TOLERANCE:
            datarounded = nlp.solve(create_rounded_data(
                data, problem.idx_x_bin), True)
            if datarounded.solved:
                stats['total_time_calc'] += toc(reset=True)
                return problem, datarounded, datarounded.x_sol

        stats['iter'] += 1
        logger.info(f"Iteration {stats['iter']} finished")

    stats['total_time_calc'] += toc(reset=True)
    data = nlp.solve(data, True)
    return problem, data, data.x_sol


def objective_feasibility_pump(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """
    Objective Feasibility Pump

        Sharma, S., Knudsen, B. R., & Grimstad, B. (2016). Towards an
        objective feasibility pump for convex MINLPs. Computational
        Optimization and Applications, 63, 737-753.
    """
    ofp = ObjectiveLinearProjection(problem, stats, s)
    return feasibility_pump(problem, data, stats, s, fp=ofp)


def random_objective_feasibility_pump(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings,
    relaxed_solution, nlp: NlpSolver, norm=1, recover=True,
) -> Tuple[MinlpData, ca.DM]:
    """
    Random objective FP.

    returns all feasible integer solutions
    returns problem, data, best_solution, is_last_relaxed, lower_bound
    """
    rnlp = RandomDirectionNlpSolver(problem, stats, s, norm=norm)
    logger.info("Solver initialized.")

    feasible_solutions = []
    best_solution = None
    MAX_ACCEPT_ITER = 3
    TOLERANCE = 1e-2
    MAX_ITER = 1000
    MAX_TRY_ITER = 10
    last_restart = 0
    stats['iterate_data'] = []
    stats['best_itr'] = -1
    done = False
    best_obj = ca.inf
    lb = data.obj_val
    prev_int_error = ca.inf
    while not done:
        logger.info(f"Starting iteration: {stats['iter']}")
        data = rnlp.solve(data)
        random_obj_f = float(rnlp.f(data.x_sol, data.p))
        lb = min(random_obj_f, lb)

        colored(f"Current random NLP objective: {random_obj_f:.3e}", "blue")
        if random_obj_f < best_obj:
            datarounded = nlp.solve(create_rounded_data(
                data, problem.idx_x_bin), set_x_bin=True)
            if datarounded.solved:
                logger.debug(f"NLP f={datarounded.obj_val:.3e} (iter {stats['iter']}) "
                             f"vs old f={best_obj:.3e} (itr {stats['best_itr']})")
                feasible_solutions.append(datarounded._sol)
                if best_obj > datarounded.obj_val:
                    best_obj = datarounded.obj_val
                    colored(
                        f"New best f={best_obj:.3e} found in iter={stats['iter']}", "green")
                    stats['best_itr'] = stats['iter']
                    best_solution = datarounded._sol
            else:
                colored("Infeasible")
        else:
            colored("Not better than best found yet")

        int_error = integer_error(data.x_sol[problem.idx_x_bin])
        stats['iterate_data'].append(stats.create_iter_dict(
            stats['iter'], stats['best_itr'], datarounded.solved,
            ub=best_obj, nlp_obj=datarounded.obj_val, last_benders=None,
            lb=int_error, x_sol=to_0d(datarounded.x_sol))
        )
        if s.WITH_LOG_DATA:
            stats.save()

        stats['iter'] += 1
        done = (
            int_error < TOLERANCE or (not np.isinf(best_obj) and (
                (stats['iter'] > MAX_ACCEPT_ITER and best_obj < random_obj_f)
                or stats['iter'] > MAX_ACCEPT_ITER + stats['best_itr']
            ))
        )
        retry = (stats['iter'] - last_restart > MAX_TRY_ITER and prev_int_error < int_error)
        prev_int_error = int_error
        if not data.solved or retry:
            if len(feasible_solutions) > 0:
                done = True
            elif not recover:
                return problem, relaxed_solution, relaxed_solution.x_sol, True, lb
            else:
                last_restart = stats['iter']
                rnlp.alpha = 1.0
                # If progress is frozen (unsolvable), try to fix it!
                data = rnlp.solve(deepcopy(relaxed_solution))
                logger.info(
                    f"Current random NLP (restoration): f={random_obj_f:.3e}")
        if stats['iter'] > MAX_ITER:
            if len(feasible_solutions) > 0:
                done = True
            else:
                return problem, data, None, False, lb

    # Construct nlpdata again!
    data.prev_solutions = [best_solution] + feasible_solutions
    data.solved_all = [True for _ in range(len(feasible_solutions) + 1)]
    return problem, data, best_solution['x'], False, lb


def random_direction_rounding_algorithm(
    problem: MinlpProblem, data: MinlpData, stats: Stats, s: Settings,
    norm=1
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """
    Random direction algorithm.

    parameters:
        - norm: 1 for L1-norm based, 2 for squared L2-norm based penalization
    """
    nlp = NlpSolver(problem, stats, s)
    stats['total_time_loading'] = toc(reset=True)
    data = nlp.solve(data)
    problem, best, x_sol, relaxed, lb = random_objective_feasibility_pump(
        problem, data, stats, s, data, nlp)
    if x_sol is None or relaxed:
        raise Exception("Problem can not be solved")
    stats['total_time_calc'] = toc(reset=True)
    return problem, best, x_sol
