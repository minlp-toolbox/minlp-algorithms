"""An random direction search NLP solver."""

import casadi as ca
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import Tuple
from benders_exp.solvers.nlp import NlpSolver
from benders_exp.solvers import SolverClass, Stats, MinlpProblem, MinlpData, \
    regularize_options
from benders_exp.defines import WITH_JIT, IPOPT_SETTINGS, CASADI_VAR, WITH_LOG_DATA
from benders_exp.utils import to_0d, toc, logging
from copy import deepcopy

logger = logging.getLogger(__name__)


def integer_error(x_int, norm=1):
    """Compute integer error."""
    if norm == 1:
        ret = np.sum(np.abs(np.round(x_int) - x_int))
    else:
        ret = np.linalg.norm(np.round(x_int) - x_int)

    logger.info(f"Integer error {ret:.3f} / {x_int.shape[0]:.3f}")
    return ret


def create_rounded_data(data, idx_x_bin):
    x_var = to_0d(data.x_sol)
    # Round the continuous solution
    x_var[idx_x_bin] = np.round(x_var[idx_x_bin])
    datarounded = deepcopy(data)
    datarounded.prev_solutions[0]['x'] = x_var
    return datarounded


def random_objective_feasibility_pump(
    problem: MinlpProblem, data: MinlpData, stats: Stats, relaxed_solution,
    nlp: NlpSolver, norm=1, recover=True,
) -> Tuple[MinlpData, ca.DM]:
    """
    Random objective FP.

    returns all feasible integer solutions
    returns proble, data, best_solution, is_last_relaxed
    """
    rnlp = RandomDirectionNlpSolver(problem, stats, norm=norm)
    logger.info("Solver initialized.")

    feasible_solutions = []
    best_solution = None
    best_obj = ca.inf
    max_accept_iter = 3
    tolerance = 1e-2
    max_iter = 50
    stats['iterate_data'] = []
    stats['best_itr'] = -1
    done = False
    prev_int_error = ca.inf
    with ThreadPoolExecutor(max_workers=2) as thread:
        while not done:
            logger.info(f"Starting iteration: {stats['iter']}")
            datarounded = create_rounded_data(data, problem.idx_x_bin)
            # Multithreaded
            future_random = thread.submit(rnlp.solve, data)
            future_rounded = thread.submit(
                nlp.solve, datarounded, set_x_bin=True)
            data = future_random.result()
            logger.info(f"Current random NLP objective: {data.obj_val:.3e}")
            datarounded = future_rounded.result()

            if datarounded.solved:
                logger.debug(f"Objective rounded NLP {datarounded.obj_val:.3e} (iter {stats['iter']}) "
                             f"vs old {best_obj:.3e} (itr {stats['best_itr']})")
                feasible_solutions.append(datarounded._sol)
                if best_obj > datarounded.obj_val:
                    logger.info(
                        f"New best objective found in {stats['iter']=}")
                    best_obj = datarounded.obj_val
                    stats['best_itr'] = stats['iter']
                    best_solution = datarounded._sol

            int_error = integer_error(data.x_sol[problem.idx_x_bin])
            stats['iterate_data'].append(stats.create_iter_dict(
                stats['iter'], stats['best_itr'], datarounded.solved,
                ub=best_obj, nlp_obj=datarounded.obj_val, last_benders=None,
                lb=int_error, x_sol=to_0d(datarounded.x_sol))
            )
            if WITH_LOG_DATA:
                stats.save()
            if int_error < tolerance:
                data = nlp.solve(create_rounded_data(
                    data, problem.idx_x_bin), set_x_bin=True)
                best_solution = data._sol

            stats['iter'] += 1
            done = int_error < tolerance or (
                (stats['iter'] > max_accept_iter and best_obj <
                 data.obj_val and prev_int_error < int_error)
            )
            prev_int_error = int_error
            if not data.solved:
                if len(feasible_solutions) > 0:
                    done = True
                elif not recover:
                    return problem, relaxed_solution, relaxed_solution.x_sol, True
                else:
                    # If progress is frozen (unsolvable), try to fix it!
                    data = rnlp.solve(deepcopy(relaxed_solution))
                    logger.info(
                        f"Current random NLP objective (restoration): {data.obj_val:.3e}")
            if stats['iter'] > max_iter:
                raise Exception("Problem can not be solved")

    # Construct nlpdata again!
    data.prev_solutions = [best_solution] + feasible_solutions
    data.solved_all = [True for _ in range(len(feasible_solutions) + 1)]
    return problem, data, data.x_sol, False


def random_direction_rounding_algorithm(
    problem: MinlpProblem, data: MinlpData, stats: Stats,
    norm=1
) -> Tuple[MinlpProblem, MinlpData, ca.DM]:
    """
    Random direction algorithm.

    parameters:
        - norm: 1 for L1-norm based, 2 for squared L2-norm based penalization
    """
    nlp = NlpSolver(problem, stats)
    stats['total_time_loading'] = toc(reset=True)
    data = nlp.solve(data)
    problem, best, x_sol, _ = random_objective_feasibility_pump(
        problem, data, stats, deepcopy(data), nlp)
    stats['total_time_calc'] = toc(reset=True)
    return problem, best, x_sol


class RandomDirectionNlpSolver(SolverClass):
    """
    Create NLP solver.

    This solver solves an NLP problem. This is either relaxed or
    the binaries are fixed.
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None, norm=2, penalty_scaling=0.5):
        """Create NLP problem."""
        super(RandomDirectionNlpSolver, self).__init___(problem, stats)
        options = regularize_options(options, IPOPT_SETTINGS)
        self.penalty_weight = 1.0
        self.penalty_scaling = penalty_scaling

        self.idx_x_bin = problem.idx_x_bin
        x_bin_var = problem.x[self.idx_x_bin]
        penalty = CASADI_VAR.sym("penalty", x_bin_var.shape[0])
        rounded_value = CASADI_VAR.sym("rounded_value", x_bin_var.shape[0])

        self.norm = norm
        self.max_rounding_error = x_bin_var.shape[0]
        if norm == 1:
            penalty_term = ca.sum1(
                ca.fabs(x_bin_var - rounded_value) * penalty)
        else:
            penalty_term = ca.norm_2(
                (x_bin_var - rounded_value) * penalty) ** 2

        options.update({"jit": WITH_JIT, "ipopt.max_iter": 5000})
        self.solver = ca.nlpsol("nlpsol", "ipopt", {
            "f": problem.f + penalty_term,
            "g": problem.g, "x": problem.x,
            "p": ca.vertcat(problem.p, rounded_value, penalty)
        }, options)

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve NLP."""
        success_out = []
        sols_out = []
        for sol in nlpdata.solutions_all:
            lbx = nlpdata.lbx
            ubx = nlpdata.ubx

            x_bin_var = to_0d(sol['x'][self.idx_x_bin])
            x_rounded = np.round(x_bin_var)
            if self.norm == 1:
                rounding_error = ca.norm_1((x_bin_var - x_rounded))
            else:
                rounding_error = ca.norm_2((x_bin_var - x_rounded)) ** 2

            penalty = self.penalty_weight * np.random.rand(len(self.idx_x_bin))
            self.penalty_weight += self.penalty_scaling * abs(nlpdata.obj_val) * (
                rounding_error / self.max_rounding_error + 0.01
            )
            logger.info(
                f"{nlpdata.obj_val=:.3} rounding error={float(rounding_error):.3f} "
                f"- weight {float(self.penalty_weight):.3e}"
            )

            new_sol = self.solver(
                x0=nlpdata.x0,
                lbx=lbx, ubx=ubx,
                lbg=nlpdata.lbg, ubg=nlpdata.ubg,
                p=ca.vertcat(nlpdata.p, x_rounded, penalty)
            )

            success, _ = self.collect_stats("RNLP")
            if not success:
                logger.debug("Infeasible solution!")
            success_out.append(success)
            sols_out.append(new_sol)

        nlpdata.prev_solutions = sols_out
        nlpdata.solved_all = success_out

        return nlpdata
