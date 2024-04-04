# Adrian Buerger, Andrea Ghezzi 2022

import copy
import numpy as np

from benders_exp.solarsys.nlpsolver import NLPSolverBin
from benders_exp.solarsys.system import System
from benders_exp.solarsys.timing import TimingMPC

import logging

logger = logging.getLogger(__name__)


class Voronoi(System):
    @property
    def termination_criterion_reached(self) -> bool:
        return self._termination_criterion_reached

    @property
    def is_initialized(self) -> bool:
        return self._A_v is not None

    @property
    def A_v(self) -> np.array:
        return self._A_v

    @property
    def lb_v(self) -> float:
        return self._lb_v

    @property
    def ub_v(self) -> float:
        return self._ub_v

    def _initialize_not_improving_iteration_counter(self) -> None:

        self._not_improving_iter = None
        self._MAX_NOT_IMPROVING_ITER = 10

    def _initialize_termination_criterion_reached_flag(self) -> None:

        self._termination_criterion_reached = False

    def _define_cut_attributes(self) -> None:

        self._A_v = None
        self._lb_v = None
        self._ub_v = None

    def _initialize_list_binary_solutions(self) -> None:

        self._binary_solutions = []

    def _define_optimal_solution_attributes(self) -> None:

        self._optimal_binary_solution = None
        self._minimum_nlp_objective_value = None

    def __init__(self) -> None:

        # initialize solution containers etc. here
        super().__init__()
        self._initialize_not_improving_iteration_counter()
        self._initialize_termination_criterion_reached_flag()
        self._initialize_list_binary_solutions()
        self._define_optimal_solution_attributes()
        self._define_cut_attributes()

    def set_weight_matrix(self, timing: TimingMPC, weight_type: str) -> None:

        if (weight_type != "CIA") & (weight_type != "ID"):
            logger.error("Allowed weighting types are CIA or ID")
            raise AttributeError("Allowed weighting types are CIA or ID")

        time_steps_np = copy.deepcopy(timing.time_steps.to_numpy())
        sqrt_W = np.dstack(
            [time_steps_np * np.tri(timing.N, timing.N) for _ in range(self.nb)]
        ).reshape(timing.N, timing.N * self.nb)
        if weight_type == "CIA":
            self._W = sqrt_W.T @ sqrt_W
            self._W = 1 / np.amax(self._W) * self._W
        elif weight_type == "ID":
            self._W = np.diag(np.ones(timing.N * self.nb))

    def add_solution(self, current_nlp_solution: NLPSolverBin) -> None:

        self._b_data = copy.deepcopy(current_nlp_solution.b_data).flatten()
        self._nlp_objective_value = copy.deepcopy(
            current_nlp_solution.nlp_objective_value
        )

    def _compute_cut_restricted(self) -> None:

        b_opt_norm2 = (
            self._optimal_binary_solution.T @ self._W @ self._optimal_binary_solution
        )

        if not np.array_equal(self._b_data, self._optimal_binary_solution):

            new_row_A = 2 * self._W @ (self._b_data - self._optimal_binary_solution)
            new_row_B = (self._b_data.T @ self._W @ self._b_data) - b_opt_norm2

            # Assign correct sign to the inequality so that the linearization point is feasible
            if new_row_A.T @ self._optimal_binary_solution <= new_row_B:
                A = new_row_A[np.newaxis, ...]
                lb = np.array([-np.inf])
                ub = np.array([new_row_B])
            else:
                A = -new_row_A[np.newaxis, ...]
                lb = np.array([-np.inf])
                ub = np.array([-new_row_B])

            if self._A_v is None:
                self._A_v = copy.deepcopy(A)
                self._lb_v = copy.deepcopy(lb)
                self._ub_v = copy.deepcopy(ub)
            else:
                self._A_v = copy.deepcopy(np.vstack([self._A_v, A]))
                self._lb_v = copy.deepcopy(np.concatenate([self._lb_v, lb]))
                self._ub_v = copy.deepcopy(np.concatenate([self._ub_v, ub]))

    def _compute_cuts_standard(self) -> None:

        A = []
        lb = []
        ub = []

        b_opt_norm2 = (
            self._optimal_binary_solution.T @ self._W @ self._optimal_binary_solution
        )
        for binary_sol in self._binary_solutions:
            if not np.array_equal(binary_sol, self._optimal_binary_solution):

                new_row_A = 2 * self._W @ (binary_sol - self._optimal_binary_solution)
                new_row_B = (binary_sol.T @ self._W @ binary_sol) - b_opt_norm2

                # Assign correct sign to the inequality so that the linearization point is feasible
                if new_row_A.T @ self._optimal_binary_solution <= new_row_B:
                    A.append(new_row_A)
                    lb.append(-np.inf)
                    ub.append(new_row_B)
                else:
                    A.append(-new_row_A)
                    lb.append(-np.inf)
                    ub.append(-new_row_B)

        if A:
            self._A_v = np.array(A).copy()
            self._lb_v = np.array(lb).copy()
            self._ub_v = np.array(ub).copy()

    def compute_cuts(self, strategy: str) -> None:

        if (strategy != "STD") & (strategy != "RES"):
            logger.error("Allowed strategy to compute the Voronoi cuts are STD or RES")
            raise AttributeError(
                "Allowed strategy to compute the Voronoi cuts are STD or RES"
            )

        if strategy == "STD":
            self._compute_cuts_standard()
        elif strategy == "RES":
            self._compute_cut_restricted()

    def update_optimal_solution_so_far(self) -> None:

        if self._optimal_binary_solution is None:
            self._optimal_binary_solution = self._b_data
            self._minimum_nlp_objective_value = self._nlp_objective_value
            self._not_improving_iter = 0
        else:
            if self._nlp_objective_value < self._minimum_nlp_objective_value:
                self._optimal_binary_solution = self._b_data
                self._minimum_nlp_objective_value = self._nlp_objective_value
                self._not_improving_iter = 0
            else:
                self._not_improving_iter += 1

        logger.warning(f"Minimum NLP objective: {self._minimum_nlp_objective_value}")
        logger.warning(f"Current NLP objective: {self._nlp_objective_value}")
        logger.warning(
            f"Consecutive not improving iter: {self._not_improving_iter}/{self._MAX_NOT_IMPROVING_ITER}"
        )

    def update_termination_criterion(self) -> None:

        if self._binary_solutions:
            if np.array_equal(self._binary_solutions[-1], self._b_data):
                self._termination_criterion_reached = True
                logger.warning(f"Termination condition on binary variables reached!")
            elif self._not_improving_iter >= self._MAX_NOT_IMPROVING_ITER:
                self._termination_criterion_reached = True
                logger.warning(
                    f"Termination condition on not improving iterations reached!"
                )

        self._append_current_solution()  # NOTE: find the best place to do this!!

    def _append_current_solution(self):

        # append current binary solution value in memory

        self._binary_solutions += [self._b_data]
