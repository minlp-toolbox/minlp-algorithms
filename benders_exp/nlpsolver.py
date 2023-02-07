# Adrian Buerger, 2022

import os
import copy
import datetime as dt
import pytz
import numpy as np
import casadi as ca

from system import System
from benders_exp.nlpsetup import NLPSetupMPC
from benders_exp.defines import _PATH_TO_NLP_OBJECT, NLP_OPTIONS_GENERAL

from abc import ABCMeta, abstractmethod

import logging

logger = logging.getLogger(__name__)


class NLPSolverMPCBaseClass(NLPSetupMPC, metaclass=ABCMeta):

    _SECONDS_TIMEOUT_BEFORE_NEXT_TIME_GRID_POINT = 5.0
    _LOGFILE_LOCATION = "/tmp/voronoi-logs/"
    if not os.path.exists(_LOGFILE_LOCATION):
        os.mkdir(_LOGFILE_LOCATION)

    _INDICATOR_VARIABLE_THRESHOLD = 1e-3

    @property
    def solve_successful(self):

        try:
            return self._nlpsolver.stats()["return_status"] in [
                "Solve_Succeeded",
                "Solved_To_Acceptable_Level",
            ]
        except AttributeError:
            msg = "Solver return status not available yet, call solve() first."
            logger.error(msg)
            raise RuntimeError(msg)

    @property
    def time_points(self):
        return self._timing.time_points

    @property
    def time_grid(self):
        return self._timing.time_grid

    @property
    def solver_type(self):
        return self._SOLVER_TYPE

    @property
    def x_data(self):
        try:
            return np.asarray(self._x_data)
        except AttributeError:
            msg = "Optimized states not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    @property
    def x_hat(self):
        return np.asarray(self._predictor.x_hat)

    @property
    def u_data(self):
        try:
            return np.asarray(self._u_data)
        except AttributeError:
            msg = "Optimized continuous controls not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    @property
    def b_data(self):
        try:
            return np.asarray(self._b_data)
        except AttributeError:
            msg = "Optimized binary controls not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    @property
    def c_data(self):
        return np.asarray(self._ambient.c_data)

    @property
    def s_ac_lb_data(self):
        try:
            return np.asarray(self._s_ac_lb_data)
        except AttributeError:
            msg = "Optimized slacks for minimum AC operation temperatures not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    @property
    def s_ac_ub_data(self):
        try:
            return np.asarray(self._s_ac_ub_data)
        except AttributeError:
            msg = "Optimized slacks for maximum AC operation temperatures not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    @property
    def s_x_data(self):
        try:
            return np.asarray(self._s_x_data)
        except AttributeError:
            msg = "Optimized slacks for soft state constraints not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    @property
    def s_ppsc_data(self):
        try:
            return np.asarray(self._s_ppsc_data)
        except AttributeError:
            msg = "Optimized slacks for collector safety pump speed not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    @property
    def max_switches(self):
        return self._previous_solver.max_switches

    @property
    def solver_name(self):
        return self._solver_name

    @property
    def nlp_objective_value(self):
        try:
            return float(self.nlp_solution["f"])
        except AttributeError:
            msg = "Objective value not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    @property
    def solver_wall_time(self):
        try:
            return self._solver_wall_time
        except AttributeError:
            msg = "NLP wall time not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    def _setup_timing(self, timing):

        self._timing = timing

    def _setup_ambient(self, ambient):

        self._ambient = ambient

    def _set_previous_solver(self, previous_solver):

        self._previous_solver = previous_solver

    def _set_predictor(self, predictor):

        self._predictor = predictor

    def _setup_solver_name(self, solver_name):

        self._solver_name = solver_name

    def _setup_general_solver_options(self):

        self._nlpsolver_options = {}
        self._nlpsolver_options["ipopt.output_file"] = os.path.join(
            self._LOGFILE_LOCATION, self._solver_name + ".log"
        )
        self._nlpsolver_options.update(NLP_OPTIONS_GENERAL)

    @abstractmethod
    def _setup_additional_nlpsolver_options(self):

        pass

    def __init__(self, timing, ambient, previous_solver, predictor, solver_name):

        logger.debug(f"Initializing NLP solver {solver_name} ...")

        super().__init__(timing)

        self._setup_timing(timing=timing)
        self._setup_ambient(ambient=ambient)

        self._setup_solver_name(solver_name=solver_name)
        self._set_previous_solver(previous_solver=previous_solver)
        self._set_predictor(predictor=predictor)

        self._setup_general_solver_options()
        self._setup_additional_nlpsolver_options()
        self._setup_collocation_options()

        logger.debug("NLP solver {solver_name} initialized.")

    def set_solver_max_cpu_time(self, time_point_to_finish):

        max_cpu_time = (
            time_point_to_finish
            - dt.datetime.now(tz=pytz.utc)
            - dt.timedelta(seconds=self._SECONDS_TIMEOUT_BEFORE_NEXT_TIME_GRID_POINT)
        ).total_seconds()

        self._nlpsolver_options["ipopt.max_cpu_time"] = max_cpu_time

        logger.debug(
            f"Maximum CPU time for {self._solver_name} set to {max_cpu_time} s ..."
        )

    def _store_previous_binary_solution(self):

        self._b_data_prev = copy.deepcopy(self._previous_solver.b_data)

    def _setup_nlpsolver(self):

        path_to_nlp_object = os.path.join(
            _PATH_TO_NLP_OBJECT, self._NLP_OBJECT_FILENAME
        )

        self._nlpsolver = ca.nlpsol(
            self._solver_name, "ipopt", path_to_nlp_object, self._nlpsolver_options
        )

    def _set_states_bounds(self):

        """
        The boundary values for the states will later be defined as soft constraints.
        """

        self.x_min = self.p_op["T"]["min"] * np.ones(
            (self._timing.N + 1, self.nx - self.nx_aux)
        )
        self.x_max = self.p_op["T"]["max"] * np.ones(
            (self._timing.N + 1, self.nx - self.nx_aux)
        )

        self.x_max[:, self.x_index["T_shx_psc"][-1]] = self.p_op["T_sc"]["T_feed_max"]

        self.x_max[:, self.x_index["T_lts"]] = self.p_op["T_lts"]["max"]

    def _set_continuous_control_bounds(self):

        self.u_min = np.hstack(
            [
                self.p_op["v_ppsc"]["min_mpc"] * np.ones((self._timing.N, 1)),
                self.p_op["p_mpsc"]["min_mpc"] * np.ones((self._timing.N, 1)),
                self.p_op["v_pssc"]["min_mpc"] * np.ones((self._timing.N, 1)),
                -np.ones((self._timing.N, 1)),
                # The upcoming controls are constrained later in the NLP using inequality constraints
                np.zeros((self._timing.N, 1)),
                np.zeros((self._timing.N, 1)),
            ]
        )

        self.u_max = np.hstack(
            [
                self.p_op["v_ppsc"]["max"] * np.ones((self._timing.N, 1)),
                self.p_op["p_mpsc"]["max"] * np.ones((self._timing.N, 1)),
                self.p_op["v_pssc"]["max"] * np.ones((self._timing.N, 1)),
                np.ones((self._timing.N, 1)),
                np.inf * np.ones((self._timing.N, 1)),
                np.inf * np.ones((self._timing.N, 1)),
            ]
        )

    @abstractmethod
    def _set_binary_control_bounds(self):

        pass

    def _set_nlpsolver_bounds_and_initials(self):

        # Optimization variables bounds and initials

        V_min = []
        V_max = []
        V_init = []

        V_r_min = []
        V_r_max = []
        V_r_init = []

        # Constraints bounds

        lbg = []
        ubg = []

        lbg_r = []
        ubg_r = []

        # Time-varying parameters

        P_data = []

        # Initial states

        if self._timing.grid_position_cursor == 0:

            V_min.append(self._predictor.x_hat)
            V_max.append(self._predictor.x_hat)
            V_init.append(self._predictor.x_hat)

        else:

            V_min.append(self._previous_solver.x_data[0, :])
            V_max.append(self._previous_solver.x_data[0, :])
            V_init.append(self._previous_solver.x_data[0, :])

        V_r_min.append(V_min[-1])
        V_r_max.append(V_max[-1])
        V_r_init.append(V_init[-1])

        for k in range(self._timing.N):

            # Collocation equations

            for j in range(1, self.d + 1):

                lbg.append(np.zeros(self.nx))
                ubg.append(np.zeros(self.nx))

            if k < self._timing.grid_position_cursor:

                lbg.append(-np.inf * np.ones(self.nx))
                ubg.append(np.inf * np.ones(self.nx))

            else:

                lbg.append(np.zeros(self.nx))
                ubg.append(np.zeros(self.nx))

            lbg_r.append(lbg[-1])
            ubg_r.append(ubg[-1])

            # s_ac_lb

            lbg.append(np.zeros(self.n_s_ac_lb))
            ubg.append(np.inf * np.ones(self.n_s_ac_lb))

            lbg_r.append(lbg[-1])
            ubg_r.append(ubg[-1])

            # s_ac_ub

            lbg.append(-np.inf * np.ones(self.n_s_ac_ub))
            ubg.append(np.zeros(self.n_s_ac_ub))

            lbg_r.append(lbg[-1])
            ubg_r.append(ubg[-1])

            # State limits soft constraints

            lbg.append(self.x_min[k + 1, :])
            ubg.append(self.x_max[k + 1, :])

            lbg_r.append(lbg[-1])
            ubg_r.append(ubg[-1])

            # Assure ppsc is running at high speed when collector temperature is high

            lbg.append(-np.inf * np.ones(3))
            ubg.append(np.zeros(3))

            lbg_r.append(lbg[-1])
            ubg_r.append(ubg[-1])

            # Assure HTS bottom layer mass flows are always smaller or equal to
            # the corresponding total pump flow

            lbg.append(-np.inf * np.ones(2))
            ubg.append(np.zeros(2))

            lbg_r.append(lbg[-1])
            ubg_r.append(ubg[-1])

            # Electric power balance

            lbg.append(0)
            ubg.append(0)

            lbg_r.append(0)
            ubg_r.append(0)

            # SOS1 constraints

            lbg.append(0)
            ubg.append(1)

            lbg_r.append(lbg[-1])
            ubg_r.append(ubg[-1])

            # Append new bounds and initials

            for j in range(1, self.d + 1):

                V_min.append(-np.inf * np.ones(self.nx))
                V_max.append(np.inf * np.ones(self.nx))
                V_init.append(self._previous_solver.x_data[k, :])

            if k < self._timing.grid_position_cursor:

                V_min.append(self._previous_solver.b_data[k, :])
                V_max.append(self._previous_solver.b_data[k, :])
                V_init.append(self._previous_solver.b_data[k, :])

            else:

                V_min.append(self.b_min[k, :])
                V_max.append(self.b_max[k, :])
                V_init.append(self._previous_solver.b_data[k, :])

            V_r_min.append(V_min[-1])
            V_r_max.append(V_max[-1])
            V_r_init.append(V_init[-1])

            V_min.append(np.zeros(self.n_s_ac_lb))
            V_max.append(np.inf * np.ones(self.n_s_ac_lb))
            try:
                V_init.append(self._previous_solver.s_ac_lb_data[k, :])
            except AttributeError:
                V_init.append(np.zeros(self.n_s_ac_lb))

            V_r_min.append(V_min[-1])
            V_r_max.append(V_max[-1])
            V_r_init.append(V_init[-1])

            V_min.append(np.zeros(self.n_s_ac_ub))
            V_max.append(np.inf * np.ones(self.n_s_ac_ub))
            try:
                V_init.append(self._previous_solver.s_ac_ub_data[k, :])
            except AttributeError:
                V_init.append(np.zeros(self.n_s_ac_ub))

            V_r_min.append(V_min[-1])
            V_r_max.append(V_max[-1])
            V_r_init.append(V_init[-1])

            V_min.append(-np.inf * np.ones(self.nx - self.nx_aux))
            V_max.append(np.inf * np.ones(self.nx - self.nx_aux))
            try:
                V_init.append(self._previous_solver.s_x_data[k, :])
            except AttributeError:
                V_init.append(np.zeros(self.nx - self.nx_aux))

            V_r_min.append(V_min[-1])
            V_r_max.append(V_max[-1])
            V_r_init.append(V_init[-1])

            V_min.append(0)
            V_max.append(1)
            try:
                V_init.append(self._previous_solver.s_ppsc_data[k, :])
            except AttributeError:
                V_init.append(0.0)

            V_r_min.append(V_min[-1])
            V_r_max.append(V_max[-1])
            V_r_init.append(V_init[-1])

            if k < self._timing.grid_position_cursor:

                V_min.append(self._previous_solver.u_data[k, :])
                V_max.append(self._previous_solver.u_data[k, :])
                V_init.append(self._previous_solver.u_data[k, :])

            else:

                V_min.append(self.u_min[k, :])
                V_max.append(self.u_max[k, :])
                V_init.append(self._previous_solver.u_data[k, :])

            V_r_min.append(V_min[-1])
            V_r_max.append(V_max[-1])
            V_r_init.append(V_init[-1])

            if (k + 1) == self._timing.grid_position_cursor:

                V_min.append(self._predictor.x_hat)
                V_max.append(self._predictor.x_hat)
                V_init.append(self._predictor.x_hat)

            elif (k + 1) < self._timing.grid_position_cursor:

                V_min.append(self._previous_solver.x_data[k + 1, :])
                V_max.append(self._previous_solver.x_data[k + 1, :])
                V_init.append(self._previous_solver.x_data[k + 1, :])

            else:

                V_min.append(-np.inf * np.ones(self.nx))
                V_max.append(np.inf * np.ones(self.nx))
                V_init.append(self._previous_solver.x_data[k + 1, :])

            V_r_min.append(V_min[-1])
            V_r_max.append(V_max[-1])
            V_r_init.append(V_init[-1])

            # Append time-varying parameters

            P_data.append(self._ambient.c_data[k, :])
            P_data.append(self._timing.time_steps[k])

        # Concatenate objects

        self.V_min = ca.veccat(*V_min)
        self.V_max = ca.veccat(*V_max)
        self.V_init = ca.veccat(*V_init)

        self.lbg = np.hstack(lbg)
        self.ubg = np.hstack(ubg)

        self.V_r_min = ca.veccat(*V_r_min)
        self.V_r_max = ca.veccat(*V_r_max)
        self.V_r_init = ca.veccat(*V_r_init)

        self.lbg_r = np.hstack(lbg_r)
        self.ubg_r = np.hstack(ubg_r)

        self.P_data = ca.veccat(*P_data)

        self._nlpsolver_args = {
            "p": self.P_data,
            "x0": self.V_init,
            "lbx": self.V_min,
            "ubx": self.V_max,
            "lbg": self.lbg,
            "ubg": self.ubg,
        }

    def _run_nlpsolver(self):

        logger.info(
            (
                f"{self._solver_name}, "
                f"iter {self._timing.mpc_iteration_count}, "
                f"limit {round(self._nlpsolver_options['ipopt.max_cpu_time'],1)} s ..."
            )
        )

        self.nlp_solution = self._nlpsolver(**self._nlpsolver_args)
        self._solver_wall_time = self._nlpsolver.stats()["t_wall_total"]

        if self._nlpsolver.stats()["return_status"] == "Maximum_CpuTime_Exceeded":

            logger.warning(
                (
                    f"{self._solver_name} returned  "
                    f"'{self._nlpsolver.stats()['return_status']}' "
                    f"after {round(self._nlpsolver.stats()['t_wall_total']), 2} s"
                )
            )

        else:

            logger.info(
                (
                    f"{self._solver_name} returned  "
                    f"'{self._nlpsolver.stats()['return_status']}' "
                    f"after {round(self._nlpsolver.stats()['t_wall_total']), 2} s"
                )
            )

    def _collect_nlp_results(self):

        v_opt = np.array(self.nlp_solution["x"])

        v_r_opt = []

        x_opt = []
        u_opt = []
        b_opt = []
        s_ac_lb_opt = []
        s_ac_ub_opt = []
        s_x_opt = []
        s_ppsc_opt = []

        offset = 0

        for k in range(self._timing.N):

            x_opt.append(v_opt[offset : offset + self.nx])
            v_r_opt.append(x_opt[-1])

            for j in range(self.d + 1):

                offset += self.nx

            b_opt.append(v_opt[offset : offset + self.nb])
            v_r_opt.append(b_opt[-1])
            offset += self.nb

            s_ac_lb_opt.append(v_opt[offset : offset + self.n_s_ac_lb])
            v_r_opt.append(s_ac_lb_opt[-1])
            offset += self.n_s_ac_lb

            s_ac_ub_opt.append(v_opt[offset : offset + self.n_s_ac_ub])
            v_r_opt.append(s_ac_ub_opt[-1])
            offset += self.n_s_ac_ub

            s_x_opt.append(v_opt[offset : offset + self.nx - self.nx_aux])
            v_r_opt.append(s_x_opt[-1])
            offset += self.nx - self.nx_aux

            s_ppsc_opt.append(v_opt[offset : offset + 1])
            v_r_opt.append(s_ppsc_opt[-1])
            offset += 1

            u_opt.append(v_opt[offset : offset + self.nu])
            v_r_opt.append(u_opt[-1])
            offset += self.nu

        x_opt.append(v_opt[offset : offset + self.nx])
        v_r_opt.append(x_opt[-1])
        offset += self.nx

        self._x_data = ca.horzcat(*x_opt).T
        self._u_data = ca.horzcat(*u_opt).T
        self._b_data = ca.horzcat(*b_opt).T

        self._s_ac_lb_data = ca.horzcat(*s_ac_lb_opt).T
        self._s_ac_ub_data = ca.horzcat(*s_ac_ub_opt).T
        self._s_x_data = ca.horzcat(*s_x_opt).T
        self._s_ppsc_data = ca.horzcat(*s_ppsc_opt).T

        self.v_opt = v_opt
        self.v_r_opt = ca.veccat(*v_r_opt)

    def solve(self):

        self._store_previous_binary_solution()
        self._setup_nlpsolver()
        self._set_states_bounds()
        self._set_continuous_control_bounds()
        self._set_binary_control_bounds()
        self._set_nlpsolver_bounds_and_initials()
        self._run_nlpsolver()
        self._collect_nlp_results()

    def reduce_object_memory_size(self):

        self._previous_solver = None
        self._predictor = None


class NLPSolverBin(NLPSolverMPCBaseClass):

    _SOLVER_TYPE = "NLPbin"

    def _setup_additional_nlpsolver_options(self):

        self._nlpsolver_options["ipopt.acceptable_tol"] = 0.2
        self._nlpsolver_options["ipopt.acceptable_iter"] = 8
        self._nlpsolver_options["ipopt.acceptable_constr_viol_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_dual_inf_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_compl_inf_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_obj_change_tol"] = 1e-1

        self._nlpsolver_options["ipopt.mu_strategy"] = "adaptive"
        self._nlpsolver_options["ipopt.mu_target"] = 1e-5

    def _set_binary_control_bounds(self):

        self.b_min = self._previous_solver.b_data
        self.b_max = self._previous_solver.b_data

    def _reset_binary_control_bounds(self):

        self.b_min = np.zeros((self._timing.N, self.nb))
        self.b_max = np.ones((self._timing.N, self.nb))

    def reset_bounds_and_initials(self):

        self._reset_binary_control_bounds()
        self._set_nlpsolver_bounds_and_initials()



class NLPSolverRel(NLPSolverMPCBaseClass):

    _SOLVER_TYPE = "NLPrel"

    @property
    def b_data_prev(self):

        try:

            return np.asarray(self._b_data_prev)

        except AttributeError:

            msg = "Optimized binary controls not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)

    def _setup_additional_nlpsolver_options(self):

        self._nlpsolver_options["ipopt.acceptable_tol"] = 0.2
        self._nlpsolver_options["ipopt.acceptable_iter"] = 8
        self._nlpsolver_options["ipopt.acceptable_constr_viol_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_dual_inf_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_compl_inf_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_obj_change_tol"] = 1e-1

        self._nlpsolver_options["ipopt.mu_strategy"] = "adaptive"
        self._nlpsolver_options["ipopt.mu_target"] = 1e-5

    def _set_binary_control_bounds(self):

        self.b_min = np.zeros((self._timing.N, self.nb))
        self.b_max = np.ones((self._timing.N, self.nb))

        if self._timing._remaining_min_up_time > 0:

            locked_time_grid_points = np.where(
                self._timing.time_grid < self._timing._remaining_min_up_time
            )[0]

            self.b_min[locked_time_grid_points, :] = np.repeat(
                self._timing._b_bin_locked, len(locked_time_grid_points), 0
            )
            self.b_max[locked_time_grid_points, :] = np.repeat(
                self._timing._b_bin_locked, len(locked_time_grid_points), 0
            )


if __name__ == "__main__":

    import datetime as dt

    from timing import TimingMPC
    from state import State
    from ambient import Ambient

    from simulator import Simulator
    from predictor import Predictor

    startup_time = dt.datetime.fromisoformat("2010-08-19 06:00:00+02:00")

    timing = TimingMPC(startup_time=startup_time)

    ambient = Ambient(timing=timing)
    ambient.update()

    state = State()
    state.initialize()

    simulator = Simulator(timing=timing, ambient=ambient, state=state)
    simulator.solve()

    predictor = Predictor(
        timing=timing,
        ambient=ambient,
        state=state,
        previous_solver=simulator,
        solver_name="predictor",
    )
    predictor.solve()

    nlpsolver = NLPSolverRel(
        timing=timing,
        ambient=ambient,
        previous_solver=simulator,
        predictor=predictor,
        solver_name="nlpsolver_rel",
    )
    nlpsolver.solve()
