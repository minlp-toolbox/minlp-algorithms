# Adrian Buerger, 2022

import os
import time
import datetime as dt
import numpy as np
import casadi as ca
from benders_exp.solarsys import dmiqp

from benders_exp.solarsys.nlpsetup import NLPSetupMPC
from benders_exp.solarsys.voronoi import Voronoi
from benders_exp.solarsys.casadisolver import BendersMILP

import logging

logger = logging.getLogger(__name__)

try:
    import pycombina
except ModuleNotFoundError:
    logger.warning(
        "pycombina not found, solving binary approximation via CIA will not work"
    )


class BinaryApproximation(NLPSetupMPC):

    _BINARY_TOLERANCE = 1e-3
    _MAX_NUM_BNB_INTERATIONS = 1e7

    _SECONDS_TIMEOUT_BEFORE_NEXT_TIME_GRID_POINT = 5.0

    _LOGFILE_LOCATION = "/tmp/voronoi-logs/"
    if not os.path.exists(_LOGFILE_LOCATION):
        os.mkdir(_LOGFILE_LOCATION)

    @property
    def solve_successful(self):
        try:
            return self._solver_stats["solve_successful"]
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
        return self._solver_type

    @property
    def x_data(self):

        try:
            return np.asarray(self._x_data)
        except AttributeError:
            logger.info(
                "x_data not availabe, returning data of previous solver instead."
            )
            return self._previous_solver.x_data

    @property
    def x_hat(self):

        return self._predictor.x_hat

    @property
    def u_data(self):

        try:
            return np.asarray(self._u_data)
        except AttributeError:
            return self._previous_solver.u_data

    @property
    def b_data(self):

        try:
            return np.asarray(self._b_data)
        except AttributeError:
            msg = "Optimized binary controls not available yet, call solve() first."
            logger.error(msg)
            raise RuntimeError(msg)

    @property
    def c_data(self):
        return self._previous_solver.c_data

    @property
    def s_ac_lb_data(self):

        try:
            return np.asarray(self._s_ac_lb_data)
        except AttributeError:
            logger.info(
                "s_ac_lb_data not availabe, returning data of previous solver instead."
            )
            return self._previous_solver.s_ac_lb_data

    @property
    def s_ac_ub_data(self):

        try:
            return np.asarray(self._s_ac_ub_data)
        except AttributeError:
            logger.info(
                "s_ac_ub_data not availabe, returning data of previous solver instead."
            )
            return self._previous_solver.s_ac_ub_data

    @property
    def s_x_data(self):

        try:
            return np.asarray(self._s_x_data)
        except AttributeError:
            logger.info(
                "s_x_data not availabe, returning data of previous solver instead."
            )
            return self._previous_solver.s_x_data

    @property
    def s_ppsc_data(self):

        try:
            return np.asarray(self._s_ppsc_data)
        except AttributeError:
            logger.info(
                "s_ppsc_data not availabe, returning data of previous solver instead."
            )
            return self._previous_solver.s_ppsc_data

    @property
    def j_qp(self):
        return self._j_qp

    @property
    def solver_name(self):

        return self._solver_name

    @property
    def voronoi(self):

        try:
            return self._voronoi
        except:
            logger.warning(
                "voronoi data not available, returning nan instead.")
            return np.nan

    @property
    def solver_wall_time(self):
        try:
            return self._solver_stats["runtime"]
        except AttributeError:
            msg = "NLP wall time not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    def _setup_timing(self, timing):

        self._timing = timing

    def _set_previous_solver(self, previous_solver):

        self._previous_solver = previous_solver

    def _set_predictor(self, predictor):

        self._predictor = predictor

    def _setup_solver_name(self, solver_name):

        self._solver_name = solver_name

    def _setup_general_solver_options(self):

        self._solver_options = {}

        self._solver_options["max_cpu_time"] = 720.0

    def __init__(self, timing, previous_solver, predictor, solver_name):

        super().__init__(timing=timing)

        self._setup_timing(timing=timing)

        self._setup_solver_name(solver_name=solver_name)
        self._set_previous_solver(previous_solver=previous_solver)
        self._set_predictor(predictor=predictor)

        self.load_ode_object()
        self._setup_simulator()
        self._setup_collocation_options()

        self._setup_general_solver_options()

    def set_solver_max_cpu_time(self, time_point_to_finish: dt.datetime) -> None:

        if self._timing.time_of_day == "night":

            time_for_final_nlp_solve = self._timing.dt_night

        else:

            time_for_final_nlp_solve = self._timing.dt_day

        time_for_final_nlp_solve /= self._timing.N_short_term

        max_cpu_time = (
            time_point_to_finish
            - dt.datetime.now(tz=self._timing.timezone)
            - dt.timedelta(seconds=time_for_final_nlp_solve)
            - dt.timedelta(seconds=self._SECONDS_TIMEOUT_BEFORE_NEXT_TIME_GRID_POINT)
        ).total_seconds()

        self._solver_options["max_cpu_time"] = max_cpu_time

        logger.debug(
            f"Maximum CPU time for {self._solver_name} set to {max_cpu_time} s ..."
        )

    def _setup_solver_type(self, method: str) -> None:

        self._solver_type = method

    def _setup_ba(self) -> None:

        self._b_rel = np.vstack(
            [
                np.asarray(self._previous_solver.b_data).T,
                np.atleast_2d(1 - self._previous_solver.b_data.sum(axis=1)),
            ]
        ).T

        # Ensure values are not out of range due to numerical effects

        self._b_rel[self._b_rel < 0] = 0
        self._b_rel[self._b_rel > 1.0] = 1

        self._binapprox = pycombina.BinApprox(
            t=self._timing.time_grid,
            b_rel=self._b_rel,
            binary_threshold=self._BINARY_TOLERANCE,
        )

    def _setup_miqp(self, use_reduced_miqp: bool) -> None:

        self._dmiqp = dmiqp.DMiqp(use_reduced_miqp=use_reduced_miqp)
        self._dmiqp.set_logfile_location(
            os.path.join(self._LOGFILE_LOCATION, self._solver_name + ".log")
        )

        if use_reduced_miqp:

            self._dmiqp.set_lin_point(self._previous_solver.v_r_opt)
            self._dmiqp.set_constraint_bounds(
                lbg=self._previous_solver.lbg_r, ubg=self._previous_solver.ubg_r
            )
            self._dmiqp.set_variable_bounds(
                self._previous_solver.V_r_min, self._previous_solver.V_r_max
            )

        else:

            self._dmiqp.set_lin_point(self._previous_solver.v_opt)
            self._dmiqp.set_constraint_bounds(
                lbg=self._previous_solver.lbg, ubg=self._previous_solver.ubg
            )
            self._dmiqp.set_variable_bounds(
                self._previous_solver.V_min, self._previous_solver.V_max
            )

        self._dmiqp.set_parameter_data(self._previous_solver.P_data)

    def _setup_milp(self, use_reduced_miqp: bool) -> None:
        self._dmiqp = BendersMILP(use_reduced_miqp=use_reduced_miqp)
        self._dmiqp.set_logfile_location(
            os.path.join(self._LOGFILE_LOCATION, self._solver_name + ".log")
        )

        if use_reduced_miqp:

            self._dmiqp.set_lin_point(self._previous_solver.v_r_opt)
            self._dmiqp.set_constraint_bounds(
                lbg=self._previous_solver.lbg_r, ubg=self._previous_solver.ubg_r
            )
            self._dmiqp.set_variable_bounds(
                self._previous_solver.V_r_min, self._previous_solver.V_r_max
            )

        else:

            self._dmiqp.set_lin_point(self._previous_solver.v_opt)
            self._dmiqp.set_constraint_bounds(
                lbg=self._previous_solver.lbg, ubg=self._previous_solver.ubg
            )
            self._dmiqp.set_variable_bounds(
                self._previous_solver.V_min, self._previous_solver.V_max
            )

        self._dmiqp.set_parameter_data(self._previous_solver.P_data)

    def _set_dwell_times_ba(self, b_fixed: bool) -> None:

        if not b_fixed:
            self._binapprox.set_min_up_times(
                min_up_times=self.p_op["acm"]["min_up_time"]
                + self.p_op["hp"]["min_up_time"]
                + [0]
            )
            self._binapprox.set_min_down_times(
                min_down_times=self.p_op["acm"]["min_down_time"]
                + self.p_op["hp"]["min_down_time"]
                + [0]
            )

    def _set_dwell_times_miqp(self, b_fixed: bool) -> None:

        if not b_fixed:
            self._dmiqp.set_dwell_times(
                time_steps=self._timing.time_steps,
                min_up_times=np.asarray(
                    self.p_op["acm"]["min_up_time"] +
                    self.p_op["hp"]["min_up_time"]
                )
                - 1e-3,
                min_down_times=np.asarray(
                    self.p_op["acm"]["min_down_time"] +
                    self.p_op["hp"]["min_down_time"]
                )
                - 1e-3,
                remaining_min_up_time=self._timing.remaining_min_up_time,
                b_bin_locked=self._timing.b_bin_locked,
            )

    def _set_voronoi_miqp(self, voronoi: Voronoi) -> None:

        self._dmiqp.set_voronoi(voronoi=voronoi)
        self._voronoi = self._dmiqp._voronoi

    def _solve_ba(self):

        logger.info(
            (
                f"{self._solver_name}, "
                f"iter {self._timing.mpc_iteration_count}, "
                f"limit {self._solver_options['max_cpu_time']} s ..."
            )
        )

        t_start = time.time()

        combina = pycombina.CombinaBnB(self._binapprox)
        combina.solve(
            max_iter=int(1e8), max_cpu_time=self._solver_options["max_cpu_time"]
        )

        self._b_data = self._binapprox.b_bin[: self.nb, :].T

        logger.info(
            f"{self._solver_name} finished after {combina.solution_time} s")

    def _set_warm_start(self, warm_start: bool, shift: bool = False):

        if warm_start:

            if shift:
                b_data_ws = np.zeros(self._previous_solver.b_data_prev.shape)
                b_data_ws[
                    : self._timing.N - self._timing.N_short_term, :
                ] = b_data_prev = self._previous_solver.b_data_prev[
                    self._timing.N_short_term:, :
                ]
            else:
                b_data_ws = self._previous_solver.b_data_prev

            self._dmiqp.set_b0(b_data_ws.flatten())

    def _set_binary_controls_if_fixed(self, b_fixed: bool):

        if b_fixed:

            b_data = self._previous_solver.b_data
            self._dmiqp.set_b0(b_data.flatten())

    def _solve_miqp(self, gap: float, b_fixed: bool) -> None:

        logger.info(
            (
                f"{self._solver_name}, "
                f"iter {self._timing.mpc_iteration_count}, "
                f"limit {self._solver_options['max_cpu_time']} s ..."
            )
        )

        self._dmiqp.solve(gap=gap, b_fixed=b_fixed)

        logger.info(
            f"{self._solver_name} finished after {self._dmiqp.solver_stats} s")

    def _collect_solver_stats(self):

        self._solver_stats = {
            "runtime": None,
            "return_status": None,
            "solve_successful": None,
        }

        if self.solver_type == "miqp":

            self._solver_stats = self._dmiqp.solver_stats
            self._solver_stats["solve_successful"] = self._dmiqp.solve_successful

        else:  # TBD!

            self._solver_stats["return_status"] = 0
            self._solver_stats["runtime"] = 0.0
            self._solver_stats["solve_successful"] = True

    def _collect_nlp_results(self, use_reduced_miqp: bool) -> None:

        x_opt = []
        u_opt = []
        b_opt = []
        s_ac_lb_opt = []
        s_ac_ub_opt = []
        s_x_opt = []
        s_ppsc_opt = []

        if self.solve_successful:

            v_opt = np.array(self._dmiqp.x)

            offset = 0

            for k in range(self._timing.N):

                x_opt.append(v_opt[offset: offset + self.nx])
                offset += self.nx

                if not use_reduced_miqp:
                    for j in range(1, self.d + 1):
                        offset += self.nx

                b_opt.append(v_opt[offset: offset + self.nb])
                offset += self.nb

                s_ac_lb_opt.append(v_opt[offset: offset + self.n_s_ac_lb])
                offset += self.n_s_ac_lb

                s_ac_ub_opt.append(v_opt[offset: offset + self.n_s_ac_ub])
                offset += self.n_s_ac_ub

                s_x_opt.append(v_opt[offset: offset + self.nx - self.nx_aux])
                offset += self.nx - self.nx_aux

                s_ppsc_opt.append(v_opt[offset: offset + 1])
                offset += 1

                u_opt.append(v_opt[offset: offset + self.nu])
                offset += self.nu

            x_opt.append(v_opt[offset: offset + self.nx])
            offset += self.nx

            self.v_opt = v_opt

        self._x_data = ca.horzcat(*x_opt).T
        self._u_data = ca.horzcat(*u_opt).T
        self._b_data = ca.horzcat(*b_opt).T

        self._s_ac_lb_data = ca.horzcat(*s_ac_lb_opt).T
        self._s_ac_ub_data = ca.horzcat(*s_ac_ub_opt).T
        self._s_x_data = ca.horzcat(*s_x_opt).T
        self._s_ppsc_data = ca.horzcat(*s_ppsc_opt).T

    def _set_j_qp(self) -> None:

        self._j_qp = self._dmiqp.compute_F_qp()

    def _approximate_binary_controls_from_relaxed_solution(
        self,
        method: str,
        use_reduced_miqp: bool,
        warm_start: bool,
        gap: float,
        voronoi: Voronoi,
        b_fixed: bool,
    ) -> None:

        if method == "miqp":
            self._setup_miqp(use_reduced_miqp=use_reduced_miqp)
            self._set_dwell_times_miqp(b_fixed=b_fixed)
            self._set_voronoi_miqp(voronoi=voronoi)
            self._set_binary_controls_if_fixed(b_fixed=b_fixed)
            self._set_warm_start(warm_start=warm_start)
            self._solve_miqp(gap=gap, b_fixed=b_fixed)
            self._collect_solver_stats()
            self._collect_nlp_results(use_reduced_miqp=use_reduced_miqp)
            self._set_j_qp()
        elif method == "milp":
            self._setup_milp(use_reduced_miqp=use_reduced_miqp)
            self._set_dwell_times_miqp(b_fixed=b_fixed)
            # TODO: Remove voronoi:
            self._set_voronoi_miqp(voronoi=voronoi)
            self._set_binary_controls_if_fixed(b_fixed=b_fixed)
            self._set_warm_start(warm_start=warm_start)
            self._solve_miqp(gap=gap, b_fixed=b_fixed)
            self._collect_solver_stats()
            self._collect_nlp_results(use_reduced_miqp=use_reduced_miqp)
        else:
            self._setup_ba()
            self._set_dwell_times_ba(b_fixed=b_fixed)
            self._solve_ba()
            self._collect_solver_stats()

        self._remove_unserializable_attributes()

    def solve(
        self,
        method: str = "miqp",
        use_reduced_miqp: bool = True,
        warm_start: bool = False,
        gap: float = 1e-4,
        voronoi: Voronoi = None,
        b_fixed: bool = False,
    ):

        t_start = time.time()

        self._setup_solver_type(method)
        self._approximate_binary_controls_from_relaxed_solution(
            method=method,
            use_reduced_miqp=use_reduced_miqp,
            warm_start=warm_start,
            gap=gap,
            b_fixed=b_fixed,
            voronoi=voronoi,
        )

        logger.info(f"BnB finished in {time.time()-t_start} seconds")
