# Adrian Buerger, 2022

import os
import time
import numpy as np
import casadi as ca
import scipy.sparse as ssp
import multiprocessing as mp

from typing import Union

import gurobipy as gp

import logging

from benders_exp.voronoi import Voronoi
from benders_exp.defines import _PATH_TO_NLP_OBJECT

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DMiqp:

    _DEFAULT_LOGFILE_LOCATION = "/tmp/gurobi.log"


    @property
    def x(self) -> np.ndarray:
        try:
            return self.qp.x
        except AttributeError:
            msg = "Optimal solution not available yet, call solve() first."
            logger.error(msg)
            raise RuntimeError(msg)

    @property
    def obj(self) -> np.ndarray:
        try:
            return self.qp.obj
        except AttributeError:
            msg = "Objective value not available yet, call solve() first."
            logger.error(msg)
            raise RuntimeError(msg)

    @property
    def solve_successful(self):
        try:
            return (self.qp.status == gp.GRB.OPTIMAL) or (
                self.qp.status == gp.GRB.SUBOPTIMAL
            )
        except AttributeError:
            msg = "Solver return status not available yet, call solve() first."
            logger.error(msg)
            raise RuntimeError(msg)

    def set_lin_point(self, wlin: Union[list, np.ndarray]) -> None:

        # ToDo: add checks

        self._wlin = wlin

    def set_parameter_data(self, pd: Union[list, np.ndarray]) -> None:

        # ToDo: add checks

        self._pd = pd

    def set_constraint_bounds(
        self, lbg: Union[list, np.ndarray], ubg: Union[list, np.ndarray]
    ) -> None:

        # ToDo: add checks

        self._lbg = lbg
        self._ubg = ubg

    def set_variable_bounds(
        self, lbw: Union[list, np.ndarray], ubw: Union[list, np.ndarray]
    ) -> None:

        # ToDo: add checks

        self._lbw = lbw
        self._ubw = ubw

    def set_dwell_times(
        self,
        time_steps: Union[list, np.ndarray],
        min_up_times: Union[list, np.ndarray] = None,
        min_down_times: Union[list, np.ndarray] = None,
        remaining_min_up_time: float = None,
        b_bin_locked: Union[list, np.ndarray] = None,
    ) -> None:

        # ToDo: add checks

        self._time_steps = np.asarray(time_steps)

        self._minimum_up_times = (
            np.atleast_1d(np.asarray(min_up_times))
            if min_up_times is not None
            else None
        )
        self._minimum_down_times = (
            np.atleast_1d(np.asarray(min_down_times))
            if min_down_times is not None
            else None
        )

        self._remaining_min_up_time = remaining_min_up_time
        self._b_bin_locked = (
            np.atleast_1d(np.asarray(b_bin_locked))
            if b_bin_locked is not None
            else None
        )

    def set_max_switches(self, max_num_switches: Union[list, np.ndarray]) -> None:

        # ToDo: add checks

        self._max_num_switches = max_num_switches

    def set_b0(self, b0: Union[list, np.ndarray]) -> None:

        self._b0 = np.array(b0, dtype=float)

    def set_sb0(self, sb0: Union[list, np.ndarray]) -> None:

        self._sb0 = np.array(sb0, dtype=int)

    def set_v0(self, v0: Union[list, np.ndarray]) -> None:

        self._v0 = v0

    def set_voronoi(self, voronoi: Voronoi) -> None:

        self._voronoi = voronoi

    def _setup_default_settings(self, use_reduced_miqp: bool) -> None:

        self._use_reduced_miqp = use_reduced_miqp

        self._wlin = None
        self._lbw = None
        self._ubw = None
        self._lbg = None
        self._ubg = None
        self._pd = []
        self._b0 = None
        self._sb0 = None
        self._v0 = None
        self._max_num_switches = None
        self._minimum_up_times = None
        self._minimum_down_times = None
        self._voronoi = None

    def _load_qp_component(self, qp_component) -> None:

        qp_file_suffix = ""
        if self._use_reduced_miqp:
            qp_file_suffix = "_red"

        path_to_qp_file = os.path.join(
            _PATH_TO_NLP_OBJECT,
            f"{qp_component+qp_file_suffix}.casadi",
        )

        setattr(self, qp_component, ca.Function.load(path_to_qp_file))

    def _load_binary_variables_indices(self) -> None:

        idx_file_suffix = ""
        if self._use_reduced_miqp:
            idx_file_suffix = "_red"

        for idx in ["idx_b", "idx_sb"]:

            path_to_idx_file = os.path.join(
                _PATH_TO_NLP_OBJECT, f"{idx+idx_file_suffix}.txt"
            )

            setattr(self, idx, np.loadtxt(path_to_idx_file).astype(int))

    def _compute_binary_variables_dimensions(self) -> None:

        self._N = 1
        self._nb = [1]

        for k, b_k in enumerate(self.idx_b[:-1]):

            if self.idx_b[k + 1] - b_k == 1:
                self._nb[-1] += 1
            else:
                self._N += 1
                self._nb += [1]

        if not (np.asarray(self._nb) - self._nb[0]).sum() == 0:
            logging.error("Something is wrong with the binary variables index")

        self._nb = self._nb[0]
        self.idx_b_2d = self.idx_b.reshape(-1, self._nb).T

    def set_logfile_location(self, logfile_location: str):

        self._logfile_location = logfile_location

    def __init__(self, use_reduced_miqp: bool = False) -> None:

        self._setup_default_settings(use_reduced_miqp=use_reduced_miqp)
        self._load_binary_variables_indices()
        self._compute_binary_variables_dimensions()
        self.set_logfile_location(self._DEFAULT_LOGFILE_LOCATION)

    def _check_if_problem_definition_complete(self) -> None:

        if self._wlin is None:
            raise RuntimeError("No linearization point provided")

        if (self._lbg is None) or (self._ubg is None):
            raise RuntimeError("No constraints bounds provided")

    def _setup_A(self, part, queue):
        logging.info(f"Setting up QP component A_{part} ...")
        self._load_qp_component(f"A_qp_{part}")
        A = getattr(self, f"A_qp_{part}")(self._wlin, self._pd)
        queue.put((f"_A_{part}", A))
        logging.info(f"Done setting up A_{part}.")

    def _setup_lb(self, queue):
        logging.info("Setting up QP component lb ...")
        self._load_qp_component(f"b_qp")
        queue.put(
            ("_lb", np.squeeze(self.b_qp(self._lbg, self._wlin, self._pd, self._A)))
        )
        logging.info("Done setting up lb.")

    def _setup_ub(self, queue):
        logging.info("Setting up QP component ub ...")
        self._load_qp_component(f"b_qp")
        queue.put(
            ("_ub", np.squeeze(self.b_qp(self._ubg, self._wlin, self._pd, self._A)))
        )
        logging.info("Done setting up ub.")

    def _setup_H(self, queue):
        logging.info("Setting up QP component H ...")
        self._load_qp_component(f"H_qp")
        queue.put(("H", self.H_qp(self._wlin, self._pd).tocsc()))
        logging.info("Done setting up H.")

    def _setup_q(self, queue):
        logging.info("Setting up QP component q ...")
        self._load_qp_component(f"q_qp")
        queue.put(("_q", np.squeeze(self.q_qp(self._wlin, self._pd))))
        logging.info("Done setting up q.")

    def _setup_qp_matrices(self) -> None:

        logging.info("Setting up QP components ...")

        t_start = time.time()

        queue = mp.Queue()

        p_A_p1 = mp.Process(target=self._setup_A, args=("p1", queue))
        p_A_p2 = mp.Process(target=self._setup_A, args=("p2", queue))

        p_A_p1.start()
        p_A_p2.start()

        it = 0
        while it <= 1:
            r = queue.get()
            setattr(self, r[0], r[1])
            it += 1

        p_A_p1.join()
        p_A_p2.join()

        A = ca.vertcat(self._A_p1, self._A_p2).tocsc()
        Al = A.tolil()
        self._A = Al.tocsc()

        p_lb = mp.Process(target=self._setup_lb, args=(queue,))
        p_ub = mp.Process(target=self._setup_ub, args=(queue,))
        p_H = mp.Process(target=self._setup_H, args=(queue,))
        p_q = mp.Process(target=self._setup_q, args=(queue,))

        p_lb.start()
        p_ub.start()
        p_H.start()
        p_q.start()

        it = 0
        while it <= 3:
            r = queue.get()
            setattr(self, r[0], r[1])
            it += 1

        p_lb.join()
        p_ub.join()
        p_H.join()
        p_q.join()

        logging.info(f"Done setting up QP components after {time.time() - t_start} s")

    def _compute_num_optimization_variables(self) -> None:

        self._nw = self.H.shape[0]

    def _setup_variable_bounds(self) -> None:

        self._A_vb = None
        self._lb_vb = None
        self._ub_vb = None

        if (self._lbw is not None) or (self._ubw is not None):

            idx_vb = ~np.squeeze(
                np.isinf(np.asarray(self._lbw)) & np.isinf(np.asarray(self._ubw))
            )

            self._A_vb = ssp.csc_matrix(np.eye(self._nw)[idx_vb, :])
            self._lb_vb = np.squeeze(np.asarray(self._lbw))[idx_vb]
            self._ub_vb = np.squeeze(np.asarray(self._ubw))[idx_vb]

    def _set_max_switching_constraints(self) -> None:

        self._A_msc = None
        self._ub_msc = None
        self._lb_msc = None

        if self._max_num_switches is not None:

            self._A_msc = []

            it = 0

            c_sum = np.zeros((self._nb, self._nw + (self._nb * self._N)))

            for k in range(self._N):

                for i in range(self._nb):

                    for s in [-1, 1]:

                        c = np.zeros(self._nw + (self._nb * self._N))

                        try:
                            c[self.idx_b_2d[i, k]] = s * 1
                            c[self.idx_b_2d[i, k + 1]] = s * (-1)
                            c[self._nw + it] = -1
                            self._A_msc.append(c)

                        except IndexError:
                            pass

                    c_sum[i, self._nw + it] = 1

                    it += 1

            self._A_msc.append(c_sum)

            self._A_msc = ssp.csc_matrix(np.vstack(self._A_msc))
            self._ub_msc = np.zeros(self._A_msc.shape[0])
            self._ub_msc[-self._nb :] = np.asarray(self._max_num_switches)
            self._lb_msc = -np.inf * np.ones(self._A_msc.shape[0])

    def _set_min_up_time_constraints(self) -> None:

        self._A_mutc = None
        self._ub_mutc = None
        self._lb_mutc = None

        if self._minimum_up_times is not None:

            self._A_mutc = []
            self._ub_mutc = []
            self._lb_mutc = []

            k_init = -1

            if self._remaining_min_up_time:

                remaining_time = 0

                for m, dt in enumerate(self._time_steps):

                    if remaining_time < self._remaining_min_up_time:

                        c = np.zeros(self._nw)
                        c[self.idx_b_2d[np.where(self._b_bin_locked)[1], m]] = 1

                        self._A_mutc.append(c)
                        self._ub_mutc.append(1)
                        self._lb_mutc.append(1)

                    else:

                        k_init = m - 1
                        break

                    remaining_time += dt

            for k in range(k_init, self._N + 1):

                for i in range(self._nb):

                    uptime = 0
                    it = 0

                    for dt in self._time_steps[max(0, k) :]:

                        uptime += dt

                        if uptime < self._minimum_up_times[i]:

                            c = np.zeros(self._nw)

                            if k != -1:
                                c[self.idx_b_2d[i, k]] = -1

                            try:
                                c[self.idx_b_2d[i, k + 1]] = 1
                                c[self.idx_b_2d[i, k + it + 2]] = -1

                                self._A_mutc.append(c)

                                if k == -1:
                                    self._ub_mutc.append(
                                        0
                                    )  # only in open loop, tbd for closed loop!
                                else:
                                    self._ub_mutc.append(0)
                                self._lb_mutc.append(-np.inf)

                            except IndexError:
                                pass

                            it += 1  # check again if better positioned elsewhere

            self._A_mutc = ssp.csc_matrix(np.vstack(self._A_mutc))
            self._ub_mutc = np.asarray(self._ub_mutc)
            self._lb_mutc = np.asarray(self._lb_mutc)

    def _set_min_down_time_constraints(self) -> None:

        self._A_mdtc = None
        self._ub_mdtc = None
        self._lb_mdtc = None

        if self._minimum_down_times is not None:

            self._A_mdtc = []
            self._ub_mdtc = []
            self._lb_mdtc = []

            for k in range(-1, self._N + 1):

                for i in range(self._nb):

                    downtime = 0
                    it = 0

                    for dt in self._time_steps[max(0, k) :]:

                        downtime += dt

                        if downtime < self._minimum_down_times[i]:

                            c = np.zeros(self._nw)

                            if k != -1:
                                c[self.idx_b_2d[i, k]] = 1

                            try:
                                c[self.idx_b_2d[i, k + 1]] = -1
                                c[self.idx_b_2d[i, k + it + 2]] = 1

                                self._A_mdtc.append(c)

                                if k == -1:
                                    self._ub_mdtc.append(
                                        1
                                    )  # only in open loop, tbd for closed loop!
                                else:
                                    self._ub_mdtc.append(1)
                                self._lb_mdtc.append(-np.inf)

                            except IndexError:
                                pass

                            it += 1

            self._A_mdtc = ssp.csc_matrix(np.vstack(self._A_mdtc))
            self._ub_mdtc = np.asarray(self._ub_mdtc)
            self._lb_mdtc = np.asarray(self._lb_mdtc)

    def _set_voronoi_cut(self) -> None:

        self._A_v = None
        self._lb_v = None
        self._ub_v = None

        if self._voronoi:

            if self._voronoi.is_initialized:

                A_v = np.zeros((self._voronoi.A_v.shape[0], self._nw))
                A_v[:, self.idx_b] = self._voronoi.A_v

                self._A_v = ssp.csc_matrix(A_v)
                self._lb_v = self._voronoi.lb_v
                self._ub_v = self._voronoi.ub_v

    def _assemble_problem(self, b_fixed: bool) -> None:

        A = [self._A]
        lb = [self._lb]
        ub = [self._ub]

        x0 = np.squeeze(self._wlin)

        for A_k in [self._A_vb, self._A_mutc, self._A_mdtc, self._A_v]:
            if A_k is not None:
                A.append(A_k)
        A = ssp.vstack(A)

        for lb_k in [self._lb_vb, self._lb_mutc, self._lb_mdtc, self._lb_v]:
            if lb_k is not None:
                lb.append(lb_k)
        lb = np.hstack(lb)

        for ub_k in [self._ub_vb, self._ub_mutc, self._ub_mdtc, self._ub_v]:
            if ub_k is not None:
                ub.append(ub_k)
        ub = np.hstack(ub)

        if self._A_msc is not None:

            A0 = ssp.csc_matrix(
                (A.shape[0], self._A_msc.shape[1] - A.shape[1]), dtype=int
            )

            A = ssp.csc_matrix(ssp.vstack([ssp.hstack([A, A0]), self._A_msc]))

            lb = np.hstack([lb, self._lb_msc])
            ub = np.hstack([ub, self._ub_msc])

            x0 = np.hstack([x0, np.zeros(A.shape[1] - x0.shape[0])])

        self.H.resize((A.shape[1], A.shape[1]))
        q = np.append(self._q, np.zeros(A.shape[1] - self._nw))

        vtype = np.empty(A.shape[1], dtype=object)
        vtype[:] = gp.GRB.CONTINUOUS
        if not b_fixed:
            vtype[self.idx_b] = gp.GRB.BINARY

        lbx = -1e3 * np.ones(A.shape[1])
        if not b_fixed:
            lbx[self.idx_b] = 0.0
        else:
            lbx[self.idx_b] = self._b0

        lbx[self.idx_sb] = 0.0
        lbx[self._nw :] = 0.0

        ubx = 1e5 * np.ones(A.shape[1])
        if not b_fixed:
            ubx[self.idx_b] = 1.0
        else:
            ubx[self.idx_b] = self._b0

        ubx[self.idx_sb] = 1.0
        # ubx[self._nw:] = 1.0 # TBD!

        self.qp = gp.Model()

        x = self.qp.addMVar(A.shape[1], vtype=vtype, lb=lbx, ub=ubx)

        if (not b_fixed) and (self._b0 is not None):

            start = gp.GRB.UNDEFINED * np.ones(vtype.size)
            start[self.idx_b] = self._b0
            x.setAttr("Start", start)

        self.qp.setObjective(
            0.5 * (x @ self.H.todense() @ x) + q.T @ x, sense=gp.GRB.MINIMIZE
        )

        self.qp.addConstr(A @ x >= lb)
        self.qp.addConstr(A @ x <= ub)

    def _solve_problem(self, gap: float) -> None:

        self.qp.setParam("MIPGap", gap)
        self.qp.setParam("NumericFocus", True)
        self.qp.setParam("LogFile", self._logfile_location)
        self.qp.update()
        self.qp.optimize()

    def _collect_solver_stats(self):

        status_dict = {
            gp.GRB.LOADED: f"Loaded only ({gp.GRB.LOADED})",
            gp.GRB.OPTIMAL: f"Optimal solution ({gp.GRB.OPTIMAL})",
            gp.GRB.INFEASIBLE: f"Infeasible ({gp.GRB.INFEASIBLE})",
            gp.GRB.INF_OR_UNBD: f"Infeasible or unbounded ({gp.GRB.INF_OR_UNBD})",
            gp.GRB.UNBOUNDED: f"Unbounded ({gp.GRB.UNBOUNDED})",
            gp.GRB.NUMERIC: f"Numerical difficulties ({gp.GRB.NUMERIC})",
            gp.GRB.SUBOPTIMAL: f"Suboptimal solution ({gp.GRB.SUBOPTIMAL})",
            gp.GRB.TIME_LIMIT: f"Time limit ({gp.GRB.TIME_LIMIT})",
            gp.GRB.ITERATION_LIMIT: f"Iteration limit ({gp.GRB.ITERATION_LIMIT})",
            gp.GRB.INTERRUPTED: f"Solver interrupted ({gp.GRB.INTERRUPTED})",
        }

        try:
            solver_status = status_dict[self.qp.status]
        except KeyError:
            solver_status = "Unknown return status"

        self.solver_stats = {
            "runtime": self.qp.Runtime,
            "return_status": status_dict[self.qp.status],
        }

    def _retry_optimization_if_failed(self):

        if not self.solve_successful:
            logger.warning(
                "First MIQP solve failed, retrying with different LP solver ..."
            )
            self.qp.setParam("Method", 1)
            self.qp.optimize()

    def solve(self, gap: float, b_fixed: bool) -> None:

        self._check_if_problem_definition_complete()
        self._setup_qp_matrices()
        self._compute_num_optimization_variables()
        self._setup_variable_bounds()
        self._set_max_switching_constraints()
        self._set_min_up_time_constraints()
        self._set_min_down_time_constraints()
        self._set_voronoi_cut()
        self._assemble_problem(b_fixed=b_fixed)

        self._solve_problem(gap=gap)
        self._collect_solver_stats()
        self._retry_optimization_if_failed()
        self._collect_solver_stats()

    def compute_F_qp(self):
        logging.info("Computing F_qp...")
        self._load_qp_component("F_qp")
        return float(self.F_qp(self._wlin, self._pd, self.x))
        logging.info("Done computing F_qp.")
