# Adrian Buerger, 2022

import os
import numpy as np
import casadi as ca
import subprocess
from benders_exp.solarsys.defines import _PATH_TO_NLP_SOURCE, _PATH_TO_NLP_OBJECT, \
        _NLP_SOURCE_FILENAME, _NLP_OBJECT_FILENAME

from benders_exp.solarsys.system import System

import logging

logger = logging.getLogger(__name__)


class NLPSetupBaseClass(System):


    _CXX_COMPILERS = ["gcc-11", 'gcc']
    _CXX_FLAGS = ["-fPIC", "-v", "-shared", "-fno-omit-frame-pointer"]
    _CXX_FLAG_NO_OPT = ["-O0"]
    _CXX_FLAG_OPT = ["-O1"]

    def _initialize_nlp_set_up_flag(self):

        self._nlp_set_up = False

    def _setup_timing(self, timing):

        self._timing = timing

    def _setup_collocation_options(self):

        self.d = 2
        self.tau_root = [0] + ca.collocation_points(self.d, "radau")

    def _setup_directories(self):

        if not os.path.exists(_PATH_TO_NLP_OBJECT):
            os.mkdir(_PATH_TO_NLP_OBJECT)

        if not os.path.exists(_PATH_TO_NLP_SOURCE):
            os.mkdir(_PATH_TO_NLP_SOURCE)

    def _export_nlp_to_c_code(self):

        __dirname__ = os.path.dirname(os.path.abspath(__file__))

        nlpsolver = ca.nlpsol("nlpsol", "ipopt", self.nlp)
        nlpsolver.generate_dependencies(_NLP_SOURCE_FILENAME)

        os.rename(
            _NLP_SOURCE_FILENAME,
            os.path.join(
                __dirname__, _PATH_TO_NLP_SOURCE, _NLP_SOURCE_FILENAME
            ),
        )

    def _check_compiler_availability(self):

        logging.info("Checking if a C compiler is available ...")

        for compiler in self._CXX_COMPILERS:

            try:
                return_status = subprocess.call(
                    [" ".join([compiler, "--version"])], shell=True
                )
                if return_status == 0:
                    logging.info(f"Compiler {compiler} found.")

                    self._compiler = compiler
                    return

            except FileNotFoundError:
                logging.info(
                    f"Compiler {compiler} not found, trying other compiler ..."
                )

        raise RuntimeError("No C compiler found, NLP object cannot be generated.")

    def _compile_nlp_object(self, optimize_for_speed, overwrite_existing_object):

        """
        When optimize_for_speed=True, the NLP objects are compiled to facilitate
        faster computation; however, more time and memory is required for compiltation
        """

        __dirname__ = os.path.dirname(os.path.abspath(__file__))

        path_to_nlp_source_code = os.path.join(
            __dirname__, _PATH_TO_NLP_SOURCE, _NLP_SOURCE_FILENAME
        )
        path_to_nlp_object = os.path.join(
            __dirname__, _PATH_TO_NLP_OBJECT, _NLP_OBJECT_FILENAME
        )

        if not os.path.isfile(path_to_nlp_object) or overwrite_existing_object:

            logger.info("Compiling NLP object ...")

            compiler_flags = self._CXX_FLAGS

            if optimize_for_speed:
                compiler_flags += self._CXX_FLAG_OPT
            else:
                compiler_flags += self._CXX_FLAG_NO_OPT

            compiler_command = [
                " ".join(
                    [self._compiler]
                    + compiler_flags
                    + ["-o"]
                    + [path_to_nlp_object]
                    + [path_to_nlp_source_code]
                )
            ]

            return_status = subprocess.call(compiler_command, shell=True)

            if return_status == 0:

                logger.info("Compilation of NLP object finished successfully.")

            else:

                logger.error(
                    "A problem occurred compiling the NLP object, check compiler output for further information."
                )
                raise subprocess.CalledProcessError(
                    return_status, " ".join(compiler_command)
                )

        else:

            logger.info("NLP object already exists, not overwriting it.")

    def _setup_nlp(self, use_big_m_constraints):

        self._setup_model_variables()
        self._setup_model()
        self._setup_collocation_options()
        self._setup_nlp_functions(use_big_m_constraints=use_big_m_constraints)
        self._setup_nlp_components()

    def generate_nlp_object(
        self,
        use_big_m_constraints=True,
        optimize_for_speed=False,
        overwrite_existing_object=False,
    ):

        logger.info("Generating NLP object ...")

        if not self._nlp_set_up:
            self._setup_nlp(use_big_m_constraints=use_big_m_constraints)
        self._setup_directories()
        self._export_nlp_to_c_code()
        self._check_compiler_availability()
        self._compile_nlp_object(
            optimize_for_speed=optimize_for_speed,
            overwrite_existing_object=overwrite_existing_object,
        )

        logger.info("Finished generating NLP object.")


class NLPSetupMPC(NLPSetupBaseClass):

    def __init__(self, timing):

        super().__init__()

        self._setup_timing(timing)
        self._initialize_nlp_set_up_flag()

    def _setup_nlp_functions(self, use_big_m_constraints):

        f = ca.Function("f", [self.x, self.c, self.u, self.b], [self.f])

        C = np.zeros((self.d + 1, self.d + 1))
        D = np.zeros(self.d + 1)

        for j in range(self.d + 1):

            p = np.poly1d([1])
            for r in range(self.d + 1):
                if r != j:
                    p *= np.poly1d([1, -self.tau_root[r]]) / (
                        self.tau_root[j] - self.tau_root[r]
                    )

            D[j] = p(1.0)

            pder = np.polyder(p)
            for r in range(self.d + 1):
                C[j, r] = pder(self.tau_root[r])

        # Collocation equations

        x_k_c = [ca.MX.sym("x_k_c_" + str(j), self.nx) for j in range(self.d + 1)]
        x_k_next_c = ca.MX.sym("x_k_next_c", self.nx)
        c_k_c = ca.MX.sym("c_k_c", self.nc)
        u_k_c = ca.MX.sym("u_k_c", self.nu)
        b_k_c = ca.MX.sym("b_k_c", self.nb)
        dt_k_c = ca.MX.sym("dt_k_c")

        eq_c = []

        for j in range(1, self.d + 1):

            x_p_c = 0

            for r in range(self.d + 1):

                x_p_c += C[r, j] * x_k_c[r]

            f_k_c = f(x_k_c[j], c_k_c, u_k_c, b_k_c)

            eq_c.append(dt_k_c * f_k_c - x_p_c)

        eq_c = ca.veccat(*eq_c)

        xf_c = 0

        for r in range(self.d + 1):

            xf_c += D[r] * x_k_c[r]

            eq_d = xf_c - x_k_next_c

        self.F = ca.Function(
            "F",
            x_k_c + [x_k_next_c, c_k_c, u_k_c, b_k_c, dt_k_c],
            [eq_c, eq_d],
            ["x_k_" + str(j) for j in range(self.d + 1)]
            + ["x_k_next", "c_k", "u_k", "b_k", "dt_k"],
            ["eq_c", "eq_d"],
        )

        x_next = ca.MX.sym("x_next", self.nx)
        dt = ca.MX.sym("dt")

        ode = {
            "x": self.x,
            "p": ca.veccat(self.c, self.u, self.b, dt),
            "ode": dt * self.f,
        }

        self._integrator = ca.integrator(
            "integrator", "cvodes", ode, {"t0": 0.0, "tf": 1.0}
        )

        eq_ms = (
            self._integrator(x0=self.x, p=ca.veccat(self.c, self.u, self.b, dt))["xf"]
            - x_next
        )

        self.F_ms = ca.Function(
            "F_ms", [self.x, x_next, self.c, self.u, self.b, dt], [eq_ms]
        )

        # ACM operation condition equations

        s_ac_lb = ca.MX.sym("s_ac_lb", self.n_s_ac_lb)
        if use_big_m_constraints:

            T_ac_min = ca.veccat(
                self.x[self.x_index["T_lts"]]
                - self.p_op["T"]["min"]
                - self.b[self.b_index["b_ac"]]
                * (self.p["T_ac_lt_min"] - self.p_op["T"]["min"])
                + s_ac_lb[0],
                self.x[self.x_index["T_hts"][0]]
                - self.p_op["T"]["min"]
                - self.b[self.b_index["b_ac"]]
                * (self.p["T_ac_ht_min"] - self.p_op["T"]["min"])
                + s_ac_lb[1],
            )

        else:

            T_ac_min = self.b[self.b_index["b_ac"]] * ca.veccat(
                self.x[self.x_index["T_lts"]] - self.p["T_ac_lt_min"] + s_ac_lb[0],
                self.x[self.x_index["T_hts"][0]] - self.p["T_ac_ht_min"] + s_ac_lb[1],
            )

        self.T_ac_min_fcn = ca.Function(
            "T_ac_min_fcn", [self.x, self.c, self.b, s_ac_lb], [T_ac_min]
        )

        s_ac_ub = ca.MX.sym("s_ac_ub", self.n_s_ac_ub)

        if use_big_m_constraints:

            T_ac_max = ca.veccat(
                self.x[self.x_index["T_lts"]]
                - self.p_op["T"]["max"]
                - self.b[self.b_index["b_ac"]]
                * (self.p["T_ac_lt_max"] - self.p_op["T"]["max"])
                - s_ac_ub[0],
                self.x[self.x_index["T_hts"][0]]
                - self.p_op["T"]["max"]
                - self.b[self.b_index["b_ac"]]
                * (self.p["T_ac_ht_min"] - self.p_op["T"]["max"])
                - s_ac_ub[1],
            )

        else:

            T_ac_max = self.b[self.b_index["b_ac"]] * ca.veccat(
                self.x[self.x_index["T_lts"][0]] - self.p["T_ac_lt_max"] - s_ac_ub[0],
                self.x[self.x_index["T_hts"][0]] - self.p["T_ac_ht_max"] - s_ac_ub[1],
            )

        self.T_ac_max_fcn = ca.Function(
            "T_ac_max_fcn", [self.x, self.c, self.b, s_ac_ub], [T_ac_max]
        )

        # States limits soft constraints

        s_x = ca.MX.sym("s_x", self.nx - self.nx_aux)

        self.s_x_fcn = ca.Function(
            "s_x_fcn", [self.x, s_x], [self.x[: self.nx - self.nx_aux] + s_x]
        )

        # Assure ppsc is running at high speed when collector temperature is high

        s_ppsc = ca.MX.sym("s_ppsc")

        self.v_ppsc_so_fpsc_fcn = ca.Function(
            "v_ppsc_so_fpsc_fcn",
            [self.x, s_ppsc],
            [
                (self.x[self.x_index["T_fpsc"]] - self.p_op["T_sc"]["T_sc_so"])
                * (self.p_op["T_sc"]["v_ppsc_so"] - s_ppsc)
            ],
        )

        self.v_ppsc_so_vtsc_fcn = ca.Function(
            "v_ppsc_so_vtsc_fcn",
            [self.x, s_ppsc],
            [
                (self.x[self.x_index["T_vtsc"]] - self.p_op["T_sc"]["T_sc_so"])
                * (self.p_op["T_sc"]["v_ppsc_so"] - s_ppsc)
            ],
        )

        self.v_ppsc_so_fcn = ca.Function(
            "v_ppsc_so_fcn", [self.u, s_ppsc], [s_ppsc - self.u[self.u_index["v_ppsc"]]]
        )

        # Assure HTS bottom layer mass flows are always smaller or equal to
        # the corresponding total pump flow

        mdot_hts_b_max = ca.veccat(
            self.u[self.u_index["mdot_o_hts_b"]]
            - self.p["mdot_ssc_max"] * self.u[self.u_index["v_pssc"]],
            self.u[self.u_index["mdot_i_hts_b"]]
            - self.b[self.b_index["b_ac"]] * self.p["mdot_ac_ht"],
        )

        self.mdot_hts_b_max_fcn = ca.Function(
            "mdot_hts_b_max_fcn", [self.u, self.b], [mdot_hts_b_max]
        )

        # Electric power balance

        P_hp = (
            0.006381135707410529 * self.x[self.x_index["T_lts"]]
            + 0.06791020408163258 * (self.c[self.c_index["T_amb"]] + self.p["dT_rc"])
            + 1.7165550470428876
        ) * 1e3  # + 8e2

        electricity_consumption = (
            self.p_op["acm"]["Pmax"][0] * self.b[self.b_index["b_ac"]]
            + self.p_op["acm"]["Pmax"][1] * self.b[self.b_index["b_fc"]]
            + P_hp * self.b[self.b_index["b_hp"]]
            + self.p_op["v_ppsc"]["Pmax"] * self.u[self.u_index["v_ppsc"]]
            + self.p_op["v_pssc"]["Pmax"] * self.u[self.u_index["v_pssc"]]
        )

        electric_power_balance = (
            -electricity_consumption
            + self.p["P_pv_p"] * self.c[self.c_index["P_pv_kWp"]]
            + self.p_op["grid"]["P_g_max"] * self.u[self.u_index["P_g"]]
        )

        self.electric_power_balance_fcn = ca.Function(
            "electric_power_balance_fcn",
            [self.x, self.u, self.b, self.c],
            [electric_power_balance],
        )

        u_prev = ca.MX.sym("u_prev", self.nu)

        F1 = ca.veccat(
            5e0 * s_ac_lb,
            5e0 * s_ac_ub,
            1e1 * s_x,
            1e2 * s_x[self.x_index["T_lts"]],
            self.p_op["dp_mpsc"]["Pmax"]
            * (u_prev[self.u_index["p_mpsc"]] - self.u[self.u_index["p_mpsc"]]),
            self.p_op["dp_mssc"]["Pmax"]
            * (
                u_prev[self.u_index["mdot_o_hts_b"]]
                - self.u[self.u_index["mdot_o_hts_b"]]
            ),
            self.p_op["dp_macm"]["Pmax"]
            * (
                u_prev[self.u_index["mdot_i_hts_b"]]
                - self.u[self.u_index["mdot_i_hts_b"]]
            ),
        )

        self.F1_fcn = ca.Function(
            "F1_fcn", [s_ac_lb, s_ac_ub, s_x, self.u, u_prev], [F1]
        )

        F2 = (
            self.c[self.c_index["p_g"]]
            * self.p_op["grid"]["P_g_max"]
            * self.u[self.u_index["P_g"]]
        )

        self.F2_fcn = ca.Function("F2_fcn", [self.u, self.c], [F2])

    def _setup_nlp_components(self):

        # Optimization variables

        V = []
        V_r = []

        # Index counter for binary optimization varialbes (req. for MIQP)

        idx = 0
        idx_red = 0

        self.idx_b = []
        self.idx_b_red = []

        self.idx_sb = []
        self.idx_sb_red = []

        # Constraints

        g = []
        g_r = []

        # Objective

        F1 = []
        F2 = []

        # Parametric controls

        P = []

        X0 = ca.MX.sym("x_0_0", self.nx)

        V.append(X0)
        V_r.append(X0)
        idx += self.nx
        idx_red += self.nx

        x_k_0 = X0

        u_k_prev = None

        for k in range(self._timing.N):

            # Add new states

            x_k_j = [
                ca.MX.sym("x_" + str(k) + "_" + str(j), self.nx)
                for j in range(1, self.d + 1)
            ]

            x_k = [x_k_0] + x_k_j
            x_k_next_0 = ca.MX.sym("x_" + str(k + 1) + "_0", self.nx)

            # Add new binary controls

            b_k = ca.MX.sym("b_" + str(k), self.nb)

            # Add new continuous controls

            u_k = ca.MX.sym("u_" + str(k), self.nu)

            if u_k_prev is None:

                u_k_prev = u_k

            # Add new parametric controls

            c_k = ca.MX.sym("c_" + str(k), self.nc)

            # Add parameter for time step at current point

            dt_k = ca.MX.sym("dt_" + str(k))

            # Add collocation equations

            F_k_inp = {"x_k_" + str(i): x_k_i for i, x_k_i in enumerate(x_k)}
            F_k_inp.update(
                {
                    "x_k_next": x_k_next_0,
                    "c_k": c_k,
                    "u_k": u_k,
                    "b_k": b_k,
                    "dt_k": dt_k,
                }
            )

            F_k = self.F(**F_k_inp)

            g.append(F_k["eq_c"])
            g.append(F_k["eq_d"])

            # Add multiple shooting equations (for reduced QP)

            g_r.append(self.F_ms(x_k_0, x_k_next_0, c_k, u_k, b_k, dt_k))

            # Add new slack variable for T_ac_min condition

            s_ac_lb_k = ca.MX.sym("s_ac_lb_" + str(k), self.n_s_ac_lb)

            # Setup T_ac_min conditions

            g.append(self.T_ac_min_fcn(x_k_0, c_k, b_k, s_ac_lb_k))
            g_r.append(self.T_ac_min_fcn(x_k_0, c_k, b_k, s_ac_lb_k))

            # Add new slack variable for T_ac_max condition

            s_ac_ub_k = ca.MX.sym("s_ac_ub_" + str(k), self.n_s_ac_ub)

            # Setup T_ac_max conditions

            g.append(self.T_ac_max_fcn(x_k_0, c_k, b_k, s_ac_ub_k))
            g_r.append(self.T_ac_max_fcn(x_k_0, c_k, b_k, s_ac_ub_k))

            # Add new slack variable for state limits soft constraints

            s_x_k = ca.MX.sym("s_x_" + str(k), self.nx - self.nx_aux)

            # Setup state limits as soft constraints to prevent infeasibility

            g.append(self.s_x_fcn(x_k_next_0, s_x_k))
            g_r.append(self.s_x_fcn(x_k_next_0, s_x_k))

            # Assure ppsc is running at high speed when collector temperature is high

            s_ppsc_k = ca.MX.sym("s_ppsc_fpsc_" + str(k))

            g.append(self.v_ppsc_so_fpsc_fcn(x_k_0, s_ppsc_k))
            g_r.append(self.v_ppsc_so_fpsc_fcn(x_k_0, s_ppsc_k))

            g.append(self.v_ppsc_so_vtsc_fcn(x_k_0, s_ppsc_k))
            g_r.append(self.v_ppsc_so_vtsc_fcn(x_k_0, s_ppsc_k))

            g.append(self.v_ppsc_so_fcn(u_k, s_ppsc_k))
            g_r.append(self.v_ppsc_so_fcn(u_k, s_ppsc_k))

            # Assure HTS bottom layer mass flows are always smaller or equal to
            # the corresponding total pump flow

            g.append(self.mdot_hts_b_max_fcn(u_k, b_k))
            g_r.append(self.mdot_hts_b_max_fcn(u_k, b_k))

            # Electric power balance

            g.append(self.electric_power_balance_fcn(x_k_0, u_k, b_k, c_k))
            g_r.append(self.electric_power_balance_fcn(x_k_0, u_k, b_k, c_k))

            # SOS1 constraint

            g.append(ca.sum1(b_k))
            g_r.append(ca.sum1(b_k))

            # Append new optimization variables, boundaries and initials

            for x_j in x_k_j:

                V.append(x_j)
                idx += self.nx

            V.append(b_k)
            V_r.append(b_k)
            for i in range(self.nb):
                self.idx_b += [idx]
                idx += 1
                self.idx_b_red += [idx_red]
                idx_red += 1

            V.append(s_ac_lb_k)
            V_r.append(s_ac_lb_k)
            idx += s_ac_lb_k.numel()
            idx_red += s_ac_lb_k.numel()

            V.append(s_ac_ub_k)
            V_r.append(s_ac_ub_k)
            idx += s_ac_ub_k.numel()
            idx_red += s_ac_ub_k.numel()

            V.append(s_x_k)
            V_r.append(s_x_k)
            idx += s_x_k.numel()
            idx_red += s_x_k.numel()

            V.append(s_ppsc_k)
            V_r.append(s_ppsc_k)
            for i in range(s_ppsc_k.numel()):
                self.idx_sb += [idx]
                idx += 1
                self.idx_sb_red += [idx_red]
                idx_red += 1

            V.append(u_k)
            V_r.append(u_k)
            idx += self.nu
            idx_red += self.nu

            V.append(x_k_next_0)
            V_r.append(x_k_next_0)
            idx += self.nx
            idx_red += self.nx

            F1.append(
                np.sqrt(dt_k / 3600)
                * self.F1_fcn(s_ac_lb_k, s_ac_ub_k, s_x_k, u_k, u_k_prev)
            )
            F2.append((dt_k / 3600) * self.F2_fcn(u_k, c_k))

            P.append(c_k)
            P.append(dt_k)

            x_k_0 = V[-1]
            u_prev = u_k

            # Concatenate optimization variables

            self.V = ca.veccat(*V)
            self.V_r = ca.veccat(*V_r)
            idb_b_2d = np.asarray(self.idx_b).reshape(-1, self.nb)

        # Concatenate objects

        self.g = ca.veccat(*g)
        self.g_r = ca.veccat(*g_r)
        self.F1 = 0.1 * ca.veccat(*F1)
        self.F2 = 0.01 * ca.sum1(ca.veccat(*F2))
        self.Pi = ca.veccat(*P)

        self.idx_b = np.asarray(self.idx_b)
        self.idx_b_red = np.asarray(self.idx_b_red)
        self.idx_sb = np.asarray(self.idx_sb)
        self.idx_sb_red = np.asarray(self.idx_sb_red)

        # Setup objective

        self.fx = 0.5 * ca.mtimes(self.F1.T, self.F1) + self.F2

        self.nlp = {"x": self.V, "p": self.Pi, "f": self.fx, "g": self.g}

        self._nlp_set_up = True

    def _setup_qp_components(self):

        dF1 = ca.jacobian(self.F1, self.V)
        dF2 = ca.jacobian(self.F2, self.V)
        B = ca.mtimes(dF1.T, dF1)

        self.H_qp = ca.Function("H_qp", [self.V, self.Pi], [B])

        self.q_qp = ca.Function(
            "q_qp",
            [self.V, self.Pi],
            [ca.mtimes(dF1.T, self.F1) - ca.mtimes(B, self.V) + dF2.T],
        )

        dg = ca.jacobian(self.g, self.V)

        self.A_qp = ca.Function("A_qp", [self.V, self.Pi], [dg])

        g_b = ca.MX.sym("V_b", self.g.shape)
        self.b_qp = ca.Function(
            "b_qp", [g_b, self.V, self.Pi], [g_b - self.g + ca.mtimes(dg, self.V)]
        )

        V_lin = ca.MX.sym("V_lin", self.V.shape)

        self.F_qp = ca.Function(
            "F_qp",
            [self.V, self.Pi, V_lin],
            [
                0.5 * ca.mtimes(self.F1.T, self.F1)
                + ca.mtimes([(V_lin - self.V).T, dF1.T, self.F1])
                + 0.5 * ca.mtimes([(V_lin - self.V).T, B, (V_lin - self.V)])
                + self.F2
                + ca.mtimes(dF2, (V_lin - self.V))
            ],
        )

    def _setup_reduced_qp_components(self):

        dF1 = ca.jacobian(self.F1, self.V_r)
        dF2 = ca.jacobian(self.F2, self.V_r)
        B = ca.mtimes(dF1.T, dF1)

        self.H_qp_red = ca.Function("H_qp", [self.V_r, self.Pi], [B])

        self.q_qp_red = ca.Function(
            "q_qp",
            [self.V_r, self.Pi],
            [ca.mtimes(dF1.T, self.F1) - ca.mtimes(B, self.V_r) + dF2.T],
        )

        dg = ca.jacobian(self.g_r, self.V_r)

        dg_p1 = ca.jacobian(self.g_r[: self.g_r.numel() // 2], self.V_r)
        dg_p2 = ca.jacobian(self.g_r[self.g_r.numel() // 2 :], self.V_r)

        self.A_qp_p1_red = ca.Function("A_qp_p1", [self.V_r, self.Pi], [dg_p1])
        self.A_qp_p2_red = ca.Function("A_qp_p2", [self.V_r, self.Pi], [dg_p2])

        g_b = ca.MX.sym("V_b", self.g_r.shape)
        dgi = ca.MX.sym("V_b", dg.shape)
        self.b_qp_red = ca.Function(
            "b_qp",
            [g_b, self.V_r, self.Pi, dgi],
            [g_b - self.g_r + ca.mtimes(dgi, self.V_r)],
        )

        V_r_lin = ca.MX.sym("V_lin", self.V_r.shape)

        self.F_qp_red = ca.Function(
            "F_qp",
            [self.V_r, self.Pi, V_r_lin],
            [
                0.5 * ca.mtimes(self.F1.T, self.F1)
                + ca.mtimes([(V_r_lin - self.V_r).T, dF1.T, self.F1])
                + 0.5 * ca.mtimes([(V_r_lin - self.V_r).T, B, (V_r_lin - self.V_r)])
                + self.F2
                + ca.mtimes(dF2, (V_r_lin - self.V_r))
            ],
        )

    def _save_qp_components_to_file(self):

        __dirname__ = os.path.dirname(os.path.abspath(__file__))

        for qp_component in [
            "H_qp",
            "H_qp_red",
            "q_qp",
            "q_qp_red",
            "A_qp",
            "A_qp_p1_red",
            "A_qp_p2_red",
            "b_qp",
            "b_qp_red",
            "F_qp",
            "F_qp_red",
        ]:

            getattr(self, qp_component).save(f"{qp_component}.casadi")

            os.rename(
                f"{qp_component}.casadi",
                os.path.join(
                    __dirname__, _PATH_TO_NLP_OBJECT, f"{qp_component}.casadi"
                ),
            )

    def _save_binary_variables_indices(self):

        __dirname__ = os.path.dirname(os.path.abspath(__file__))

        for idx in ["idx_b", "idx_b_red", "idx_sb", "idx_sb_red"]:

            np.savetxt(
                os.path.join(__dirname__, _PATH_TO_NLP_OBJECT, f"{idx}.txt"),
                getattr(self, idx),
            )

    def generate_qp_files(self):

        logger.info("Generating QP objects ...")

        if not self._nlp_set_up:
            self._setup_nlp()
        self._setup_qp_components()
        self._setup_reduced_qp_components()
        self._setup_directories()
        self._save_qp_components_to_file()
        self._save_binary_variables_indices()

        logger.info("Finished generating QP object.")

    def _remove_unserializable_attributes(self):

        super()._remove_unserializable_attributes()

        for attrib in [
            "V",
            "V_r",
            "g",
            "g_r",
            "F1",
            "F2",
            "Pi",
            "fx",
            "nlp",
            "H_qp",
            "q_qp",
            "A_qp",
            "b_qp",
            "H_qp_red",
            "q_qp_red",
            "A_qp_red",
            "b_qp_red",
            "_dmiqp",
        ]:

            try:
                delattr(self, attrib)
            except AttributeError:
                print("No attribute ", attrib, ", therefore not deleted")


if __name__ == "__main__":

    import datetime as dt
    from timing import TimingMPC

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch = logging.StreamHandler()

    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    startup_time = dt.datetime.fromisoformat("2010-08-19 06:00:00+02:00")

    timing_mpc = TimingMPC(startup_time=startup_time)
    nlpsetup_mpc = NLPSetupMPC(timing=timing_mpc)

    system = System()
    system.generate_ode_file()

    nlpsetup_mpc.generate_nlp_object(
        use_big_m_constraints=True,  # use Big M constraints for ACM operation conditions, can improve results
        optimize_for_speed=True,  # takes longer to compile, but improves runtime
        overwrite_existing_object=True,
    )

    nlpsetup_mpc.generate_qp_files()
