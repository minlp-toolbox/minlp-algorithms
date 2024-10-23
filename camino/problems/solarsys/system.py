"""
Solar Thermal Climate System (STCS) at Karsruhe University of Applied Sciences.

Adrian Buerger, 2022
Adapted by Wim Van Roy and Andrea Ghezzi, 2023
"""

import numpy as np
import casadi as ca

from camino.settings import GlobalSettings
from camino.utils.cache import CachedFunction
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class System:
    """System dynamics."""

    @staticmethod
    def get_default_initial_state():
        """Get default initial states."""
        return {
            'T_hts': [70.0, 65.0, 63.0, 60.0],
            'T_lts': 14.0,
            'T_fpsc': 20.0,
            'T_fpsc_s': 20.0,
            'T_vtsc': 22.0,
            'T_vtsc_s': 22.0,
            'T_pscf': 18.0,
            'T_pscr': 20.0,
            'T_shx_psc': [11.0, 11.0, 11.0, 11.0],
            'T_shx_ssc': [32.0, 32.0, 32.0, 32.0],
        }

    def _setup_system_dimensions(self):
        self.nx = 19
        self.nb = 3
        self.nu = 6
        self.nc = 6

        self.n_s_ac_lb = 2
        self.n_s_ac_ub = 2

    def _setup_system_components(self):
        # Model parameters
        self.p = {
            # Media
            "rho_w": 1.0e3,
            "c_w": 4.182e3,
            "rho_sl": 1.0e3,
            "c_sl": 3.8e3,
            # Storages
            "V_hts": 2.0,
            "V_lts": 1.0,
            "lambda_hts": [1.15482, 2.89951, 1.195370, 1.000000],
            "eps_hts": 1e-2,
            # Cooling system
            "dT_lc": 4.0,
            "mdot_lc_max": 0.5,
            # Flat plate collectors
            "eta_fpsc": 0.857461,
            "A_fpsc": 53.284,
            "C_fpsc": 248552.0,
            "alpha_fpsc": 1.97512,
            "V_fpsc_s": 2.0e-3,
            "lambda_fpsc_s": 1.0,
            # Tube collectors
            "eta_vtsc": 0.835968,
            "A_vtsc": 31.136,
            "C_vtsc": 445931.0,
            "alpha_vtsc": 1.76747,
            "V_vtsc_s": 2.0e-3,
            "lambda_vtsc_s": 1.0,
            # Pipes connecting solar collectors and solar heat exchanger
            "lambda_psc": 133.144,
            "C_psc": 378160.0,
            # Solar heat exchanger
            "V_shx": 3.8e-3,
            "A_shx": 4.02,
            "alpha_shx": 22535.7,
            "mdot_ssc_max": 0.625,
            # ACM
            "mdot_ac_ht": 0.693,
            "T_ac_ht_min": 55.0,
            "T_ac_ht_max": 95.0,
            "dT_ht_min": 5.0,
            "mdot_ac_lt": 0.8,
            "T_ac_lt_min": 10.0,
            "T_ac_lt_max": 26.0,
            "dT_lt_min": 2.0,
            "T_ac_mt_min": 15.0,
            "T_ac_mt_max": 40.0,
            "Qdot_ac_lb": 0.1,
            "Qdot_ac_ub": 15.0,
            "COP_ac_lb": 0.1,
            "COP_ac_ub": 0.6,
            # HP
            "mdot_hp_lt": 0.5,
            # Recooling tower
            "dT_rc": 2.0,
            # Free cooling
            "mdot_fc_lt": 0.693,
            # PV collectors
            "P_pv_p": 2.0,
        }

        # States

        self.x_index = OrderedDict(
            [
                ("T_hts", [0, 1, 2, 3]),
                ("T_lts", 4),
                ("T_fpsc", 5),
                ("T_fpsc_s", 6),
                ("T_vtsc", 7),
                ("T_vtsc_s", 8),
                ("T_pscf", 9),
                ("T_pscr", 10),
                ("T_shx_psc", [11, 12, 13, 14]),
                ("T_shx_ssc", [15, 16, 17, 18]),
            ]
        )

        self.x_aux_index = {}

        # Continuous controls

        self.u_index = OrderedDict(
            [
                ("v_ppsc", 0),
                ("p_mpsc", 1),
                ("v_pssc", 2),
                ("P_g", 3),
                ("mdot_o_hts_b", 4),
                ("mdot_i_hts_b", 5),
            ]
        )

        # Discrete controls

        self.b_index = OrderedDict(
            [
                ("b_ac", 0),
                ("b_fc", 1),
                ("b_hp", 2),
            ]
        )

        # Time-varying parameters

        self.c_index = OrderedDict(
            [
                ("T_amb", 0),
                ("I_fpsc", 1),
                ("I_vtsc", 2),
                ("Qdot_c", 3),
                ("P_pv_kWp", 4),
                ("p_g", 5),
            ]
        )

    def _setup_operation_specifications_and_limits(self):

        self.p_op = {
            "T": {"min": 8, "max": 98.0},
            "T_lts": {"max": 18.0},
            "T_sc": {
                "T_sc_so": 65.0,
                "M_sc_so": 60.0,
                "v_ppsc_so": 1.0,
                "T_feed_max": 85.0,
            },
            "p_mpsc": {
                "min_mpc": 0.25,
                "min_real": 0.25,
                "max": 0.884,
            },
            "v_ppsc": {"min_mpc": 0.0, "min_real": 0.3, "max": 1.0, "Pmax": 4e2},
            "dp_mpsc": {
                "Pmax": 1e0,
            },
            "dp_mssc": {
                "Pmax": 1e0,
            },
            "dp_macm": {
                "Pmax": 1e0,
            },
            "v_pssc": {"min_mpc": 0.0, "min_real": 0.3, "max": 1.0, "Pmax": 3e2},
            "acm": {
                "min_up_time": [3600.0, 900.0],
                "min_down_time": [1800.0, 900.0],
                "Pmax": [1.5e3, 1.2e3]
                # "Pmax": [1.5e3 + 1.6e3, 1.2e3 + 1.6e3]
            },
            "hp": {"min_up_time": [3600.0], "min_down_time": [1800.0]},
            "grid": {"P_g_max": 1e4},
        }

    def _setup_simulation_control_parameters(self):

        self.p_csim = {
            "dT_ac_ht": 5.0,
            "dT_ac_lt": 1.0,
            "dT_ac_mt": 1.0,
            "dT_sc_ub": 15.0,
            "dT_sc_lb": 5.0,
            "dT_vtsc_fpsc_ub": 5.0,
            "dT_vtsc_fpsc_lb": -5.0,
            "dT_o_hts_b_ub": 10.0,
            "dT_o_hts_b_lb": 0.0,
            "T_i_hts_b_active": 80.0,
            "dT_i_hts_b_ub": 10.0,
            "dT_i_hts_b_lb": 0.0,
        }

    def __init__(self):
        """Create system."""
        logger.debug("Creating system")
        self._setup_system_dimensions()
        self._setup_system_components()
        self._setup_operation_specifications_and_limits()
        self._setup_simulation_control_parameters()
        logger.debug("Creating system variables")
        self._setup_model_variables()
        logger.debug("System initialized")

    def _setup_model_variables(self):
        """Set up variables."""
        self.x = GlobalSettings.CASADI_VAR.sym("x", self.nx)
        self.b = GlobalSettings.CASADI_VAR.sym("b", self.nb)
        self.u = GlobalSettings.CASADI_VAR.sym("u", self.nu)
        self.c = GlobalSettings.CASADI_VAR.sym("c", self.nc)

    def get_f_fcn(self):
        """Create solar thermal climate system model."""
        logger.debug("Creating system model")
        # States
        T_hts = self.x[self.x_index["T_hts"]]
        T_lts = self.x[self.x_index["T_lts"]]

        T_fpsc = self.x[self.x_index["T_fpsc"]]
        T_fpsc_s = self.x[self.x_index["T_fpsc_s"]]
        T_vtsc = self.x[self.x_index["T_vtsc"]]
        T_vtsc_s = self.x[self.x_index["T_vtsc_s"]]

        T_pscf = self.x[self.x_index["T_pscf"]]
        T_pscr = self.x[self.x_index["T_pscr"]]

        T_shx_psc = self.x[self.x_index["T_shx_psc"]]
        T_shx_ssc = self.x[self.x_index["T_shx_ssc"]]

        # Discrete controls
        b_ac = self.b[self.b_index["b_ac"]]
        b_fc = self.b[self.b_index["b_fc"]]
        b_hp = self.b[self.b_index["b_hp"]]

        # Continuous controls
        v_ppsc = self.u[self.u_index["v_ppsc"]]
        p_mpsc = self.u[self.u_index["p_mpsc"]]
        v_pssc = self.u[self.u_index["v_pssc"]]

        mdot_o_hts_b = self.u[self.u_index["mdot_o_hts_b"]]
        mdot_i_hts_b = self.u[self.u_index["mdot_i_hts_b"]]

        # Time-varying parameters
        T_amb = self.c[self.c_index["T_amb"]]
        I_vtsc = self.c[self.c_index["I_vtsc"]]
        I_fpsc = self.c[self.c_index["I_fpsc"]]

        Qdot_c = self.c[self.c_index["Qdot_c"]]

        # Modeling

        f = []

        # Grey box ACM model (spline version)

        def qdot_ac(T_lts, T_hts, T_mts):

            if (
                (T_lts < self.p["T_ac_lt_min"] - self.p["dT_lt_min"])
                or (T_hts < self.p["T_ac_ht_min"] - self.p["dT_ht_min"])
                or (T_mts > self.p["T_ac_mt_max"])
            ):

                return self.p["Qdot_ac_lb"]

            T_mts = max(T_mts, self.p["T_ac_mt_min"])

            char_curve_acm = np.asarray(
                [
                    1.0,
                    T_lts,
                    T_hts,
                    (T_mts),
                    T_lts**2,
                    T_hts**2,
                    (T_mts) ** 2,
                    T_lts * T_hts,
                    T_lts * (T_mts),
                    T_hts * (T_mts),
                    T_lts * T_hts * (T_mts),
                ]
            )

            params_Qdot_ac_lt = np.asarray(
                [
                    -5.6924,
                    1.12102,
                    0.291654,
                    -0.484546,
                    -0.00585722,
                    -0.00140483,
                    0.00795341,
                    0.00399118,
                    -0.0287113,
                    -0.00221606,
                    5.42825e-05,
                ]
            )

            return np.dot(char_curve_acm, params_Qdot_ac_lt)

        def cop_ac(T_lts, T_hts, T_mts):

            if (
                (T_lts < self.p["T_ac_lt_min"] - self.p["dT_lt_min"])
                or (T_hts < self.p["T_ac_ht_min"] - self.p["dT_ht_min"])
                or (T_mts > self.p["T_ac_mt_max"])
            ):

                return self.p["COP_ac_lb"]

            T_mts = max(T_mts, self.p["T_ac_mt_min"])

            char_curve_acm = np.asarray(
                [
                    1.0,
                    T_lts,
                    T_hts,
                    (T_mts),
                    T_lts**2,
                    T_hts**2,
                    (T_mts) ** 2,
                    T_lts * T_hts,
                    T_lts * (T_mts),
                    T_hts * (T_mts),
                    T_lts * T_hts * (T_mts),
                ]
            )

            params_COP_ac = np.asarray(
                [
                    2.03268,
                    -0.116526,
                    -0.0165648,
                    -0.043367,
                    -0.00074309,
                    -0.000105659,
                    -0.00172085,
                    0.00113422,
                    0.00540221,
                    0.00116735,
                    -3.87996e-05,
                ]
            )

            return np.dot(char_curve_acm, params_COP_ac)

        def setup_acm_spline(name, data_fcn, lb, ub):

            T_lts = np.arange(-4, 50, 2)
            T_hts = np.arange(0, 140, 2)
            T_mts = np.arange(-5, 45, 2)

            data = []

            for T_lts_k in T_lts:
                for T_hts_l in T_hts:
                    for T_mts_m in T_mts:

                        T_lts_s = max(
                            self.p["T_ac_lt_min"], min(
                                self.p["T_ac_lt_max"], T_lts_k)
                        )
                        T_hts_s = max(
                            self.p["T_ac_ht_min"], min(
                                self.p["T_ac_ht_max"], T_hts_l)
                        )
                        T_mts_s = max(
                            self.p["T_ac_mt_min"], min(
                                self.p["T_ac_mt_max"], T_mts_m)
                        )

                        if (
                            (T_lts_k < self.p["T_ac_lt_min"])
                            or (T_hts_l < self.p["T_ac_ht_min"])
                            or (T_mts_m > self.p["T_ac_mt_max"])
                        ):
                            data.append(lb)
                        else:
                            data.append(
                                data_fcn(T_lts=T_lts_s,
                                         T_hts=T_hts_s, T_mts=T_mts_s)
                            )

            data = np.asarray(data)
            data[data <= lb] = lb
            data[data >= ub] = ub

            return ca.interpolant(name, "bspline", [T_mts, T_hts, T_lts], data)

        iota_qdot = setup_acm_spline(
            "qdot", qdot_ac, self.p["Qdot_ac_lb"], self.p["Qdot_ac_ub"]
        )
        iota_cop = setup_acm_spline(
            "cop", cop_ac, self.p["COP_ac_lb"], self.p["COP_ac_ub"]
        )

        Qdot_ac_lt = (
            iota_qdot(
                ca.veccat(T_amb + self.p["dT_rc"], T_hts[0], T_lts)) * 1e3
        )
        # Better would be to limit COP_ac!
        COP_ac = iota_cop(ca.veccat(T_amb + self.p["dT_rc"], T_hts[0], T_lts))

        # TODO: Possible error
        T_ac_ht = T_hts[0] - (
            (Qdot_ac_lt) / ca.fmax(COP_ac, 1e-6) /
            (self.p["mdot_ac_ht"] * self.p["c_w"])
        )
        T_ac_lt = T_lts - (Qdot_ac_lt / (self.p["mdot_ac_lt"] * self.p["c_w"]))

        # Free cooling

        T_fc_lt = T_amb + self.p["dT_rc"]

        # Heat pump

        Qdot_hp_lt = (
            0.35246631376323156 * T_lts
            - 0.07341224489795989 * (T_amb + self.p["dT_rc"])
            + 10.882680173898295
        ) * 1e3
        T_hp_lt = T_lts - (Qdot_hp_lt / (self.p["mdot_hp_lt"] * self.p["c_w"]))

        # HT storage model

        mdot_ssc = self.p["mdot_ssc_max"] * v_pssc

        # Parameter hts:
        m_hts = (self.p["V_hts"] * self.p["rho_w"]) / T_hts.numel()

        mdot_hts_t_s = mdot_ssc - b_ac * self.p["mdot_ac_ht"]
        mdot_hts_t_sbar = ca.sqrt(mdot_hts_t_s**2 + self.p["eps_hts"])

        mdot_hts_b_s = mdot_i_hts_b - mdot_o_hts_b
        mdot_hts_b_sbar = ca.sqrt(mdot_hts_b_s**2 + self.p["eps_hts"])

        f.append(
            (1.0 / m_hts)
            * (
                mdot_ssc * T_shx_ssc[-1]
                - b_ac * self.p["mdot_ac_ht"] * T_hts[0]
                - (
                    mdot_hts_t_s * ((T_hts[0] + T_hts[1]) / 2.0)
                    + mdot_hts_t_sbar * ((T_hts[0] - T_hts[1]) / 2.0)
                )
                - (self.p["lambda_hts"][0] /
                   self.p["c_w"] * (T_hts[0] - T_amb))
            )
        )

        f.append(
            (1.0 / m_hts)
            * (
                (b_ac * self.p["mdot_ac_ht"] - mdot_i_hts_b) * T_ac_ht
                - (mdot_ssc - mdot_o_hts_b) * T_hts[1]
                + (
                    mdot_hts_t_s * ((T_hts[0] + T_hts[1]) / 2.0)
                    + mdot_hts_t_sbar * ((T_hts[0] - T_hts[1]) / 2.0)
                )
                + (
                    mdot_hts_b_s * ((T_hts[2] + T_hts[1]) / 2.0)
                    + mdot_hts_b_sbar * ((T_hts[2] - T_hts[1]) / 2.0)
                )
                - (self.p["lambda_hts"][3] /
                   self.p["c_w"] * (T_hts[1] - T_amb))
            )
        )

        f.append(
            (1.0 / m_hts)
            * (
                mdot_hts_b_s
                * (((T_hts[-1] + T_hts[-2]) / 2.0) - ((T_hts[-2] + T_hts[-3]) / 2.0))
                + mdot_hts_b_sbar
                * (((T_hts[-1] - T_hts[-2]) / 2.0) - ((T_hts[-2] - T_hts[-3]) / 2.0))
                - (self.p["lambda_hts"][-2] /
                   self.p["c_w"] * (T_hts[-2] - T_amb))
            )
        )

        f.append(
            (1.0 / m_hts)
            * (
                mdot_i_hts_b * T_ac_ht
                - mdot_o_hts_b * T_hts[-1]
                - (
                    mdot_hts_b_s * ((T_hts[-1] + T_hts[-2]) / 2.0)
                    + mdot_hts_b_sbar * ((T_hts[-1] - T_hts[-2]) / 2.0)
                )
                - (self.p["lambda_hts"][-1] /
                   self.p["c_w"] * (T_hts[-1] - T_amb))
            )
        )

        # LT storage and cooling system model

        mdot_lc = Qdot_c / (self.p["c_w"] * self.p["dT_lc"])

        m_lts = self.p["V_lts"] * self.p["rho_w"]

        f.append(
            (1.0 / m_lts)
            * (
                mdot_lc * self.p["dT_lc"]
                + b_ac * self.p["mdot_ac_lt"] * (T_ac_lt - T_lts)
                + b_fc * self.p["mdot_fc_lt"] * (T_fc_lt - T_lts)
                + b_hp * self.p["mdot_hp_lt"] * (T_hp_lt - T_lts)
            )
        )

        # Flat plate collectors

        data_v_ppsc = [-0.1, 0.0, 0.4, 0.6, 0.8, 1.0, 1.1]
        data_p_mpsc = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        data_mdot_fpsc = np.array(
            [
                0.00116667,
                0.0,
                0.02765,
                0.03511667,
                0.04258333,
                0.04993333,
                0.04993333,
                0.00116667,
                0.0,
                0.02765,
                0.03511667,
                0.04258333,
                0.04993333,
                0.04993333,
                0.0035,
                0.0,
                0.08343,
                0.11165333,
                0.13568,
                0.15895333,
                0.15895333,
                0.005,
                0.0,
                0.12508167,
                0.16568,
                0.20563333,
                0.2397,
                0.2397,
                0.0055,
                0.0,
                0.13999333,
                0.1859,
                0.22790167,
                0.26488,
                0.26488,
                0.006,
                0.0,
                0.14969167,
                0.19844,
                0.24394167,
                0.28695333,
                0.28695333,
                0.00633333,
                0.0,
                0.15706667,
                0.21070833,
                0.25807,
                0.310775,
                0.310775,
                0.0075,
                0.0,
                0.17047833,
                0.229775,
                0.28826667,
                0.34190833,
                0.34190833,
                0.0095,
                0.0,
                0.20687333,
                0.27500667,
                0.331775,
                0.37235333,
                0.37235333,
                0.013,
                0.0,
                0.24111667,
                0.31581,
                0.38300833,
                0.44705167,
                0.44705167,
                0.013,
                0.0,
                0.24111667,
                0.31581,
                0.38300833,
                0.44705167,
                0.44705167,
            ]
        )

        iota_mdot_fpsc = ca.interpolant(
            "iota_mdot_fpsc", "bspline", [
                data_v_ppsc, data_p_mpsc], data_mdot_fpsc
        )

        mdot_fpsc = iota_mdot_fpsc(ca.veccat(v_ppsc, p_mpsc))

        Qdot_fpsc = self.p["eta_fpsc"] * self.p["A_fpsc"] * I_fpsc
        Qdot_fpsc_amb = self.p["alpha_fpsc"] * \
            self.p["A_fpsc"] * (T_fpsc - T_amb)

        f.append(
            (1.0 / self.p["C_fpsc"])
            * (
                mdot_fpsc * self.p["c_sl"] * (T_pscf - T_fpsc)
                + Qdot_fpsc
                - Qdot_fpsc_amb
            )
        )

        f.append(
            (1.0 / (self.p["V_fpsc_s"] * self.p["rho_sl"] * self.p["c_sl"]))
            * (
                mdot_fpsc * self.p["c_sl"] * (T_fpsc - T_fpsc_s)
                - self.p["lambda_fpsc_s"] * (T_fpsc_s - T_amb)
            )
        )

        # Tube collectors

        data_mdot_vtsc = np.array(
            [
                0.0155,
                0.0,
                0.36735,
                0.46655,
                0.56575,
                0.6634,
                0.6634,
                0.0155,
                0.0,
                0.36735,
                0.46655,
                0.56575,
                0.6634,
                0.6634,
                0.01316667,
                0.0,
                0.32157,
                0.41501333,
                0.50432,
                0.59438,
                0.59438,
                0.01166667,
                0.0,
                0.29325167,
                0.37932,
                0.4577,
                0.54363333,
                0.54363333,
                0.01116667,
                0.0,
                0.28167333,
                0.3641,
                0.44043167,
                0.52345333,
                0.52345333,
                0.01066667,
                0.0,
                0.271975,
                0.34822667,
                0.42439167,
                0.50138,
                0.50138,
                0.01033333,
                0.0,
                0.25626667,
                0.33095833,
                0.39859667,
                0.464225,
                0.464225,
                0.00916667,
                0.0,
                0.217855,
                0.275225,
                0.3384,
                0.39975833,
                0.39975833,
                0.00716667,
                0.0,
                0.15479333,
                0.19832667,
                0.243225,
                0.30098,
                0.30098,
                0.00366667,
                0.0,
                0.06721667,
                0.08752333,
                0.10865833,
                0.13128167,
                0.13128167,
                0.00366667,
                0.0,
                0.06721667,
                0.08752333,
                0.10865833,
                0.13128167,
                0.13128167,
            ]
        )

        iota_mdot_vtsc = ca.interpolant(
            "iota_mdot_vtsc", "bspline", [
                data_v_ppsc, data_p_mpsc], data_mdot_vtsc
        )

        mdot_vtsc = iota_mdot_vtsc(ca.veccat(v_ppsc, p_mpsc))

        Qdot_vtsc = self.p["eta_vtsc"] * self.p["A_vtsc"] * I_vtsc
        Qdot_vtsc_amb = self.p["alpha_vtsc"] * \
            self.p["A_vtsc"] * (T_vtsc - T_amb)

        f.append(
            (1.0 / self.p["C_vtsc"])
            * (
                mdot_vtsc * self.p["c_sl"] * (T_pscf - T_vtsc)
                + Qdot_vtsc
                - Qdot_vtsc_amb
            )
        )

        f.append(
            (1.0 / (self.p["V_vtsc_s"] * self.p["rho_sl"] * self.p["c_sl"]))
            * (
                mdot_vtsc * self.p["c_sl"] * (T_vtsc - T_vtsc_s)
                - self.p["lambda_vtsc_s"] * (T_vtsc_s - T_amb)
            )
        )

        # Pipes connecting solar collectors and solar heat exchanger

        f.append(
            1.0
            / self.p["C_psc"]
            * (
                (mdot_fpsc + mdot_vtsc) *
                self.p["c_sl"] * (T_shx_psc[-1] - T_pscf)
                - self.p["lambda_psc"] * (T_pscf - T_amb)
            )
        )

        f.append(
            1.0
            / self.p["C_psc"]
            * (
                mdot_fpsc * self.p["c_sl"] * T_fpsc_s
                + mdot_vtsc * self.p["c_sl"] * T_vtsc_s
                - (mdot_fpsc + mdot_vtsc) * self.p["c_sl"] * T_pscr
                - self.p["lambda_psc"] * (T_pscr - T_amb)
            )
        )

        # Solar heat exchanger

        m_shx_psc = self.p["V_shx"] * self.p["rho_sl"] / T_shx_psc.numel()
        m_shx_ssc = self.p["V_shx"] * self.p["rho_w"] / T_shx_psc.numel()

        A_shx_k = self.p["A_shx"] / T_shx_psc.numel()

        f.append(
            (1.0 / (m_shx_psc * self.p["c_sl"]))
            * (
                (mdot_fpsc + mdot_vtsc) *
                self.p["c_sl"] * (T_pscr - T_shx_psc[0])
                - (A_shx_k * self.p["alpha_shx"] *
                   (T_shx_psc[0] - T_shx_ssc[-1]))
            )
        )

        for k in range(1, T_shx_psc.numel()):

            f.append(
                (1.0 / (m_shx_psc * self.p["c_sl"]))
                * (
                    (mdot_fpsc + mdot_vtsc)
                    * self.p["c_sl"]
                    * (T_shx_psc[k - 1] - T_shx_psc[k])
                    - (
                        A_shx_k
                        * self.p["alpha_shx"]
                        * (T_shx_psc[k] - T_shx_ssc[-1 - k])
                    )
                )
            )

        f.append(
            (1.0 / (m_shx_ssc * self.p["c_w"]))
            * (
                (mdot_ssc - mdot_o_hts_b) *
                self.p["c_w"] * (T_hts[1] - T_shx_ssc[0])
                + mdot_o_hts_b * self.p["c_w"] * (T_hts[-1] - T_shx_ssc[0])
                + (A_shx_k * self.p["alpha_shx"] *
                   (T_shx_psc[-1] - T_shx_ssc[0]))
            )
        )

        for k in range(1, T_shx_ssc.numel()):

            f.append(
                (1.0 / (m_shx_ssc * self.p["c_w"]))
                * (
                    mdot_ssc * self.p["c_w"] *
                    (T_shx_ssc[k - 1] - T_shx_ssc[k])
                    + (
                        A_shx_k
                        * self.p["alpha_shx"]
                        * (T_shx_psc[-1 - k] - T_shx_ssc[k])
                    )
                )
            )

        return ca.Function(
            "f", [self.x, self.c, self.u, self.b], [ca.veccat(*f)]
        )

    def get_integrator(self):
        """Set up simulator."""
        dt = ca.MX.sym("dt")
        f = CachedFunction("stcs_f", self.get_f_fcn)

        ode = {
            "x": self.x,
            "p": ca.veccat(dt, self.c, self.u, self.b),
            "ode": dt * f(self.x, self.c, self.u, self.b),
        }
        return ca.integrator(
            "integrator", "cvodes", ode, 0.0, 1.0
        )

    def get_t_ac_min_function(self, use_big_m_constraints=True):
        """
        Create a function to represent the minimum uptime.

        Function has as argument x, c, b, slack_variables
        The output is the T_ac_min constraint (>0).
        """
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
                self.x[self.x_index["T_lts"]] -
                self.p["T_ac_lt_min"] + s_ac_lb[0],
                self.x[self.x_index["T_hts"][0]] -
                self.p["T_ac_ht_min"] + s_ac_lb[1],
            )

        return ca.Function(
            "T_ac_min_fcn", [self.x, self.c, self.b, s_ac_lb], [T_ac_min]
        )

    def get_t_ac_max_function(self, use_big_m_constraints=True):
        """
        Create a function to represent the maximum uptime.

        Function has as argument x, c, b, slack_variables
        The output is the T_ac_min constraint (>0).
        """
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
                self.x[self.x_index["T_lts"]] -
                self.p["T_ac_lt_max"] - s_ac_ub[0],
                self.x[self.x_index["T_hts"][0]] -
                self.p["T_ac_ht_max"] - s_ac_ub[1],
            )

        return ca.Function(
            "T_ac_max_fcn", [self.x, self.c, self.b, s_ac_ub], [T_ac_max]
        )

    def get_slacked_state_fcn(self):
        s_x = ca.MX.sym("s_x", self.nx)
        return ca.Function("s_x_fcn", [self.x, s_x], [self.x + s_x])

    def get_v_ppsc_so_fpsc_fcn(self):
        """Get v_ppsc so function."""
        # Assure ppsc is running at high speed when collector temperature is high
        s_ppsc = ca.MX.sym("s_ppsc")
        return ca.Function(
            "v_ppsc_so_fpsc_fcn",
            [self.x, s_ppsc],
            [
                (self.x[self.x_index["T_fpsc"]] - self.p_op["T_sc"]["T_sc_so"])
                * (self.p_op["T_sc"]["v_ppsc_so"] - s_ppsc)
            ],
        )

    def get_v_ppsc_so_vtsc_fcn(self):
        """V ppsc."""
        s_ppsc = ca.MX.sym("s_ppsc")
        return ca.Function(
            "v_ppsc_so_vtsc_fcn",
            [self.x, s_ppsc],
            [
                (self.x[self.x_index["T_vtsc"]] - self.p_op["T_sc"]["T_sc_so"])
                * (self.p_op["T_sc"]["v_ppsc_so"] - s_ppsc)
            ],
        )

    def get_v_ppsc_so_fcn(self):
        """Get v ppsc so function."""
        s_ppsc = ca.MX.sym("s_ppsc")
        return ca.Function(
            "v_ppsc_so_fcn", [self.u, s_ppsc], [
                s_ppsc - self.u[self.u_index["v_ppsc"]]]
        )

    def get_mdot_hts_b_max_fcn(self):
        """Get mdot hts b max function."""
        # Assure HTS bottom layer mass flows are always smaller or equal to
        # the corresponding total pump flow

        mdot_hts_b_max = ca.veccat(
            self.u[self.u_index["mdot_o_hts_b"]]
            - self.p["mdot_ssc_max"] * self.u[self.u_index["v_pssc"]],
            self.u[self.u_index["mdot_i_hts_b"]]
            - self.b[self.b_index["b_ac"]] * self.p["mdot_ac_ht"],
        )

        return ca.Function(
            "mdot_hts_b_max_fcn", [self.u, self.b], [mdot_hts_b_max]
        )

    def get_electric_power_balance_fcn(self):
        """Get electric power balance fcn."""
        P_hp = (
            0.006381135707410529 * self.x[self.x_index["T_lts"]]
            + 0.06791020408163258 *
            (self.c[self.c_index["T_amb"]] + self.p["dT_rc"])
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

        return ca.Function(
            "electric_power_balance_fcn",
            [self.x, self.u, self.b, self.c],
            [electric_power_balance],
        )

    def get_F1_fcn(self):
        """Get F1."""
        s_ac_lb = ca.MX.sym("s_ac_lb", self.n_s_ac_lb)
        s_ac_ub = ca.MX.sym("s_ac_ub", self.n_s_ac_ub)
        s_x = ca.MX.sym("s_x", self.nx)
        u_prev = ca.MX.sym("u_prev", self.nu)

        F1 = ca.veccat(
            5e0 * s_ac_lb,
            5e0 * s_ac_ub,
            1e1 * s_x,
            1e2 * s_x[self.x_index["T_lts"]],
            self.p_op["dp_mpsc"]["Pmax"]
            * (u_prev[self.u_index["p_mpsc"]] -
               self.u[self.u_index["p_mpsc"]]),
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

        return ca.Function(
            "F1_fcn", [s_ac_lb, s_ac_ub, s_x, self.u, u_prev], [F1]
        )

    def get_F2_fcn(self):
        """Get F2."""
        F2 = (
            self.c[self.c_index["p_g"]]
            * self.p_op["grid"]["P_g_max"]
            * self.u[self.u_index["P_g"]]
        )

        return ca.Function("F2_fcn", [self.u, self.c], [F2])

    def get_system_dynamics_collocation(system, d=2):
        """Create collocation."""
        tau_root = [0] + ca.collocation_points(d, "radau")

        f = CachedFunction("stcs_f", system.get_f_fcn)
        C = np.zeros((d + 1, d + 1))
        D = np.zeros(d + 1)

        for j in range(d + 1):
            p = np.poly1d([1])
            for r in range(d + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (
                        tau_root[j] - tau_root[r]
                    )

            D[j] = p(1.0)
            pder = np.polyder(p)
            for r in range(d + 1):
                C[j, r] = pder(tau_root[r])

        # Collocation equations
        x_k_c = [ca.MX.sym("x_k_c_" + str(j), system.nx) for j in range(d + 1)]
        x_k_next_c = ca.MX.sym("x_k_next_c", system.nx)
        c_k_c = ca.MX.sym("c_k_c", system.nc)
        u_k_c = ca.MX.sym("u_k_c", system.nu)
        b_k_c = ca.MX.sym("b_k_c", system.nb)
        dt_k_c = ca.MX.sym("dt_k_c")

        eq_c = []
        for j in range(1, d + 1):
            x_p_c = 0
            for r in range(d + 1):
                x_p_c += C[r, j] * x_k_c[r]
            f_k_c = f(x_k_c[j], c_k_c, u_k_c, b_k_c)
            eq_c.append(dt_k_c * f_k_c - x_p_c)

        eq_c = ca.veccat(*eq_c)
        xf_c = 0

        for r in range(d + 1):
            xf_c += D[r] * x_k_c[r]
            eq_d = xf_c - x_k_next_c

        return ca.Function(
            "F",
            x_k_c + [x_k_next_c, c_k_c, u_k_c, b_k_c, dt_k_c],
            [eq_c, eq_d],
            ["x_k_" + str(j) for j in range(d + 1)]
            + ["x_k_next", "c_k", "u_k", "b_k", "dt_k"],
            ["eq_c", "eq_d"],
        )


if __name__ == "__main__":
    system = System()
