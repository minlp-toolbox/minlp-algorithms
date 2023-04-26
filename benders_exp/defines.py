"""Global defines."""

from os import path
import casadi as ca

SOURCE_FOLDER = path.dirname(path.abspath(__file__))
_DATA_FOLDER = path.join(SOURCE_FOLDER, "../data")

WITH_JIT = False
WITH_LOGGING = True
WITH_PLOT = False
CASADI_VAR = ca.MX
IPOPT_SETTINGS = {
    # "ipopt.tol": 1e-2,
    # "ipopt.dual_inf_tol": 2,
    # "ipopt.constr_viol_tol": 1e-3,
    # "ipopt.compl_inf_tol": 1e-3,
    "ipopt.linear_solver": "ma27",
    "ipopt.max_cpu_time": 3600.0,
    "ipopt.max_iter": 6000,
    "ipopt.acceptable_tol": 0.2,
    "ipopt.acceptable_iter": 8,
    "ipopt.acceptable_constr_viol_tol": 10.0,
    "ipopt.acceptable_dual_inf_tol": 10.0,
    "ipopt.acceptable_compl_inf_tol": 10.0,
    "ipopt.acceptable_obj_change_tol": 1e-1,
    "ipopt.mu_strategy": "adaptive",
    "ipopt.mu_target": 1e-4,
    "ipopt.print_level": 1,
}
