"""Defines."""

from benders_exp.defines import path, SOURCE_FOLDER, _DATA_FOLDER  # noqa: F401

PICKLE_FOLDER = path.join(SOURCE_FOLDER, "../results/voronoi")

_PATH_TO_NLP_SOURCE = path.join(SOURCE_FOLDER, "../.src/")
_PATH_TO_NLP_OBJECT = path.join(SOURCE_FOLDER, "../.lib/")
_PATH_TO_ODE_OBJECT = path.join(SOURCE_FOLDER, "../.lib/")
_PATH_TO_ODE_FILE = _PATH_TO_ODE_OBJECT

_NLP_SOURCE_FILENAME = "nlp_mpc.c"
_NLP_OBJECT_FILENAME = "nlp_mpc.so"


RESULTS_FOLDER = path.join(SOURCE_FOLDER, "../results/standard")

NLP_OPTIONS_GENERAL = {
    "ipopt.linear_solver": "ma57",
    "ipopt.mumps_mem_percent": 10000,
    "ipopt.mumps_pivtol": 0.001,
    "ipopt.print_level": 5,
    "ipopt.file_print_level": 5,
    "ipopt.max_cpu_time": 3600.0,
    "ipopt.max_iter": 600000
}

NLP_OPTIONS_ADDITIONAL = {
    "ipopt.acceptable_tol": 0.2,
    "ipopt.acceptable_iter": 8,
    "ipopt.acceptable_constr_viol_tol": 10.0,
    "ipopt.acceptable_dual_inf_tol": 10.0,
    "ipopt.acceptable_compl_inf_tol": 10.0,
    "ipopt.acceptable_obj_change_tol": 1e-1,
    "ipopt.mu_strategy": "adaptive",
    "ipopt.mu_target": 1e-5,
}