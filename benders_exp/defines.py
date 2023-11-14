"""Global defines."""

from os import path, makedirs, environ
import casadi as ca


def to_bool(val):
    """String or bool value to bool."""
    if isinstance(val, bool):
        return val
    else:
        return not (val.lower() in ["0", "false", "false", "no"])


TIME_LIMIT = ca.inf  # 600.0
SOURCE_FOLDER = path.dirname(path.abspath(__file__))
_DATA_FOLDER = path.join(SOURCE_FOLDER, "../data")
IMG_DIR = path.join(SOURCE_FOLDER, "../results/figures")
OUT_DIR = path.join(SOURCE_FOLDER, "../results")
CACHE_FOLDER = path.join(SOURCE_FOLDER, "../data/cache")
USE_SOLUTION_POOL = False

if not path.exists(IMG_DIR):
    makedirs(IMG_DIR)

if not path.exists(CACHE_FOLDER):
    makedirs(CACHE_FOLDER)


WITH_JIT = False
WITH_PLOT = False
EPS = 1e-5
OBJECTIVE_TOL = 1e-2
CONSTRAINT_INT_TOL = 1e-2
CONSTRAINT_TOL = 1e-3
BENDERS_LB = -1e16
CASADI_VAR = ca.MX
MIP_SOLVER = environ.get("MIP_SOLVER", "gurobi")
WITH_DEBUG = to_bool(environ.get("DEBUG", False))
WITH_LOG_DATA = to_bool(environ.get("LOG_DATA", False))
# WITH_DEFAULT_SETTINGS = to_bool(environ.get("DEFAULT", True))

IPOPT_SETTINGS = {  # TODO: make ipopt setting change according to the problem called
    "ipopt.linear_solver": "ma57",
    "ipopt.mumps_mem_percent": 10000,
    "ipopt.mumps_pivtol": 0.001,
    "ipopt.print_level": 5,
    "ipopt.max_cpu_time": 3600.0,
    "ipopt.max_iter": 600000,
    "ipopt.acceptable_tol": 1e-1,
    "ipopt.acceptable_iter": 8,
    "ipopt.acceptable_constr_viol_tol": 10.0,
    "ipopt.acceptable_dual_inf_tol": 10.0,
    "ipopt.acceptable_compl_inf_tol": 10.0,
    "ipopt.acceptable_obj_change_tol": 1e-1,
    "ipopt.mu_strategy": "adaptive",
    "ipopt.mu_target": 1e-3,
}
BONMIN_SETTINGS = {}
GUROBI_SETTINGS = {
    "gurobi.MIPGap": 0.05,
    "gurobi.NumericFocus": 1,
    "gurobi.FeasibilityTol": CONSTRAINT_TOL,
    "gurobi.IntFeasTol": OBJECTIVE_TOL,
    # "gurobi.PoolSearchMode": 2,  # Default 0
    # "gurobi.PoolSolutions": 100,  # Default 10
    # "gurobi.PoolObjBound" # Discard avoce this value
}
if USE_SOLUTION_POOL:
    GUROBI_SETTINGS["gurobi.PoolSearchMode"] = 1
    GUROBI_SETTINGS["gurobi.PoolSolutions"] = 5

BONMIN_SETTINGS = {}
HIGHS_SETTINGS = {}
MIP_SETTINGS_ALL = {
    "gurobi": GUROBI_SETTINGS,
    "highs": HIGHS_SETTINGS
}
if MIP_SOLVER not in MIP_SETTINGS_ALL:
    raise Exception("Configure a MIP_SOLVER from the list: {}" % ", ".join(
        MIP_SETTINGS_ALL.keys()
    ))

# Adapt settings based on configuration
MIP_SETTINGS = MIP_SETTINGS_ALL[MIP_SOLVER]
if not WITH_DEBUG:
    IPOPT_SETTINGS.update({
        "ipopt.print_level": 0,
    })
    GUROBI_SETTINGS.update({
        "gurobi.output_flag": 0,
    })
else:
    IPOPT_SETTINGS.update({
        "ipopt.print_level": 5,
    })
