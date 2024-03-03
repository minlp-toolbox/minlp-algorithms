"""Global defines."""

from typing import Any, Dict
from dataclasses import dataclass, field
from os import path, makedirs, environ
import casadi as ca


def to_bool(val):
    """String or bool value to bool."""
    if isinstance(val, bool):
        return val
    else:
        return not (val.lower() in ["0", "false", "false", "no"])


CASADI_VAR = ca.MX
SOURCE_FOLDER = path.dirname(path.abspath(__file__))
_DATA_FOLDER = path.join(SOURCE_FOLDER, "../data")
IMG_DIR = path.join(SOURCE_FOLDER, "../results/figures")
OUT_DIR = path.join(SOURCE_FOLDER, "../results")
CACHE_FOLDER = path.join(SOURCE_FOLDER, "../data/cache")
if not path.exists(IMG_DIR):
    makedirs(IMG_DIR)

if not path.exists(CACHE_FOLDER):
    makedirs(CACHE_FOLDER)


@dataclass(init=True)
class Settings:
    TIME_LIMIT: float = ca.inf  # 60.0
    TIME_LIMIT_SOLVER_ONLY: bool = False
    WITH_JIT: bool = False
    WITH_PLOT: bool = False
    EPS: float = 1e-6
    OBJECTIVE_TOL: float = 1e-5
    CONSTRAINT_INT_TOL: float = 1e-2  # Due to rounding, this will be almost EPS
    CONSTRAINT_TOL: float = 1e-5
    BENDERS_LB: float = -1e16
    _MIP_SOLVER: str = "gurobi"

    WITH_DEBUG: bool = to_bool(environ.get("DEBUG", False))
    WITH_LOG_DATA: bool = to_bool(environ.get("LOG_DATA", False))
    MINLP_TOLERANCE: float = 0.01
    MINLP_TOLERANCE_ABS: float = 0.01
    # WITH_DEFAULT_SETTINGS = to_bool(environ.get("DEFAULT", True))

    AMPL_EXPORT_SETTINGS: Dict[str, Any] = field(default_factory=lambda: {})
    IPOPT_SETTINGS: Dict[str, Any] = field(default_factory=lambda: {
        "ipopt.linear_solver": "ma57",
        "ipopt.mumps_mem_percent": 10000,
        "ipopt.mumps_pivtol": 0.001,
        "ipopt.max_cpu_time": 3600.0,
        "ipopt.max_iter": 600000,
        "ipopt.acceptable_tol": 1e-1,
        "ipopt.acceptable_iter": 8,
        "ipopt.acceptable_constr_viol_tol": 10.0,
        "ipopt.acceptable_dual_inf_tol": 10.0,
        "ipopt.acceptable_compl_inf_tol": 10.0,
        "ipopt.acceptable_obj_change_tol": 1e-1,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.mu_target": 1e-5,
    })
    BONMIN_SETTINGS: Dict[str, Any] = field(default_factory=lambda: {
        "bonmin.time_limit": Settings.TIME_LIMIT,
        "bonmin.allowable_fraction_gap": Settings.MINLP_TOLERANCE,
    })
    MIP_SETTINGS_ALL: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "highs": {},
        "cbc": {},
        "gurobi": {
            "gurobi.NumericFocus": 1,
            # Note since this is only the dual, and since rounding occurs,
            # this low tolerance does not affect the solution!
            "gurobi.FeasibilityTol": Settings.CONSTRAINT_INT_TOL,
            "gurobi.IntFeasTol": Settings.CONSTRAINT_INT_TOL,
            "gurobi.PoolSearchMode": 1,
            "gurobi.PoolSolutions": 1,
        }
    })

    def __post_init__(self, *args, **kwargs):
        """Settings."""
        super(Settings, self).__init__(*args, **kwargs)
        if self.WITH_DEBUG:
            self.IPOPT_SETTINGS.update({"ipopt.print_level": 5})
            self.MIP_SETTINGS_ALL['gurobi'].update({"gurobi.output_flag": 1})
        else:
            self.IPOPT_SETTINGS.update({"ipopt.print_level": 0})
            self.MIP_SETTINGS_ALL['gurobi'].update({"gurobi.output_flag": 0})
        self.MIP_SOLVER = environ.get("MIP_SOLVER", "gurobi")

    @property
    def USE_SOLUTION_POOL(self):
        return (self.MIP_SETTINGS_ALL["gurobi"]["gurobi.PoolSolutions"] > 1)

    @USE_SOLUTION_POOL.setter
    def USE_SOLUTION_POOL(self, value):
        if value:
            self.MIP_SETTINGS_ALL["gurobi"]["gurobi.PoolSearchMode"] = 1
            self.MIP_SETTINGS_ALL["gurobi"]["gurobi.PoolSolutions"] = 5
        else:
            self.MIP_SETTINGS_ALL["gurobi"]["gurobi.PoolSearchMode"] = 0
            self.MIP_SETTINGS_ALL["gurobi"]["gurobi.PoolSolutions"] = 1

    @property
    def MIP_SOLVER(self):
        return self._MIP_SOLVER

    @MIP_SOLVER.setter
    def MIP_SOLVER(self, value):
        if value not in self.MIP_SETTINGS_ALL:
            raise Exception("Configure a MIP_SOLVER from the list: %s" % str(", ".join(
                self.MIP_SETTINGS_ALL.keys()
            )))
        else:
            self._MIP_SOLVER = value

    @property
    def MIP_SETTINGS(self):
        return self.MIP_SETTINGS_ALL[self.MIP_SOLVER]
