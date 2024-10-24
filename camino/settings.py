# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Global settings."""

from typing import Any, Dict
from dataclasses import dataclass, field
from os import path, makedirs, environ
import casadi as ca


def create_and_return(folder):
    if not path.exists(folder):
        makedirs(folder)
    return folder


@dataclass
class _GlobalSettings:
    _lock: bool = False
    _CASADI_VAR: Any = ca.SX
    SOURCE_FOLDER: str = path.dirname(path.abspath(__file__))
    _DATA_FOLDER: str = path.join(SOURCE_FOLDER, "../data")
    _IMG_DIR: str = path.join(SOURCE_FOLDER, "../results/figures")
    _OUT_DIR: str = path.join(SOURCE_FOLDER, "../results")
    _CACHE_FOLDER: str = path.join(SOURCE_FOLDER, "../data/cache")

    @property
    def CASADI_VAR(self):
        self._lock = True
        return self._CASADI_VAR

    @CASADI_VAR.setter
    def CASADI_VAR(self, value: Any):
        if self._lock:
            raise RuntimeError(
                "Casadi var already used! You can not change it anymore!"
            )
        self._CASADI_VAR = value

    @property
    def IMG_DIR(self):
        return create_and_return(self._IMG_DIR)

    @IMG_DIR.setter
    def IMG_DIR(self, value):
        """Set image directory."""
        self._IMG_DIR = value

    @property
    def OUT_DIR(self):
        return create_and_return(self._OUT_DIR)

    @OUT_DIR.setter
    def OUT_DIR(self, value):
        """Set output dir directory."""
        self._OUT_DIR = value

    @property
    def CACHE_FOLDER(self):
        return create_and_return(self._CACHE_FOLDER)

    @CACHE_FOLDER.setter
    def CACHE_FOLDER(self, value):
        """Set output dir directory."""
        self._CACHE_FOLDER = value

    @property
    def DATA_FOLDER(self):
        return create_and_return(self._DATA_FOLDER)

    @DATA_FOLDER.setter
    def DATA_FOLDER(self, value):
        """Set output dir directory."""
        self._DATA_FOLDER = value


GlobalSettings = _GlobalSettings()


@dataclass(init=True)
class Settings:
    from camino.utils.conversion import to_bool

    TIME_LIMIT: float = ca.inf  # 60.0
    TIME_LIMIT_SOLVER_ONLY: bool = False
    WITH_JIT: bool = False
    WITH_PLOT: bool = False
    INF: float = 1e8
    BIGM: float = 1e6
    EPS: float = 1e-6
    OBJECTIVE_TOL: float = 1e-5
    CONSTRAINT_INT_TOL: float = 1e-2  # Due to rounding, this will be almost EPS
    CONSTRAINT_TOL: float = 1e-5
    BENDERS_LB: float = -1e16
    _MIP_SOLVER: str = "gurobi"

    WITH_DEBUG: bool = to_bool(environ.get("DEBUG", False))
    WITH_LOG_DATA: bool = to_bool(environ.get("LOG_DATA", False))
    WITH_SAVE_FIG = to_bool(environ.get("SAVE_FIG", False))
    MINLP_TOLERANCE: float = 0.01
    MINLP_TOLERANCE_ABS: float = 0.01
    BRMIQP_GAP: float = 1e-4
    LBMILP_GAP: float = 1e-4
    RHO_AMPLIFICATION: float = 1.5
    ALPHA_KRONQVIST: float = 0.5
    USE_RELAXED_AS_WARMSTART = False
    # WITH_DEFAULT_SETTINGS = to_bool(environ.get("DEFAULT", True))

    PUMP_MAX_STEP_IMPROVEMENTS = 5
    PUMP_MAX_ITER = 1000
    PUMP_MAX_TRY = 10
    PARALLEL_SOLUTIONS = 5

    AMPL_EXPORT_SETTINGS: Dict[str, Any] = field(default_factory=lambda: {})
    IPOPT_SETTINGS: Dict[str, Any] = field(default_factory=lambda: {
        "ipopt.linear_solver": "ma27",
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
