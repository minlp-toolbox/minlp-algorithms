"""
This file contains the data types related to the problem description.

The MinlpProblem contain the equations, indices of integer variables, special
data to speed up the calculations and meta data.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from minlp_algorithms.defines import CASADI_VAR
import casadi as ca


@dataclass
class MetaData:
    """Meta data class."""


@dataclass
class MetaDataOcp(MetaData):
    """Meta data in case the problem is an OCP."""

    N_horizon: Optional[int] = None
    n_state: Optional[int] = None
    n_continuous_control: Optional[int] = None
    n_discrete_control: Optional[int] = None
    idx_state: Optional[List[float]] = None
    idx_bin_state: Optional[List[float]] = None
    idx_control: Optional[List[float]] = None
    idx_bin_control: Optional[List[float]] = None
    idx_other: Optional[Dict[str, List[float]]] = None
    idx_param: Optional[dict] = None
    f_dynamics: Optional[Callable] = None
    # TODO: initial_state needs to become an index list of p
    initial_state: Optional[List[float]] = None
    dt: Optional[float] = None
    scaling_coeff_control: Optional[List[float]] = None
    min_uptime: Optional[int] = None
    min_downtime: Optional[int] = None
    dump_solution: ca.Function = None
    idx_t: Optional[int] = None


@dataclass
class MetaDataMpc(MetaData):
    """Meta data in case the problem is an OCP."""

    plot: Callable[[MetaData, ca.DM], bool] = None
    shift: Callable[[MetaData, ca.DM], bool] = None


@dataclass
class MinlpProblem:
    """Minlp problem description."""

    f: CASADI_VAR
    g: CASADI_VAR
    x: CASADI_VAR
    p: CASADI_VAR
    idx_x_bin: List[float]

    h: Optional[CASADI_VAR] = None  # for soc-type constraints
    gn_hessian: Optional[CASADI_VAR] = None
    idx_g_lin: Optional[List[int]] = None
    idx_g_lin_bin: Optional[List[int]] = None
    idx_g_conv: Optional[List[int]] = None
    idx_g_other: Optional[List[int]] = None
    precompiled_nlp: Optional[str] = None

    hessian_not_psd: bool = False
    f_qp: Optional[CASADI_VAR] = None
    meta: MetaData = MetaData()
