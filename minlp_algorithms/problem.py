"""
This file contains the data types related to the problem description.

The MinlpProblem contain the equations, indices of integer variables, special
data to speed up the calculations and meta data.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Any
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
    """
    Minlp problem description.

    The variables and equations should be defined
    using the GlobaLSettings.CASADI_VAR.

    You need to at least fill in f, g, x, p and idx_x_bin. Note that if
    your hessian is not positive semi-definite, you might want to set the flag
    to avoid computational problems.
    """
    # Casadi objective expression (1x1)
    f: Any
    # Casadi constraints expression (1xM)
    g: Any
    # Casadi variables (1xN)
    x: Any
    # Casadi parameters
    p: Any
    # Indices of the integer variables
    idx_x_bin: List[float]
    # Flag if the hessian is not positive semi-definite
    hessian_not_psd: bool = False

    # SOC-type constraints
    h: Optional[Any] = None

    # Guass-Newton Hessian (NXN)
    gn_hessian: Optional[Any] = None
    # Indices of the linear constraints
    idx_g_lin: Optional[List[int]] = None
    # Indices of the linear integer constraints
    idx_g_lin_bin: Optional[List[int]] = None
    # Indices of the convex constraints
    idx_g_conv: Optional[List[int]] = None
    # Indices of the constraints that do not belong to any other category
    idx_g_other: Optional[List[int]] = None

    # A precompiled NLP of the problem formulation
    precompiled_nlp: Optional[str] = None
    # Quadratic form or approximation of the objective function
    f_qp: Optional[Any] = None

    # Meta data of the problem
    meta: MetaData = MetaData()
