"""General problem structure."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from benders_exp.defines import CASADI_VAR, ca, Settings
from copy import deepcopy
import numpy as np


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


@dataclass
class MinlpData:
    """Nlp data."""

    p: List[float]
    x0: ca.DM
    _lbx: ca.DM
    _ubx: ca.DM
    _lbg: ca.DM
    _ubg: ca.DM
    prev_solutions: Optional[List[Dict[str, Any]]] = None
    solved_all: List[bool] = field(default_factory=lambda: [True])
    best_solutions: Optional[list] = field(default_factory=lambda: [])

    @property
    def nr_sols(self):
        if self.prev_solutions:
            return len(self.prev_solutions)
        else:
            return 1

    @property
    def solved(self):
        return self.solved_all[0]

    @solved.setter
    def solved(self, value):
        self.solved_all = [value]

    @property
    def _sol(self):
        """Get safely previous solution."""
        if self.prev_solutions is not None:
            return self.prev_solutions[0]
        else:
            return {"f": -ca.inf, "x": ca.DM(self.x0)}

    @property
    def solutions_all(self):
        if self.prev_solutions is not None:
            return self.prev_solutions
        else:
            return [{"f": -ca.inf, "x": ca.DM(self.x0)}]

    @property
    def prev_solution(self):
        """Previous solution."""
        raise Exception("May not be used!")

    @property
    def obj_val(self):
        """Get float value."""
        return float(self._sol['f'])

    @property
    def x_sol(self):
        """Get x solution."""
        return self._sol['x']

    @property
    def lam_g_sol(self):
        """Get lambda g solution."""
        return self._sol['lam_g']

    @property
    def lam_x_sol(self):
        """Get lambda g solution."""
        return self._sol['lam_x']

    @prev_solution.setter
    def prev_solution(self, value):
        self.prev_solutions = [value]

    @property
    def lbx(self):
        """Get lbx."""
        return deepcopy(self._lbx)

    @property
    def ubx(self):
        """Get ubx."""
        return deepcopy(self._ubx)

    @property
    def lbg(self):
        """Get lbx."""
        return deepcopy(self._lbg)

    @property
    def ubg(self):
        """Get ubx."""
        return deepcopy(self._ubg)

    @lbg.setter
    def lbg(self, value):
        self._lbg = value

    @ubg.setter
    def ubg(self, value):
        self._ubg = value

    @lbx.setter
    def lbx(self, value):
        self._lbx = value

    @ubx.setter
    def ubx(self, value):
        self._ubx = value


def to_float(val):
    """To single float."""
    if isinstance(val, np.ndarray):
        return to_float(val[0])
    elif isinstance(val, list):
        return to_float(val[0])
    return val


def check_solution(problem: MinlpProblem, data: MinlpData, x_star, s: Settings, throws=True, check_objval=True):
    """Check a solution."""
    f = ca.Function("f", [problem.x, problem.p], [problem.f])
    g = ca.Function("g", [problem.x, problem.p], [problem.g])
    f_val = to_float(f(x_star, data.p).full())
    g_val = g(x_star, data.p).full().squeeze()
    lbg, ubg = data.lbg.squeeze(), data.ubg.squeeze()
    print(f"Objective value {f_val} (real) vs {data.obj_val}")
    msg = []
    if check_objval and abs(to_float(data.obj_val) - f_val) > s.OBJECTIVE_TOL:
        msg.append("Objective value wrong!")
    if np.any(data.lbx > x_star + s.CONSTRAINT_TOL):
        msg.append(f"Lbx > x* for indices:\n{np.nonzero(data.lbx > x_star).T}")
    if np.any(data.ubx < x_star - s.CONSTRAINT_TOL):
        msg.append(f"Ubx > x* for indices:\n{np.nonzero(data.ubx < x_star).T}")
    if np.any(lbg > g_val + s.CONSTRAINT_TOL):
        msg.append(f"{g_val=}  {lbg=}")
        msg.append("Lbg > g(x*,p) for indices:\n"
                   f"{np.nonzero(lbg > g_val)}")
    if np.any(ubg < g_val - s.CONSTRAINT_TOL):
        msg.append(f"{g_val=}  {ubg=}")
        msg.append("Ubg < g(x*,p) for indices:\n"
                   f"{np.nonzero(ubg < g_val)}")

    check_integer_feasible(problem.idx_x_bin, x_star, s, throws=throws)

    if msg:
        msg = "\n".join(msg)
        if throws:
            raise Exception(msg)
        else:
            print(msg)


def check_integer_feasible(idx_x_bin, x_star, s: Settings, throws=True):
    """Check if the solution is integer feasible."""
    x_bin = np.array(x_star)[idx_x_bin].squeeze()
    x_bin_rounded = np.round(x_bin)
    if np.any(np.abs(x_bin_rounded - x_bin) > s.CONSTRAINT_TOL):
        idx = np.nonzero(np.abs(x_bin_rounded - x_bin) > s.CONSTRAINT_INT_TOL)
        msg = f"Integer infeasible: {x_bin[idx]} {idx}"
        if throws:
            raise Exception(msg)
        else:
            print(throws)
