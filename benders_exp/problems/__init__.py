"""General problem structure."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from benders_exp.defines import CASADI_VAR, ca
from copy import deepcopy


@dataclass
class MetaData:
    """Meta data class."""


@dataclass
class MetaDataOcp(MetaData):
    """Meta data in case the problem is an OCP."""

    n_state: Optional[int] = None
    n_control: Optional[int] = None
    idx_state: Optional[List[float]] = None
    idx_control: Optional[List[float]] = None
    initial_state: Optional[List[float]] = None  # TODO: initial_state needs to become an index list of p
    dt: Optional[float] = None
    scaling_coeff_control: Optional[List[float]] = None
    min_uptime: Optional[int] = None


@dataclass
class MinlpProblem:
    """Minlp problem description."""

    f: CASADI_VAR
    g: CASADI_VAR
    x: CASADI_VAR
    p: CASADI_VAR
    idx_x_bin: List[float]
    precompiled_nlp: Optional[str] = None

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
    solved: bool
    prev_solution: Optional[Dict[str, Any]] = None

    @property
    def _sol(self):
        """Get safely previous solution."""
        if self.prev_solution is not None:
            return self.prev_solution
        else:
            return {"f": -ca.inf, "x": ca.DM(self.x0)}

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
