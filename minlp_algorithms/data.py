"""
Data for a given problem.

This file contains the data of a problem. This includes lower and upper
bounds of the variables and equations, an initial value and
the previous solution if existing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import casadi as ca
from copy import deepcopy


@dataclass
class MinlpData:
    """Nlp data."""

    p: List[float]
    x0: ca.DM
    _lbx: ca.DM
    _ubx: ca.DM
    _lbg: ca.DM
    _ubg: ca.DM
    _prev_solutions: Optional[List[Dict[str, Any]]] = None
    solved_all: List[bool] = field(default_factory=lambda: [True])
    best_solutions: Optional[list] = field(default_factory=lambda: [])
    relaxed: bool = False

    @property
    def nr_sols(self):
        if self._prev_solutions:
            return len(self._prev_solutions)
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
        if self._prev_solutions is not None:
            return self._prev_solutions[0]
        else:
            return {"f": -ca.inf, "x": ca.DM(self.x0)}

    @property
    def solutions_all(self):
        if self._prev_solutions is not None:
            return self._prev_solutions
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
        self.relaxed = False

    @property
    def prev_solutions(self):
        """Get all previous solutions."""
        return self._prev_solutions

    @prev_solutions.setter
    def prev_solutions(self, value):
        self._prev_solutions = value
        self.relaxed = False

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
