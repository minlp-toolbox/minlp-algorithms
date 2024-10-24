# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""An easier way to define a problem, similar to casadi opti."""


import copy
from typing import Optional, Union, List, Tuple
from camino.settings import GlobalSettings
from casadi import inf, vertcat, hessian, Function, DM, reshape, jacobian, vcat, Sparsity, blockcat
from math import isinf
from camino.problems import MinlpProblem, MinlpData
import numpy as np


def extract(sol, lst):
    """Extract a value."""
    return [float(sol['x'][i]) for i in lst]


def make_list(value, nr=1):
    """Make list."""
    if isinstance(value, np.ndarray):
        return list(np.reshape(value, (nr,)))
    elif not isinstance(value, list):
        return [value] * nr
    elif isinstance(value, list):
        if len(value) != nr:
            raise Exception("Value error!")
        return value


def count_shape(shape):
    """Shape to nr."""
    if isinstance(shape, (tuple, list)):
        return shape[0] * shape[1]
    else:
        return shape


def as_shape(x, shape):
    """Make shape."""
    if isinstance(shape, (tuple, list)):
        return reshape(x, shape)
    else:
        return x


def make_soc_matrix(h, new_h):
    if isinstance(h, list):
        if h == []:
            return new_h
    else:
        upper_left = copy.deepcopy(h)
        lower_right = copy.deepcopy(new_h)
        upper_right = Sparsity(upper_left.shape[0], lower_right.shape[1])
        lower_left = Sparsity(lower_right.shape[0], upper_left.shape[1])
        return blockcat(upper_left, upper_right, lower_left, lower_right)


class Description:
    """Description for Casadi."""

    def __init__(self):
        """Create description."""
        self.h = []  # for soc-type constraints
        self.g = []
        self.ubg = []
        self.lbg = []
        self.g_lin = []
        self.g_dis = []
        self.w = []
        self.w0 = []
        self.indices = {}  # Names with indices
        self.p = []
        self.p0 = []
        self.indices_p = {}  # Names with indices
        self.lbw = []
        self.ubw = []
        self.discrete = []
        self.f = GlobalSettings.CASADI_VAR(0)
        self.r = []  # residual for GN hessian computation
        self.check_illconditioned = False

    def add_w(
        self,
        lb: Union[float, List[float]],
        w: GlobalSettings.CASADI_VAR,
        ub: Union[float, List[float]],
        w0: Optional[Union[float, List[float]]],
        discrete: Union[bool, List[int]] = False
    ):
        """Add an existing symbolic variable."""
        if self.check_illconditioned:
            too_small = [i for i in lb
                         if abs(i) > 1e5 and not isinf(i)]
            too_large = [i for i in ub
                         if abs(i) > 1e5 and not isinf(i)]

            if len(too_small) > 0 or len(too_large) > 0:
                raise Exception(
                    "Illconditioned {lb} <= {w} <= {ub}".format(
                        ub=ub, lb=lb, w=w
                    )
                )

        idx = [i + len(self.lbw) for i in range(len(lb))]
        self.w += [w]
        self.lbw += lb
        self.ubw += ub
        self.w0 += w0
        if isinstance(discrete, List):
            if len(discrete) != len(lb):
                raise Exception(
                    "Discrete list doesn't have the same size as lb!")
            self.discrete += discrete
        else:
            self.discrete += [1 if discrete else 0] * len(lb)
        return idx

    def sym(
        self,
        name: str,
        shape: Union[int, Tuple[int, int]],
        lb: Union[float, List[float]],
        ub: Union[float, List[float]],
        w0: Optional[Union[float, List[float]]] = None,
        discrete: Union[bool, List[int]] = False
    ) -> GlobalSettings.CASADI_VAR:
        """Create a symbolic variable."""
        nr = count_shape(shape)

        # Gather Data
        if name not in self.indices:
            self.indices[name] = []
        idx_list = self.indices[name]

        # Create
        x = GlobalSettings.CASADI_VAR.sym("%s[%d]" % (name, len(idx_list)), nr)
        lb = make_list(lb, nr)
        ub = make_list(ub, nr)

        if w0 is None:
            w0 = lb
        w0 = make_list(w0, nr)

        if len(lb) != nr:
            raise Exception("Lower bound error!")
        if len(ub) != nr:
            raise Exception("Upper bound error!")
        if len(w0) != nr:
            raise Exception("Estimation length error (w0 %d vs nr %d)!"
                            % (len(w0), nr))

        # Collect
        out = self.add_w(lb, x, ub, w0, discrete)
        idx_list.append(out)
        return as_shape(x, shape)

    def add_parameters(self, name, shape: Union[int, Tuple[int, int]], values=0):
        """Add some parameters."""
        nr = count_shape(shape)
        # Gather Data
        if name not in self.indices_p:
            self.indices_p[name] = []
        idx_list = self.indices_p[name]

        # Create
        p = GlobalSettings.CASADI_VAR.sym("%s[%d]" % (name, len(idx_list)), nr)
        values = make_list(values, nr)

        # Create & Collect
        new_idx = [i + len(self.p0) for i in range(nr)]
        self.p += [p]
        self.p0 += values

        self.indices_p[name].extend(new_idx)
        return as_shape(p, shape)

    def add_g(self, mini: float, equation: GlobalSettings.CASADI_VAR, maxi: float,
              is_linear=0, is_discrete=0) -> int:
        """
        Add to g.

        :param mini: minimum
        :param equation: constraint equation
        :param maxi: maximum
        :return: index of constraint
        """
        if self.check_illconditioned:
            if ((abs(maxi) > 1e5 and not isinf(maxi))
                    or (abs(mini) > 1e5 and not isinf(mini))):
                raise Exception(
                    "Illconditioned {mini} <= {eq} <= {maxi}".format(
                        mini=mini, eq=equation, maxi=maxi
                    )
                )

        nr = equation.shape[0] * equation.shape[1]
        equation = reshape(equation, (nr, 1))
        self.lbg += make_list(mini, nr)
        self.g += make_list(equation)
        self.ubg += make_list(maxi, nr)
        self.g_lin += make_list(int(is_linear), nr)
        self.g_dis += make_list(int(is_discrete), nr)
        return len(self.ubg) - 1

    def add_soc(self, equation: GlobalSettings.CASADI_VAR):
        self.h = make_soc_matrix(self.h, equation)

    def leq(self, op1, op2, is_linear=0, is_discrete=0):
        """Lower or equal."""
        if isinstance(op1, (float, int, list)):
            self.add_g(op1, op2, inf,
                       is_linear=is_linear, is_discrete=is_discrete)
        elif isinstance(op2, (float, int, list, np.ndarray)):
            self.add_g(-inf, op1, op2,
                       is_linear=is_linear, is_discrete=is_discrete)
        else:
            diff = op1 - op2
            self.add_g(-inf, diff, 0,
                       is_linear=is_linear, is_discrete=is_discrete)

    def eq(self, op1, op2, is_linear=0, is_discrete=0):
        """Equal."""
        self.add_g(0, op1 - op2, 0,
                   is_linear=is_linear, is_discrete=is_discrete)

    def sym_bool(
        self, name: str, nr: int = 1, w0=1
    ) -> GlobalSettings.CASADI_VAR:
        """Create a symbolic boolean."""
        return self.sym(name, nr, 0, 1, w0=w0, discrete=True)

    def get_gauss_newton_hessian(self) -> GlobalSettings.CASADI_VAR:
        x = vertcat(*self.w)
        if isinstance(self.r, list):
            self.r = vertcat(*self.r)

        dr = jacobian(self.r, x)
        return dr.T @ dr

    def create_f(self):
        """Create function f."""
        return Function("f", [vcat(self.w), vcat(self.p)], [self.f])

    def create_g(self):
        """Create function f."""
        return Function("g", [vcat(self.w), vcat(self.p)], [vcat(self.g)])

    def _r_exist(self):
        return not (isinstance(self.r, list) and len(self.r) == 0)

    def get_problem(self, with_gn=True) -> MinlpProblem:
        """Extract problem."""
        idx_x_integer = [i for i, v in enumerate(self.discrete) if v == 1]
        if self._r_exist() and with_gn:
            gn_hessian = self.get_gauss_newton_hessian()
        else:
            gn_hessian = None
        return MinlpProblem(f=self.f, g=vertcat(*self.g), h=self.h,
                            x=vertcat(*self.w), idx_x_integer=idx_x_integer, p=vertcat(*self.p), gn_hessian=gn_hessian)

    def get_data(self) -> MinlpData:
        """Get data structure."""
        return MinlpData(p=self.p0, x0=DM(self.w0), _lbx=DM(self.lbw), _ubx=DM(self.ubw),
                         _lbg=DM(self.lbg), _ubg=DM(self.ubg))

    def get_indices(self, name: str, is_param: bool = False):
        """
        Get indices of a certain variable.

        :param name: name
        """
        if is_param:
            return self.indices_p[name]
        else:
            return self.indices[name]

    def check(self):
        """Test properties."""
        nr_of_w = vertcat(*(self.w)).size()[0]
        if nr_of_w != len(self.lbw):
            raise Exception("Length lbw (%d) incorrect, need %d." % (
                len(self.lbw), nr_of_w
            ))
        if nr_of_w != len(self.ubw):
            raise Exception("Length ubw (%d) incorrect, need %d." % (
                len(self.ubw), nr_of_w
            ))

        nr_of_g = vertcat(*(self.g)).size()[0]
        if nr_of_g != len(self.lbg):
            raise Exception("Length lbg (%d) incorrect, need %d." % (
                len(self.lbg), nr_of_w
            ))
        if nr_of_g != len(self.ubg):
            raise Exception("Length ubg (%d) incorrect, need %d." % (
                len(self.ubg), nr_of_w
            ))
        print("Nr of w %d - Nr of g %d" % (nr_of_w, nr_of_g))

    def check_hessian(self, plot=True, tolerance=1e-8):
        """Check hessian, returns the hessian h, jacobian j and eigenvalues."""
        import numpy as np
        w = vertcat(*self.w)
        h = hessian(self.f, w)
        h_func = Function('h', [w], [h[0]])
        h_val = h_func(np.ones((w.size(1), w.size(2))))
        eig_values = np.unique(np.linalg.eig(h_val)[0])
        print(f"Eigenvalues: {eig_values}")
        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.spy(h_val)
            plt.subplot(1, 2, 2)
            plt.imshow(h_val, interpolation='none', cmap='binary')
            plt.colorbar()
        if (eig_values < -tolerance).any():
            print("Hessian is not semi-definite!")
        return h[0], h[1], eig_values
