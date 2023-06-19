"""Old utilities."""

import casadi as ca
import numpy as np


class Nlp:
    """Generate an nlp class using CasADi."""

    def __init__(self, name):
        """Initialize values."""
        self.parameter = ca.SX.sym('p')
        self.var = ca.SX.sym('x')
        self.var_0 = ca.DM.zeros(self.var.shape)
        self.f = 0
        self.cons = [(0, self.var, 0)]
        self.solver_name = 'ipopt'
        self.solver_opt = {}
        self.name = name

    def build(self):
        """Build the nlp."""
        self.g, self.lbg, self.ubg = self.g_lb_ub(self.cons)
        self.nlp_definition = {'x': self.var, 'f': self.f,
                               'g': self.g, 'p': self.parameter}

        self.opt = ca.nlpsol(self.name, self.solver_name,
                             self.nlp_definition, self.solver_opt)

    def solve(self, parameter_value=0):
        """Solve the nlp using the solver specified."""
        self.solution = self.opt(x0=self.var_0,
                                 lbg=self.lbg, ubg=self.ubg,
                                 p=parameter_value)

    def return_stats(self):
        """Return the stats of the solver."""
        return self.opt.stats()

    @staticmethod
    def linearize(exp, var, var_lin, name):
        """
        Return a function which is linearization of a function from var to exp.

        Since var might include parameters, var_lin specifies the linearization
        variable. Example:

            f(x, y) = y * sin(x)
            F_l = linearize(y * sin(x), [x, y], x)
            F_l(x, x0, y0) = y0 * sin(x0) + y0 * cos(x0) * (x - x0)
        """
        func = ca.Function('func', var, [exp])
        func_grad = ca.Function('func_grad', var, [ca.jacobian(exp, var_lin)])

        tmp_var = ca.SX.sym('var', var_lin.shape)
        exp_lin = func(*var) + func_grad(*var) @ (tmp_var - var_lin)

        return ca.Function(name, [tmp_var] + var, [exp_lin])

    @staticmethod
    def g_lb_ub(cons):
        """Generate g based on a list of [(lbg, g, ubg)] constraints."""
        g = ca.vertcat(*[c[1] for c in cons])
        lbg = ca.vertcat(*[x[0]*np.ones(x[1].shape) for x in cons])
        ubg = ca.vertcat(*[x[2]*np.ones(x[1].shape) for x in cons])

        return g, lbg, ubg
