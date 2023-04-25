import numpy as np
import casadi as ca
import logging
from time import perf_counter
from benders_exp.defines import _DATA_FOLDER

CALLBACK_INPUTS = dict()
for i, label in enumerate(ca.nlpsol_out()):
    CALLBACK_INPUTS[label] = i


def setup_logger():
    """Set up the logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)


def tic():
    """Tic."""
    global perf_ti
    perf_ti = perf_counter()


def toc(reset=False):
    """Toc."""
    global perf_ti
    tim = perf_counter()
    dt = tim - perf_ti
    print("  Elapsed time: %s s." % (dt))
    if reset:
        perf_ti = tim
    return dt


def to_0d(array):
    """To zero dimensions."""
    if isinstance(array, np.ndarray):
        return array.squeeze()
    else:
        return array.full().squeeze()


class DebugCallBack(ca.Callback):
    """
    Create a debug callback.

    :param name: name of the callback function
    :param nx: Nr of x variables
    :param ng: Nr of g constraints
    :param np: Nr of parameters p
    """

    def __init__(self, name, nx: int, ng: int, np: int, opts=None):
        """Create the debug call back for casadi."""
        ca.Callback.__init__(self)
        if opts is None:
            opts = {}

        self.nx = nx
        self.ng = ng
        self.np = np
        self.iter = 0
        self.name = name
        # Initialize internal objects
        self.construct(name, opts)

    def add_to_solver_opts(self, options, iter_per_callback=50):
        """
        Add to the solver options.

        :param options: options of the nlp
        :param iter_per_callback: nr of iters per callback
        """
        options["iteration_callback"] = self
        options['iteration_callback_step'] = iter_per_callback
        return options

    def get_n_in(self):
        """Get number of inputs."""
        return ca.nlpsol_n_out()

    def get_n_out(self):
        """Get number of outputs."""
        return 1

    def get_sparsity_in(self, i):
        """Get sparsity of matrices."""
        n = ca.nlpsol_out(i)
        if n == 'f':
            return ca.Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return ca.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return ca.Sparsity.dense(self.ng)
        elif n in ('p', 'lam_p'):
            return ca.Sparsity.dense(self.np)
        else:
            return ca.Sparsity(0, 0)

    def save(self, x):
        """Save the x variable."""
        self.iter += 1
        np.save(_DATA_FOLDER + f"/x_{self.name}_{self.iter}", x.full())

    def eval(self, arg):
        """Evaluate the callback."""
        x = arg[CALLBACK_INPUTS["x"]]
        self.save(x)
        return [0]


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
