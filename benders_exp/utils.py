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
