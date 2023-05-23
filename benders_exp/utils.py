"""Set of utilities for MINLP optimization and visualization."""

from math import sqrt
from typing import Union
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import casadi as ca
import logging
from time import perf_counter
from benders_exp.defines import _DATA_FOLDER
from benders_exp.problems import MinlpData, MinlpProblem, MetaDataOcp

perf_ti = None
CALLBACK_INPUTS = dict()
for i, label in enumerate(ca.nlpsol_out()):
    CALLBACK_INPUTS[label] = i


def make_bounded(problem: MinlpProblem, data: MinlpData, new_inf=1e3):
    """Make bounded."""
    lbx, ubx = data.lbx, data.ubx
    lbg, ubg = data.lbg, data.ubg

    # # Move all continmous bounds to g!
    # when constraints are one sided the convention is -inf <= g <= ubg
    g_extra, g_lb, g_ub = [], [], []
    for i in range(data.lbx.shape[0]):
        if i not in problem.idx_x_bin:
            if lbx[i] > -new_inf:
                g_extra.append(-problem.x[i] + max(lbx[i], -new_inf))
                g_lb.append(-np.inf)
                g_ub.append(0)
                lbx[i] = -1e9  # TODO: for -ca.inf it raises an error (OverflowError: cannot convert float infinity to integer)
            if ubx[i] < new_inf:
                g_extra.append(problem.x[i] - min(ubx[i], new_inf))
                g_lb.append(-np.inf)
                g_ub.append(0)
                ubx[i] = new_inf
        else:
            if lbx[i] < -new_inf:
                lbx[i] = -new_inf
            if ubx[i] > new_inf:
                ubx[i] = new_inf

    data.lbx, data.ubx = lbx, ubx
    # Update
    problem.g = ca.vertcat(problem.g, ca.vertcat(*g_extra))
    data.lbg = np.concatenate((ca.DM(lbg), ca.DM(g_lb)))
    data.ubg = np.concatenate((ca.DM(ubg), ca.DM(g_ub)))


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
    if perf_ti is None:
        tic()
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


def latexify(fig_width=None, fig_height=None):
    """
    Set up matplotlib's RC params for LaTeX plotting.

    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """
    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf
    if fig_width is None:
        fig_width = 5  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    params = {
        # "backend": "ps",
        "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
        "axes.labelsize": 10,  # fontsize for x and y labels (was 10)
        "axes.titlesize": 10,
        "lines.linewidth": 2,
        "legend.fontsize": 10,  # was 10
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
    }

    matplotlib.rcParams.update(params)


def plot_trajectory(
    s_collection: Union[np.ndarray, list],
    a_collection: Union[np.ndarray, list],
    meta: MetaDataOcp,
    title: str,
):
    """Plot a trajectory of an OCP."""
    alpha = 0.4
    if isinstance(a_collection, np.ndarray):
        a_collection = [a_collection]
        alpha = 0.7
    if isinstance(s_collection, np.ndarray):
        s_collection = [s_collection]
    N = a_collection[0].shape[0]
    dt = meta.dt
    time_array = np.linspace(0, N * dt, N + 1)

    latexify()
    fig, axs = plt.subplots(meta.n_state + meta.n_control,
                            1, figsize=(4.5, 8), sharex=True)
    for s in range(meta.n_state):
        axs[s].grid()
        for s_traj in s_collection:
            if len(s_traj.shape) == 1:
                s_traj = s_traj[..., np.newaxis]
            axs[s].plot(time_array, s_traj[:, s], "-",
                        alpha=alpha, color="tab:blue")
        # axs[s].axhline(s_max[s], linestyle=":", color="k", alpha=0.7)
        # axs[s].axhline(s_min[s], linestyle=":", color="k", alpha=0.7)
        axs[s].set_ylim(0, )
        axs[s].set_ylabel(f"$x_{s}$")

    for a in range(meta.n_control):
        axs[meta.n_state + a].set_ylim(0, 10.5)
        for a_traj in a_collection:
            if len(a_traj.shape) == 1:
                a_traj = a_traj[..., np.newaxis]
            axs[meta.n_state + a].step(
                time_array,
                meta.scaling_coeff_control[a] * np.append([a_traj[0, a]], a_traj[:, a]),
                alpha=alpha,
                color="tab:orange",
            )
        axs[meta.n_state + a].set_ylabel(f"$u_{a}$")
        axs[meta.n_state + a].grid()

    axs[0].set_title(title)
    axs[-1].set_xlabel(r"time [sec]")
    axs[-1].set_xlim(0, time_array[-1])
    # axs[-1].grid()
    # axs[-1].axhline(model.a_max, linestyle=":", color="k", alpha=0.5)
    # axs[-1].axhline(model.a_min, linestyle=":", color="k", alpha=0.5)

    plt.tight_layout()
    # fig.savefig(f"{IMG_DIR}/acados_test_loop.pdf", bbox_inches='tight')
    return fig, axs


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
