# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Set of utilities for MINLP optimization and visualization."""

from math import sqrt
from typing import Union
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import casadi as ca
import logging
from time import perf_counter
from camino.problems import MinlpData, MinlpProblem, MetaDataOcp

logger = logging.getLogger(__name__)

try:
    from colored import fg, stylize

    def colored(text, color="red"):
        """Color a text."""
        logger.info(stylize(text, fg(color)))
except Exception:
    def colored(text, color=None):
        """Color a text."""
        logger.info(text)

perf_ti = None


def make_bounded(problem: MinlpProblem, data: MinlpData, new_inf=1e3):
    """Make bounded."""
    lbx, ubx = data.lbx, data.ubx
    lbg, ubg = data.lbg, data.ubg

    # # Move all continuous bounds to g!
    # when constraints are one sided the convention is -inf <= g <= ubg
    g_extra, g_lb, g_ub = [], [], []
    for i in range(data.lbx.shape[0]):
        if i in problem.idx_x_integer:
            if lbx[i] < -new_inf:
                lbx[i] = -new_inf
            if ubx[i] > new_inf:
                ubx[i] = new_inf

    data.lbx, data.ubx = lbx, ubx
    # Update
    problem.g = ca.vertcat(problem.g, ca.vertcat(*g_extra))
    data.lbg = np.concatenate((ca.DM(lbg), ca.DM(g_lb)))
    data.ubg = np.concatenate((ca.DM(ubg), ca.DM(g_ub)))


def setup_logger(level=logging.WARNING):
    """Set up the logging."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=level
    )
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)


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
    logger.info(f"Elapsed time: {dt} s")
    if reset:
        perf_ti = tim
    return dt


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
    x_star: np.ndarray,
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
    if isinstance(meta.scaling_coeff_control, type(None)):
        meta.scaling_coeff_control = [
            1 for _ in range(a_collection[0].shape[0])]

    N = a_collection[0].shape[0]
    dt = meta.dt
    if dt is None:
        time_array = np.hstack((np.array([0]), x_star[meta.idx_t]))
    else:
        time_array = np.linspace(0, N * dt, N + 1)
    n_controls = meta.n_continuous_control + meta.n_discrete_control

    latexify()
    fig, axs = plt.subplots(meta.n_state + n_controls,
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

    for a in range(n_controls):
        if a >= meta.n_continuous_control:
            # Discrete control
            axs[meta.n_state + a].set_ylim(-1.5, 1.5)
        else:
            axs[meta.n_state + a].set_ylim(-10.5, 10.5)

        for a_traj in a_collection:
            if len(a_traj.shape) == 1:
                a_traj = a_traj[..., np.newaxis]
            axs[meta.n_state + a].step(
                time_array,
                meta.scaling_coeff_control[a] *
                np.append([a_traj[0, a]], a_traj[:, a]),
                alpha=alpha,
                color="tab:orange",
                marker='.',
                where='post'
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
    return fig, axs
