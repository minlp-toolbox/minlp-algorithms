# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import casadi as ca
from camino.settings import GlobalSettings
from camino.problems import MetaDataOcp
from camino.problems.dsc import Description
from camino.utils.integrators import integrate_rk4


def time_opt_car():
    """Time optimal car with a goal."""
    v_turbo = 10
    v_max = 100
    N = 20
    p_des = 500
    v_des = 0.0

    p = GlobalSettings.CASADI_VAR.sym("p")
    v = GlobalSettings.CASADI_VAR.sym("v")
    t = GlobalSettings.CASADI_VAR.sym("t")
    x = ca.vcat([p, v, t])
    x_lb = [0, 0, 0]
    x_ub = [ca.inf, v_max, ca.inf]
    x_0 = [0, 0, 0]
    x_discrete = [0, 0, 0]

    # Controls
    a = GlobalSettings.CASADI_VAR.sym("a")
    turbo = GlobalSettings.CASADI_VAR.sym("turbo")  # Boolean
    reverse = GlobalSettings.CASADI_VAR.sym("reverse")  # gear
    u = ca.vcat([a, turbo, reverse])
    u_lb = [0, 0, -1]
    u_ub = [10, 1, 1]
    u_0 = [0, 0, 0]
    u_discrete = [0, 1, 1]
    u_path_weights = [0.5, 0, 0]

    # Dynamics
    x_dot = ca.vertcat(
        v,
        a * (1 + turbo) * reverse,
        1
    )

    # Path constraints 0 < constraint
    p_g = ca.Function("g", [x, u], [
        v / v_turbo - turbo,  # Turbo only on at v > v_turbo,
    ])
    p_g_lb = 0
    p_g_ub = ca.inf

    terminal_g = ca.Function("g_term", [x], [ca.vertcat(
        p - p_des,
        v - v_des
    )])
    terminal_g_lb = 0
    terminal_g_ub = 0

    # Terminal cost
    stage_cost = ca.Function("stage_cost", [x, u], [0])
    terminal_cost = ca.Function("T", [x], [t])

    # Discretize
    dt = GlobalSettings.CASADI_VAR.sym("dt")
    dt_lb = 0.0
    dt_ub = 1.0
    F = integrate_rk4(x, u, x_dot, dt)

    # Create OCP
    dsc = Description()
    xp = dsc.add_parameters("x0", x.numel(), x_0)
    up = dsc.add_parameters("u0", u.numel(), u_0)
    dtk = dsc.sym("dt", 1, dt_lb, dt_ub)
    for i in range(N):
        xk = dsc.sym("x", x.numel(), x_lb, x_ub, x_0, x_discrete)
        uk = dsc.sym("u", u.numel(), u_lb, u_ub, discrete=u_discrete)
        dsc.eq(F(xp, uk, dtk), xk)
        dsc.add_g(p_g_lb, p_g(xk, uk), p_g_ub)
        dsc.f += dtk * stage_cost(xk, uk)
        dsc.f += sum(
            [v * (uk[id] - up[id]) ** 2 for id, v in enumerate(u_path_weights)]
        )
        xp, up = xk, uk
    # Final cost
    dsc.add_g(terminal_g_lb, terminal_g(xp), terminal_g_ub)
    dsc.f += terminal_cost(xp)

    problem = dsc.get_problem()
    problem.meta = MetaDataOcp(
        dt=None,
        initial_state=x_0[:-1],
        n_state=x.numel()-1,
        n_continuous_control=u.numel(),
        n_discrete_control=0,
        idx_state=np.vstack(dsc.get_indices("x"))[:, :-1],
        idx_control=np.vstack(dsc.get_indices("u")),
        idx_bin_control=[],
        idx_t=np.vstack(dsc.get_indices("x"))[:, -1]
    )
    problem.hessian_not_psd = True
    data = dsc.get_data()
    return problem, data
