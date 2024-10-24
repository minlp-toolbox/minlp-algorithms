# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import casadi as ca
from camino.settings import GlobalSettings, Settings
from camino.problems import MinlpProblem, MinlpData, \
    MetaDataOcp
from camino.problems.dsc import Description
from camino.utils.integrators import integrate_ie


def particle_trajectory():
    N = 100
    t_f = 10
    dt = t_f / N
    N_uptime = int(0.5 // dt)
    t = GlobalSettings.CASADI_VAR.sym("t")
    r = ca.Function("r", [t], [ca.sin(t) * 0.5 + 1])

    x = GlobalSettings.CASADI_VAR.sym("x")
    v = GlobalSettings.CASADI_VAR.sym("v")
    X = ca.vertcat(x, v)
    X_lb = [-100, -100]
    X_ub = [100, 100]
    X_0 = [0, 0]
    X_discrete = [0, 0]

    # Controls
    U = GlobalSettings.CASADI_VAR.sym("u", 3)
    U_lb = [-1, 0, 0]
    U_ub = [1, 1, 1]
    U_discrete = [1, 0, 0]

    # Dynamics
    x_dot = ca.vertcat(
        v,
        U[0],
    )

    # Terminal cost
    stage_cost = ca.Function("stage_cost", [X, U, t], [(x - r(t))**2])

    # Discretize
    F = integrate_ie(X, U, x_dot, dt)

    # Create OCP
    dsc = Description()
    xp = dsc.add_parameters("x0", X.numel(), X_0)
    Up = 0
    u_vector = []
    for i in range(N):
        xk = dsc.sym("x", X.numel(), X_lb, X_ub, X_0, X_discrete)
        uk = dsc.sym("u", U.numel(), U_lb, U_ub, discrete=U_discrete)
        dsc.eq(Up, uk[0] - uk[1] + uk[2])
        u_vector.append(uk[1] + uk[2])
        dsc.eq(F(xp, uk, xk), 0)
        dsc.f += dt * stage_cost(xk, uk, (i + 1) * dt)
        xp, Up = xk, uk[0]

    for i in range(N-N_uptime):
        dsc.leq(sum(u_vector[i:i+N_uptime+1]), 1)

    problem = dsc.get_problem()
    problem.meta = MetaDataOcp(
        dt=1,
        initial_state=X_0,
        n_state=X.numel(),
        n_continuous_control=1,
        n_discrete_control=0,
        idx_state=np.vstack(dsc.get_indices("x")),
        idx_control=np.vstack(dsc.get_indices("u"))[:, 0],
    )
    data = dsc.get_data()
    s = Settings()
    s.MIP_SETTINGS_ALL["gurobi"].update({
        "gurobi.PoolSearchMode": 1,
        "gurobi.PoolSolutions": 3,
    })
    return problem, data, s
