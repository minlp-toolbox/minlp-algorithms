# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Overview of all problems."""

from camino.settings import Settings, GlobalSettings
from camino.problems import MinlpProblem, MinlpData, \
    MetaDataOcp
import casadi as ca
import numpy as np
from camino.problems.dsc import Description
from camino.solvers import Stats
from camino.utils.integrators import integrate_rk4  # integrate_ee
from camino.problems.double_tank import create_double_tank_problem2
from camino.problems.solarsys import create_stcs_problem
from camino.problems.gearbox import create_simple_gearbox, create_gearbox, \
    create_gearbox_int
from camino.problems.minlp import MINLP_PROBLEMS
from camino.utils.conversion import to_bool
from camino.problems.time_opt import time_opt_car
from camino.problems.sto_based import particle_trajectory


def create_ocp_unstable_system(p_val=[0.9, 0.7]):
    """
    OCP of a unstable system subject to min uptime constraints.

    Example taken from preprint of A. Buerger. Inspired by a textbook example of the MPC book by Rawlings, Mayne, Diehl
    """
    dt = 0.05
    N = 30
    min_uptime = 2  # in time steps

    dsc = Description()
    x = GlobalSettings.CASADI_VAR.sym('x')  # state
    u = GlobalSettings.CASADI_VAR.sym('u')  # control
    Xref = dsc.add_parameters("Xk_ref", 1, p_val[1])

    xdot = x ** 3 - u
    r = ca.Function('residual_obj', [x, Xref], [x - Xref])
    F = integrate_rk4(x, u, xdot, dt, m_steps=1)

    Xk = dsc.add_parameters("Xk0", 1, p_val[0])
    BigM = 1e3
    Uprev = None
    for k in range(N):
        Uk = dsc.sym("Uk", 1, lb=0, ub=1, w0=1, discrete=True)
        if Uprev is not None and min_uptime > 0:
            uptime_step = 0
            while uptime_step < min_uptime:
                idx_1 = k - 1
                idx_2 = k - (uptime_step + 2)

                if idx_1 >= 0:
                    b_idx_1 = int(np.array(dsc.get_indices("Uk")[idx_1]))
                else:
                    b_idx_1 = 0
                b_idx_1 = dsc.w[b_idx_1]

                if idx_2 >= 0:
                    b_idx_2 = int(np.array(dsc.get_indices("Uk")[idx_2]))
                else:
                    b_idx_2 = 0
                b_idx_2 = dsc.w[b_idx_2]

                dsc.leq(-Uk + b_idx_1 - b_idx_2, 0)
                uptime_step += 1

        # Integrate till the end of the interval
        Xk_end = F(Xk, Uk)
        dsc.f += 0.5 * (Xk - Xref) ** 2
        dsc.r += [r(Xk, Xref)]

        # New NLP variable for state at end of interval
        Xk = dsc.sym("Xk", 1, lb=-BigM, ub=BigM, w0=p_val[0])
        dsc.eq(Xk_end, Xk)

        Uprev = Uk
    dsc.f += 0.5 * (Xk - Xref) ** 2

    problem = dsc.get_problem()
    meta = MetaDataOcp(
        dt=dt, n_state=1, n_discrete_control=1, n_continuous_control=0,
        initial_state=p_val[0], idx_control=np.hstack(dsc.get_indices("Uk")),
        idx_state=np.hstack(dsc.get_indices("Xk")),
        idx_bin_control=problem.idx_x_integer,
        scaling_coeff_control=[1],
        min_uptime=min_uptime,
        min_downtime=min_uptime,
        f_dynamics=F,
    )
    problem.meta = meta
    data = dsc.get_data()
    s = Settings()
    s.USE_RELAXED_AS_WARMSTART = False
    s.CONSTRAINT_INT_TOL = 1e-5
    s.MIP_SETTINGS_ALL["gurobi"].update(
        {"gurobi.FeasibilityTol": s.CONSTRAINT_INT_TOL,
         "gurobi.IntFeasTol": s.CONSTRAINT_INT_TOL, })
    return problem, data, s


def create_check_sign_lagrange_problem():
    """Create a problem to check the sign of the multipliers."""
    x = GlobalSettings.CASADI_VAR.sym("x")
    p = GlobalSettings.CASADI_VAR.sym("p")

    problem = MinlpProblem(
        x=x, f=(x - 2)**2, g=ca.vcat([x]), p=p, idx_x_integer=[0])
    data = MinlpData(
        x0=np.array([0]), _ubx=np.array([np.inf]), _lbx=np.array([-np.inf]),
        _ubg=np.array([-1]), _lbg=np.array([-7]), p=np.array([0]))

    return problem, data


def create_dummy_problem(p_val=[1000, 3]):
    """
    Create a dummy problem.

    This problem corresponds to the tutorial example in the GN-Voronoi paper.
    (apart from the upper bound)
    """
    opti = Description()
    x0 = np.array([0, 4, 100])
    lbx = np.array([0, 0, 0])
    ubx = np.array([4, 4, np.inf])
    x = opti.sym("x", shape=3, lb=lbx, ub=ubx, w0=x0, discrete=[1, 1, 0])
    # idx_x_integer = [0, 1]
    p = opti.add_parameters("p", shape=2, values=p_val)
    opti.f = (x[0] - 4.1)**2 + (x[1] - 4.0)**2 + x[2] * p[0]
    ubg = np.array([ca.inf, ca.inf])
    lbg = np.array([0, 0])
    opti.add_g(lbg, ca.vertcat(
        x[2], -(x[0]**2 + x[1]**2 - x[2] - p[1]**2)), ubg, is_linear=1)
    problem = opti.get_problem()
    data = opti.get_data()
    return problem, data


def create_dummy_problem_2():
    """Create a dummy problem."""
    x = GlobalSettings.CASADI_VAR.sym("x", 2)
    x0 = np.array([2, 4])
    idx_x_integer = [0]
    p = GlobalSettings.CASADI_VAR.sym("p", 1)
    f = x[0]**2 + x[1]
    g = ca.vertcat(
        x[1],
        -x[0]**2 - x[1] / 4 + p[0]**2
    )
    ubg = np.array([np.inf, np.inf])
    lbg = np.array([-3.0, 0.0])
    lbx = -np.inf * np.ones((2,))
    ubx = np.array([ca.inf, ca.inf])

    problem = MinlpProblem(x=x, f=f, g=g, p=p, idx_x_integer=idx_x_integer)
    data = MinlpData(x0=x0, _ubx=ubx, _lbx=lbx,
                     _ubg=ubg, _lbg=lbg, p=[100])
    return problem, data


def create_double_pipe_problem(p_val=[1, 5, 1, 10]):
    """Create double pipe problem."""
    y = GlobalSettings.CASADI_VAR.sym("y", 1)  # integers
    z = GlobalSettings.CASADI_VAR.sym("z", 2)  # continuous
    x0 = np.array([1, 0, 0])
    x = ca.vertcat(*[y, z])
    idx_x_integer = [0]

    alpha = GlobalSettings.CASADI_VAR.sym("alpha", 2)
    r = GlobalSettings.CASADI_VAR.sym("r", 1)
    gamma = GlobalSettings.CASADI_VAR.sym("gamma", 1)
    p = ca.vertcat(*[alpha, r, gamma])

    f = alpha[0] * z[0] + alpha[1] * y[0]
    g = ca.vertcat(*[
        z[0] + y[0] * z[1] - r,
        gamma - z[1]
    ])
    lbg = np.array([0, 0])
    ubg = np.array([ca.inf, ca.inf])
    lbx = np.array([0, 0, -ca.inf])
    ubx = np.array([1, ca.inf, ca.inf])

    problem = MinlpProblem(x=x, f=f, g=g, p=p, idx_x_integer=idx_x_integer)
    data = MinlpData(x0=x0, _ubx=ubx, _lbx=lbx,
                     _ubg=ubg, _lbg=lbg, p=p_val)
    return problem, data


def create_double_tank_problem(p_val=[2, 2.5]):
    """
    Implement the double tank problem.

    Taken from Abbasi et al. ECC 23, reimplemented to achieve nice sparsity pattern.

    """
    eps = 1e-3
    N = 300  # NOTE In paper the is set to 300
    dt = 1/30
    T = N * dt
    alpha = 100
    beta = np.array([1., 1.2])
    gamma = 10
    scaling_coeff = [gamma, 1]
    demand = np.array([2 + 0.5 * np.sin(x)
                      for x in np.linspace(0, T, N+1)])

    nx = 2
    nu = 2
    x_0 = GlobalSettings.CASADI_VAR.sym('x0', nx)
    x = GlobalSettings.CASADI_VAR.sym('x', nx)  # state
    u = GlobalSettings.CASADI_VAR.sym('u', nu)  # control
    x1dot = scaling_coeff[0]*u[0] + u[1] - ca.sqrt(x[0] + eps)
    x2dot = ca.sqrt(x[0] + eps) - ca.sqrt(x[1] + eps)
    xdot = ca.vertcat(*[x1dot, x2dot])

    # F = integrate_ee(x, u, xdot, dt, m_steps=1)
    F = integrate_rk4(x, u, xdot, dt, m_steps=1)

    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []
    p = []
    idx_x_integer = []
    idx_state = []
    idx_control = []
    idx_var = 0

    Xk = x_0
    p += [Xk]
    for k in range(N):
        Uk = GlobalSettings.CASADI_VAR.sym(f"u_{k}", nu)
        w += [Uk]
        idx_var += nu
        lbw += [0, 0]
        ubw += [1, gamma]
        w0 += [0.5, 0.5]
        idx_x_integer.append(np.arange(idx_var-nu, idx_var-1))
        idx_control.append(np.arange(idx_var-nu+1, idx_var))

        # Integrate till the end of the interval
        Xk_end = F(Xk, Uk)
        J += dt * alpha * (Xk[1] - demand[k]) ** 2
        J += dt * (beta[0] * scaling_coeff[0] * Uk[0] + beta[1] * Uk[1])

        # New NLP variable for state at end of interval
        Xk = GlobalSettings.CASADI_VAR.sym(f"x_{k+1}", nx)
        idx_var += nx
        idx_state.append(np.arange(idx_var-nx, idx_var))
        w += [Xk]
        lbw += [0, 0]
        ubw += [1e3, 1e3]
        w0 += [0.5, 0.5]

        # Add equality constraint
        g += [Xk_end - Xk]
        lbg += [0, 0]
        ubg += [0, 0]

    meta = MetaDataOcp(
        dt=dt, n_state=nx, n_continuous_control=1, n_discrete_control=1,
        initial_state=p_val,
        idx_control=np.hstack(idx_control),
        idx_bin_control=np.hstack(idx_x_integer),
        idx_state=np.hstack(idx_state),
        scaling_coeff_control=scaling_coeff
    )
    problem = MinlpProblem(x=ca.vcat(w), f=J, g=ca.vcat(
        g), p=x_0, idx_x_integer=np.hstack(idx_x_integer), meta=meta)
    data = MinlpData(x0=ca.vcat(w0), _ubx=ca.vcat(ubw), _lbx=ca.vcat(lbw),
                     _ubg=np.array(ubg), _lbg=np.array(lbg), p=p_val)
    return problem, data


def counter_example_nonconvexity():
    """Nonconvexity example."""
    x = GlobalSettings.CASADI_VAR.sym('x')
    y = GlobalSettings.CASADI_VAR.sym('y')

    f = ca.atan(x-0.3)**2 + x/10 + x**2/50 + y**2
    problem = MinlpProblem(x=ca.vcat([x, y]), f=f, g=ca.MX(
        []), p=ca.MX([]), idx_x_integer=[0])
    data = MinlpData(x0=np.array([-4, 2]), _lbx=np.array([-5, -5]), _ubx=np.array([5, 5]),
                     _ubg=[], _lbg=[], p=[])
    return problem, data


def create_from_nl_file(file, compiled=True):
    """Load from NL file."""
    from camino.utils.cache import CachedFunction, return_func
    import hashlib
    # Create an NLP instance
    nl = ca.NlpBuilder()

    # Parse an NL-file
    nl.import_nl(file, {"verbose": False})
    print(f"Loading MINLP with: {nl.repr()}")

    if not isinstance(nl.x[0], GlobalSettings.CASADI_VAR):
        raise Exception(
            f"Set GlobalSettings.CASADI_VAR to {type(nl.x[0])} in defines!")

    idx = np.where(np.array(nl.discrete))

    if compiled:
        key = str(hashlib.md5(file.encode()).hexdigest())[:64]
        x = ca.vcat(nl.x)
        problem = MinlpProblem(
            x=x,
            f=CachedFunction(f"f_{key}", return_func(
                ca.Function("f", [x], [nl.f])))(x),
            g=CachedFunction(f"g_{key}", return_func(
                ca.Function("g", [x], [ca.vcat(nl.g)])))(x),
            idx_x_integer=idx[0].tolist(),
            p=[]
        )
    else:
        problem = MinlpProblem(
            x=ca.vcat(nl.x),
            f=nl.f, g=ca.vcat(nl.g),
            idx_x_integer=idx[0].tolist(),
            p=[]
        )
    if nl.f.is_constant():
        raise Exception("No objective!")

    problem.hessian_not_psd = True
    data = MinlpData(x0=np.array(nl.x_init),
                     _lbx=np.array(nl.x_lb),
                     _ubx=np.array(nl.x_ub),
                     _lbg=np.array(nl.g_lb),
                     _ubg=np.array(nl.g_ub), p=[])

    from camino.solvers import inspect_problem, set_constraint_types
    set_constraint_types(problem, *inspect_problem(problem, data))
    s = Settings()

    s.OBJECTIVE_TOL = 1e-5
    s.CONSTRAINT_TOL = 1e-5
    s.CONSTRAINT_INT_TOL = 1e-2
    s.MINLP_TOLERANCE = 0.01
    s.MINLP_TOLERANCE_ABS = 0.01
    s.TIME_LIMIT = 300
    s.TIME_LIMIT_SOLVER_ONLY = True
    s.IPOPT_SETTINGS = {
        "ipopt.linear_solver": "ma27",
        "ipopt.max_cpu_time": s.TIME_LIMIT / 4,
        "ipopt.mu_target": 1e-3,
        "ipopt.max_iter": 1000,
        "ipopt.print_level": 0,
    }
    s.MIP_SETTINGS_ALL["gurobi"] = {
        "gurobi.MIPGap": 0.10,
        "gurobi.FeasibilityTol": s.CONSTRAINT_INT_TOL,
        "gurobi.IntFeasTol": s.CONSTRAINT_INT_TOL,
        "gurobi.PoolSearchMode": 0,
        "gurobi.PoolSolutions": 1,
        "gurobi.Threads": 1,
        "gurobi.TimeLimit": s.TIME_LIMIT / 2,
        "gurobi.output_flag": 0,
    }
    s.BONMIN_SETTINGS["bonmin.time_limit"] = s.TIME_LIMIT
    return problem, data, s


def reduce_list(data):
    """Reduce list."""
    out = []
    for el in data:
        if isinstance(el, list):
            out.extend(reduce_list(el))
        else:
            out.append(el)
    return out


def create_from_nosnoc(file, compiled=False):
    """Create a problem from nosnoc."""
    from camino.utils.data import load_pickle
    from camino.utils.cache import CachedFunction, cache_data, return_func
    from camino.solvers import get_lin_bounds
    from os import path
    name = path.basename(file)
    data = load_pickle(file)
    x = GlobalSettings.CASADI_VAR.sym("x", data['w_shape'][0])
    p = GlobalSettings.CASADI_VAR.sym("p", data['p_shape'][0])
    if to_bool(compiled):
        f = CachedFunction(f"{name}_f", return_func(data['f']))(x, p)
        g = CachedFunction(f"{name}_g", return_func(data['g']))(x, p)
    else:
        f = data['f'](x, p)
        g = data['g'](x, p)

    print("Loaded Functions")
    idx_x_integer = reduce_list(data['ind_bool'])
    problem = MinlpProblem(
        x=x, p=p, f=f, g=g,
        idx_x_integer=idx_x_integer,
        hessian_not_psd=True
    )
    if to_bool(compiled):
        problem.idx_g_lin, problem.idx_g_lin_bin = cache_data(
            f"{name}_id", get_lin_bounds, problem)
    else:
        # Probably this is just overkill, set them to 0
        problem.idx_g_lin, problem.idx_g_lin_bin = [], []

    data['w0'][idx_x_integer] = np.round(data['w0'][idx_x_integer])
    data = MinlpData(
        p=data['p0'], x0=data['w0'],
        _lbx=data['lbw'], _ubx=data['ubw'],
        _lbg=data['lbg'], _ubg=data['ubg']
    )
    return problem, data


def create_from_sto(file, with_uptime=True):
    """Create a problem from STO."""
    from camino.utils.data import load_pickle
    from camino.utils.conversion import to_bool
    with_uptime = to_bool(with_uptime)
    data = load_pickle(file)
    dt = data['dt']
    N = data['N']

    nx = data['lbx'].shape[0] - 1
    p = GlobalSettings.CASADI_VAR.sym("p", 1)
    p0 = [dt * N]
    x = GlobalSettings.CASADI_VAR.sym("x", nx)
    x0 = data['init'][1:]
    ubx, lbx = data['ubx'][1:], data['lbx'][1:]
    f = data['f'](ca.vertcat(p, x))
    g = data['g'](ca.vertcat(p, x))
    idx_x_integer = np.where(data['int'][1:].full() == 1)[0].tolist()
    ub_idx = data['ub_idx'] - 1
    u_idx = data['u_idx']-1
    x_idx = data['x_idx']-1

    # Add extra switching constraints:
    switches = []
    constraints_eq = []  # 0 < item
    constraints_leq = []  # 0 < item
    min_uptime = [int(x/dt) for x in data['min_uptime']]
    min_downtime = [int(x/dt) for x in data['min_downtime']]
    ub_idx_c = ub_idx.reshape((-1, data['nub']))
    if with_uptime:
        for i, (min_up, min_down) in enumerate(zip(min_uptime[:-1], min_downtime[:-1])):
            if min_up > 1 or min_down > 1:
                switch = GlobalSettings.CASADI_VAR.sym(f"switch_x{i}", N)
                switches.append(switch)
                for idxt in range(N-1):
                    constraints_eq.append(
                        (x[ub_idx_c[idxt, i]] + switch[idxt]) - x[ub_idx_c[idxt+1, i]])

                if min_up > 1:
                    for idxt in range(N-1):
                        for idxt2 in range(idxt+1, min(N, idxt+min_up)):
                            # switch < x -> 0 < x - switch
                            constraints_leq.append(
                                x[ub_idx_c[idxt2, i]] - switch[idxt])

                if min_down > 1:
                    for idxt in range(N-1):
                        for idxt2 in range(idxt+1, min(N, idxt+min_down)):
                            # 1 + switch < 1 - x -> 0 < 1 - x - switch - 1
                            constraints_leq.append(
                                x[ub_idx_c[idxt2, i]] + switch[idxt])
    # Add to problem!
    nr_switches = len(switches)
    switches = ca.vcat(switches)
    constraints_eq = ca.vcat(constraints_eq)
    constraints_leq = ca.vcat(constraints_leq)
    nub = data['nub'] + nr_switches
    if nr_switches > 0:
        idx_switches = [
            [nx + N * swi + i for i in range(N)] for swi in range(nr_switches)
        ]
        ub_idx_c = np.hstack((ub_idx_c, np.array(idx_switches).T))
        min_uptime = min_uptime[:-1] + [0] * nr_switches + [min_uptime[-1]]
        min_downtime = min_downtime[:-1] + [0] * \
            nr_switches + [min_downtime[-1]]

    ub_idx_c = ub_idx_c.reshape((-1,))
    x_total = ca.vertcat(x, switches)
    g = ca.vertcat(g, constraints_eq, constraints_leq)
    lbx = ca.vertcat(lbx, -np.ones(switches.shape))
    ubx = ca.vertcat(ubx, np.ones(switches.shape))
    x0 = ca.vertcat(x0, np.zeros(switches.shape))
    x0[idx_x_integer] = np.round(x0[idx_x_integer])
    lbg = ca.vertcat(data['lbg'], np.zeros(
        constraints_eq.shape), np.zeros(constraints_leq.shape))
    ubg = ca.vertcat(data['ubg'], np.zeros(
        constraints_eq.shape), ca.inf * np.ones(constraints_leq.shape))

    problem = MinlpProblem(
        x=x_total, p=p, f=f, g=g,
        idx_x_integer=idx_x_integer,
        hessian_not_psd=True
    )
    problem.idx_g_lin, problem.idx_g_lin_bin = [], []

    meta = MetaDataOcp(
        dt=data['dt'],
        n_state=data['nx'],
        n_continuous_control=data['nu']-data['nub'],
        n_discrete_control=nub,
        initial_state=data['x0'],
        idx_control=u_idx,
        idx_state=x_idx,
        idx_bin_control=ub_idx_c,
        min_uptime=min_uptime,
        min_downtime=min_downtime,
        dump_solution=ca.Function("dump", [x_total], [ca.vertcat(p0, x)])
    )

    problem.meta = meta

    data = MinlpData(
        p=p0,
        x0=x0, _lbx=lbx, _ubx=ubx,
        _lbg=lbg, _ubg=ubg
    )
    return problem, data


def create_from_nlpsol_description(file):
    """Create a problem from casadi nlpsol description."""
    from camino.utils.data import load_pickle

    data = load_pickle(file)

    x = GlobalSettings.CASADI_VAR.sym("x", data['nx'])
    p = GlobalSettings.CASADI_VAR.sym("p", data['np'])
    f = data['f_fun'](x)
    g = data['g_fun'](x)
    idx_x_integer = data['idx_x_integer']

    problem = MinlpProblem(
        x=x, p=p, f=f, g=g,
        idx_x_integer=idx_x_integer,
        hessian_not_psd=True
    )

    data['x0'][idx_x_integer] = np.round(data['x0'][idx_x_integer])
    data = MinlpData(
        p=data['p0'], x0=data['x0'],
        _lbx=data['lbx'], _ubx=data['ubx'],
        _lbg=data['lbg'], _ubg=data['ubg']
    )
    return problem, data


PROBLEMS = {
    "sign_check": create_check_sign_lagrange_problem,
    "dummy": create_dummy_problem,
    "dummy2": create_dummy_problem_2,
    "doublepipe": create_double_pipe_problem,
    "doubletank": create_double_tank_problem,
    "doubletank2": create_double_tank_problem2,
    "stcs": create_stcs_problem,
    "gearbox": create_simple_gearbox,
    "gearbox_int": create_gearbox_int,
    "gearbox_complx": create_gearbox,
    "nonconvex": counter_example_nonconvexity,
    "unstable_ocp": create_ocp_unstable_system,
    "nl_file": create_from_nl_file,
    "nosnoc": create_from_nosnoc,
    "from_sto": create_from_sto,
    "from_nlpsol_dsc": create_from_nlpsol_description,
    "to_car": time_opt_car,
    "particle": particle_trajectory
}
PROBLEMS.update(MINLP_PROBLEMS)


if __name__ == '__main__':
    from camino.solvers.subsolvers.nlp import NlpSolver
    from camino.utils import plot_trajectory
    from camino.utils.conversion import to_0d
    import matplotlib.pyplot as plt
    from camino.solvers.decomposition.voronoi_master import VoronoiTrustRegionMIQP

    stats = Stats(mode='custom', problem_name='unstable_ocp')
    prob, data, settings = create_ocp_unstable_system()

    nlp = NlpSolver(prob, stats, settings)
    data = nlp.solve(data, set_x_bin=False)
    print(f"Relaxed  {data.obj_val=}")

    # Solve MIQP around MINLP solution
    miqp = VoronoiTrustRegionMIQP(prob, data, stats, settings)
    data = miqp.solve(data, prev_feasible=True, is_qp=True)

    data = nlp.solve(data, set_x_bin=True)
    x_star = data.x_sol
    print(f"{data.obj_val=}")

    if isinstance(prob.meta, MetaDataOcp):
        meta = prob.meta
        state = to_0d(x_star)[meta.idx_state].reshape(-1, meta.n_state)
        state = np.vstack([meta.initial_state, state])
        control = to_0d(x_star)[
            meta.idx_control].reshape(-1, meta.n_continuous_control)
        fig, axs = plot_trajectory(
            to_0d(x_star), state, control, meta, title='problem name')
        plt.show()

    # grad_f = ca.Function('grad_f', [prob.x, prob.p], [ca.gradient(prob.f, prob.x)])
    # grad_g = ca.Function('grad_g', [prob.x, prob.p], [ca.jacobian(prob.g, prob.x).T])
    # lambda_k = grad_f(data.x_sol, data.p) + grad_g(data.x_sol, data.p) @ data.lam_g_sol
    # assert np.allclose(-data.lam_x_sol, lambda_k)
