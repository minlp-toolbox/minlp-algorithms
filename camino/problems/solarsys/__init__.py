"""
Solar Thermal Climate System (STCS) at Karsruhe University of Applied Sciences.

Adrian Buerger, 2022
Adapted by Wim Van Roy and Andrea Ghezzi, 2023
"""

import numpy as np
from camino.settings import Settings
from camino.problems import MetaDataOcp
from camino.problems.solarsys.system import System, ca
from camino.problems.solarsys.ambient import Ambient, Timing
from camino.problems.solarsys.simulator import Simulator
from camino.problems.dsc import Description
from camino.settings import GlobalSettings
from camino.utils.cache import CachedFunction
import logging

from camino.utils.conversion import to_0d, convert_to_flat_list

logger = logging.getLogger(__name__)


def create_stcs_problem(n_steps=None, with_slack=True):
    """Build problem."""
    logger.debug("Start processing")
    system = System()
    timing = Timing()
    ambient = Ambient(timing)
    dsc = Description()
    if n_steps is None:
        n_steps = timing.N
    else:
        n_steps = int(n_steps)

    # Run simulator and predictor and use those output to warm start
    simulator = Simulator(ambient=ambient, N=n_steps)
    simulator.solve()
    # x_hat = simulator.predict()

    collocation_nodes = 2

    logger.debug("Constructing bounds")
    x_min = system.p_op["T"]["min"] * np.ones((n_steps + 1, system.nx))
    x_max = system.p_op["T"]["max"] * np.ones((n_steps+1, system.nx))
    x_max[:, system.x_index["T_shx_psc"][-1]
          ] = system.p_op["T_sc"]["T_feed_max"]
    x_max[:, system.x_index["T_lts"]] = system.p_op["T_lts"]["max"]

    u_min = np.hstack(
        [
            system.p_op["v_ppsc"]["min_mpc"] * np.ones((n_steps, 1)),
            system.p_op["p_mpsc"]["min_mpc"] * np.ones((n_steps, 1)),
            system.p_op["v_pssc"]["min_mpc"] * np.ones((n_steps, 1)),
            -np.ones((n_steps, 1)),
            # The upcoming controls are constrained later in the NLP using inequality constraints
            np.zeros((n_steps, 1)),
            np.zeros((n_steps, 1)),
        ]
    )
    u_max = np.hstack(
        [
            system.p_op["v_ppsc"]["max"] * np.ones((n_steps, 1)),
            system.p_op["p_mpsc"]["max"] * np.ones((n_steps, 1)),
            system.p_op["v_pssc"]["max"] * np.ones((n_steps, 1)),
            np.ones((n_steps, 1)),
            np.inf * np.ones((n_steps, 1)),
            np.inf * np.ones((n_steps, 1)),
        ]
    )
    min_up_times = np.asarray(
        system.p_op["acm"]["min_up_time"] +
        system.p_op["hp"]["min_up_time"]
    ) - 1e-3
    min_down_times = np.asarray(
        system.p_op["acm"]["min_down_time"] +
        system.p_op["hp"]["min_down_time"]
    ) - 1e-3

    logger.debug("Creating basic equations")
    F = system.get_system_dynamics_collocation(collocation_nodes)
    T_ac_min_fcn = CachedFunction("stcs", system.get_t_ac_min_function)
    T_ac_max_fcn = CachedFunction("stcs", system.get_t_ac_max_function)
    slacked_state_fcn = CachedFunction("stcs", system.get_slacked_state_fcn)
    v_ppsc_so_fpsc_fcn = CachedFunction("stcs", system.get_v_ppsc_so_fpsc_fcn)
    v_ppsc_so_vtsc_fcn = CachedFunction("stcs", system.get_v_ppsc_so_vtsc_fcn)
    v_ppsc_so_fcn = CachedFunction("stcs", system.get_v_ppsc_so_fcn)
    mdot_hts_b_max_fcn = CachedFunction("stcs", system.get_mdot_hts_b_max_fcn)
    electric_power_balance_fcn = CachedFunction(
        "stcs", system.get_electric_power_balance_fcn)
    F1_fcn = CachedFunction("stcs", system.get_F1_fcn)
    F2_fcn = system.get_F2_fcn()  # CachedFunction("stcs", system.get_F2_fcn)

    logger.debug("Create NLP problem")
    x_k_0 = dsc.add_parameters(
        "x0", system.nx, to_0d(simulator.x_data[0, :]).tolist())
    u_k_prev = None
    tk = ambient.get_t0()
    F1 = []
    F2 = []

    for k in range(n_steps):
        logger.debug(f"Create NLP step {k}")
        dt = ambient.time_steps[k]
        tk += dt

        x_k_full = [x_k_0] + [
            dsc.sym(f"x_{k}_c", system.nx,
                    lb=-np.inf, ub=np.inf, w0=to_0d(simulator.x_data[k, :]).tolist())
            for j in range(1, collocation_nodes + 1)
        ]
        x_k_next_0 = dsc.sym("x", system.nx, lb=-ca.inf, ub=ca.inf,
                             w0=to_0d(simulator.x_data[k+1, :]).tolist())

        # Add new binary controls
        b_k = dsc.sym_bool("b", system.nb)

        # Add new continuous controls
        u_k = dsc.sym("u", system.nu, lb=u_min[k, :], ub=u_max[k, :],  w0=to_0d(
            simulator.u_data[k, :]).tolist())

        if u_k_prev is None:
            u_k_prev = u_k

        # Add new parametric controls
        params = convert_to_flat_list(system.nc, system.c_index,
                                      ambient.interpolate(tk-dt))
        c_k = dsc.add_parameters("c", system.nc, params)
        dt_k = dsc.add_parameters("dt", 1, dt.total_seconds())

        # Add collocation equations
        F_k_inp = {"x_k_" + str(i): x_k_i for i, x_k_i in enumerate(x_k_full)}
        F_k_inp.update(
            {
                "x_k_next": x_k_next_0,
                "c_k": c_k,
                "u_k": u_k,
                "b_k": b_k,
                "dt_k": dt_k,
            }
        )
        F_k = F(**F_k_inp)
        dsc.eq(F_k["eq_c"], 0)
        dsc.eq(F_k["eq_d"], 0)

        # Add new slack variable for T_ac_min condition
        if with_slack:
            s_ac_lb_k = dsc.sym("s_ac_lb", system.n_s_ac_lb, 0, ca.inf)
        else:
            s_ac_lb_k = np.zeros((system.n_s_ac_lb,))

        # Setup T_ac_min conditions
        dsc.leq(0, T_ac_min_fcn(x_k_0, c_k, b_k, s_ac_lb_k))

        # Add new slack variable for T_ac_max condition
        if with_slack:
            s_ac_ub_k = dsc.sym("s_ac_ub", system.n_s_ac_ub, 0, ca.inf)
        else:
            s_ac_ub_k = np.zeros((system.n_s_ac_ub,))

        # Setup T_ac_max conditions
        dsc.leq(T_ac_max_fcn(x_k_0, c_k, b_k, s_ac_ub_k), 0)

        # Add new slack variable for state limits soft constraints
        if with_slack:
            s_x_k = dsc.sym("s_x", system.nx, -ca.inf, ca.inf, w0=0)
        else:
            s_x_k = np.zeros((system.nx,))

        # Setup state limits as soft constraints to prevent infeasibility
        dsc.add_g(x_min[k + 1, :],
                  slacked_state_fcn(x_k_next_0, s_x_k), x_max[k + 1, :])

        # Assure ppsc is running at high speed when collector temperature is high
        s_ppsc_k = dsc.sym("s_ppsc_fpsc", 1, lb=0, ub=1)

        dsc.leq(v_ppsc_so_fpsc_fcn(x_k_0, s_ppsc_k), 0)
        dsc.leq(v_ppsc_so_vtsc_fcn(x_k_0, s_ppsc_k), 0)
        dsc.leq(v_ppsc_so_fcn(u_k, s_ppsc_k), 0)

        # Assure HTS bottom layer mass flows are always smaller or equal to
        # the corresponding total pump flow
        dsc.leq(mdot_hts_b_max_fcn(u_k, b_k), 0)

        # Electric power balance
        dsc.eq(electric_power_balance_fcn(x_k_0, u_k, b_k, c_k), 0)

        # SOS1 constraint
        dsc.add_g(0, ca.sum1(b_k), 1)

        F1.append(
            np.sqrt(dt_k / 3600)
            * F1_fcn(s_ac_lb_k, s_ac_ub_k, s_x_k, u_k, u_k_prev)
        )
        F2.append((dt_k / 3600) * F2_fcn(u_k, c_k))

        x_k_0 = x_k_next_0
        u_k_prev = u_k

    # Concatenate objects
    F1 = 0.1 * ca.veccat(*F1)
    F2 = 0.01 * sum(F2)

    # Specify residual for GN Hessian computation
    dsc.r = F1

    idx_b_2d = np.asarray(dsc.get_indices('b')).T
    w = ca.vertcat(*dsc.w)

    # Add min uptime
    for k in range(-1, n_steps + 1):
        for i in range(system.nb):
            uptime = 0
            it = 0
            for dt in ambient.time_steps[max(0, k):]:
                uptime += dt.total_seconds()
                if uptime < min_up_times[i]:
                    if k != -1:
                        idx_k = idx_b_2d[i, k]

                    try:
                        idx_k_1 = idx_b_2d[i, k + 1]
                        idx_k_dt = idx_b_2d[i, k + it + 2]
                    except IndexError:
                        pass
                    if k != -1:
                        dsc.leq(- w[idx_k] + w[idx_k_1] - w[idx_k_dt], 0)
                    else:
                        dsc.leq(w[idx_k_1] - w[idx_k_dt], 0)

                    it += 1
    # Add min downtime
    for k in range(-1, n_steps + 1):
        for i in range(system.nb):
            downtime = 0
            it = 0
            for dt in ambient.time_steps[max(0, k):]:
                downtime += dt.total_seconds()
                if downtime < min_down_times[i]:
                    if k != -1:
                        idx_k = idx_b_2d[i, k]

                    try:
                        idx_k_1 = idx_b_2d[i, k + 1]
                        idx_k_dt = idx_b_2d[i, k + it + 2]
                    except IndexError:
                        pass
                    if k != -1:
                        dsc.leq(w[idx_k] - w[idx_k_1] + w[idx_k_dt], 1)
                    else:
                        dsc.leq(- w[idx_k_1] + w[idx_k_dt], 1)

                    it += 1

    # Setup objective
    dsc.f = 0.5 * ca.mtimes(F1.T, F1) + F2
    logger.debug("NLP created")
    prob = dsc.get_problem()
    x_bar = GlobalSettings.CASADI_VAR.sym("x_bar", prob.x.shape)
    fun_F1 = ca.Function("F1", [prob.x, prob.p], [F1])
    fun_F2 = ca.Function("F2", [prob.x, prob.p], [F2])
    fun_grad_F1 = ca.Function("F1_grad", [prob.x, prob.p], [
                              ca.jacobian(F1, prob.x).T])
    fun_grad_F2 = ca.Function("F2_grad", [prob.x, prob.p], [
                              ca.jacobian(F2, prob.x).T])

    x = prob.x
    prob.f_qp = ca.Function("F_qp", [x, x_bar, prob.p], [
        0.5 * ca.mtimes(fun_F1(x_bar, prob.p).T, fun_F1(x_bar, prob.p))
        + ca.mtimes([(x - x_bar).T,  fun_grad_F1(x_bar, prob.p),
                    fun_F1(x_bar, prob.p)])
        + 0.5 * ca.mtimes([(x - x_bar).T, fun_grad_F1(x_bar, prob.p),
                          fun_grad_F1(x_bar, prob.p).T, (x - x_bar)])
        + fun_F2(x_bar, prob.p) +
        ca.mtimes(fun_grad_F2(x_bar, prob.p).T, x - x_bar)
    ])
    meta = MetaDataOcp(
        n_state=system.nx, n_continuous_control=system.nu, n_discrete_control=system.nb,
        idx_param=dsc.indices_p,
        idx_state=np.hstack(dsc.indices['x']).tolist(),
        idx_control=np.hstack(dsc.indices['u']).tolist(),
        idx_bin_control=np.hstack(dsc.indices['b']).tolist(),
        initial_state=to_0d(simulator.x_data[0, :]).tolist(),
        dt=ambient.time_steps,
        min_uptime=min_up_times,
        min_downtime=min_down_times,
    )
    prob.meta = meta
    data = dsc.get_data()
    data.x0[prob.idx_x_integer] = to_0d(simulator.b_data).flatten().tolist()
    s = Settings()
    s.BRMIQP_GAP = 0.15
    s.LBMILP_GAP = 0.15
    s.ALPHA_KRONQVIST = 0.2
    s.IPOPT_SETTINGS.update({
        "ipopt.linear_solver": "ma57",
        "ipopt.mumps_mem_percent": 10000,
        "ipopt.mumps_pivtol": 0.001,
        "ipopt.max_cpu_time": 3600.0,
        "ipopt.max_iter": 5000,
        "ipopt.acceptable_tol": 1e-1,
        "ipopt.acceptable_iter": 8,
        "ipopt.acceptable_constr_viol_tol": 10.0,
        "ipopt.acceptable_dual_inf_tol": 10.0,
        "ipopt.acceptable_compl_inf_tol": 10.0,
        "ipopt.acceptable_obj_change_tol": 1e-1,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.mu_target": 1e-5,
        "ipopt.print_frequency_iter": 100,
    })
    s.MIP_SETTINGS_ALL["gurobi"].update({
        "gurobi.PoolSearchMode": 0,
        "gurobi.PoolSolutions": 3,
        "gurobi.TimeLimit": 600,
    })
    # 7 days...
    s.TIME_LIMIT = 7 * 24 * 3600

    # Improve calculation speed by getting indices:
    # set_constraint_types(prob, *cache_data(
    #     f"scts_{n_steps}_{with_slack}", inspect_problem, prob, data
    # ))

    return prob, data, s


if __name__ == "__main__":
    from camino.utils import setup_logger, logging
    from camino.utils.validate import check_solution
    from camino.stats import Stats
    from camino.solvers.subsolvers.nlp import NlpSolver
    from datetime import datetime
    import pickle

    setup_logger(logging.DEBUG)
    prob, data = create_stcs_problem()
    s = Settings()
    s.IPOPT_SETTINGS
    stats = Stats(mode='custom', problem_name='stcs',
                  datetime=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), data={})
    nlp = NlpSolver(prob, stats, s)

    # with open("data/nlpargs_adrian.pickle", 'rb') as f:
    #     nlpargs_adrian = pickle.load(f)
    # data.x0 = nlpargs_adrian['x0']
    # data.p = nlpargs_adrian['p']
    # data.lbx = nlpargs_adrian['lbx']
    # data.ubx = nlpargs_adrian['ubx']
    # data.lbg = nlpargs_adrian['lbg']
    # data.ubg = nlpargs_adrian['ubg']

    data = nlp.solve(data, set_x_bin=False)  # solve relaxed problem
    with open("results/x_star_rel_test.pickle", "wb") as f:
        pickle.dump(data.x_sol, f)
    breakpoint()
    check_solution(prob, data, data.prev_solution['x'], s)
