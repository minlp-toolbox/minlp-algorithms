"""
Solar Thermal Climate System (STCS) at Karsruhe University of Applied Sciences.

# Adrian Buerger, 2022
# Adapted by Wim Van Roy, 2023
"""

import numpy as np
from benders_exp.defines import CASADI_VAR
from benders_exp.problems import MetaDataOcp
from benders_exp.problems.solarsys.system import System, ca
from benders_exp.problems.solarsys.ambient import Ambient
from benders_exp.problems.dsc import Description
from benders_exp.utils.cache import CachedFunction
from datetime import timedelta
import logging

from benders_exp.utils import convert_to_flat_list

logger = logging.getLogger(__name__)


def create_stcs_problem():
    """Build problem."""
    logger.debug("Start processing")
    system = System()
    ambient = Ambient()
    dsc = Description()
    n_steps = 83
    dt = timedelta(seconds=1800)


    # TODO run simulator and predictor and use those output to warm start


    x0 = convert_to_flat_list(system.nx, system.x_index, system.get_default_initial_state())
    collocation_nodes = 2

    logger.debug("Constructing bounds")
    x_min = system.p_op["T"]["min"] * np.ones((n_steps + 1, system.nx))
    x_max = system.p_op["T"]["max"] * np.ones((n_steps+1, system.nx))
    x_max[:, system.x_index["T_shx_psc"][-1]] = system.p_op["T_sc"]["T_feed_max"]
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

    logger.debug("Creating basic equations")
    F = system.get_system_dynamics_collocation(collocation_nodes)
    T_ac_min_fcn = CachedFunction("stcs", system.get_t_ac_min_function)
    T_ac_max_fcn = CachedFunction("stcs", system.get_t_ac_max_function)
    slacked_state_fcn = CachedFunction("stcs", system.get_slacked_state_fcn)
    v_ppsc_so_fpsc_fcn = CachedFunction("stcs", system.get_v_ppsc_so_fpsc_fcn)
    v_ppsc_so_vtsc_fcn = CachedFunction("stcs", system.get_v_ppsc_so_vtsc_fcn)
    v_ppsc_so_fcn = CachedFunction("stcs", system.get_v_ppsc_so_fcn)
    mdot_hts_b_max_fcn = CachedFunction("stcs", system.get_mdot_hts_b_max_fcn)
    electric_power_balance_fcn = CachedFunction("stcs", system.get_electric_power_balance_fcn)
    F1_fcn = CachedFunction("stcs", system.get_F1_fcn)
    F2_fcn = CachedFunction("stcs", system.get_F2_fcn)

    logger.debug("Create NLP problem")
    # x_k_prev = dsc.add_parameters("x0", system.nx, x0) # TODO add it as parameter when debugging is finished!
    x_k_0 = dsc.sym("x_0_0", system.nx, lb=x0, ub=x0)
    u_k_prev = None
    tk = ambient.get_t0()
    F1 = []
    F2 = []
    for k in range(n_steps):
        logger.debug(f"Create NLP step {k}")
        tk += dt

        x_k_full = [x_k_0] + [
            dsc.sym(f"x_{k}_c", system.nx,
                    lb=-np.inf, ub=np.inf)
            for j in range(1, collocation_nodes + 1)
        ]
        x_k_next_0 = CASADI_VAR.sym(f"x_{k+1}_0", system.nx) # NOTE: append later to decision variables
        # x_k_next_0 = dsc.sym(f"x_{k+1}_0", system.nx,
                    #   lb=-ca.inf, ub=ca.inf)

        # Add new binary controls
        b_k = dsc.sym_bool("b", system.nb)

        # Add new continuous controls
        u_k = CASADI_VAR.sym(f"u_{k}", system.nu) # NOTE: append later to decision variables
        # u_k = dsc.sym("u", system.nu, lb=u_min[k, :], ub=u_max[k, :])

        if u_k_prev is None:
            u_k_prev = u_k

        # Add new parametric controls
        params = convert_to_flat_list(system.nc, system.c_index,
                                      ambient.interpolate(tk))
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
        s_ac_lb_k = dsc.sym("s_ac_lb", system.n_s_ac_lb, 0, ca.inf)

        # Setup T_ac_min conditions
        dsc.leq(0, T_ac_min_fcn(x_k_0, c_k, b_k, s_ac_lb_k))

        # Add new slack variable for T_ac_max condition
        s_ac_ub_k = dsc.sym("s_ac_ub", system.n_s_ac_ub, 0, ca.inf)

        # Setup T_ac_max conditions
        dsc.leq(T_ac_max_fcn(x_k_0, c_k, b_k, s_ac_ub_k), 0)

        # Add new slack variable for state limits soft constraints
        s_x_k = dsc.sym("s_x", system.nx, -ca.inf, ca.inf)

        # Setup state limits as soft constraints to prevent infeasibility
        dsc.add_g(x_min[k + 1, :], slacked_state_fcn(x_k_next_0, s_x_k), x_max[k + 1, :])

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

        # Append u_k and x_k_next_0 to the vector of variables
        dsc.add_w(lb=list(u_min[k, :]), ub=list(u_max[k, :]), w=u_k, w0=np.zeros(system.nu).tolist())
        dsc.add_w(lb=(-np.inf * np.ones(system.nx)).tolist(),
                  w=x_k_next_0,
                  ub=(np.inf * np.ones(system.nx)).tolist(),
                  w0=(0 * np.ones(system.nx)).tolist())

        F1.append(
            np.sqrt(dt_k / 3600)
            * F1_fcn(s_ac_lb_k, s_ac_ub_k, s_x_k, u_k, u_k_prev)
        )
        F2.append((dt_k / 3600) * F2_fcn(u_k, c_k))

        x_k_0 = x_k_next_0
        u_k_prev = u_k

    # Specify residual for GN Hessian computation
    dsc.r = F1

    # Concatenate objects
    F1 = 0.1 * ca.veccat(*F1)
    F2 = 0.01 * ca.sum1(ca.veccat(*F2))

    # Setup objective
    dsc.f = 0.5 * ca.mtimes(F1.T, F1) + F2
    logger.debug("NLP created")
    prob = dsc.get_problem()
    meta = MetaDataOcp(
        # n_state=len(dsc.indices['x'][0]), n_control=len(dsc.indices['u'][0]),
        idx_param=dsc.indices_p,
        # idx_state=np.hstack(dsc.indices['x']),
        # idx_control=np.hstack(dsc.indices['u']),
        )
    prob.meta = meta
    return prob, dsc.get_data()


if __name__ == "__main__":
    from benders_exp.utils import setup_logger, logging
    from benders_exp.problems import check_solution
    from benders_exp.solvers import Stats
    from benders_exp.solvers.nlp import NlpSolver
    import pickle

    setup_logger(logging.DEBUG)
    prob, data = create_stcs_problem()

    stats = Stats({})
    nlp = NlpSolver(prob, stats)

    with open("data/nlpargs_adrian.pickle", 'rb') as f:
        nlpargs_adrian = pickle.load(f)
    data.x0 = nlpargs_adrian['x0']
    data.p = nlpargs_adrian['p']
    data.lbx = nlpargs_adrian['lbx']
    data.ubx = nlpargs_adrian['ubx']
    data.lbg = nlpargs_adrian['lbg']
    data.ubg = nlpargs_adrian['ubg']

    data = nlp.solve(data, set_x_bin=False) # solve relaxed problem
    with open("results/x_star_rel_stcs.pickle", "wb") as f:
        pickle.dump(data.x_sol, f)
    breakpoint()
    check_solution(prob, data, data.prev_solution['x'])
