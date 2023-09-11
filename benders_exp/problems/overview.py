"""Overview of all problems."""

from benders_exp.problems import MinlpProblem, CASADI_VAR, MinlpData, \
    MetaDataOcp
import casadi as ca
import numpy as np
from benders_exp.problems.dsc import Description
from benders_exp.solarsys import extract as extract_solarsys
from benders_exp.solvers import Stats
from benders_exp.solvers.nlp import NlpSolver
from benders_exp.problems.utils import integrate_rk4  # integrate_ee
from benders_exp.problems.double_tank import create_double_tank_problem2
from benders_exp.problems.gearbox import create_simple_gearbox, create_gearbox, \
    create_gearbox_int


def create_ocp_unstable_system(p_val=[0.8, 0.7]):
    """
    OCP of a unstable system subject to min uptime constraints.

    Example taken from preprint of A. Buerger. Inspired by a textbook example of the MPC book by Rawlings, Mayne, Diehl
    """
    dt = 0.05
    N = 30
    min_uptime = 2  # in time steps

    dsc = Description()
    x = CASADI_VAR.sym('x')  # state
    u = CASADI_VAR.sym('u')  # control
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

    meta = MetaDataOcp(
        dt=dt, n_state=1, n_control=1,
        initial_state=p_val[0], idx_control=np.hstack(dsc.get_indices("Uk")),
        idx_state=np.hstack(dsc.get_indices("Xk")),
        scaling_coeff_control=[1],
        min_uptime=min_uptime
    )
    problem = dsc.get_problem()
    problem.meta = meta
    data = dsc.get_data()

    return problem, data


def create_check_sign_lagrange_problem():
    """Create a problem to check the sign of the multipliers."""
    x = CASADI_VAR.sym("x")
    p = CASADI_VAR.sym("p")

    problem = MinlpProblem(
        x=x, f=(x - 2)**2, g=ca.vcat([x]), p=p, idx_x_bin=[0])
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
    x = CASADI_VAR.sym("x", 3)
    x0 = np.array([0, 4, 100])
    idx_x_bin = [0, 1]
    p = CASADI_VAR.sym("p", 2)
    f = (x[0] - 4.1)**2 + (x[1] - 4.0)**2 + x[2] * p[0]
    g = ca.vertcat(
        x[2],
        -(x[0]**2 + x[1]**2 - x[2] - p[1]**2)
    )
    ubg = np.array([ca.inf, ca.inf])
    lbg = np.array([0, 0])
    lbx = np.array([0, 0, 0])
    ubx = np.array([4, 4, np.inf])

    problem = MinlpProblem(x=x, f=f, g=g, p=p, idx_x_bin=idx_x_bin)
    data = MinlpData(x0=x0, _ubx=ubx, _lbx=lbx,
                     _ubg=ubg, _lbg=lbg, p=p_val)
    return problem, data


def create_dummy_problem_2():
    """Create a dummy problem."""
    x = CASADI_VAR.sym("x", 2)
    x0 = np.array([0, 4])
    idx_x_bin = [0]
    p = CASADI_VAR.sym("p", 1)
    f = x[0]**2 + x[1]
    g = ca.vertcat(
        x[1],
        -(x[0]**2 + x[1] - p[0]**2)
    )
    ubg = np.array([np.inf, np.inf])
    lbg = np.array([0, 0])
    lbx = -np.inf * np.ones((2,))
    ubx = np.array([ca.inf, ca.inf])

    problem = MinlpProblem(x=x, f=f, g=g, p=p, idx_x_bin=idx_x_bin)
    data = MinlpData(x0=x0, _ubx=ubx, _lbx=lbx,
                     _ubg=ubg, _lbg=lbg, p=[3])
    return problem, data


def create_double_pipe_problem(p_val=[1, 5, 1, 10]):
    """Create double pipe problem."""
    y = CASADI_VAR.sym("y", 1)  # integers
    z = CASADI_VAR.sym("z", 2)  # continuous
    x0 = np.array([1, 0, 0])
    x = ca.vertcat(*[y, z])
    idx_x_bin = [0]

    alpha = CASADI_VAR.sym("alpha", 2)
    r = CASADI_VAR.sym("r", 1)
    gamma = CASADI_VAR.sym("gamma", 1)
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

    problem = MinlpProblem(x=x, f=f, g=g, p=p, idx_x_bin=idx_x_bin)
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
    x_0 = CASADI_VAR.sym('x0', nx)
    x = CASADI_VAR.sym('x', nx)  # state
    u = CASADI_VAR.sym('u', nu)  # control
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
    idx_x_bin = []
    idx_state = []
    idx_control = []
    idx_var = 0

    Xk = x_0
    p += [Xk]
    for k in range(N):
        Uk = CASADI_VAR.sym(f"u_{k}", nu)
        w += [Uk]
        idx_var += nu
        lbw += [0, 0]
        ubw += [1, gamma]
        w0 += [0.5, 0.5]
        idx_x_bin.append(np.arange(idx_var-nu, idx_var-1))
        idx_control.append(np.arange(idx_var-nu, idx_var))

        # Integrate till the end of the interval
        Xk_end = F(Xk, Uk)
        J += dt * alpha * (Xk[1] - demand[k]) ** 2
        J += dt * (beta[0] * scaling_coeff[0] * Uk[0] + beta[1] * Uk[1])

        # New NLP variable for state at end of interval
        Xk = CASADI_VAR.sym(f"x_{k+1}", nx)
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
        dt=dt, n_state=nx, n_control=nu,
        initial_state=p_val, idx_control=np.hstack(idx_control),
        idx_state=np.hstack(idx_state),
        scaling_coeff_control=scaling_coeff
    )
    problem = MinlpProblem(x=ca.vcat(w), f=J, g=ca.vcat(
        g), p=x_0, idx_x_bin=np.hstack(idx_x_bin), meta=meta)
    data = MinlpData(x0=ca.vcat(w0), _ubx=ca.vcat(ubw), _lbx=ca.vcat(lbw),
                     _ubg=np.array(ubg), _lbg=np.array(lbg), p=p_val)
    return problem, data


def counter_example_nonconvexity():
    """Nonconvexity example."""
    x = CASADI_VAR.sym('x')
    y = CASADI_VAR.sym('y')

    f = ca.atan(x-0.3)**2 + x/10 + x**2/50 + y**2
    problem = MinlpProblem(x=ca.vcat([x, y]), f=f, g=[], p=[], idx_x_bin=[0])
    data = MinlpData(x0=np.array([-4, 2]), _lbx=np.array([-5, -5]), _ubx=np.array([5, 5]),
                     _ubg=[], _lbg=[], p=[])
    return problem, data


def create_from_nl_file(file):
    """Load from NL file."""
    # Create an NLP instance
    nl = ca.NlpBuilder()

    # Parse an NL-file
    nl.import_nl(file, {"verbose": False})
    print(f"Loading MINLP with: {nl.repr()}")

    problem = MinlpProblem(x=nl.x, f=nl.f, g=nl.g, idx_x_bin=nl.discrete)
    data = MinlpData(x0=nl.x_init, _lbx=nl.x_lb, _ubx=nl.x_ub,
                     _lbg=nl.g_lb, _ubg=nl.g_ub, p=[])
    return problem, data


PROBLEMS = {
    "sign_check": create_check_sign_lagrange_problem,
    "dummy": create_dummy_problem,
    "dummy2": create_dummy_problem_2,
    "orig": extract_solarsys,
    "doublepipe": create_double_pipe_problem,
    "doubletank": create_double_tank_problem,
    "doubletank2": create_double_tank_problem2,
    "gearbox": create_simple_gearbox,
    "gearbox_int": create_gearbox_int,
    "gearbox_complx": create_gearbox,
    "nonconvex": counter_example_nonconvexity,
    "unstable_ocp": create_ocp_unstable_system,
    "nl_file": create_from_nl_file
}


if __name__ == '__main__':
    from benders_exp.utils import to_0d, plot_trajectory
    import matplotlib.pyplot as plt
    from benders_exp.solvers.voronoi import VoronoiTrustRegionMILP

    stats = Stats({})
    prob, data = create_ocp_unstable_system()

    nlp = NlpSolver(prob, stats)
    data = nlp.solve(data, set_x_bin=False)
    print(f"Relaxed  {data.obj_val=}")

    # Solve MIQP around MINLP solution
    miqp = VoronoiTrustRegionMILP(prob, data, stats)
    data = miqp.solve(data, prev_feasible=True, is_qp=True)

    data = nlp.solve(data, set_x_bin=True)
    x_star = data.x_sol
    print(f"{data.obj_val=}")

    if isinstance(prob.meta, MetaDataOcp):
        meta = prob.meta
        state = to_0d(x_star)[meta.idx_state].reshape(-1, meta.n_state)
        state = np.vstack([meta.initial_state, state])
        control = to_0d(x_star)[meta.idx_control].reshape(-1, meta.n_control)
        fig, axs = plot_trajectory(state, control, meta, title='problem name')
        plt.show()

    # grad_f = ca.Function('grad_f', [prob.x, prob.p], [ca.gradient(prob.f, prob.x)])
    # grad_g = ca.Function('grad_g', [prob.x, prob.p], [ca.jacobian(prob.g, prob.x).T])
    # lambda_k = grad_f(data.x_sol, data.p) + grad_g(data.x_sol, data.p) @ data.lam_g_sol
    # assert np.allclose(-data.lam_x_sol, lambda_k)
    # breakpoint()
