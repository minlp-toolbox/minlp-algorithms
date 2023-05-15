"""Overview of all problems."""

from benders_exp.problems import MinlpProblem, CASADI_VAR, MinlpData, \
    MetaDataOcp
import casadi as ca
import numpy as np
from benders_exp.solarsys import extract as extract_solarsys
from benders_exp.solvers import Stats
from benders_exp.solvers.nlp import NlpSolver


def create_check_sign_lagrange_problem():
    """Create a problem to check the sign of the multipliers."""
    x = CASADI_VAR.sym("x")
    p = CASADI_VAR.sym("p")

    problem = MinlpProblem(x=x, f=(x - 2)**2, g=x, p=p, idx_x_bin=[])
    data = MinlpData(x0=0, _ubx=np.inf, _lbx=-np.inf,
                     _ubg=-1, _lbg=-7, p=[], solved=True)

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
    lbx = -1e3 * np.ones((3,))
    ubx = np.array([ca.inf, ca.inf, ca.inf])

    problem = MinlpProblem(x=x, f=f, g=g, p=p, idx_x_bin=idx_x_bin)
    data = MinlpData(x0=x0, _ubx=ubx, _lbx=lbx,
                     _ubg=ubg, _lbg=lbg, p=p_val, solved=True)
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
                     _ubg=ubg, _lbg=lbg, p=[3], solved=True)
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
                     _ubg=ubg, _lbg=lbg, p=p_val, solved=True)
    return problem, data


def create_double_tank_problem(p_val=[2, 2.5]):
    """
    Implement the double tank problem.

    Taken from Abbasi et al. ECC 23, reimplemented to achieve nice sparsity pattern.
    """
    N = 300
    T = 10
    dt = T / N
    alpha = 100
    beta = np.array([[1., 1.1]])
    gamma = 10
    demand = np.array([2 + 0.5 * np.sin(x)
                      for x in np.linspace(0, T, N+1)])[np.newaxis, :]

    nx = 2
    ns = 2
    nq = 2
    x_0 = CASADI_VAR.sym('x0', nx)
    x = CASADI_VAR.sym('x', nx)  # state
    s = CASADI_VAR.sym('s', ns)  # binary control
    q = CASADI_VAR.sym('q', nq)  # continuous control

    x1dot = s.T @ q - ca.sqrt(x[0])
    x2dot = ca.sqrt(x[0]) - ca.sqrt(x[1])
    xdot = ca.Function('xdot', [x, s, q], [ca.vertcat(x1dot, x2dot)])
    # TODO: implement a RK4 integrator
    F = ca.Function('F', [x, s, q], [x + dt * xdot(x, s, q)])

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
        Qk = CASADI_VAR.sym(f"q_{k}", nq)
        w += [Qk]
        idx_var += nq
        lbw += [gamma, 0]
        ubw += [gamma, gamma]
        w0 += [0, 0]
        Sk = CASADI_VAR.sym(f"s_{k}", ns)
        w += [Sk]
        idx_var += ns
        idx_x_bin.append(np.arange(idx_var-ns, idx_var))
        idx_control.append(np.arange(idx_var-ns-nq, idx_var))
        lbw += [0, 0]
        ubw += [1, 1]
        w0 += [0, 0]

        # Integrate till the end of the interval
        Xk_end = F(Xk, Sk, Qk)
        J += dt * alpha * (Xk[1] - demand[:, k]) ** 2
        J += dt * ca.sum2(beta @ (Qk * Sk))

        # New NLP variable for state at end of interval
        Xk = CASADI_VAR.sym(f"x_{k+1}", nx)
        idx_var += nx
        idx_state.append(np.arange(idx_var-nx, idx_var))
        w += [Xk]
        lbw += [-np.inf, -np.inf]
        ubw += [np.inf, np.inf]
        w0 += [0, 0]

        # Add equality constraint
        g += [Xk_end - Xk]
        lbg += [0, 0]
        ubg += [0, 0]

    meta = MetaDataOcp(
        dt=dt, n_state=nx, n_control=ns+nq,
        initial_state=p_val, idx_control=np.hstack(idx_control),
        idx_state=np.hstack(idx_state)
    )
    problem = MinlpProblem(x=ca.vcat(w), f=J, g=ca.vcat(
        g), p=x_0, idx_x_bin=np.hstack(idx_x_bin), meta=meta)
    data = MinlpData(x0=0.5*np.ones(len(w0)), _ubx=np.array(ubw), _lbx=np.array(lbw),
                     _ubg=np.array(ubg), _lbg=np.array(lbg), p=p_val, solved=True)

    return problem, data


PROBLEMS = {
    "sign_check": create_check_sign_lagrange_problem,
    "dummy": create_dummy_problem,
    "dummy2": create_dummy_problem_2,
    "orig": extract_solarsys,
    "doublepipe": create_double_pipe_problem,
    "doubletank": create_double_tank_problem
}


if __name__ == '__main__':
    stats = Stats({})
    prob, data = create_check_sign_lagrange_problem()
    nlp = NlpSolver(prob, stats)
    data = nlp.solve(data)
    breakpoint()
    # prob, data = create_double_pipe_problem()
    # prob, data = create_dummy_problem()
