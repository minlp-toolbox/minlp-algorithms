"""Models with pipes."""

from typing import Union
from benders_exp.problems import MinlpProblem, CASADI_VAR, MinlpData, \
    MetaDataOcp
from benders_exp.problems.dsc import Description
from benders_exp.utils import integrate_rk4
import numpy as np
import casadi as ca


def create_double_tank_problem2(p_val=[2, 2.5]) -> Union[MinlpProblem, MinlpData]:
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
    demand = np.array([2 + 0.5 * np.sin(x)
                      for x in np.linspace(0, T, N+1)])

    nx = 2
    nu = 2
    x = CASADI_VAR.sym('x', nx)  # state
    u = CASADI_VAR.sym('u', nu)  # control
    x1dot = gamma * u[0] + u[1] - ca.sqrt(x[0] + eps)
    x2dot = ca.sqrt(x[0] + eps) - ca.sqrt(x[1] + eps)
    xdot = ca.vertcat(*[x1dot, x2dot])
    # F = integrate_ee(x, u, xdot, dt, m_steps=1)
    F = integrate_rk4(x, u, xdot, dt, m_steps=1)

    dsc = Description()
    Xk = dsc.add_parameters("Xk0", nx, p_val)
    BigM = 1e3
    for k in range(N):
        Uk = dsc.sym("Uk", nu, lb=[0, 0], ub=[1, gamma], w0=[0.5, 0.5], discrete=[1, 0])

        # Integrate till the end of the interval
        Xk_end = F(Xk, Uk)
        dsc.f += dt * alpha * (Xk[1] - demand[k]) ** 2
        dsc.f += dt * (beta[0] * gamma * Uk[0] + beta[1] * Uk[1])

        # New NLP variable for state at end of interval
        Xk = dsc.sym("Xk", nx, lb=0, ub=BigM, w0=0.5)
        dsc.eq(Xk_end, Xk)

    meta = MetaDataOcp(
        dt=dt, n_state=nx, n_control=nu,
        initial_state=p_val, idx_control=np.hstack(dsc.get_indices("Uk")),
        idx_state=np.hstack(dsc.get_indices("Xk")),
        scaling_coeff_control=[gamma, 1]
    )
    problem = dsc.get_problem()
    problem.meta = meta
    data = dsc.get_data()

    return problem, data
