"""Models with pipes."""

from typing import Union
from benders_exp.problems import MinlpProblem, CASADI_VAR, MinlpData, \
    MetaDataOcp
from benders_exp.problems.dsc import Description
from benders_exp.problems.utils import integrate_rk4
import numpy as np
import casadi as ca


def create_double_tank_problem2(p_val=[2, 2.5]) -> Union[MinlpProblem, MinlpData]:
    """
    Implement the double tank problem.

    Taken from Abbasi et al. ECC 23, reimplemented to achieve nice sparsity pattern.

    """
    eps = 1e-3
    N = 300
    dt = 1/30
    T = N * dt
    min_uptime = int(0.5 / dt)
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
    Uprev = None
    for k in range(N):
        Uk = dsc.sym("Uk", nu, lb=[0, 0], ub=[
                     1, gamma], w0=[0, 0], discrete=[1, 0])
        if Uprev is not None and min_uptime > 0:
            # Implementation details: you can switch OFF only there are
            # no switch UP happened in the last "min_uptime -1" steps
            Suk = dsc.sym("Suk", 1, lb=0, ub=1, w0=0, discrete=True)  # switching up variable
            dsc.eq(Uk[0] - Uprev[0] - Suk, [0])  # switch up detection

            # retrieve the indexes of Suk for window [k-min_uptime+1, k]
            slicing_idx = np.array(dsc.get_indices(
                "Suk"))[-min(min_uptime, k) + 1:]
            # min-uptime constraint: sum of switch UP in the window must be \leq than Uk
            dsc.leq(ca.sum1(ca.vcat(*[dsc.w])[slicing_idx]) - Uk[0], [0])
            # --> Uk can be 0 only if the min_uptime window there are no switch UP actions.

        # Integrate till the end of the interval
        Xk_end = F(Xk, Uk)
        dsc.f += dt * alpha * (Xk[1] - demand[k]) ** 2
        dsc.f += dt * (beta[0] * gamma * Uk[0] + beta[1] * Uk[1])

        # New NLP variable for state at end of interval
        Xk = dsc.sym("Xk", nx, lb=0, ub=BigM, w0=0.5)
        dsc.eq(Xk_end, Xk)

        Uprev = Uk

    meta = MetaDataOcp(
        dt=dt, n_state=nx, n_control=nu,
        initial_state=p_val, idx_control=np.hstack(dsc.get_indices("Uk")),
        idx_state=np.hstack(dsc.get_indices("Xk")),
        scaling_coeff_control=[gamma, 1], min_uptime=min_uptime
    )
    problem = dsc.get_problem()
    problem.meta = meta
    data = dsc.get_data()

    return problem, data
