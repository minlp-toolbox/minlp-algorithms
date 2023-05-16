"""Set of utilities to model a problem."""

from benders_exp.defines import CASADI_VAR
import casadi as ca


def integrate_rk4(x: CASADI_VAR, u: CASADI_VAR, x_dot: CASADI_VAR, dt: float, m_steps: int = 1):
    """
    Implement RK4 integrator for ODE.

    x: state
    u: control
    x_dot: ODE that describes the continuous time dynamics of the system
    dt: integration time
    m_steps: number of integration steps per interval
    """
    if m_steps > 1:
        dt = dt / m_steps

    f = ca.Function("f", [x, u], [x_dot])
    X0 = CASADI_VAR.sym("X0", x.shape[0])
    U = CASADI_VAR.sym("U", u.shape[0])
    X = X0
    for _ in range(m_steps):
        k1 = f(X, U)
        k2 = f(X + dt / 2 * k1, U)
        k3 = f(X + dt / 2 * k2, U)
        k4 = f(X + dt * k3, U)
        X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function("I_rk4", [X0, U], [X], ["x0", "u"], ["xf"])


def integrate_ee(x: CASADI_VAR, u: CASADI_VAR, x_dot: CASADI_VAR, dt: float, m_steps: int = 1):
    """
    Implement explicit euler integrator for ODE.

    x: state
    u: control
    x_dot: ODE that describes the continuous time dynamics of the system
    dt: integration time
    m_steps: number of integration steps per interval
    """
    if m_steps > 1:
        dt = dt / m_steps

    f = ca.Function("f", [x, u], [x_dot])
    X0 = CASADI_VAR.sym("X0", x.shape[0])
    U = CASADI_VAR.sym("U", u.shape[0])
    X = X0
    for _ in range(m_steps):
        k1 = f(X, U)
        X = X + dt * k1
    return ca.Function("I_ee", [X0, U], [X], ["x0", "u"], ["xf"])
