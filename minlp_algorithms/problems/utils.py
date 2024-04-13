"""Set of utilities to model a problem."""

# TODO: Move to utils folder

from typing import Union
from minlp_algorithms.settings import GlobalSettings as GS
import casadi as ca


def integrate_rk4(x: GS.CASADI_VAR, u: GS.CASADI_VAR, x_dot: GS.CASADI_VAR,
                  dt: Union[float, GS.CASADI_VAR], m_steps: int = 1):
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
    X0 = GS.CASADI_VAR.sym("X0", x.shape[0])
    U = GS.CASADI_VAR.sym("U", u.shape[0])
    X = X0
    if isinstance(dt, GS.CASADI_VAR):
        DT = GS.CASADI_VAR.sym("DT")
    else:
        DT = dt
    for _ in range(m_steps):
        k1 = f(X, U)
        k2 = f(X + DT / 2 * k1, U)
        k3 = f(X + DT / 2 * k2, U)
        k4 = f(X + DT * k3, U)
        X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    if isinstance(dt, GS.CASADI_VAR):
        return ca.Function("I_rk4", [X0, U, DT], [X], ["x0", "u", "dt"], ["xf"])
    else:
        return ca.Function("I_rk4", [X0, U], [X], ["x0", "u"], ["xf"])


def integrate_ee(x: GS.CASADI_VAR, u: GS.CASADI_VAR, x_dot: GS.CASADI_VAR,
                 dt: Union[float, GS.CASADI_VAR], m_steps: int = 1):
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
    X0 = GS.CASADI_VAR.sym("X0", x.shape[0])
    U = GS.CASADI_VAR.sym("U", u.shape[0])
    X = X0
    if isinstance(dt, GS.CASADI_VAR):
        DT = GS.CASADI_VAR.sym("DT")
    else:
        DT = dt
    for _ in range(m_steps):
        k1 = f(X, U)
        X = X + DT * k1
    if isinstance(dt, GS.CASADI_VAR):
        return ca.Function("I_ee", [X0, U, DT], [X], ["x0", "u", "dt"], ["xf"])
    else:
        return ca.Function("I_ee", [X0, U], [X], ["x0", "u"], ["xf"])


def integrate_ie(x: GS.CASADI_VAR, u: GS.CASADI_VAR, x_dot: GS.CASADI_VAR, dt: Union[float, GS.CASADI_VAR]):
    """Integrate Implicit Euler."""
    f = ca.Function("f", [x, u], [x_dot])
    X0 = GS.CASADI_VAR.sym("X0", x.shape[0])
    Xk = GS.CASADI_VAR.sym("Xk", x.shape[0])
    U = GS.CASADI_VAR.sym("U", u.shape[0])
    if isinstance(dt, GS.CASADI_VAR):
        DT = GS.CASADI_VAR.sym("DT")
    else:
        DT = dt
    f = X0 + DT * f(Xk, U) - Xk
    if isinstance(dt, GS.CASADI_VAR):
        return ca.Function("I_ie", [X0, U, Xk, DT], [f],
                           ["x0",  "u", "x1", "dt"], ["xf"])
    else:
        return ca.Function("I_ie", [X0, U, Xk], [f], ["x0", "u", "x1"], ["xf"])
