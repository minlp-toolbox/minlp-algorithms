# @Author: Massimo De Mauri <massimo>
# @Date:   2020-12-30T12:40:22+01:00
# @Email:  massimo.demauri@protonmail.com
# @Filename: models.py
# @Last modified by:   massimo
# @Last modified time: 2020-12-30T12:46:07+01:00
# @License: LGPL-3.0

import casadi as cs
from camino.settings import GlobalSettings

# engine


def insight_50kw_power_jc():

    # properties : jointly-convex
    w = GlobalSettings.CASADI_VAR.sym('w')
    P = GlobalSettings.CASADI_VAR.sym('P')

    wrat = GlobalSettings.CASADI_VAR.sym('wrat')
    Prat = GlobalSettings.CASADI_VAR.sym('Prat')

    Trat = 1.1349*Prat/wrat

    # scaled inputs
    wn = w/wrat
    Pn = P/Prat

    # generic data
    dFmax = 0.1*Prat/1000
    # fuel_volume = 1.3353  # ml/g or l/kg
    # fuel_density = 0.7489  # Kg/l
    mass = (1.8 + 0.6)*Prat/1000
    inertia = 0.1  # guessed
    minw = 0.1*wrat  # guessed
    dFi = 1e-3*Prat*0.045665168578973/1.1

    # fuel consumption constraint
    # 1
    a00 = 0.198 - 0.5054 + 0.5006 + 0.0234
    a01 = 2.642 - 0.03629 - 1.438 - 0.09664 + 0.5312 - 0.3161
    a02 = 0.6187 + 0.0307 + 1.51 - 0.2206 + 1.585 - 1.498
    a10 = -1.01 - 0.02489
    a11 = -0.1102 + 0.005099 + 0.2856 + 0.2132 - 1.662 + 1.374
    a20 = 1.867 + 0.01006
    dFr1 = Prat/50e3*(a20*wn**2 + a02*Pn**2 + a11 *
                      wn*Pn + a10*wn + a01*Pn + a00)
    # 2
    a00 = 0.1662
    a01 = 1 - 1.438 - 0.09664 + 0.5312 - 0.3161 - 0.112 - 0.7949
    a02 = 0.1 + 1.51 - 0.2206 + 1.585 - 1.498 + 3.476 + 4.362
    a10 = 0.05455
    a11 = 0.4 + 0.2856 + 0.2132 - 1.662 + 1.374 - 3.029 - 3
    a20 = 1
    dFr2 = Prat/50e3*(a20*wn**2 + a02*Pn**2 + a11 *
                      wn*Pn + a10*wn + a01*Pn + a00)
    # 3
    a00 = -0.04825
    a01 = 2.274
    a02 = 0
    a10 = 0.6964
    a11 = 0
    a20 = 0
    dFr3 = Prat/50e3*(a20*wn**2 + a02*Pn**2 + a11 *
                      wn*Pn + a10*wn + a01*Pn + a00)
    # 4
    a00 = -2.507e-10
    a01 = 1.589
    a02 = 4 + 0.0246
    a10 = -2.867e-12
    a11 = -1
    a20 = 0.05 + 0.0246
    dFr4 = Prat/50e3*(a20*wn**2 + a02*Pn**2 + a11 *
                      wn*Pn + a10*wn + a01*Pn + a00)
    # 5
    a00 = -0.07
    a01 = 6
    a02 = 0
    a10 = -2.5
    a11 = 0
    a20 = 0
    dFr5 = Prat/50e3*(a20*wn**2 + a02*Pn**2 + a11 *
                      wn*Pn + a10*wn + a01*Pn + a00)
    # result
    dFr = cs.vertcat(dFr1, dFr2, dFr3, dFr4, dFr5)

    # max torque
    # 1
    p1 = 5.6e+04
    p0 = 0
    Pmax1 = Prat/50e3*(p1*wn + p0)
    # 2
    p1 = 2.68e+04
    p0 = 2.334e+04
    Pmax2 = Prat/50e3*(p1*wn + p0)
    # result
    Pmax = cs.vertcat(Pmax1, Pmax2)

    return {'dFin': cs.Function('dFin', [wrat, Prat, w, P], [dFr]),
            'Fstart': cs.Function('Fstart', [Prat], [dFi]),
            'Pmax':  cs.Function('Pmax', [wrat, Prat, w], [Pmax]),
            'Trat':  cs.Function('Trat', [wrat, Prat], [Trat]),
            'mass':  cs.Function('mass', [Prat], [mass]),
            'minw':  cs.Function('minw', [wrat], [minw]),
            'inertia': cs.Function('inertia', [Prat], [inertia]),
            'dFmax': cs.Function('dFmax', [Prat], [dFmax])}


# electric motor
def advisor_em_pwr_sc():
    # properties : semi-convex
    w = GlobalSettings.CASADI_VAR.sym('w')
    P = GlobalSettings.CASADI_VAR.sym('P')

    wrat = GlobalSettings.CASADI_VAR.sym('wrat')
    Prat = GlobalSettings.CASADI_VAR.sym('Prat')

    Trat = 3.7858*Prat/wrat

    # scaled inputs
    wn = w/wrat
    Pn = P/Prat

    # generic data
    mass = 1.21*Prat/1000
    Pin_max = 1.2*Prat
    inertia = 0.1

    c01 = -1.604e-12*0.85
    c02 = 0.1719*0.85
    c10 = 0.02417*0.85
    c20 = 2.895e-05*0.85
    Ploss0 = Prat*(c20*wn**2 + c02*Pn**2 + c10*wn + c01*Pn)

    c01 = 0
    c02 = 0.2425
    c10 = 0.6099
    c20 = -2.669
    Ploss1 = Prat*(c20*wn**2 + c02*Pn**2 + c10*wn + c01*Pn)

    c01 = -0.175
    c02 = 0.08767
    c10 = -0.3903
    c20 = 0.3944
    Ploss2 = Prat*(c20*wn**2 + c02*Pn**2 + c10*wn + c01*Pn)

    c01 = 0.175
    c02 = 0.08767
    c10 = -0.3903
    c20 = 0.3944
    Ploss3 = Prat*(c20*wn**2 + c02*Pn**2 + c10*wn + c01*Pn)

    Pin = cs.vertcat(P+Ploss0, P+Ploss1, P+Ploss2, P+Ploss3)
    # maps
    Pmax = cs.vertcat(Prat, Trat*wrat*wn)

    return {'Pin': cs.Function('Pin', [wrat, Prat, w, P], [Pin]),
            'Pmax': cs.Function('Pmax', [wrat, Prat, w], [Pmax]),
            'Trat': cs.Function('Trat', [wrat, Prat], [Trat]),
            'mass': cs.Function('mass', [Prat], [mass]),
            'inertia': cs.Function('inertia', [Prat], [inertia]),
            'Pin_max': cs.Function('Pin_max', [Prat], [Pin_max]),
            }


# battery
def battery_hu(Erat=GlobalSettings.CASADI_VAR.sym('Erat')):

    Q = 8280  # amp*s
    C = 51782  # farad
    R = 0.01  # ohm
    m = 0.07  # kilograms per cell
    imax = 70  # amp
    imin = -35  # amp
    U0 = 3.3  # volts

    SoC = GlobalSettings.CASADI_VAR.sym('Soc')       # state of charge
    P = GlobalSettings.CASADI_VAR.sym('P')           # internal power in watt

    n = Erat/(.5*Q**2/C + Q*U0)   # num cells
    mass = m*n                   # total battery mass
    E = Erat*SoC                 # current energy

    dSoC = -P/Erat
    Pout = P - (R*C*P**2)/(2*E + n*C*U0**2)
    # Pmax = imax*cs.sqrt(n*(2*Erat/C + n*U0**2))
    # Pmin = imin*cs.sqrt(n*(2*Erat/C + n*U0**2))
    Pmax = imax*cs.sqrt(n*(2*E/C + n*U0**2))
    Pmin = imin*cs.sqrt(n*(2*E/C + n*U0**2))

    return {'dSoC': cs.Function('dSoC', [Erat, SoC, P], [dSoC]),
            'Pout': cs.Function('Pout', [Erat, SoC, P], [Pout]),
            'Pmax': cs.Function('Pmax', [Erat, SoC], [Pmax]),
            'Pmin': cs.Function('Pmin', [Erat, SoC], [Pmin]),
            'SoC_max': .8,
            'SoC_min': .2,
            'mass': cs.Function('mass', [Erat], [mass])}
