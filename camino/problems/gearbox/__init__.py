#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Car model.

Based on car model of Massimo De Mauri
"""

import csv
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from os import path

from typing import Union
from camino.settings import GlobalSettings
from camino.problems import MinlpData, MetaDataMpc, MinlpProblem
from camino.problems.gearbox.models import insight_50kw_power_jc, \
    advisor_em_pwr_sc, battery_hu
from camino.problems.dsc import Description
from camino.utils.cache import CachedFunction


def get_cycle_data(dt=1, N=30):
    """Get cycle data."""
    current_directory = path.abspath(path.dirname(__file__))
    # import the driving cycle
    with open(current_directory + '/cycle.csv', 'r') as csvfile:
        cycle_dat = [
            float(s) for s in list(csv.reader(csvfile, delimiter=','))[0]
        ][20:]

    xgrid = np.linspace(0, len(cycle_dat), len(cycle_dat))
    f = ca.interpolant("data", "bspline", [xgrid], cycle_dat)
    cycle = f(np.arange(0, dt * (N+1), dt))
    return cycle, cycle[1:] - cycle[:-1]


def create_gear_step(dsc, nr_gears, gears_prev, gears_prev2, allow_stop=False):
    """Create a setp of gears."""
    gears = dsc.sym_bool("gears", nr_gears)
    dsc.eq(ca.sum1(gears), 1)

    if allow_stop:
        dsc.leq(gears[0], gears_prev[0] + gears_prev[1])
        dsc.leq(gears[0], gears_prev2[0] + gears_prev2[1])

    for j in range(1, nr_gears - 1):
        dsc.leq(gears[j], gears_prev[j-1] + gears_prev[j] + gears_prev[j+1])
        dsc.leq(gears[j], gears_prev2[j-1] + gears_prev2[j] + gears_prev2[j+1])

    j = nr_gears - 1
    dsc.leq(gears[j], gears_prev[j-1] + gears_prev[j])
    dsc.leq(gears[j], gears_prev2[j-1] + gears_prev2[j])
    return gears


def create_gearbox_int(N=10, traject=None):
    """Create integer gearbox."""
    if traject is None:
        traject = [np.sin(i / 10) * 4 + 5 for i in range(N)]

    def cost_gear(j, setpoint):
        """Cost to run at setpoint for gear j."""
        return (10 * (setpoint - j*2 + 1))**2 + 5 * setpoint + j

    dsc = Description()

    traject = dsc.add_parameters("traject", N, traject)
    nr_gears = 5
    gears_prev2 = dsc.add_parameters("gears-1", 1, [0])
    gears_prev = dsc.add_parameters("gears0", 1, [0])
    throttle_prev = dsc.add_parameters("throttle", 1, [0])
    for i in range(N):
        gears = dsc.sym("gear", 1, 0, 5, 0, discrete=True)
        throttle = dsc.sym("throttle", 1, 0, 10)
        dsc.leq(gears - gears_prev, 1)
        dsc.leq(gears_prev - gears, 1)
        dsc.leq(gears - gears_prev2, 1)
        dsc.leq(gears_prev2 - gears, 1)
        dsc.leq(throttle - throttle_prev, 0.5)
        dsc.leq(throttle_prev - throttle, 0.5)

        dsc.leq(throttle, gears + 1)
        dsc.leq(gears - 1, throttle)

        for j in range(nr_gears):
            dsc.f += (traject[i] - throttle)**4 + \
                (traject[i] - throttle)**2 + gears * 10

        # swcost = dsc.sym("swcost", 1, 0, 1)
        # dsc.f += swcost
        # dsc.leq(gears - gears_prev, swcost)

        gears_prev2 = gears_prev
        gears_prev = gears
        throttle_prev = throttle

    def plot_gearbox(data: MinlpData, x_star):
        """Plot gearbox."""
        gear_idcs = dsc.get_indices("gear")
        gears = [
            float(x_star[gear_idx])
            for gear_idx in gear_idcs
        ]
        plt.scatter(list(range(len(gears))), gears)
        plt.show()

    problem = dsc.get_problem()
    problem.meta = MetaDataMpc()
    problem.meta.plot = plot_gearbox
    problem.meta.shift = plot_gearbox

    return problem, dsc.get_data()


def create_simple_gearbox(N=10, traject=None):
    """Very simple gearbox."""
    if traject is None:
        traject = [np.sin(i / 10) * 4 + 5 for i in range(N)]

    def cost_gear(j, setpoint):
        """Cost to run at setpoint for gear j."""
        return (10 * (setpoint - j*2 + 1))**2 + 5 * setpoint + j

    dsc = Description()

    traject = dsc.add_parameters("traject", N, traject)
    nr_gears = 5
    gears_prev2 = dsc.add_parameters("gears-1", nr_gears, [1, 0, 0, 0, 0])
    gears_prev = dsc.add_parameters("gears0", nr_gears, [1, 0, 0, 0, 0])
    for i in range(N):
        gears = create_gear_step(dsc, nr_gears, gears_prev, gears_prev2)
        for j in range(nr_gears):
            dsc.f += gears[j] * cost_gear(j, traject[i])

        swcost = dsc.sym("swcost", 1, 0, 1)
        dsc.f += swcost
        for j in range(1, nr_gears):
            dsc.leq(gears[j] - gears_prev[j-1], swcost)

        gears_prev2 = gears_prev
        gears_prev = gears

    def plot_gearbox(data: MinlpData, x_star):
        """Plot gearbox."""
        gear_idcs = dsc.get_indices("gears")
        gears = [
            float(sum([x_star[k] * (1 + i) for i, k in enumerate(gear_idx)]))
            for gear_idx in gear_idcs
        ]
        plt.scatter(list(range(len(gears))), gears)
        plt.show()

    problem = dsc.get_problem()
    problem.meta = MetaDataMpc()
    problem.meta.plot = plot_gearbox
    problem.meta.shift = plot_gearbox

    return problem, dsc.get_data()


def create_gearbox(N=10, gearbox_type="gasoline", dt=1.0, switch_cost=False) -> Union[MinlpProblem, MinlpData]:
    """Create a gearbox model."""
    N = int(N)
    cycle, dcycle = get_cycle_data(dt, N)

    # ------------------------------------------------
    # Model
    # ------------------------------------------------
    base_mass = 800
    wheelR = 0.285
    drag_area = 2
    air_density = 1.225
    drag_coeff = .35

    # temporary variables
    w = GlobalSettings.CASADI_VAR.sym('w')
    P = GlobalSettings.CASADI_VAR.sym('P')
    Prat = GlobalSettings.CASADI_VAR.sym('Prat')
    Erat = GlobalSettings.CASADI_VAR.sym('Erat')
    SoC = GlobalSettings.CASADI_VAR.sym('SoC')

    # ICE
    ICE_wrat = 6000 * np.pi / 30
    ICE_data = insight_50kw_power_jc()
    ICE_dFin = ca.Function('ICE_dFin', [Prat, w, P], [
                           ICE_data['dFin'](ICE_wrat, 1000 * Prat, w, 1000 * P)])
    ICE_Fstart = ca.Function('ICE_Fstart', [Prat], [
        ICE_data['Fstart'](1000 * Prat)])
    ICE_Pmax = ca.Function('ICE_Pmax', [Prat, w], [
                           ICE_data['Pmax'](ICE_wrat, 1000 * Prat, w) * 1e-3])
    ICE_mass = ca.Function('ICE_mass', [Prat], [
                           ICE_data['mass'](1000 * Prat)])
    ICE_minw = max(ICE_data['minw'](ICE_wrat), 1000 * np.pi / 30)

    # EM
    EM_wrat = 10000 * np.pi / 30
    EM_data = advisor_em_pwr_sc()
    EM_Pin = ca.Function('EM_Pin', [Prat, w, P], [
                         EM_data['Pin'](EM_wrat, 1000 * Prat, w, 1000 * P) * 1e-3])
    # EM_Pmax = ca.Function('EM_Pmax', [Prat, w], [
    #                       EM_data['Pmax'](EM_wrat, 1000 * Prat, w) * 1e-3])
    EM_mass = ca.Function('EM_mass', [Prat], [EM_data['mass'](1000 * Prat)])

    # battery
    BT_data = battery_hu()
    SoC_max = BT_data['SoC_max']
    SoC_min = BT_data['SoC_min']
    BT_dSoC = ca.Function('BT_dSoC', [Erat, SoC, P], [
                          BT_data['dSoC'](Erat * 3.6e6, SoC, 1000 * P)])
    BT_Pout = ca.Function('BT_Pout', [Erat, SoC, P], [
                          BT_data['Pout'](Erat * 3.6e6, SoC, 1000 * P) * 1e-3])
    BT_Pmax = ca.Function('BT_Pmax', [Erat, SoC], [
                          BT_data['Pmax'](Erat * 3.6e6, SoC) * 1e-3])
    BT_Pmin = ca.Function('BT_Pmin', [Erat, SoC], [
                          BT_data['Pmin'](Erat * 3.6e6, SoC) * 1e-3])
    BT_mass = ca.Function('BT_mass', [Erat], [BT_data['mass'](Erat * 3.6e6)])

    # ------------------------------------------------
    # Parameters initial values
    # ------------------------------------------------

    # speeds to guess ratios
    reference_speeds = [15, 30, 55, 85, 115]
    # reference_speeds = [30, 55, 85, 115, 155]

    # for diesel:
    referece_ICEspeed_diesel = 2000
    R_diesel = [(referece_ICEspeed_diesel * np.pi / 30)
                / ((reference_speeds[k] / 3.6) / wheelR) for k in range(len(reference_speeds))]

    # for gasoline
    referece_ICEspeed_gasoline = 2500
    R_gasoline = [(referece_ICEspeed_gasoline * np.pi / 30)
                  / ((reference_speeds[k] / 3.6) / wheelR) for k in range(len(reference_speeds))]

    # # from advisor default parallel
    # Rem_advisor = 0.99 * (EM_wrat / ICE_wrat)
    # R_advisor = [13.33, 7.57, 5.01, 3.77, 2]

    # ------------------------------------------------
    # Problem definition
    # ------------------------------------------------

    # fixed parameters
    if gearbox_type == "diesel":
        R = R_diesel
    else:
        R = R_gasoline

    Rem = (3000 * np.pi / 30) / ((60 / 3.6) / wheelR)
    ICE_Prat = 55
    EM_Prat = 25
    BT_Erat = 1.024  # kWh
    TNK_Frat = 30000
    TNK_Finit = TNK_Frat / 2
    # SoC_opt = SoC_min + 0.75 * (SoC_max - SoC_min)
    SoC_start = SoC_min + 0.1 * (SoC_max - SoC_min)

    dsc = Description()
    SoC = dsc.sym('SoC', N + 1, float(SoC_min),
                  float(SoC_max), float(SoC_start))
    dsc.eq(float(SoC_start), SoC[0])
    Fuel = dsc.sym('Fuel', N + 1, 0, float(TNK_Frat), float(TNK_Finit))
    dsc.add_g(float(TNK_Finit), Fuel[0], float(TNK_Finit))

    # # Switch on/off
    # # Create gearbox
    cost_switch = 0.4 * float(ICE_Fstart(ICE_Prat))

    # cost_switch = 0  # 0.4 * ICE_Fstart(ICE_Prat)
    # # float(0.4 * 45.6 * ICE_Fstart(ICE_Prat))

    # Continous variables
    dFuel_max = 20.0
    dFuel = dsc.sym('dFuel', N, 0, dFuel_max, 0)
    Pb = dsc.sym('Pb', N,
                 float(BT_Pmin(BT_Erat, 1.0)), float(BT_Pmax(BT_Erat, 1.0)), 0)
    Pice = dsc.sym('Pice', N, 0, ICE_Prat, 0)
    Pem = dsc.sym('Pem', N, -EM_Prat, EM_Prat, 0)

    nr_gears = len(R) + 1
    gears_init = [0] * nr_gears
    gears_init[0] = 1
    gearbox_prev = dsc.add_parameters("gears0", nr_gears, gears_init)
    gearbox_prev2 = dsc.add_parameters("gears-1", nr_gears, gears_init)

    # precalculations
    mass = base_mass + ICE_mass(ICE_Prat) + EM_mass(EM_Prat) + BT_mass(BT_Erat)

    for i in range(N):
        gearbox = create_gear_step(dsc, nr_gears, gearbox_prev, gearbox_prev2)
        if switch_cost:
            dsc.f += ca.sum1((gearbox_prev - gearbox)**2 * cost_switch / 2)
        gearbox_prev2 = gearbox_prev
        gearbox_prev = gearbox

        # gearbox = create_gear_step(dsc, nr_gears, gears_prev, gears_prev, allow_stop=True)
        off = gearbox[0]
        gears = gearbox[1:]

        Freq = mass * dcycle[i] + .5 * air_density * \
            drag_coeff * drag_area * cycle[i]**2
        # Treq = Freq * wheelR
        Paux = 0.3  # accessory power load kW
        Preq = Freq * cycle[i] / 1000
        wem = Rem * cycle[i] / wheelR
        wice = sum([gears[k] * R[k] * cycle[i]
                    / wheelR for k in range(len(R))])

        # # ODE
        dsc.eq(BT_dSoC(BT_Erat, SoC[i], Pb[i]) + (SoC[i] - SoC[i + 1]) / dt, 0)
        dsc.eq((Fuel[i] - Fuel[i + 1]) / dt - dFuel[i], 0)
        # max_fuel * (1- off) > dFuel)
        dsc.leq(dFuel[i] - (1 - off) * dFuel_max, 0)

        # EQ
        # Power required = P_em + P_ice
        dsc.leq(Preq - Pem[i] - Pice[i], 0)

        # wice < ICE_wrat
        dsc.leq(wice, ICE_wrat * 0.5)
        dsc.leq(ICE_minw * (1 - off) - wice, 0)
        dsc.leq(Pice[i] - ICE_Prat * (1 - off), 0)
        for k in range(ICE_Pmax(0, 0).numel()):
            dsc.leq(Pice[i] - ICE_Pmax(ICE_Prat, wice)[k], 0)

        for k in range(ICE_dFin(0, 0, 0).numel()):
            dsc.leq(
                (
                    ICE_dFin(ICE_Prat, wice, Pice[i])[k]
                    - off * ICE_dFin(ICE_Prat, 0, 0)[k] - dFuel[i]
                ),
                0
            )

        # Following equations are OK!
        dsc.leq(Pb[i] - BT_Pmax(BT_Erat, SoC[i]), 0)
        dsc.leq(0, Pb[i] - BT_Pmin(BT_Erat, SoC[i]))
        for k in range(EM_Pin(0, 0, 0).numel()):
            dsc.leq(
                (
                    EM_Pin(EM_Prat, wem, Pem[i])[k] + Paux
                    - BT_Pout(BT_Erat, SoC[i], Pb[i])
                ),
                0
            )
        dsc.add_g(-EM_Prat, Pem[i], EM_Prat)

        # Pb = battery consumption (Power battery)
        dsc.f += (0.4 * 45.6 * dFuel[i] + Pb[i])

    dsc.f += 200 * (
        - ca.log((SoC[-1] - SoC_min) / (SoC_max - SoC_min))
        - ca.log((SoC_max - SoC[-1]) / (SoC_max - SoC_min))
        + 2 * ca.log(.5)
    )

    def plot_gearbox(data: MinlpData, x_star):
        """Plot gearbox."""
        # from camino.utils import latexify
        # latexify()

        t = np.arange(0, (N + 1) * dt, dt)
        nr_plots = 6
        fig, axs = plt.subplots(nr_plots, 1, figsize=(6, 8))

        axs[0].plot(t, cycle, label="Traject")
        plot_names = [['SoC'], ['Fuel'], ['dFuel'], ['Pb', 'Pice', 'Pem']]
        for i, ax in enumerate(axs[1:-1]):
            for name in plot_names[i]:
                y_data = x_star[dsc.get_indices(name)[0]]
                ax.plot(t[:y_data.numel()], y_data, label=name)

        gear_idcs = dsc.get_indices("gears")
        gears = [
            float(sum([x_star[k] * i for i, k in enumerate(gear_idx)]))
            for gear_idx in gear_idcs
        ]

        axs[-1].scatter(t[:-1], gears, label='gear shift')
        for i in range(nr_gears):
            axs[-1].plot([0, N*dt], [i, i], ':', color='tab:grey')
        axs[-1].set_ylim([0, nr_gears])

        for ax in axs:
            ax.set_xlim(0, (N + 1) * dt)
            ax.legend(loc='upper right', fontsize='small').set_draggable(True)

        fig.subplots_adjust(hspace=0.5)
        plt.show()

    dsc.w = [ca.vcat(dsc.w)]
    dsc.p = [ca.vcat(dsc.p)]
    dsc.f = CachedFunction(f"gearbox{N}", dsc.create_f)(dsc.w[0], dsc.p[0])
    dsc.g = [CachedFunction(f"gearbox{N}", dsc.create_g)(dsc.w[0], dsc.p[0])]

    problem = dsc.get_problem()
    problem.meta = MetaDataMpc()
    problem.meta.plot = plot_gearbox
    problem.meta.shift = plot_gearbox

    return problem, dsc.get_data()
