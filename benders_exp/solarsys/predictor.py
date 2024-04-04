# Adrian Buerger, 2022

import numpy as np
import casadi as ca

from benders_exp.solarsys.system import System

import logging

logger = logging.getLogger(__name__)

class Predictor(System):

    @property
    def time_grid(self):
        return self._timing.time_grid


    @property
    def x_data(self):
        return self._previous_solver.x_data


    @property
    def x_hat(self):
        try:
            return self._x_hat
        except AttributeError:
            msg = "Estimated states not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def u_data(self):
        return self._previous_solver.u_data


    @property
    def b_data(self):
        return self._previous_solver.b_data


    @property
    def c_data(self):
        return self._ambient.c_data


    @property
    def s_ac_lb_data(self):
        return self._previous_solver.s_ac_lb_data


    @property
    def s_ac_ub_data(self):
        return self._previous_solver.s_ac_ub_data


    @property
    def s_x_data(self):
        return self._previous_solver.s_x_data


    @property
    def s_ppsc_fpsc_data(self):
        return self._previous_solver.s_ppsc_fpsc_data


    @property
    def s_ppsc_vtsc_data(self):
        return self._previous_solver.s_ppsc_vtsc_data


    @property
    def solver_name(self):
        return self._solver_name


    def _setup_timing(self, timing):

        self._timing = timing


    def _setup_state(self, state):

        self._state = state


    def _setup_ambient(self, ambient):

        self._ambient = ambient


    def _set_previous_solver(self, previous_solver):

        self._previous_solver = previous_solver


    def _setup_solver_name(self, solver_name):

        self._solver_name = solver_name


    def __init__(self, timing, state, ambient, previous_solver, solver_name):

        super().__init__()

        self._setup_timing(timing=timing)
        self._setup_state(state=state)
        self._setup_ambient(ambient=ambient)

        self._setup_solver_name(solver_name=solver_name)
        self._set_previous_solver(previous_solver=previous_solver)

        self.load_ode_object()
        self._setup_simulator()
        self._remove_unserializable_attributes()


    def solve(self, n_steps = 1):

        x_hat = self._state.x_hat

        for k in range(n_steps):

            pos_grid = self._timing.grid_position_cursor + k

            x_hat = self._integrator(x0=x_hat, p=ca.veccat( \
                    self._timing.time_steps[pos_grid], self.c_data[pos_grid,:], \
                    self.u_data[pos_grid,:], self.b_data[pos_grid,:]))["xf"]

        self._x_hat = x_hat

