# Adrian Buerger, 2022

import numpy as np
import pandas as pd
import datetime as dt
import time
import pytz

from system import System

from abc import ABCMeta, abstractmethod

import logging

logger = logging.getLogger(__name__)

class TimingBaseClass(System, metaclass = ABCMeta):


    @property
    def current_time(self):
        return self._current_time


    @property
    def time_grid(self):
        return np.round((self.time_points - \
            self.time_points[0]).total_seconds())


    @property
    def time_steps(self):
        return (self.time_grid[1:] - self.time_grid[:-1])


    @property
    def time_until_next_time_point(self):
        return (self.time_points[self.grid_position_cursor] \
            - dt.datetime.now()).total_seconds()


    @property
    def time_of_day(self):
        return self._time_of_day


    @property
    def remaining_min_up_time(self):
        return self._remaining_min_up_time


    @property
    def b_bin_locked(self):
        return self._b_bin_locked


    def _define_startup_time(self, startup_time):

        self._startup_time = startup_time.astimezone(pytz.utc)


    def _update_current_time(self, current_time):

        self._current_time = current_time.astimezone(pytz.utc)


    def _set_initial_time_grid_point(self, initial_time_grid_point):

        self._initial_time_grid_point = initial_time_grid_point


    def __init__(self, startup_time: dt.datetime):

        super().__init__()

        self._define_startup_time(startup_time=startup_time)
        self._setup_grid_dimensions()
        self._update_current_time(current_time=startup_time)
        self._set_initial_time_grid_point(initial_time_grid_point=self._current_time)


    def _sleep_until_given_time_point(self, process_name, time_grid_position, current_time):

        self._update_current_time(current_time)

        sleep_duration = (self.time_points[time_grid_position] - \
            self._current_time).total_seconds()

        logger.info(process_name + " waits for " + str(round(sleep_duration, 2)) + " s ...")

        try:

            time.sleep(sleep_duration)

        except:

            logger.warning(process_name + " is " + str(round(sleep_duration, 2)) \
                + " s behind schedule.")


    def sleep_until_time_grid_point(self, process_name: str, \
        time_grid_position: int, current_time: dt.datetime):

        self._sleep_until_given_time_point( \
            process_name=process_name, time_grid_position=time_grid_position, 
            current_time=current_time)


class TimingMPC(TimingBaseClass):

    @property
    def next_long_term_time_point(self):

        return self._next_long_term_time_point


    @property
    def next_short_term_time_point(self):

        return self._next_short_term_time_point


    @property
    def mpc_iteration_count(self):

        return self._mpc_iteration_count


    @property
    def grid_position_cursor(self):

        return self._grid_position_cursor


    def _setup_mpc_iteration_count(self):

        self._mpc_iteration_count = 0


    def _reset_grid_position_cursor(self):

        self._grid_position_cursor = 0


    def increment_grid_position_cursor(self, n_steps = 1):

        self._grid_position_cursor += n_steps


    def _define_earliest_sunset_and_latest_sunrise(self):

        '''
        Depending on the location, these times need to be adapted
        '''

        self._earliest_sunrise_time = (4,30)
        self._latest_sunset_time = (21,30)


    def _get_time_of_day(self):

        initial_time = (self._initial_time_grid_point.hour, self._initial_time_grid_point.minute)

        self._time_of_day = "night"

        self._earliest_sunrise = self._initial_time_grid_point.replace( \
            hour = self._earliest_sunrise_time[0], \
            minute = self._earliest_sunrise_time[1]) + dt.timedelta(days = 1)
        self._latest_sunset = self._initial_time_grid_point.replace( \
            hour = self._latest_sunset_time[0], \
            minute = self._latest_sunset_time[1]) + dt.timedelta(days = 1)


        if initial_time < (self._latest_sunset.hour, self._latest_sunset.minute):

            self._time_of_day = "day"
            self._latest_sunset += dt.timedelta(days = -1)

        if initial_time < (self._earliest_sunrise.hour, self._earliest_sunrise.minute):

            self._time_of_day = "night"
            self._earliest_sunrise += dt.timedelta(days = -1)


    def _setup_grid_dimensions(self):

        '''
        The MPC timegrid is set up from time steps of size dt_day for day times,
        and dt_night for night times; the first upcoming discrete interval is
        further divided into N_short_term smaller steps
        '''

        self.dt_day = 900
        self.dt_night = 2 * self.dt_day

        self.N_day = 4 * 4
        self.N_night = 4 * 2

        self.N_short_term = 2

        self.t_f = self.N_day * self.dt_day + self.N_night * self.dt_night

        self.N = self.N_day + self.N_night + (self.N_short_term - 1)


    def _setup_time_grid(self):

        if self.dt_night % self.dt_day:

            raise NotImplementedError("In the following, we assume that dt_night is a multiple of dt_day.")

        if self.dt_day % self.N_short_term:

            raise NotImplementedError("In the following, we assume that dt_day is a multiple of N_short_term.")

        if self.dt_night % self.N_short_term:

            raise NotImplementedError("In the following, we assume that dt_day is a multiple of N_short_term.")

        # For time horizons different from 24 h, the following needs to be adapted

        self.time_points = pd.DatetimeIndex([self._initial_time_grid_point])

        if self._time_of_day == "night":

            self.time_points = self.time_points.append(pd.date_range(self.time_points[-1], \
                self._earliest_sunrise, freq = str(self.dt_night)+"s")[1:])

            self.time_points = self.time_points.append(pd.date_range(self.time_points[-1], \
                self._latest_sunset, freq = str(self.dt_day)+"s")[1:])

            self.time_points = self.time_points.append(pd.date_range(self.time_points[-1], \
                self._initial_time_grid_point + dt.timedelta(days = 1), \
                freq = str(self.dt_night)+"s")[1:])

        elif self._time_of_day == "day":

            self.time_points = self.time_points.append(pd.date_range(self.time_points[-1], \
                self._latest_sunset, freq = str(self.dt_day)+"s")[1:])

            self.time_points = self.time_points.append(pd.date_range(self.time_points[-1], \
                self._earliest_sunrise, freq = str(self.dt_night)+"s")[1:])

            self.time_points = self.time_points.append(pd.date_range(self.time_points[-1], \
                self._initial_time_grid_point + dt.timedelta(days = 1), \
                freq = str(self.dt_day)+"s")[1:])

        self.time_points = pd.date_range(self.time_points[0], self.time_points[1], \
            periods = self.N_short_term+1).append(self.time_points[2:])


    def _update_next_long_term_time_point(self):

        self._next_long_term_time_point = self.time_points[self.N_short_term]


    def _update_next_short_term_time_point(self):

        self._next_short_term_time_point = self.time_points[ \
            self.short_term_grid_position_cursor]


    def _setup_min_up_time_structs(self):

        self._remaining_min_up_time = 0
        self._b_bin_locked = np.zeros((1, self.nb))
        self._b_bin_prev = np.zeros((1, self.nb))


    def __init__(self, startup_time):

        super().__init__(startup_time)

        self._setup_mpc_iteration_count()
        self._reset_grid_position_cursor()
        self._define_earliest_sunset_and_latest_sunrise()
        self._get_time_of_day()
        self._setup_time_grid()
        self._setup_min_up_time_structs()


    def sleep_until_grid_position_cursor_time_grid_point(self, process_name):

        self._sleep_until_given_time_point( \
            process_name = process_name, \
            time_grid_position = self.grid_position_cursor)


    def shift_time_grid(self):

        self._reset_grid_position_cursor()
        self._set_initial_time_grid_point( \
            initial_time_grid_point = self.time_points[self.N_short_term])
        self._get_time_of_day()
        self._setup_time_grid()


    def increment_mpc_iteration_count(self):

        self._mpc_iteration_count += 1


    def define_initial_switch_positions(self, solver):

        self._b_bin_prev = solver.b_data[self.N_short_term-1,:]
