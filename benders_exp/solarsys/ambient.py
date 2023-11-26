# Adrian Buerger, 2022

import os
import numpy as np
import pandas as pd
import datetime as dt
from abc import ABCMeta, abstractmethod

from benders_exp.defines import _DATA_FOLDER
from benders_exp.solarsys.system import System

import logging

logger = logging.getLogger(__name__)


class Ambient(System):

    @property
    def time_grid(self):

        return self._timing.time_grid

    @property
    def c_data(self):

        try:

            return self._df_ambient[list(self.c_index.keys())].values

        except AttributeError:

            msg = "Ambient conditions (parametric inputs) not available yet, call update() first."

            logging.error(msg)
            raise RuntimeError(msg)

    def _setup_timing(self, timing):

        self._timing = timing

    def __init__(self, timing):

        super().__init__()

        self._setup_timing(timing=timing)

    def _load_forecast_data(self, method="csv"):

        self._df_ambient = pd.read_csv(
            os.path.join(_DATA_FOLDER, "forecast.csv"), index_col=0
        )
        self._df_ambient.index = pd.DatetimeIndex(self._df_ambient.index)

    def _compute_load_forecast(self):

        self._df_ambient["Qdot_c"] = 9e2 * (self._df_ambient["T_amb"] - 18.0)
        self._df_ambient.loc[self._df_ambient["Qdot_c"] <= 0.0, ["Qdot_c"]] = 0.0
        self._df_ambient["Qdot_c"] += 1e3

    def _set_ambient_time_range(self):

        self.timestamp_ambient_start = self._timing.time_points[0]
        self.timestamp_ambient_end = self.timestamp_ambient_start + dt.timedelta(
            seconds=self._timing.t_f
        )

    def _adjust_ambient_to_mpc_grid(self):

        self._df_ambient = self._df_ambient.reindex(
            self._df_ambient.index.union(
                pd.date_range(
                    start=self.timestamp_ambient_start,
                    end=self.timestamp_ambient_end,
                    freq=f"{self._timing.dt_day}s",
                )
            )
        )

        self._df_ambient.interpolate(method="linear", inplace=True)

        self._df_ambient = self._df_ambient.reindex(
            pd.date_range(
                start=self.timestamp_ambient_start,
                end=self.timestamp_ambient_end,
                freq=f"{self._timing.dt_day}s",
            )
        )

    def _convert_time_points_to_time_grid(self):

        self._df_ambient.index = np.round(
            (self._df_ambient.index - self._df_ambient.index[0]).total_seconds()
        )

        self._df_ambient = self._df_ambient.reindex(self._timing.time_grid)

    def _fill_remaining_nan_values(self):

        self._df_ambient.interpolate(method="linear", inplace=True)
        self._df_ambient.bfill(inplace=True)

    def _manual_tuning_of_forecasts(self):

        self._df_ambient[["I_vtsc", "I_fpsc"]] *= 0.9
        self._df_ambient["T_amb"] += 3

    def update(self):

        self._load_forecast_data()
        self._compute_load_forecast()

        self._set_ambient_time_range()
        self._adjust_ambient_to_mpc_grid()
        self._convert_time_points_to_time_grid()

        self._fill_remaining_nan_values()
        self._manual_tuning_of_forecasts()


if __name__ == "__main__":

    import time
    import datetime as dt
    import matplotlib.pyplot as plt

    from timing import TimingMPC

    startup_time = dt.datetime.fromisoformat("2010-08-19 06:00:00+02:00")

    timing = TimingMPC(startup_time=startup_time)
    ambient = Ambient(timing)

    ambient.update()

    plt.figure()
    plt.plot(range(ambient.c_data.shape[0]), ambient.c_data[:, 0])
    plt.plot(range(ambient.c_data.shape[0]), ambient.c_data[:, 1])
    plt.plot(range(ambient.c_data.shape[0]), ambient.c_data[:, 2])
    plt.plot(range(ambient.c_data.shape[0]), ambient.c_data[:, 3])
    plt.plot(range(ambient.c_data.shape[0]), ambient.c_data[:, 4])
    plt.plot(range(ambient.c_data.shape[0]), ambient.c_data[:, 5])
    plt.yscale("log")

    # plt.figure()
    # ax1 = plt.gca()

    # for c in ["I_vtsc", "I_fpsc"]:
    #     ax1.plot(ambient.time_grid, ambient.c_data[:, ambient.c_index[c]], label=c)

    # ax1.legend(loc="upper left")

    # ax2 = ax1.twinx()
    # ax2.plot(
    #     ambient.time_grid, ambient.c_data[:, ambient.c_index["T_amb"]], label="T_amb"
    # )
    # ax2.legend(loc="upper right")

    plt.show()
