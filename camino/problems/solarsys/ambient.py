"""
Ambient conditions model.

Adrian Buerger, 2022
Adapted by Wim Van Roy and Andrea Ghezzi, 2023
"""

from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from camino.settings import GlobalSettings
import logging


logger = logging.getLogger(__name__)


@dataclass
class Timing:
    dt_day = 900
    earliest_sunrise_time_UTC = (4, 30)
    latest_sunset_time_UTC = (21, 30)

    def __post_init__(self):
        self.dt_night = 2 * Timing.dt_day

        self.N_day = 17 * 4
        self.N_night = 7 * 2
        self.N_short_term = 2

        self.t_f = self.N_day * self.dt_day + self.N_night * self.dt_night
        self.N = self.N_day + self.N_night + (self.N_short_term - 1)


class Ambient:
    """Ambient information."""

    def __init__(self, timing: Timing):
        """Create ambient information."""
        self.timing = timing
        self._load_forecast_data()
        self._fill_remaining_nan_values()
        self._compute_load_forecast()
        self._manual_tuning_of_forecasts()
        self._set_time_steps()

    def _load_forecast_data(self, method="csv"):
        """Load forecast data."""
        self._df_ambient = pd.read_csv(
            os.path.join(GlobalSettings.DATA_FOLDER, "forecast.csv"), index_col=0
        )
        self._df_ambient.index = pd.DatetimeIndex(self._df_ambient.index)
        self._time = [idx.timestamp() for idx in self._df_ambient.index]

    def _compute_load_forecast(self):
        """Compute load forecast."""
        self._df_ambient["Qdot_c"] = 9e2 * (self._df_ambient["T_amb"] - 18.0)
        self._df_ambient.loc[self._df_ambient["Qdot_c"]
                             <= 0.0, ["Qdot_c"]] = 0.0
        self._df_ambient["Qdot_c"] += 1e3

    def _fill_remaining_nan_values(self):
        self._df_ambient.interpolate(method="linear", inplace=True)
        self._df_ambient.bfill(inplace=True)

    def _manual_tuning_of_forecasts(self):
        self._df_ambient[["I_vtsc", "I_fpsc"]] *= 0.9
        self._df_ambient["T_amb"] += 3

    def get_t0(self):
        """Get start time."""
        return self._df_ambient.index[6]

    def _set_time_steps(self):
        # TODO make it general, based on csv file!
        self.time_steps = [timedelta(seconds=self.timing.dt_day) if idx < self.timing.N_day +
                           2 else timedelta(seconds=self.timing.dt_night) for idx in range(self.timing.N)]

    def interpolate(self, value: datetime):
        """Interpolate a value."""
        return {
            k: np.interp(value.timestamp(), self._time, self._df_ambient[k])
            for k in self._df_ambient.keys()
        }


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    timing = Timing()
    ambient = Ambient(timing=timing)

    interp_df = []
    interp_df.append(ambient._df_ambient.loc[ambient.get_t0()].to_dict())
    tmp = timedelta(0)
    for i in range(len(ambient.time_steps)):
        tmp += ambient.time_steps[i]
        interp_df.append(ambient.interpolate(ambient.get_t0() + tmp))
    interp_df = pd.DataFrame(interp_df)

    time_grid = [0]
    for i in range(timing.N):
        time_grid.append(time_grid[i] + ambient.time_steps[i].total_seconds())
    interp_df['time_grid'] = time_grid
    print(interp_df)

    plt.figure()
    plt.plot(range(ambient.timing.N+1), interp_df.iloc[:, 0])
    plt.plot(range(ambient.timing.N+1), interp_df.iloc[:, 1])
    plt.plot(range(ambient.timing.N+1), interp_df.iloc[:, 2])
    plt.plot(range(ambient.timing.N+1), interp_df.iloc[:, 3])
    plt.plot(range(ambient.timing.N+1), interp_df.iloc[:, 4])
    plt.plot(range(ambient.timing.N+1), interp_df.iloc[:, 5])
    plt.yscale("log")
    plt.show()
    # plt.figure()
    # ax1 = plt.gca()
    # ax1.plot(interp_df.time_grid, interp_df.I_vtsc)
    # ax1.plot(interp_df.time_grid, interp_df.I_fpsc)

    # # ax1.legend(loc="upper left")

    # ax2 = ax1.twinx()
    # ax2.plot(interp_df.time_grid, interp_df.T_amb)
    # ax2.legend(loc="upper right")

    # plt.show()
