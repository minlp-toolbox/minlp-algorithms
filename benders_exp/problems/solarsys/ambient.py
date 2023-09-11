"""
Ambient conditions model.

Adrian Buerger, 2022
Adapted by Wim Van Roy, 2023
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from benders_exp.defines import _DATA_FOLDER
import logging


logger = logging.getLogger(__name__)


class Ambient:
    """Ambient information."""

    def __init__(self):
        """Create ambient information."""
        self._load_forecast_data()
        self._fill_remaining_nan_values()
        self._compute_load_forecast()
        self._manual_tuning_of_forecasts()

    def _load_forecast_data(self, method="csv"):
        """Load forecast data."""
        self._df_ambient = pd.read_csv(
            os.path.join(_DATA_FOLDER, "forecast.csv"), index_col=0
        )
        self._df_ambient.index = pd.DatetimeIndex(self._df_ambient.index)
        self._time = [idx.timestamp() for idx in self._df_ambient.index]

    def _compute_load_forecast(self):
        """Compute load forecast."""
        self._df_ambient["Qdot_c"] = 9e2 * (self._df_ambient["T_amb"] - 18.0)
        self._df_ambient.loc[self._df_ambient["Qdot_c"] <= 0.0, ["Qdot_c"]] = 0.0
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

    def interpolate(self, value: datetime):
        """Interpolate a value."""
        return {
            k: np.interp(value.timestamp(), self._time, self._df_ambient[k])
            for k in self._df_ambient.keys()
        }


if __name__ == "__main__":
    dt = timedelta(minutes=30)
    ambient = Ambient()
    for i in range(10):
        print(ambient.interpolate(ambient.get_t0() + i * dt))
