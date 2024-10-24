# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Statistics class."""

from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional
from camino.utils import toc
from camino.utils.conversion import to_0d
from camino.settings import GlobalSettings
from camino.utils.data import save_pickle
from camino.data import MinlpData


@dataclass()
class Stats:
    """
    Collecting statistics


    There are some shared properties between the algorithms:
        iter_nr: Number of iterations performed
        lb: Lower bound value
        ub: Upper bound value
        relaxed: Relaxed solution
    """

    mode: str
    problem_name: str
    nr_reset: int = 0
    datetime: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    )
    data: Dict[str, float] = field(default_factory=lambda: {})
    out_dir: str = GlobalSettings.OUT_DIR
    _full_stats_to_pickle: List = field(
        default_factory=lambda: []
    )
    _relaxed: Optional[MinlpData] = None

    def __getitem__(self, key):
        """Get attribute."""
        if key not in self.data:
            return 0
        return self.data[key]

    def __setitem__(self, key, value):
        """Set item."""
        self.data[key] = value

    @property
    def relaxed_solution(self):
        """Relaxed value."""
        return deepcopy(self._relaxed)

    @relaxed_solution.setter
    def relaxed_solution(self, value):
        """Set relaxed solution value."""
        self._relaxed = deepcopy(value)

    def print(self):
        """Print statistics."""
        print("Statistics")
        for k, v in sorted(self.data.items()):
            if k not in [
                "iterate_data", "solutions_all", "solved_all", "solutions", "mip_solutions_all",
                "mip_solved_all"
            ]:
                if k == 'sol_x':
                    if len(self.data['sol_x']) < 10:
                        print(f"\t{k}: {v}")
                    else:
                        print(f"\t{k}: {v[:5]}, ...")
                else:
                    print(f"\t{k}: {v}")

    def reset(self):
        """Reset the statistics."""
        self.data = {}
        self._full_stats_to_pickle = {}
        self.nr_reset += 1

    def save(self, dest=None):
        """Save statistics."""
        time = toc()  # TODO add time
        if dest is None:
            dest = os.path.join(
                self.out_dir,
                f'{self.datetime}_{self.mode}_{self.problem_name}_{self.nr_reset}.pkl'
            )
        print(f"Saving to {dest}")
        data = self.data.copy()
        to_pickle = []
        general_stats = {}
        for key, value in data.items():
            if key not in ["solutions_all", "solved_all", "mip_solutions_all", "mip_solved_all"]:
                general_stats[key] = value
        general_stats["time"] = time
        try:
            for idx, (elm, mip_elm) in enumerate(zip(data["solutions_all"], data["mip_solutions_all"])):
                tmp_dict = {}
                tmp_dict.update(general_stats)
                tmp_dict["sol_pool_idx"] = idx
                tmp_dict["sol_pool_success"] = data["solved_all"][idx]
                tmp_dict["sol_pool_objective"] = float(elm["f"])
                tmp_dict["sol_pool_x"] = to_0d(elm["x"])
                tmp_dict["mip_sol_pool_idx"] = idx
                tmp_dict["mip_sol_pool_success"] = data["mip_solved_all"][idx]
                tmp_dict["mip_sol_pool_objective"] = float(mip_elm["f"])
                tmp_dict["mip_sol_pool_x"] = to_0d(mip_elm["x"])
                to_pickle.append(tmp_dict)
        except Exception:
            tmp_dict = {}
            tmp_dict.update(general_stats)
            to_pickle.append(tmp_dict)

        self._full_stats_to_pickle += to_pickle
        save_pickle(self._full_stats_to_pickle, dest)
