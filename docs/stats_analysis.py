# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import pandas as pd
from camino.settings import GlobalSettings
import pickle

filename = "2024-05-17_12:44:55_fp_particle_generic.pkl"

file_path = os.path.join(GlobalSettings.OUT_DIR, filename)
with open(file_path, 'rb') as f:
    stats = pickle.load(f)

stats_df = pd.DataFrame(stats)
print(stats_df.columns)
print(stats_df.head())
