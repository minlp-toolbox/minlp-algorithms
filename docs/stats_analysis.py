import os
import pandas as pd
from minlp_algorithms.settings import GlobalSettings
import pickle

filename = "2024-05-17_12:44:55_fp_particle_generic.pkl"

file_path = os.path.join(GlobalSettings.OUT_DIR, filename)
with open(file_path, 'rb') as f:
    stats = pickle.load(f)

stats_df = pd.DataFrame(stats)
print(stats_df.columns)
print(stats_df.head())
