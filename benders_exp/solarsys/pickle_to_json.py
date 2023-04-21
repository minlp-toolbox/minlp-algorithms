import os
import copy
import pickle
import glob
import json
import numpy as np
from benders_exp.solarsys.defines import PICKLE_FOLDER

sol_pkl_list = glob.glob(os.path.join(PICKLE_FOLDER, "*.pickle"))

for sol_pkl_name in sol_pkl_list:

    with open(sol_pkl_name, "rb") as f:
        sol_pkl = pickle.load(f)

    data = {}

    for attr in [
        "x_data",
        "u_data",
        "b_data",
        "c_data",
        "s_ac_lb_data",
        "s_ac_ub_data",
        "s_x_data",
        "s_ppsc_data",
    ]:

        data[attr] = copy.deepcopy(getattr(sol_pkl, attr)).tolist()

    if "nlpsolver" in sol_pkl_name:
        attr = "nlp_objective_value"
        data[attr] = copy.deepcopy(getattr(sol_pkl, attr))
    if "binapprox" in sol_pkl_name:
        attr = "voronoi"
        obj = copy.deepcopy(getattr(sol_pkl, attr))
        if obj is not np.nan:
            if obj.A_v is not None:
                data["A_v"] = copy.deepcopy(getattr(obj, "A_v")).tolist()
                data["lb_v"] = copy.deepcopy(getattr(obj, "lb_v")).tolist()
                data["ub_v"] = copy.deepcopy(getattr(obj, "ub_v")).tolist()
        else:
            data["A_v"] = []
            data["lb_v"] = []
            data["ub_v"] = []

    post = {
        "solver_name": sol_pkl.solver_name,
        "solver_type": sol_pkl.solver_type,
        "time_points": [tp.isoformat() for tp in sol_pkl._timing.time_points],
        "grid_position_cursor": sol_pkl._timing.grid_position_cursor,
        "data": data,
    }
    try:
        wall_time = sol_pkl.solver_wall_time
    except:
        wall_time = -1.0
    post["solver_wall_time"] = wall_time

    json_path = sol_pkl_name.split(".pickle")[0] + ".json"
    with open(json_path, "w") as f:
        print(f"saving to {json_path}")
        json.dump(post, f)
