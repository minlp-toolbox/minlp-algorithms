# Adrian Buerger, 2022

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

LOGFILE_LOCATION = "/tmp/voronoi-logs"

LOGFILE_REL = "nlpsolver_rel_voronoi.log"
LOGFILE_BIN_PREFIX = "nlpsolver_bin_miqp_voronoi_iter_"
LOGFILE_BINAPPROX_PREFIX = "binapprox_miqp_voronoi_iter_"

PLOTFILE_NAME = "/tmp/voronoi.png"


def get_objective_from_ipopt_log(folder, filename):

    OBJECTIVE_KEYWORD = "Objective"
    OPTIMALITY_KEYWORD = "Optimal Solution Found."
    POS_SCALED_OBJECTIVE = 2

    logfile_path = os.path.join(folder, filename)

    with open(logfile_path) as f:
        s = f.readlines()

    obj_line = [e for e in s if OBJECTIVE_KEYWORD in e][0]
    obj_value = float(obj_line.split("   ")[POS_SCALED_OBJECTIVE])

    solution_optimal = bool([e for e in s if OPTIMALITY_KEYWORD in e])

    if not solution_optimal:
        print(f"Solution from file {filename} not optimal, check that")

    return obj_value


def get_runtime_from_ipopt_log(folder, filename):

    RUNTIME_KEYWORD = "Total CPU secs "

    runtime = 0

    logfile_path = os.path.join(folder, filename)

    with open(logfile_path) as f:
        s = f.readlines()

    runtime_lines = [e for e in s if RUNTIME_KEYWORD in e]

    for l in runtime_lines:
        runtime += float(l.split("=")[1])

    return runtime


def get_runtime_from_gurobi_log(folder, filename):

    LINE_KEYWORD = "Explored"

    logfile_path = os.path.join(folder, filename)

    with open(logfile_path) as f:
        s = f.readlines()

    runtime_line = [e for e in s if LINE_KEYWORD in e][0]
    runtime = float(runtime_line.split(") in")[1].split(" seconds")[0])

    return runtime


obj_value_rel = get_objective_from_ipopt_log(LOGFILE_LOCATION, LOGFILE_REL)

obj_values_bin = []

runtimes_gurobi = []
runtimes_ipopt = []

n_log_files_bin = len(
    glob.glob(f"{os.path.join(LOGFILE_LOCATION, LOGFILE_BIN_PREFIX)}*.log")
)

for k in range(n_log_files_bin):

    obj_values_bin.append(
        get_objective_from_ipopt_log(LOGFILE_LOCATION, f"{LOGFILE_BIN_PREFIX}{k}.log")
    )
    runtimes_gurobi.append(
        get_runtime_from_gurobi_log(
            LOGFILE_LOCATION, f"{LOGFILE_BINAPPROX_PREFIX}{k}.log"
        )
    )
    runtimes_ipopt.append(
        get_runtime_from_ipopt_log(LOGFILE_LOCATION, f"{LOGFILE_BIN_PREFIX}{k}.log")
    )

obj_values_bin = np.asarray(obj_values_bin)
runtimes_gurobi = np.asarray(runtimes_gurobi)
runtimes_ipopt = np.asarray(runtimes_ipopt)

iter_min_obj = np.argmin(obj_values_bin)
min_obj = obj_values_bin[iter_min_obj]

print(f"Minimum objective value {min_obj} in iteration {iter_min_obj}")

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax[0].plot(
    [0, obj_values_bin.size - 1],
    [obj_value_rel, obj_value_rel],
    color="C1",
    linestyle="dashed",
    label="NLP_rel",
)
ax[0].plot(obj_values_bin, color="C0", label="NLP_bin")
ax[0].scatter(0, obj_values_bin[0], color="C0", marker="o", label="NLP_bin_init")
ax[0].scatter(iter_min_obj, min_obj, color="C0", marker="x", label="NLP_bin_min")
ax[0].legend(loc="upper center")
ax[0].set_ylabel("Ipopt objective value")

ax0a = ax[0].twinx()
ax0a.plot(
    obj_values_bin / obj_value_rel,
    linestyle="dotted",
    color="C7",
    label="NLP_bin/NLP_rel",
)
ax0a.legend(loc="upper right")
ax0a.set_yscale("log")
ax0a.set_ylabel("Relation NLP_bin/NLP_rel (log scale)")

ax[1].plot(runtimes_gurobi, color="C0", label="gurobi")
ax[1].plot(runtimes_ipopt, color="C1", label="ipopt")
ax[1].set_xlim(0, runtimes_ipopt.size - 1)
ax[1].set_yscale("log")
ax[1].grid(visible=True, which="both", axis="y", linestyle="dotted")
ax[1].legend(loc="best")
ax[1].set_ylabel("Runtime (s)")
ax[1].set_xlabel("Iteration")

for ax_k in ax:
    for pos in ["top", "right"]:
        ax_k.spines[pos].set_visible(False)

ax0a.spines["top"].set_visible(False)

plt.savefig(PLOTFILE_NAME, format="png", bbox_inches="tight")
