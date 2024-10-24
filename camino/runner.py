# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main running scripts."""

import casadi as ca
from camino.solver import MinlpSolver
from camino.problem import MetaDataOcp
from camino.settings import Settings
from camino.problems.overview import PROBLEMS
from camino.utils import make_bounded, logger
from camino.utils.data import write_json, read_json
from camino.utils.validate import check_solution


def batch_runner(algorithm, target, nl_files):
    """Run a batch of problems."""
    from os import makedirs, path
    from time import time

    def do_write(overview_target, start, i, mode_name, total_stats):
        time_now = time() - start
        total_time = time_now / (i + 1) * total_to_compute
        write_json({
            "time": time_now,
            "total": total_to_compute,
            "done": (i+1),
            "progress": (i+1) / total_to_compute,
            "time_remaining_est": total_time - time_now,
            "time_total_est": total_time,
            "mode": mode_name,
            "data": total_stats
        }, overview_target)

    overview_target = path.join(target, "overview.json")
    start = time()
    total_to_compute = len(nl_files)
    if path.exists(overview_target):
        data = read_json(overview_target)
        total_stats = data['data']
        mode_name = data["mode"]
        i_start = data['done']
        start -= data["time"]
        if total_stats[-1][0] != i_start:
            total_stats.append([
                i_start,
                nl_files[i_start],
                -ca.inf, "FAILED", "CRASH"
            ])
        do_write(overview_target, start, i_start, mode_name, total_stats)
        i_start += 1
    else:
        makedirs(target, exist_ok=True)
        total_stats = [["id", "path", "obj",
                        "load_time", "calctime",
                        "solvertime", "iter_nr", "nr_int"]]
        i_start = 0

    for i in range(i_start, len(nl_files)):
        nl_file = nl_files[i]
        try:
            stats, data = runner(mode_name, "nl_file", None, [nl_file])
            stats["x_star"] = data.x_sol
            stats["f_star"] = data.obj_val
            stats.print()
            stats.save(path.join(target, f"stats_{i}.pkl"))
        except Exception as e:
            print(f"{e}")
            total_stats.append(
                [i, nl_file, -ca.inf, "FAILED", f"{e}"]
            )
        do_write(overview_target, start, i, mode_name, total_stats)


def runner(solver_name, problem_name, target_file, args):
    """General runner."""
    if problem_name not in PROBLEMS:
        raise Exception(f"No {problem_name=}, available: {PROBLEMS.keys()}")

    logger.info(f"Load problem {problem_name}")
    if args is None:
        args = []

    output = PROBLEMS[problem_name](*args)
    if len(output) == 2:
        problem, data = output
        settings = Settings()
        logger.info("Using default settings")
    else:
        logger.info("Using custom settings")
        problem, data, settings = output

    if problem == "orig":
        new_inf = 1e5
    else:
        new_inf = 1e3
    make_bounded(problem, data, new_inf=new_inf)

    if len(problem.idx_x_integer) == 0:
        solver_name = "relaxed"

    solver = MinlpSolver(
        solver_name, problem, data, settings=settings, problem_name=problem_name
    )

    logger.info(f"Start mode {solver_name}")
    data = solver.solve(data)
    solved, stats = solver.collect_stats()
    stats.print()

    if target_file is not None:
        if isinstance(problem.meta, MetaDataOcp) and problem.meta.dump_solution is not None:
            output_data = problem.meta.dump_solution(data.x_sol)
        else:
            output_data = data.x_sol

        write_json({"w0": output_data}, target_file)

    logger.info(f"Objective value: {data.obj_val}")
    check_solution(problem, data, data.x_sol, settings)
    return stats, data
