# Adrian Buerger, Andrea Ghezzi 2022

import argparse
import datetime as dt
import logging
import os
import pickle

from benders_exp.ambient import Ambient
from benders_exp.binapprox import BinaryApproximation
from benders_exp.predictor import Predictor
from benders_exp.simulator import Simulator
from benders_exp.state import State
from benders_exp.timing import TimingMPC
from benders_exp.voronoi import Voronoi
from benders_exp.casadisolver import NLPSolverBin2


def main(args):
    RESULTS_FOLDER = "../results/voronoi"

    USE_STORED_NLP_REL = False
    STORE_RESULTS = True
    USE_STORED_ITER = False

    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    startup_time = dt.datetime.fromisoformat("2010-08-19 06:00:00+02:00")
    timing = TimingMPC(startup_time=startup_time)

    ambient = Ambient(timing=timing)
    ambient.update()

    state = State()
    state.initialize()

    voronoi = Voronoi()
    voronoi.set_weight_matrix(timing=timing, weight_type=args.weight)

    simulator = Simulator(timing=timing, ambient=ambient, state=state)
    simulator.solve()

    predictor = Predictor(
        timing=timing,
        ambient=ambient,
        state=state,
        previous_solver=simulator,
        solver_name="predictor",
    )
    predictor.solve(n_steps=0)

    if USE_STORED_NLP_REL:

        with open(os.path.join(RESULTS_FOLDER, "nlpsolver_rel.pickle"), "rb") as f:
            nlpsolver_rel = pickle.load(f)

    else:

        nlpsolver_rel = NLPSolverBin2(
            timing=timing,
            ambient=ambient,
            previous_solver=simulator,
            predictor=predictor,
            solver_name="nlpsolver_rel_voronoi",
        )
        nlpsolver_rel.solve()

        if STORE_RESULTS:
            with open(os.path.join(RESULTS_FOLDER, "nlpsolver_rel.pickle"), "wb") as f:
                pickle.dump(nlpsolver_rel, f)

    previous_solver = nlpsolver_rel
    n_iter = 0
    while not voronoi.termination_criterion_reached:

        if (USE_STORED_ITER) & ((n_iter <= 1)):
            with open(
                os.path.join(
                    RESULTS_FOLDER, f"binapprox_miqp_CIA_STD_iter_{n_iter}.pickle"
                ),
                "rb",
            ) as f:
                binapprox_miqp = pickle.load(f)
        else:
            binapprox_miqp = BinaryApproximation(
                timing=timing,
                previous_solver=previous_solver,
                predictor=predictor,
                solver_name=f"binapprox_miqp_{args.weight}_{args.strategy}_iter_{n_iter}",
            )
            binapprox_miqp.solve(
                method="miqp",
                use_reduced_miqp=True,
                warm_start=False,
                voronoi=voronoi,
                gap=0.25,
            )

        if STORE_RESULTS:
            with open(
                os.path.join(
                    RESULTS_FOLDER,
                    f"binapprox_miqp_{args.weight}_{args.strategy}_iter_{n_iter}.pickle",
                ),
                "wb",
            ) as f:
                pickle.dump(binapprox_miqp, f)

        if (USE_STORED_ITER) & ((n_iter <= 1)):
            with open(
                os.path.join(
                    RESULTS_FOLDER, f"nlpsolver_bin_miqp_CIA_STD_iter_{n_iter}.pickle"
                ),
                "rb",
            ) as f:
                nlpsolver_bin_miqp = pickle.load(f)
        else:
            nlpsolver_bin_miqp = nlpsolver_rel
            nlpsolver_rel.update(binapprox_miqp)
            nlpsolver_bin_miqp.solve()

        if STORE_RESULTS:
            with open(
                os.path.join(
                    RESULTS_FOLDER,
                    f"nlpsolver_bin_miqp_{args.weight}_{args.strategy}_iter_{n_iter}.pickle",
                ),
                "wb",
            ) as f:
                pickle.dump(nlpsolver_bin_miqp, f)

        voronoi.add_solution(current_nlp_solution=nlpsolver_bin_miqp)
        logger.warning(f"Relaxed NLP objective: {nlpsolver_rel.nlp_objective_value}")
        voronoi.update_optimal_solution_so_far()
        voronoi.update_termination_criterion()
        voronoi.compute_cuts(strategy=args.strategy)
        nlpsolver_bin_miqp.reset_bounds_and_initials()
        nlpsolver_bin_miqp.reduce_object_memory_size()
        previous_solver = nlpsolver_bin_miqp
        n_iter += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pass experiment parameters via command line"
    )
    parser.add_argument(
        "--weight",
        type=str,
        default="CIA",
        help="weighting for the Voronoi cuts, available options: CIA (CIA matrix), ID (identity matrix)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="STD",
        help="strategy to compute the Voronoi cuts, available options: STD (standard), RES (restricted)",
    )
    args = parser.parse_args()

    main(args)
