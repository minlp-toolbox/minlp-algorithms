# Adrian Buerger, 2022

import os
import datetime as dt
import pickle

import logging

from timing import TimingMPC
from state import State
from ambient import Ambient

from simulator import Simulator
from predictor import Predictor
from nlpsolver import NLPSolverRel, NLPSolverBin
from binapprox import BinaryApproximation


def main():
    RESULTS_FOLDER = "results/standard"

    USE_STORED_NLP_REL = False
    STORE_RESULTS = True

    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    startup_time = dt.datetime.fromisoformat("2010-08-19 06:00:00+02:00")
    timing = TimingMPC(startup_time=startup_time)

    ambient = Ambient(timing=timing)
    ambient.update()

    state = State()
    state.initialize()

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

        nlpsolver_rel = NLPSolverRel(
            timing=timing,
            ambient=ambient,
            previous_solver=simulator,
            predictor=predictor,
            solver_name="nlpsolver_rel",
        )
        nlpsolver_rel.solve()

        if STORE_RESULTS:
            with open(os.path.join(RESULTS_FOLDER, "nlpsolver_rel.pickle"), "wb") as f:
                pickle.dump(nlpsolver_rel, f)

    binapprox_miqp = BinaryApproximation(
        timing=timing,
        previous_solver=nlpsolver_rel,
        predictor=predictor,
        solver_name="binapprox_miqp",
    )
    binapprox_miqp.solve(
        method="miqp", use_reduced_miqp=True, warm_start=False, gap=0.05
    )

    if STORE_RESULTS:
        with open(os.path.join(RESULTS_FOLDER, "binapprox_miqp.pickle"), "wb") as f:
            pickle.dump(binapprox_miqp, f)

    nlpsolver_bin_miqp = NLPSolverBin(
        timing=timing,
        ambient=ambient,
        previous_solver=binapprox_miqp,
        predictor=predictor,
        solver_name="nlpsolver_bin_miqp",
    )
    nlpsolver_bin_miqp.solve()

    if STORE_RESULTS:
        with open(os.path.join(RESULTS_FOLDER, "nlpsolver_bin_miqp.pickle"), "wb") as f:
            pickle.dump(nlpsolver_bin_miqp, f)

    # For comparison

    binapprox_cia = BinaryApproximation(
        timing=timing,
        previous_solver=nlpsolver_rel,
        predictor=predictor,
        solver_name="binapprox_cia",
    )
    binapprox_cia.solve(method="cia")

    if STORE_RESULTS:
        with open(os.path.join(RESULTS_FOLDER, "binapprox_cia.pickle"), "wb") as f:
            pickle.dump(binapprox_cia, f)

    nlpsolver_bin_cia = NLPSolverBin(
        timing=timing,
        ambient=ambient,
        previous_solver=binapprox_cia,
        predictor=predictor,
        solver_name="nlpsolver_bin_cia",
    )
    nlpsolver_bin_cia.solve()

    if STORE_RESULTS:
        with open(os.path.join(RESULTS_FOLDER, "nlpsolver_bin_cia.pickle"), "wb") as f:
            pickle.dump(nlpsolver_bin_cia, f)


if __name__ == "__main__":
    main()
