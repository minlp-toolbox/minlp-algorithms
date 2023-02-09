# Andrea Ghezzi, 2023

import os
import datetime as dt
import logging
import pickle

from benders_exp.utils import setup_logger
from benders_exp.timing import TimingMPC
from benders_exp.state import State
from benders_exp.ambient import Ambient

from benders_exp.simulator import Simulator
from benders_exp.predictor import Predictor
# from benders_exp.nlpsolver import NLPSolverRel, NLPSolverBin
from benders_exp.binapprox import BinaryApproximation
from benders_exp.defines import RESULTS_FOLDER
from benders_exp.casadisolver import NLPSolverBin2, BendersMILP

setup_logger()
logger = logging.getLogger(__name__)


# def end_criteria(*kwargs):
#     return True
# 
# 
# def solve_benders(**data):
#     """Solve benders."""
#     pass
# 
# 
# def solve_nlp(**data):
#     """Solve NLP."""
# 
# 
# def main_loop():
#     Y = []
#     j_bar = np.inf
#     while not end_criteria(**data):
#         y_k_star = solve_benders(**data)
#         z_k_star, j_k = solve_nlp(**data)
#         if j_k < j_bar:
#             # Updat ehest solution
#             Y.append(y_k_star)
#             y_bar = y_k_star,
#             z_bar = z_k_star
#             j_bar = j_k
#         else:
#             # store integer solution
#             y_bar = y.pop()
#             y.append([y_k_star, y_bar])
# 


def main():
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

    logger.info("Simulator setup.")
    nlpsolver_rel = NLPSolverBin2(
        timing=timing,
        ambient=ambient,
        previous_solver=simulator,
        predictor=predictor,
        solver_name="nlpsolver_rel",
    )
    nlpsolver_rel.solve()

    # binapprox_miqp = BinaryApproximation(
    #     timing=timing,
    #     previous_solver=nlpsolver_rel,  # nlpsolver_rel,
    #     predictor=predictor,
    #     solver_name="binapprox_miqp",
    # )
    # binapprox_miqp.solve(
    #     method="milp", use_reduced_miqp=True, warm_start=False, gap=0.5
    # )


if __name__ == "__main__":
    main()
