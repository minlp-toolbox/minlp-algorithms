from benders_exp.solarsys.nlpsolver import NLPSolverRel  # NLPSolverBin
from benders_exp.solarsys.ambient import Ambient
from benders_exp.solarsys.defines import _PATH_TO_NLP_OBJECT, _NLP_OBJECT_FILENAME
from benders_exp.solarsys.nlpsetup import NLPSetupMPC
from benders_exp.solarsys.predictor import Predictor
from benders_exp.solarsys.simulator import Simulator
from benders_exp.solarsys.state import State
from benders_exp.solarsys.timing import TimingMPC
from benders_exp.problems import MinlpProblem, MinlpData
import datetime as dt
from os import path


def extract():
    """Extract original problem."""
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

    # simulator.b_data
    nlpsolver_rel = NLPSolverRel(
        timing=timing,
        ambient=ambient,
        previous_solver=simulator,
        predictor=predictor,
        solver_name="nlpsolver_rel",
    )

    nlpsetup_mpc = NLPSetupMPC(timing=timing)
    nlpsetup_mpc._setup_nlp(True)

    binary_values = []
    binary_values.extend(nlpsetup_mpc.idx_b)
    # binary_values.extend(nlpsetup_mpc.idx_sb)
    # binary_values.extend(nlpsetup_mpc.idx_sb)
    # binary_values.extend(nlpsetup_mpc.idx_sb_red)

    nlpsolver_rel._store_previous_binary_solution()
    nlpsolver_rel._setup_nlpsolver()
    nlpsolver_rel._set_states_bounds()
    nlpsolver_rel._set_continuous_control_bounds()
    nlpsolver_rel._set_binary_control_bounds()
    nlpsolver_rel._set_nlpsolver_bounds_and_initials()

    nlp_args = nlpsolver_rel._nlpsolver_args
    path_to_nlp_object = path.join(
        _PATH_TO_NLP_OBJECT, _NLP_OBJECT_FILENAME
    )

    problem = MinlpProblem(
        **nlpsetup_mpc.nlp, idx_x_bin=binary_values,
        precompiled_nlp=path_to_nlp_object
    )
    data = MinlpData(
        x0=nlp_args['x0'],
        _lbx=nlp_args['lbx'],
        _ubx=nlp_args['ubx'],
        _lbg=nlp_args['lbg'],
        _ubg=nlp_args['ubg'],
        p=nlp_args['p'],
    )

    return problem, data


if __name__ == "__main__":
    from benders_exp.utils import setup_logger, logging
    from benders_exp.solvers import Stats
    from benders_exp.solvers.nlp import NlpSolver
    from benders_exp.problems import check_solution
    import pickle

    setup_logger(logging.DEBUG)
    prob, data = extract()

    stats = Stats({})
    nlp = NlpSolver(prob, stats)

    with open("data/nlpargs_adrian.pickle", 'rb') as f:
        nlpargs_adrian = pickle.load(f)
    data.x0 = nlpargs_adrian['x0']
    data.p = nlpargs_adrian['p']
    data.lbx = nlpargs_adrian['lbx']
    data.ubx = nlpargs_adrian['ubx']
    data.lbg = nlpargs_adrian['lbg']
    data.ubg = nlpargs_adrian['ubg']

    data = nlp.solve(data, set_x_bin=False) # solve relaxed problem
    with open("results/x_star_rel_orig.pickle", "wb") as f:
        pickle.dump(data.x_sol, f)
    breakpoint()
    check_solution(prob, data, data.prev_solution['x'])
    breakpoint()