import datetime as dt

# from benders_exp.nlpsolver import NLPSolverRel, NLPSolverBin
from benders_exp.ambient import Ambient
from benders_exp.casadisolver import NLPSolverBin2
from benders_exp.defines import RESULTS_FOLDER
from benders_exp.nlpsetup import NLPSetupMPC
from benders_exp.predictor import Predictor
from benders_exp.simulator import Simulator
from benders_exp.state import State
from benders_exp.timing import TimingMPC
import casadi as ca


def extract():
    startup_time = dt.datetime.fromisoformat("2010-08-19 06:00:00+02:00")
    timing = TimingMPC(startup_time=startup_time)

    ambient = Ambient(timing=timing)
    ambient.update()

    state = State()
    state.initialize()

    simulator = Simulator(timing=timing, ambient=ambient, state=state)
    simulator.solve()

    nlpsetup_mpc = NLPSetupMPC(timing=timing)
    nlpsetup_mpc._setup_nlp(True)

    binary_values = []
    binary_values.extend(nlpsetup_mpc.idx_b)
    binary_values.extend(nlpsetup_mpc.idx_b_red)
    binary_values.extend(nlpsetup_mpc.idx_sb)
    binary_values.extend(nlpsetup_mpc.idx_sb_red)

    predictor = Predictor(
        timing=timing,
        ambient=ambient,
        state=state,
        previous_solver=simulator,
        solver_name="predictor",
    )
    predictor.solve(n_steps=0)

    nlpsolver_rel = NLPSolverBin2(
        timing=timing,
        ambient=ambient,
        previous_solver=simulator,
        predictor=predictor,
        solver_name="nlpsolver_rel",
    )
    nlpsolver_rel._store_previous_binary_solution()
    nlpsolver_rel._setup_nlpsolver()
    nlpsolver_rel._set_states_bounds()
    nlpsolver_rel._set_continuous_control_bounds()
    nlpsolver_rel._set_binary_control_bounds()
    nlpsolver_rel._set_nlpsolver_bounds_and_initials()

    nlp_args = nlpsolver_rel._nlpsolver_args
    # print(nlpsolver_rel._nlpsolver_args)
    # print(f"{nlpsolver_rel._nlpsolver_args['lbx'].shape=}")
    # print(f"{nlpsetup_mpc.nlp['x'].shape=}")

    # x = nlp
    # H, q = ca.hessian(nlpsetup_mpc.nlp['f'], nlpsetup_mpc.nlp['x'])
    # a = ca.jacobian(nlpsetup_mpc.nlp['g'], nlpsetup_mpc.nlp['x'])

    nlpsol = ca.nlpsol("nlpsol", "ipopt", nlpsetup_mpc.nlp, {
        "jit": True
    })
    # qpsol = ca.qpsol("qpsolver", "gurobi", {
    #     "f": 1/ 2 * x.T @ H @ x + q.T @ x
    #     "g": a
    # }

    # breakpoint()
    result = nlpsol(**nlp_args)
    # stats = nlpsol.stats()


#         "p": self.P_data,
#         "x0": self.V_init,
#         "lbx": self.V_min,
#         "ubx": self.V_max,
#         "lbg": self.lbg,
#         "ubg": self.ubg,
#     }

extract()
