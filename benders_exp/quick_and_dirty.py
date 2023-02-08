"""Quick and dirty implementation."""

from dataclasses import dataclass
import datetime as dt
from typing import List, Dict, Any, Optional
import numpy as np

# from benders_exp.nlpsolver import NLPSolverRel, NLPSolverBin
from benders_exp.ambient import Ambient
from benders_exp.casadisolver import NLPSolverBin2
# from benders_exp.defines import RESULTS_FOLDER
from benders_exp.nlpsetup import NLPSetupMPC
from benders_exp.predictor import Predictor
from benders_exp.simulator import Simulator
from benders_exp.state import State
from benders_exp.timing import TimingMPC
import casadi as ca


@dataclass
class Stats:
    """Collect stats."""

    data: Dict[str, float]
    runtime: float = 0


@dataclass
class MinlpProblem:
    """Minlp problem description."""

    f: ca.SX
    g: ca.SX
    x: ca.SX
    p: ca.SX
    idx_x_bin: List[float]


@dataclass
class MinlpData:
    """Nlp data."""

    p: List[float]
    x0: ca.DM
    lbx: ca.DM
    ubx: ca.DM
    lbg: ca.DM
    ubg: ca.DM
    solved: bool
    prev_solution: Optional[Dict[str, Any]] = None

    @property
    def _sol(self):
        """Get safely previous solution."""
        if self.prev_solution is not None:
            return self.prev_solution
        else:
            return {"f": -ca.inf, "x": self.x0}

    @property
    def obj_val(self):
        """Get float value."""
        return float(self._sol['f'])

    @property
    def x_sol(self):
        """Get x solution."""
        return self._sol['x']

    @property
    def lam_g_sol(self):
        """Get lambda g solution."""
        return self._sol['lam_g']


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
    # Hessian, gradient_f = ca.hessian(nlpsetup_mpc.nlp['f'], nlpsetup_mpc.nlp['x'])
    # jacobian_g = ca.jacobian(nlpsetup_mpc.nlp['g'], nlpsetup_mpc.nlp['x'])

    # nlpsol = ca.nlpsol("nlpsol", "ipopt", nlpsetup_mpc.nlp, {
    #     "jit": True
    # })
    # nlpsetup_mpc.nlp['f']
    # nlpsetup_mpc.nlp['g']
    # bin_x = nlpsetup_mpc.nlp['x'][binary_values]
    # x = nlpsetup_mpc.nlp['x']
    # qpsol = ca.qpsol("qpsolver", "gurobi", {
    #     "f": 1/ 2 * x.T @ H @ x + q.T @ x
    #     "g": a
    # }

    # breakpoint()
    # result = nlpsol(**nlp_args)
    # stats = nlpsol.stats()
    problem = MinlpProblem(*nlpsetup_mpc.nlp, idx_x_bin=binary_values)
    data = MinlpData(**nlp_args, solved=True)
    return problem, data


def create_dummy_problem(p_val=[1000, 3]):
    """Create a dummy problem."""
    x = ca.MX.sym("x", 3)
    x0 = np.array([0, 4, 100])
    idx_x_bin = [0, 1]
    p = ca.MX.sym("p", 2)
    f = (x[0] - 4.1)**2 + (x[1] - 4.0)**2 + p[0]
    g = ca.vertcat(
        -x[2],
        x[0]**2 + x[1]**2 - x[2] - p[1]**2
    )
    lbg = -np.array([ca.inf, ca.inf])
    ubg = np.array([0, 0])
    lbx = -np.array([ca.inf, ca.inf, ca.inf])
    ubx = np.array([ca.inf, ca.inf, ca.inf])

    problem = MinlpProblem(x=x, f=f, g=g, p=p, idx_x_bin=idx_x_bin)
    data = MinlpData(x0=x0, ubx=ubx, lbx=lbx,
                     ubg=ubg, lbg=lbg, p=p_val, solved=True)
    return problem, data


class SolverClass:
    """Create solver class."""

    def __init___(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create a solver class."""
        self.stats = stats
        self.solver = None

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve the problem."""

    def collect_stats(self):
        """Collect statistics."""
        stats = self.solver.stats()
        self.stats.runtime += stats["t_wall_total"]
        return stats["success"], stats


class NlpSolver(SolverClass):
    """Create NLP solver."""

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create NLP problem."""
        super(NlpSolver, self).__init___(problem, stats)
        if options is None:
            options = {}

        self.idx_x_bin = problem.idx_x_bin
        options.update({"jit": True})
        self.solver = ca.nlpsol("nlpsol", "ipopt", {
            "f": problem.f, "g": problem.g, "x": problem.x, "p": problem.p
        }, options)

    def solve(self, nlpdata: MinlpData, set_x_bin=False) -> MinlpData:
        """Solve NLP."""
        lbx = nlpdata.lbx.copy()
        ubx = nlpdata.ubx.copy()
        if set_x_bin:
            lbx[nlpdata.idx_x_bin] = nlpdata.x_sol[nlpdata.idx_x_bin]
            ubx[nlpdata.idx_x_bin] = nlpdata.x_sol[nlpdata.idx_x_bin]

        nlpdata.prev_solution = self.solver(
            p=nlpdata.p, x0=nlpdata.x_sol,
            lbx=lbx, ubx=ubx,
            lbg=nlpdata.lbg, ubg=nlpdata.ubg
        )
        nlpdata.solved = self.collect_stats()[0]
        return nlpdata


class BendersMasterMILP(SolverClass):
    """Create benders master problem."""

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create benders master MILP."""
        super(BendersMasterMILP, self).__init___(problem, stats)
        if options is None:
            options = {}

        self.grad_f_x_bin = ca.Function(
            "gradient_f_x_bin",
            [problem.x, problem.p], [ca.gradient(
                problem.f, problem.x
            )[problem.idx_x_bin]],
            {"jit": True}
        )
        self.f = ca.Function(
            "f", [problem.x, problem.p], [problem.f],
            {"jit": True}
        )
        self.jac_g = ca.Function(
            "jac_g", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)[:, problem.idx_x_bin]],
            {"jit": True}
        )
        self.idx_x_bin = problem.idx_x_bin
        self.nr_x_bin = len(problem.idx_x_bin)
        self._x_bin = ca.SX.sym("x_bin", self.nr_x_bin)
        self._nu = ca.SX.sym("nu", 1)
        self._g = np.array([])
        self.nr_g = 0
        self.options = options.copy()
        self.options["discrete"] = [1] * (self.nr_x_bin + 1)
        self.options["discrete"][-1] = 0
        self.options["gurobi.MIPGap"] = 0.05

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """solve."""
        grad_f_k = self.grad_f_x_bin(nlpdata.x_sol, nlpdata.p)
        jac_g_k = self.jac_g(nlpdata.x_sol, nlpdata.p)
        lambda_k = grad_f_k - jac_g_k @ nlpdata.lam_g_sol
        f_k = self.f(nlpdata.x_sol, nlpdata.p)
        g_k = (
            f_k + lambda_k.T @ (self._x_bin - nlpdata.x_sol[self.idx_x_bin])
            - self._nu
        )
        self._g = ca.vertcat(self._g, g_k)
        self.nr_g += 1

        self.solver = ca.qpsol(f"benders{self.nr_g}", "gurobi", {
            "f": self._nu, "g": self._g,
            "x": ca.vertcat(self._x_bin, self._nu),
        }, self.options)

        solution = self.solver(
            x0=ca.vertcat(nlpdata.x_sol[self.idx_x_bin], nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx[self.idx_x_bin], -ca.inf),
            ubx=ca.vertcat(nlpdata.ubx[self.idx_x_bin], ca.inf),
            lbg=-ca.inf * np.ones(self.nr_g),
            ubg=np.zeros(self.nr_g)
        )
        x_full = nlpdata.x_sol.copy()
        x_full[self.idx_x_bin] = solution['x'][:-1]
        solution['x'] = x_full
        nlpdata_out = nlpdata.copy()
        nlpdata_out.prev_solution = solution
        return nlpdata_out


if __name__ == "__main__":
    problem, data = create_dummy_problem()
    stats = Stats({})
    nlp = NlpSolver(problem, stats)
    benders_milp = BendersMasterMILP(problem, stats)
    data = nlp.solve(data)

    # Benders algorithm
    lb = -ca.inf
    ub = ca.inf
    tolerance = 0.04
    feasible = True
    data = nlp.solve(data)
    x_bar = data.x_sol
    x_star = x_bar
    while lb + tolerance < ub and feasible:
        # Solve MILP-BENDERS and set lower bound:
        data = benders_milp.solve(data)  # Linearization point = previous solution!
        feasible = data.solved
        lb = data.obj_val
        # x_hat = data.x_sol

        # Obtain new linearization point for NLP:
        data = nlp.solve(data, set_x_bin=True)
        x_bar = data.x_sol
        if not data.solved:
            # Solve NLPF!
            raise NotImplementedError()
        elif data.obj_val < ub:
            ub = data.obj_val
            x_star = x_bar
