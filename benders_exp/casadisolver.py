import logging
import numpy as np
import casadi as ca
import scipy.sparse as ssp

from benders_exp.nlpsolver import NLPSolverMPCBaseClass
from benders_exp.dmiqp import DMiqp

CASADI_CONTINOUS = 0
CASADI_BINARY = 1

logger = logging.getLogger(__name__)


class NLPSolverBin2(NLPSolverMPCBaseClass):

    _SOLVER_TYPE = "NLPbin2"

    def __init__(self, *args, **kwargs):
        """Create solver 2."""
        NLPSolverMPCBaseClass.__init__(self, *args, **kwargs)
        self._store_previous_binary_solution()
        self._setup_nlpsolver()
        self._set_states_bounds()
        self._set_continuous_control_bounds()
        self._set_binary_control_bounds()
        self._set_nlpsolver_bounds_and_initials()

    def update(self, prev_solver):
        """Update using previous solver the binary controls."""
        self.set_binary_controls(prev_solver.b_data)
        self._set_previous_solver(prev_solver)

    def set_binary_controls(self, b_data):
        """Set binary control bounds."""
        self.b_min = b_data
        self.b_max = b_data

    def get_x(self):
        """Get x solution."""
        return np.array(self.nlp_solution["x"])

    def solve(self):
        """Solve this problem."""
        self._set_nlpsolver_bounds_and_initials()
        self._run_nlpsolver()
        self._collect_nlp_results()

    def _setup_additional_nlpsolver_options(self):
        """Setup additional NLP solver options."""
        self._nlpsolver_options["ipopt.acceptable_tol"] = 0.2
        self._nlpsolver_options["ipopt.acceptable_iter"] = 8
        self._nlpsolver_options["ipopt.acceptable_constr_viol_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_dual_inf_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_compl_inf_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_obj_change_tol"] = 1e-1

        self._nlpsolver_options["ipopt.mu_strategy"] = "adaptive"
        self._nlpsolver_options["ipopt.mu_target"] = 1e-5

    def _set_binary_control_bounds(self):
        """Set up initial binary controls."""
        self.b_min = np.zeros((self._timing.N, self.nb))
        self.b_max = np.ones((self._timing.N, self.nb))

        if self._timing._remaining_min_up_time > 0:
            locked_time_grid_points = np.where(
                self._timing.time_grid < self._timing._remaining_min_up_time
            )[0]

            self.b_min[locked_time_grid_points, :] = np.repeat(
                self._timing._b_bin_locked, len(locked_time_grid_points), 0
            )
            self.b_max[locked_time_grid_points, :] = np.repeat(
                self._timing._b_bin_locked, len(locked_time_grid_points), 0
            )


def filter(lst):
    """Stack a matrix."""
    out = []
    for el in lst:
        if el is not None:
            out.append(el)
    return out


class BendersMILP(DMiqp):
    """Create benders MILP"""

    @property
    def x(self) -> np.ndarray:
        try:
            return np.array(self.solution["x"])
        except AttributeError:
            msg = "Optimal solution not available yet, call solve() first."
            logger.error(msg)
            raise RuntimeError(msg)

    @property
    def obj(self) -> np.ndarray:
        try:
            return self.qp.obj
            return self.solution["f"]
        except AttributeError:
            msg = "Objective value not available yet, call solve() first."
            logger.error(msg)
            raise RuntimeError(msg)

    @property
    def solve_successful(self):
        try:
            return self._stats["success"]
        except AttributeError:
            msg = "Solver return status not available yet, call solve() first."
            logger.error(msg)
            raise RuntimeError(msg)


    def _assemble_problem(self, b_fixed: bool) -> None:
        """Create the problem."""
        A = [self._A]
        lb = [self._lb]
        ub = [self._ub]

        x0 = np.squeeze(self._wlin)

        for A_k in [self._A_vb, self._A_mutc, self._A_mdtc, self._A_v]:
            if A_k is not None:
                A.append(A_k)
        A = ssp.vstack(A)

        for lb_k in [self._lb_vb, self._lb_mutc, self._lb_mdtc, self._lb_v]:
            if lb_k is not None:
                lb.append(lb_k)
        lb = np.hstack(lb)

        for ub_k in [self._ub_vb, self._ub_mutc, self._ub_mdtc, self._ub_v]:
            if ub_k is not None:
                ub.append(ub_k)
        ub = np.hstack(ub)

        if self._A_msc is not None:

            A0 = ssp.csc_matrix(
                (A.shape[0], self._A_msc.shape[1] - A.shape[1]), dtype=int
            )

            A = ssp.csc_matrix(ssp.vstack([ssp.hstack([A, A0]), self._A_msc]))

            lb = np.hstack([lb, self._lb_msc])
            ub = np.hstack([ub, self._ub_msc])

            x0 = np.hstack([x0, np.zeros(A.shape[1] - x0.shape[0])])

        self.H.resize((A.shape[1], A.shape[1]))

        # self.H.resize((A.shape[1], A.shape[1]))
        q = self._q  # np.append(self._q, np.zeros(A.shape[1] - self._nw))

        vtype = np.empty(A.shape[1], dtype=object)
        vtype[:] = CASADI_CONTINOUS
        if not b_fixed:
            vtype[self.idx_b] = CASADI_BINARY

        lbx = -1e3 * np.ones(A.shape[1])
        if not b_fixed:
            lbx[self.idx_b] = 0.0
        else:
            lbx[self.idx_b] = self._b0

        lbx[self.idx_sb] = 0.0
        lbx[self._nw:] = 0.0

        ubx = 1e5 * np.ones(A.shape[1])
        if not b_fixed:
            ubx[self.idx_b] = 1.0
        else:
            ubx[self.idx_b] = self._b0

        ubx[self.idx_sb] = 1.0
        # ubx[self._nw:] = 1.0 # TBD!

        qpproblem = {'a': ca.DM(A).sparsity(), 'h': ca.DM.eye(q.shape[0]).sparsity()}

        qpproblem = {
            # 'h':
            'a': ca.DM(A).sparsity(),
            'h': ca.DM(self.H).sparsity(),
        }
        self.solver = ca.conic('S', 'gurobi', qpproblem, {
            'discrete': vtype,
            "gurobi": {"MIPGap": 0.05}
        })

        self._qpsolver_args = {
            # "p": self.P_data,
            "x0": x0,
            "lbx": lbx,
            "ubx": ubx,
            "lba": lb,
            "uba": ub,
            "g": q,
            "h": 0 * ca.DM(self.H),
            'a': ca.DM(A),
        }

    def _solve_problem(self, gap: float) -> None:
        self.solution = self.solver(**self._qpsolver_args)
        self._stats = self.solver.stats()

    def _collect_solver_stats(self):
        self.solver_stats = {
            "runtime": self._stats["t_wall_solver"],
            "return_status": self._stats["return_status"],
        }
