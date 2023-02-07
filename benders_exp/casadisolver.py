from benders_exp.nlpsolver import NLPSolverMPCBaseClass
from benders_exp.dmiqp import DMiqp
import numpy as np
import casadi as ca

CASADI_CONTINOUS = 0
CASADI_BINARY = 1


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

    def _assemble_problem(self, b_fixed: bool) -> None:
        """Create the problem."""
        x0 = np.squeeze(self._wlin)

        A = np.vstack(filter([
            self._A, self._A_vb, self._A_mutc, self._A_mdtc
        ]))
        lb = np.hstack(filter([
            self._lb, self._lb_vb, self._lb_mutc, self._lb_mdtc
        ]))

        ub = np.hstack(filter([
            self._ub, self._ub_vb, self._ub_mutc, self._ub_mdtc
        ]))

        if self._A_msc is not None:
            A0 = np.csc_matrix(
                (A.shape[0], self._A_msc.shape[1] - A.shape[1]), dtype=int
            )

            A = np.csc_matrix(np.vstack([np.hstack([A, A0]), self._A_msc]))

            lb = np.hstack([lb, self._lb_msc])
            ub = np.hstack([ub, self._ub_msc])

            x0 = np.hstack([x0, np.zeros(A.shape[1] - x0.shape[0])])

        # self.H.resize((A.shape[1], A.shape[1]))
        q = np.append(self._q, np.zeros(A.shape[1] - self._nw))

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

        qpproblem = {
            # 'h':
            'a': A,
            'g': q,
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
            "lbg": lb,
            "ubg": ub,
        }

    def _solve_problem(self, gap: float) -> None:
        self.solution = self.solver(**self.qpsolver_args)
        self.stats = self.solver.stats()

    def _collect_solver_stats(self):
        pass
