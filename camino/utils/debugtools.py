# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

import casadi as ca
import numpy as np
from camino.solvers import MinlpProblem, MinlpData
from camino.settings import Settings
from camino.settings import GlobalSettings

CALLBACK_INPUTS = dict()
for i, label in enumerate(ca.nlpsol_out()):
    CALLBACK_INPUTS[label] = i


class CheckNoDuplicate:
    """Check duplicates."""

    def __init__(self, problem: MinlpProblem, s: Settings):
        """Check if no duplicates pass through."""
        self.idx_x_integer = problem.idx_x_integer
        self.s = s
        self.old = []
        self.count = 0

    def __call__(self, nlpdata: MinlpData):
        """Check if no old solutions pass through."""
        if nlpdata.prev_solutions is None:
            return
        for sol in nlpdata.prev_solutions:
            new = sol['x'][self.idx_x_integer]
            for el in self.old:
                if np.allclose(el, new, equal_nan=False, atol=self.s.EPS):
                    print("Duplicate!")
                    self.count += 1
            self.old.append(new)

        if self.count > 10:
            raise Exception()


class DebugCallBack(ca.Callback):
    """
    Create a debug callback.

    Usage:
        options = {}
        mycallback = DebugCallBack('Name', nx, ng, np, options)
        mycallback.add_to_solver_opts(options)
        ... Construct your solver as usual with options 'options' ...

        Every few iterations, the values for x will be written to a file in
        the datafolder.

    :param name: name of the callback function
    :param nx: Nr of x variables
    :param ng: Nr of g constraints
    :param np: Nr of parameters p
    :param opts: Additional options
    """

    def __init__(self, name, nx: int, ng: int, np: int, opts=None):
        """Create the debug call back for casadi."""
        ca.Callback.__init__(self)
        if opts is None:
            opts = {}

        self.nx = nx
        self.ng = ng
        self.np = np
        self.iter_nr = 0
        self.name = name
        # Initialize internal objects
        self.construct(name, opts)

    def add_to_solver_opts(self, options, iter_per_callback=50):
        """
        Add to the solver options.

        :param options: options of the nlp
        :param iter_per_callback: nr of iters per callback
        """
        options["iteration_callback"] = self
        options['iteration_callback_step'] = iter_per_callback
        return options

    def get_n_in(self):
        """Get number of inputs."""
        return ca.nlpsol_n_out()

    def get_n_out(self):
        """Get number of outputs."""
        return 1

    def get_sparsity_in(self, i):
        """Get sparsity of matrices."""
        n = ca.nlpsol_out(i)
        if n == 'f':
            return ca.Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return ca.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return ca.Sparsity.dense(self.ng)
        elif n in ('p', 'lam_p'):
            return ca.Sparsity.dense(self.np)
        else:
            return ca.Sparsity(0, 0)

    def save(self, x):
        """Save the x variable."""
        self.iter_nr += 1
        np.save(GlobalSettings.DATA_FOLDER +
                f"/x_{self.name}_{self.iter}", x.full())

    def eval(self, arg):
        """Evaluate the callback."""
        x = arg[CALLBACK_INPUTS["x"]]
        self.save(x)
        return [0]
