# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Problems from the MINLP lib."""

from camino.problems.dsc import Description
import casadi as ca


def alan():
    dsc = Description()
    x1 = dsc.sym("x1", 1, 0, ca.inf, 0.302884615384618)
    x2 = dsc.sym("x2", 1, 0, ca.inf, 0.0865384615384593)
    x3 = dsc.sym("x3", 1, 0, ca.inf, 0.504807692307693)
    x4 = dsc.sym("x4", 1, 0, ca.inf, 0.10576923076923)
    b5 = dsc.sym_bool("b5", 1, 0)
    b6 = dsc.sym_bool("b6", 1, 0)
    b7 = dsc.sym_bool("b7", 1, 0)
    b8 = dsc.sym_bool("b8", 1, 0)

    dsc.f = x1 * (4 * x1 + 3 * x2 - x3) + x2 * (3 * x1 +
                                                6 * x2 + x3) + x3 * (-x1 + x2 + 10 * x3)
    dsc.eq(x1 + x2 + x3 + x4, 1)
    dsc.eq(8 * x1 + 9 * x2 + 12 * x3 + 7 * x4, 10)
    dsc.leq(x1 - b5, 0)
    dsc.leq(x2 - b6, 0)
    dsc.leq(x3 - b7, 0)
    dsc.leq(x4 - b8, 0)
    dsc.leq(b5 + b6 + b7 + b8, 3)
    return dsc.get_problem(), dsc.get_data()


MINLP_PROBLEMS = {
    "alan": alan
}
