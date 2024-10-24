# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test if all combinations work."""

import unittest
from os import path, makedirs, listdir
from shutil import rmtree
from camino.utils.cache import CachedFunction, ca


FILE_DIR = ".test"
FILE_PATH = path.join(FILE_DIR, "test")


def dummy_casadi_function():
    """Create a simple function."""
    x = ca.SX.sym("x", 2)
    f = x[0] ** 2 + x[1]
    return ca.Function("f", [x], [f])


class TestSolver(unittest.TestCase):
    """Test."""

    def setUp(self):
        """Setup a folder."""
        makedirs(FILE_DIR, exist_ok=True)

    def tearDown(self):
        """Remove cached files."""
        try:
            rmtree(FILE_DIR)
        except Exception:
            pass

    def test_cached_function(self):
        """Test a cached function."""
        f1 = CachedFunction("test", dummy_casadi_function,
                            FILE_PATH, do_compile=False)
        out = f1([2.0, 3.0])
        self.assertEqual(out[0], 7.0)

        files = listdir(FILE_DIR)
        for f in files:
            self.assertFalse(f.endswith(".so"))

        f2 = CachedFunction("test", dummy_casadi_function,
                            FILE_PATH, do_compile=False)
        out = f2([2.0, 3.0])
        self.assertEqual(out[0], 7.0)
        f2.f.jacobian().jacobian()

    def test_cached_function_compiled(self):
        """Test a cached function."""
        f1 = CachedFunction("test", dummy_casadi_function,
                            FILE_PATH, do_compile=True)
        out = f1([2.0, 3.0])
        self.assertEqual(out[0], 7.0)

        files = listdir(FILE_DIR)
        has_so = False
        for f in files:
            if f.endswith(".so"):
                has_so = True

        self.assertTrue(has_so)

        f2 = CachedFunction("test", dummy_casadi_function,
                            FILE_PATH, do_compile=True)
        out = f2([2.0, 3.0])
        self.assertEqual(out[0], 7.0)
        f2.f.jacobian().jacobian()


if __name__ == "__main__":
    unittest.main()
