# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cached function."""

import logging
from os import path, rename
from typing import Callable
import casadi as ca
from subprocess import call

from camino.settings import GlobalSettings
from camino.utils.data import read_json, write_json, load_pickle, save_pickle

logger = logging.getLogger(__name__)
_COMPILERS = ["gcc"]
_COMPILER = None


def get_compiler():
    """Get available compiler."""
    global _COMPILER
    if _COMPILER is None:
        for compiler in _COMPILERS:
            try:
                call([compiler, "--version"])
                _COMPILER = compiler
                break
            except Exception:
                pass

    return _COMPILER


def compile(input_file, output_file, options=None):
    """Compile a c file to an so file."""
    compiler = get_compiler()

    _CXX_FLAGS = ["-fPIC", "-v", "-shared", "-fno-omit-frame-pointer", "-O2"]
    call([compiler] + _CXX_FLAGS + ["-o", output_file, input_file])


def cache_data(name, generator_func, *args, **kwargs):
    """Cache data."""
    name = name + "_" + getattr(generator_func, '__name__', 'Unknown')
    filename = path.join(GlobalSettings.CACHE_FOLDER, name + ".pkl")
    if path.exists(filename):
        return load_pickle(filename)
    else:
        data = generator_func(*args, **kwargs)
        save_pickle(data, filename)
        return data


def return_func(func):
    """Create a function."""
    def r():
        return func
    return r


class CachedFunction:
    """A cached function."""

    def __init__(self, name, func: Callable[[], ca.Function], filename=None, do_compile=None):
        """Load or create a cached function."""
        if do_compile is None:
            do_compile = True

        self.name = name + "_" + getattr(func, '__name__', 'Unknown')
        if filename is None:
            self.filename = path.join(GlobalSettings.CACHE_FOLDER, self.name)
        else:
            self.filename = filename

        if self._exist_so():
            logger.debug(f"Loading function from so-file {self.name}")
            self._load_so()
        elif self._exist():
            logger.debug(f"Loading function from disc {self.name}")
            self._load()
        else:
            logger.debug(f"Creating function {self.name}")
            self._create(func)
            if do_compile:
                logger.debug(f"Compiling function {self.name}")
                self._save_so()
                self._load_so()
            else:
                self._save()

    def _create(self, func):
        """Create a function."""
        self.f = func()

    def _exist(self):
        """Check if the file exist."""
        return path.exists(self.filename + ".cache")

    def _save(self):
        """Save to a file."""
        self.f.save(self.filename + ".cache")

    def _load(self):
        """Load the function."""
        self.f = ca.Function.load(self.filename + ".cache")

    def _exist_so(self):
        """Check if the file exist."""
        return path.exists(self.filename + ".json")

    def _save_so(self):
        """Save as so file."""
        data = {
            "lib_file": self.filename + ".so",
            "c_file": self.filename + ".c",
            "func_name": self.f.name()
        }

        tmp_file = data["func_name"] + ".c"
        cg = ca.CodeGenerator(tmp_file)
        cg.add(self.f)
        cg.add(self.f.jacobian())
        cg.add(self.f.jacobian().jacobian())
        cg.generate()
        rename(tmp_file, data["c_file"])
        compile(data["c_file"], data["lib_file"])
        write_json(data, self.filename + ".json")

    def _load_so(self):
        """Load an SO file."""
        data = read_json(self.filename + ".json")
        self.f = ca.external(data["func_name"], data["lib_file"])

    def __call__(self, *args, **kwargs):
        """Call the function."""
        return self.f(*args, **kwargs)
