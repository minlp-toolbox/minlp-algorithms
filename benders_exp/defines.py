"""Defines."""

from os import path

SOURCE_FOLDER = path.dirname(path.abspath(__file__))

PICKLE_FOLDER = path.join(SOURCE_FOLDER, "../results/voronoi")

_PATH_TO_NLP_SOURCE = path.join(SOURCE_FOLDER, "../.src/")
_PATH_TO_NLP_OBJECT = path.join(SOURCE_FOLDER, "../.lib/")
_PATH_TO_ODE_OBJECT = path.join(SOURCE_FOLDER, "../.lib/")
_PATH_TO_ODE_FILE = _PATH_TO_ODE_OBJECT
