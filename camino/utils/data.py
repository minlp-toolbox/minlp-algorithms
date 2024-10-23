# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Data utilities for loading and writing data."""

import json
import pickle


def write_json(data, file):
    """Write json file."""
    with open(file, "w") as f:
        json.dump(
            data,
            f, indent=4
        )


def read_json(file):
    """Load json file."""
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def load_pickle(file):
    """Load a pickle file."""
    with open(file, "rb") as f:
        return pickle.load(f)


def save_pickle(data, file):
    """Write a pickle file."""
    with open(file, "wb") as handle:
        pickle.dump(data, handle)
