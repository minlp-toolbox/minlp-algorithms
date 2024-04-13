"""A set of conversion tools."""

import numpy as np


def to_bool(val):
    """String or bool value to bool."""
    if isinstance(val, bool):
        return val
    else:
        return not (val.lower() in ["0", "false", "false", "no"])


def to_float(val):
    """To single float."""
    if isinstance(val, np.ndarray):
        return to_float(val[0])
    elif isinstance(val, list):
        return to_float(val[0])
    return val
