"""Json tools."""

import json


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
