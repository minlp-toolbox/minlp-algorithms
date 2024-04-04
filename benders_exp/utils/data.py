"""Data utilities."""

import pickle


def load_pickle(file):
    """Load a pickle file."""
    with open(file, "rb") as f:
        return pickle.load(f)


def save_pickle(data, file):
    """Write a pickle file."""
    with open(file, "wb") as handle:
        pickle.dump(data, handle)
