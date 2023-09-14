"""Bender experiment."""


def load_pickle(file):
    """Load a pickle file."""
    import pickle
    with open(file, "rb") as f:
        return pickle.load(f)


def save_pickle(data, file):
    """Write a pickle file."""
    import pickle
    with open(file, "wb") as handle:
        pickle.dump(data, handle)
