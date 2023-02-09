import logging
from time import perf_counter


def setup_logger():
    """Set up the logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)


def tic():
    """Tic."""
    global perf_ti
    perf_ti = perf_counter()


def toc(reset=False):
    """Toc."""
    global perf_ti
    tim = perf_counter()
    dt = tim - perf_ti
    print("  Elapsed time: %s s." % (dt))
    if reset:
        perf_ti = tim
    return dt
