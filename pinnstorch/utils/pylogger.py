import logging
from pathlib import Path

from lightning.pytorch.utilities import rank_zero_only


def get_pylogger(name: str = __name__, output_path: Path = None) -> logging.Logger:
    """Initializes a multi-GPU-friendly python command line logger.

    :param name: The name of the logger, defaults to ``__name__``.
    :param output_path: Optional path to a log file. If provided, logs are
        also written to this file (only from rank 0 in distributed mode).

    :return: A logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Ensures all logging levels get marked with the rank zero decorator,
    # so messages are emitted only from rank 0 in multi-GPU setups.
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    # Console handler (logs to stdout) — guarded by rank_zero_only implicitly
    # via the decorated logging methods above.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_handler)

    # File handler (logs to file) — also guarded by rank_zero_only
    # because it uses the same decorated logger methods.
    if output_path:
        file_handler = logging.FileHandler(output_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(file_handler)

    return logger
