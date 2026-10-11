import logging
from pathlib import Path

# Lightning re-exports this decorator at runtime, but its stubs omit that export.
from lightning.pytorch.utilities import rank_zero_only  # type: ignore[attr-defined]


class _ConsoleHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Identify console handlers installed by this module."""

    # StreamHandler is generic in type stubs but not subscriptable at runtime.


class _FileHandler(logging.FileHandler):
    """Identify file handlers installed by this module."""


def get_pylogger(name: str = __name__, output_path: Path | None = None) -> logging.Logger:
    """Initializes a multi-GPU-friendly python command line logger.

    :param name: The name of the logger, defaults to ``__name__``.

    :return: A logger object.
    """
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    if not logger.__dict__.get("_pinnstorch_rank_wrapped", False):
        for level in logging_levels:
            setattr(logger, level, rank_zero_only(getattr(logger, level)))
        logger.__dict__["_pinnstorch_rank_wrapped"] = True

    if not any(isinstance(handler, _ConsoleHandler) for handler in logger.handlers):
        console_handler = _ConsoleHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(console_handler)

    if output_path is not None and not any(
        isinstance(handler, _FileHandler) and handler.baseFilename == str(output_path.resolve())
        for handler in logger.handlers
    ):
        file_handler = _FileHandler(output_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

    return logger
