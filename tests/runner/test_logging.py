"""Exercise real logging handlers with only Lightning's decorator mocked."""

import importlib.util
import logging
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest


@pytest.fixture
def logger_module() -> Iterator[tuple[ModuleType, Mock, str]]:
    path = Path(__file__).resolve().parents[2] / "pinnstorch/utils/pylogger.py"
    spec = importlib.util.spec_from_file_location("logger_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    decorator = Mock(side_effect=lambda method: method)
    utilities = ModuleType("lightning.pytorch.utilities")
    utilities.rank_zero_only = decorator
    with patch.dict("sys.modules", {"lightning.pytorch.utilities": utilities}):
        spec.loader.exec_module(module)

    name = f"runner-tests.{uuid4()}"
    yield module, decorator, name

    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def test_console_logger_does_not_open_files(
    logger_module: tuple[ModuleType, Mock, str],
) -> None:
    module, _, name = logger_module
    with patch.object(logging.FileHandler, "_open", side_effect=AssertionError("File opened")):
        logger = module.get_pylogger(name)
    assert len(logger.handlers) == 1
    assert not isinstance(logger.handlers[0], logging.FileHandler)


def test_repeated_setup_does_not_duplicate_handlers_or_decorators(
    logger_module: tuple[ModuleType, Mock, str], tmp_path: Path
) -> None:
    module, decorator, name = logger_module
    output = tmp_path / "log.log"
    logger = module.get_pylogger(name)
    info_method = logger.info
    initial_decorator_calls = decorator.call_count
    logger.setLevel(logging.INFO)
    module.get_pylogger(name, output_path=output)
    module.get_pylogger(name, output_path=tmp_path / "." / "log.log")

    logger.info("one message")

    assert len(logger.handlers) == 2
    assert initial_decorator_calls > 0
    assert decorator.call_count == initial_decorator_calls
    assert logger.info is info_method
    assert output.read_text().count("one message") == 1


def test_keeps_existing_application_handler(
    logger_module: tuple[ModuleType, Mock, str],
) -> None:
    module, _, name = logger_module
    existing = logging.NullHandler()
    logging.getLogger(name).addHandler(existing)

    logger = module.get_pylogger(name)
    module.get_pylogger(name)

    assert existing in logger.handlers
    assert len(logger.handlers) == 2
