"""The public file lifecycle on the unforced physical domain, with exact data."""

from pathlib import Path

import pytest

from tests.integration.test_nonheat_runner import (
    test_nonheat_train_exact_load_predict_and_retrain as lifecycle,
)

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("case", ["burgers_unforced_nu010", "burgers_unforced_nu001"])
def test_unforced_public_lifecycle(tmp_path: Path, case: str) -> None:
    lifecycle(tmp_path, case)
