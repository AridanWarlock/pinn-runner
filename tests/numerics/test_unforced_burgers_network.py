"""Actual unforced networks preserve physical and every-weight derivatives."""

import os
from pathlib import Path

import pytest
import torch

from benchmarks.catalog import UNFORCED_VISCOSITIES
from scripts.audit_nonheat import audit
from tests.numerics.test_unforced_burgers_reference import (
    test_reference_independent_kernel_and_physical_domain as check_reference,
)


@pytest.mark.parametrize("case", ["burgers_unforced_nu010", "burgers_unforced_nu001"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [11, 29])
def test_unforced_operator_and_every_weight(
    tmp_path: Path, case: str, dtype: torch.dtype, seed: int
) -> None:
    check_reference(case, UNFORCED_VISCOSITIES[case])
    result = audit(
        tmp_path / "audit",
        case,
        dtype,
        seed,
        os.environ.get("PINN_TEST_DEVICE", "cpu"),
        "hard_initial_dirichlet",
    )
    assert result["parameters"] == 2241 and result["state_unchanged"]
    assert result["initial_max"] < 2e-6 and result["boundary_max"] < 2e-6
    assert result["detached_gradient_relative_l2"] > 1e-4
    assert (result["finite_difference_error"] is not None) == (dtype == torch.float64)
