"""Real transport/Burgers networks validate every weight on the selected device."""

import os
from pathlib import Path

import pytest
import torch

from scripts.audit_nonheat import audit


@pytest.mark.parametrize("case", ["advection", "burgers"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [19, 73])
def test_nonheat_network_operator_and_every_weight(
    tmp_path: Path, case: str, dtype: torch.dtype, seed: int
) -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    result = audit(tmp_path / "audit", case, dtype, seed, device)
    assert result["device"] == device
    assert result["parameters"] == 2241
    assert result["state_unchanged"]
    assert (result["finite_difference_error"] is not None) == (dtype == torch.float64)
    if case == "advection":
        assert result["unused_constant_bias"] == "net.model.output.bias"
        assert result["detached_gradient_relative_l2"] is None
    else:
        assert result["unused_constant_bias"] == ""
        assert result["detached_gradient_relative_l2"] > 1e-4
