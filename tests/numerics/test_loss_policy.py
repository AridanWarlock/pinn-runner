"""Objective weights and diagnostics must preserve optimizer gradients."""

import pytest
import torch

from pinnstorch.runner.settings import LossSettings, Reader
from pinnstorch.training.loss import EqualWeights, FixedWeights, gradient_norms


def test_equal_and_fixed_objectives_and_gradients() -> None:
    parameter = torch.tensor(2.0, requires_grad=True)
    terms = {"initial": parameter.square(), "boundaries": (parameter - 1).square()}
    equal = EqualWeights().combine(terms)
    fixed = FixedWeights({"boundaries": 3}).combine(terms)
    assert equal.item() == 5
    assert fixed.item() == 7
    assert torch.autograd.grad(equal, parameter, retain_graph=True)[0].item() == 6
    assert torch.autograd.grad(fixed, parameter)[0].item() == 10
    with pytest.raises(ValueError, match="at least"):
        EqualWeights().combine({})


def test_diagnostic_norms_preserve_existing_grads_and_backward() -> None:
    parameter = torch.tensor(2.0, requires_grad=True)
    unused = torch.tensor(5.0, requires_grad=True)
    parameter.grad = torch.tensor(7.0)
    terms = {"initial": parameter.square(), "equations": (parameter - 1).square()}
    norms = gradient_norms(terms, [parameter, unused])
    assert norms["initial"].item() == 4
    assert norms["equations"].item() == 2
    assert parameter.grad.item() == 7
    assert unused.grad is None
    EqualWeights().combine(terms).backward()
    assert parameter.grad.item() == 13


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_weights_fail_early(value: object) -> None:
    with pytest.raises(ValueError):
        LossSettings.read(Reader({"kind": "fixed", "boundaries": value}, "training.loss"))


def test_loss_recipe_defaults_and_strict_keys() -> None:
    assert LossSettings.read(Reader({}, "loss")) == LossSettings()
    assert LossSettings.read(Reader({"kind": "fixed", "boundaries": 3}, "loss")).boundaries == 3
    for mapping in (
        {"kind": "equal", "boundaries": 3},
        {"target": "python"},
        {"gradient_norm_every": None},
    ):
        with pytest.raises(ValueError):
            LossSettings.read(Reader(mapping, "loss"))


def test_benchmark_adapters_do_not_change_control_or_pde() -> None:
    import numpy as np

    from benchmarks.cases import CASES
    from benchmarks.variants import BenchmarkVariant, BoundaryWeight

    base = CASES["advection"]
    control = BenchmarkVariant(base.name, base.dimensions, base=base, adapter=BoundaryWeight(1))
    variant = BenchmarkVariant(base.name, base.dimensions, base=base, adapter=BoundaryWeight(3))
    assert control.config("cpu", 3, 19) == base.config("cpu", 3, 19)
    configured = variant.config("cpu", 3, 19)
    assert configured["training"].pop("loss")["boundaries"] == 3
    assert configured == base.config("cpu", 3, 19)
    spatial = {"x": np.array([[0.0], [1.0]])}
    np.testing.assert_array_equal(variant.boundary_mask(spatial), base.boundary_mask(spatial))
