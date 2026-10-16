"""Fingerprint content, layout independence, and ordered objective identity."""

import os

import torch

from benchmarks.fingerprints import objective_fingerprint, tensor_fingerprint
from pinnstorch.contracts import SamplerBatch


def test_tensor_identity_ignores_layout_and_preserves_gradient() -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    value = torch.arange(12.0, device=device).reshape(3, 4).requires_grad_()
    state = {"weights": value.T, "integer": torch.tensor(7), "half": value.bfloat16()}
    clone = {name: tensor.cpu().clone() for name, tensor in reversed(list(state.items()))}
    assert tensor_fingerprint(state) == tensor_fingerprint(clone)
    value.square().sum().backward()
    torch.testing.assert_close(value.grad, 2 * value)


def test_tensor_identity_includes_names_shapes_and_dtypes() -> None:
    reference = torch.ones(2, 2)
    expected = tensor_fingerprint({"weights": reference})
    for alternate in (
        {"other": reference},
        {"weights": reference.flatten()},
        {"weights": reference.double()},
        {"weights": reference + 1},
    ):
        assert tensor_fingerprint(alternate) != expected


def test_objective_identity_includes_point_order_targets_and_condition() -> None:
    x, t = torch.arange(3.0)[:, None], torch.zeros(3, 1)
    batch: SamplerBatch = ([x], t, {"temperature": x.square()})
    expected = objective_fingerprint({"equations": batch})
    assert expected == objective_fingerprint({"equations": ([x.clone()], t.clone(), batch[2])})
    alternatives: list[dict[str, SamplerBatch]] = [
        {"equations": ([x.flip(0)], t, batch[2])},
        {"equations": ([x], t + 1, batch[2])},
        {"equations": ([x], t, {"temperature": x})},
        {"boundary": batch},
    ]
    for alternate in alternatives:
        assert objective_fingerprint(alternate) != expected
