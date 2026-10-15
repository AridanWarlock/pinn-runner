"""Manufactured errors verify eigenmode forcing, boundary sign and time quadrature."""

import math
from functools import partial

import pytest
import torch

from benchmarks.modal_balance import HeatEigenmode, modal_heat_balance


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("robin", [False, True])
def test_modal_error_ode_matches_prescribed_amplitude(dimensions: int, robin: bool) -> None:
    frequency = math.pi / 2 if robin else math.pi
    shift = 0.5 if robin else 0
    alpha = frequency if robin else 0
    mode = HeatEigenmode((frequency,) * dimensions, (shift,) * dimensions)

    def reference(spatial: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        background = torch.ones_like(time)
        for x in spatial:
            background = background + x * time
        return background + mode.spatial(spatial) * time.mul(-mode.decay_rate).exp()

    def field(spatial: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        return reference(spatial, time) + 0.02 * time.square() * mode.spatial(spatial)

    def source(spatial: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        value = torch.zeros_like(time)
        for x in spatial:
            value = value + x
        return value

    def target(
        spatial: list[torch.Tensor], time: torch.Tensor, axis: int, side: int
    ) -> torch.Tensor:
        background = torch.ones_like(time)
        for x in spatial:
            background = background + x * time
        return alpha * background + (-1 if side == 0 else 1) * time

    targets = tuple(
        (partial(target, axis=i, side=0), partial(target, axis=i, side=1))
        for i in range(dimensions)
    )
    result = modal_heat_balance(
        field,
        reference,
        source,
        targets,
        (((alpha, 1), (alpha, 1)),) * dimensions,
        mode,
        ((0, 1),) * dimensions,
        (0, 0.5),
        "cpu",
        space_points=9,
        time_points=41,
    )
    metrics = result.metrics
    expected = 0.02 * 0.5**2 * math.exp(mode.decay_rate * 0.5)
    assert metrics["final_signed_relative_amplitude_error"] == pytest.approx(
        expected, rel=1e-4, abs=2e-7
    )
    assert metrics["ode_identity_rmse"] < 3e-6
    assert metrics["max_boundary_forcing"] < 3e-6
    # Composite trapezoid bound: T*h^2/12 * max |(exp(lambda*t)*drive)''|.
    rate = mode.decay_rate
    second_derivative_bound = (
        0.02 * math.exp(rate * 0.5) * (6 * rate + 6 * rate**2 * 0.5 + rate**3 * 0.5**2)
    )
    temporal_bound = 0.5 * (0.5 / 40) ** 2 / 12 * second_derivative_bound
    assert metrics["reconstruction_max_error"] < temporal_bound + 1e-6


def test_robin_boundary_defect_contributes_with_positive_sign() -> None:
    frequency = math.pi / 2
    mode = HeatEigenmode((frequency,), (0.5,))

    def reference(spatial: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        return mode.spatial(spatial) * time.mul(-mode.decay_rate).exp()

    def field(spatial: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        return reference(spatial, time) + 0.1 * time

    def zero(spatial: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(time)

    result = modal_heat_balance(
        field,
        reference,
        zero,
        ((zero, zero),),
        (((frequency, 1), (frequency, 1)),),
        mode,
        ((0, 1),),
        (0, 0.5),
        "cpu",
        space_points=13,
        time_points=41,
    )
    assert result.metrics["max_boundary_forcing"] > 0.01
    assert result.metrics["ode_identity_rmse"] < 1e-6
    assert result.metrics["reconstruction_max_error"] < 1e-6
    assert result.rows[-1]["boundary_forcing"] > 0


def test_incompatible_eigenmode_is_rejected() -> None:
    mode = HeatEigenmode((1,), (0,))

    def zero(spatial: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        return spatial[0] * 0 + time * 0

    with pytest.raises(ValueError, match="eigenmode does not satisfy"):
        modal_heat_balance(
            zero,
            zero,
            zero,
            ((zero, zero),),
            (((0, 1), (0, 1)),),
            mode,
            ((0, 1),),
            (0, 0.5),
            "cpu",
            space_points=3,
            time_points=3,
        )
