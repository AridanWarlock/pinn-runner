"""Compare formula and table coefficients through the same CPU PDE/backward path."""

from __future__ import annotations

import statistics
import time

import torch

from pinnstorch.api import Coordinates, Equation
from pinnstorch.coefficients import RegularGridCoefficient
from pinnstorch.models import FCN
from pinnstorch.runner.coefficients import CoefficientBank


def conductivity(c: Coordinates) -> torch.Tensor:
    return 1 + c.x + c.y


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    network = FCN([3, 32, 32, 1], [0, 0, 0], [1, 1, 1], ("T",))
    source = [torch.rand(1024, 1) for _ in range(3)]
    x, y = torch.linspace(0, 1, 65), torch.linspace(0, 1, 49)
    table = RegularGridCoefficient({"x": x, "y": y}, 1 + x[:, None] + y[None, :])
    banks = {
        "formula": CoefficientBank({"k": conductivity}),
        "table": CoefficientBank({"k": table}),
    }

    def evaluate(bank: CoefficientBank) -> tuple[torch.Tensor, torch.Tensor]:
        columns = [column.detach().requires_grad_() for column in source]
        c = bank.context(Coordinates({"x": columns[0], "y": columns[1]}, columns[2]))
        network.zero_grad(set_to_none=True)
        e = Equation(network(columns[:-1], columns[-1]), c)
        flux = {
            axis: e.coefficients["k"] * derivative for axis, derivative in e.gradient("T").items()
        }
        residual = e.dt("T") - e.divergence(flux)
        residual.square().mean().backward()
        parameters = list(network.parameters())
        assert any(parameter.grad is not None for parameter in parameters)
        # PDE derivatives do not depend on a constant output bias. Treat its missing
        # gradient as zero when comparing the two coefficient representations.
        gradients = [
            parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            for parameter in parameters
        ]
        return residual.detach(), torch.cat([gradient.reshape(-1) for gradient in gradients])

    expected, expected_gradient = evaluate(banks["formula"])
    actual, actual_gradient = evaluate(banks["table"])
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-4, atol=2e-5)
    measurements: dict[str, float] = {}
    for name, bank in banks.items():
        for _ in range(5):
            evaluate(bank)
        runs: list[float] = []
        for _ in range(5):
            started = time.perf_counter()
            for _ in range(50):
                evaluate(bank)
            runs.append((time.perf_counter() - started) / 50)
        measurements[name] = statistics.median(runs)
        print(f"{name}: {measurements[name] * 1000:.3f} ms/step")
    print(f"table/formula: {measurements['table'] / measurements['formula']:.3f}")
    print(
        "CPU only, 1024 points; coefficient + PDE derivatives + backward; all weight gradients checked."
    )


if __name__ == "__main__":
    main()
