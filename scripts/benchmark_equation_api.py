"""Compare the problem operators with direct autograd on the same CPU network."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch

from pinnstorch.api import Coordinates, Equation
from pinnstorch.models import FCN
from pinnstorch.utils import gradient

Residual = Callable[[torch.Tensor, list[torch.Tensor]], torch.Tensor]


def direct(field: torch.Tensor, columns: list[torch.Tensor]) -> torch.Tensor:
    x, y, conductivity, time = columns
    dx, dy, dt = gradient(field, [x, y, time])
    return dt - conductivity * (gradient(dx, x)[0] + gradient(dy, y)[0])


def problem_language(field: torch.Tensor, columns: list[torch.Tensor]) -> torch.Tensor:
    x, y, conductivity, time = columns
    equation = Equation({"T": field}, Coordinates({"x": x, "y": y}, time, {"k": conductivity}))
    return equation.dt("T") - equation.parameters["k"] * equation.laplacian("T")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    options = parser.parse_args()
    if options.points < 1 or options.steps < 1:
        parser.error("points and steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    network = FCN([4, 32, 32, 1], [0, 0, 0.1, 0], [1, 1, 0.5, 1], ("T",))
    source = [torch.rand(options.points, 1) for _ in range(4)]
    source[2] = 0.1 + 0.4 * source[2]

    def evaluate(function: Residual) -> tuple[torch.Tensor, torch.Tensor]:
        columns = [column.detach().requires_grad_() for column in source]
        network.zero_grad(set_to_none=True)
        output = network(columns[:-1], columns[-1])["T"]
        residual = function(output, columns)
        residual.square().mean().backward()
        weight = next(network.parameters())
        assert weight.grad is not None
        return residual.detach(), weight.grad.detach().clone()

    expected, expected_gradient = evaluate(direct)
    actual, actual_gradient = evaluate(problem_language)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    for function in (direct, problem_language):
        for _ in range(5):
            evaluate(function)
    measurements: dict[str, float] = {}
    for name, function in (("direct", direct), ("problem", problem_language)):
        runs = []
        for _ in range(5):
            started = time.perf_counter()
            for _ in range(options.steps):
                evaluate(function)
            runs.append((time.perf_counter() - started) / options.steps)
        measurements[name] = statistics.median(runs)
    for name, elapsed in measurements.items():
        print(f"{name}: {elapsed * 1000:.3f} ms/step")
    print(f"problem/direct: {measurements['problem'] / measurements['direct']:.3f}")
    print("CPU only; forward + PDE derivatives + backward; not a CUDA throughput test.")


if __name__ == "__main__":
    main()
