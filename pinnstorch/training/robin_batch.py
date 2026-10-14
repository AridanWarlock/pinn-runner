"""Same tensor-product projection with one network batch for boundary strata."""

from dataclasses import dataclass, field
from itertools import combinations, product
from math import prod

import torch

from pinnstorch.contracts import TensorOutputs
from pinnstorch.training.robin_box import BoxRobinProjection
from pinnstorch.training.robin_initial import NetworkField


@dataclass(frozen=True)
class BatchedBoxRobinProjection(BoxRobinProjection):
    nodes: tuple[tuple[int, ...], ...] = field(init=False)
    derivatives: tuple[tuple[int, ...], ...] = field(init=False)
    coefficients: tuple[tuple[float, ...], ...] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        nodes = tuple(node for node in product((0, 1, 2), repeat=len(self.axes)) if any(node))
        derivatives = tuple(
            subset
            for degree in range(len(self.axes) + 1)
            for subset in combinations(range(len(self.axes)), degree)
        )
        coefficients = tuple(
            tuple(self._coefficient(node, subset) for node in nodes) for subset in derivatives
        )
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "derivatives", derivatives)
        object.__setattr__(self, "coefficients", coefficients)

    def _coefficient(self, node: tuple[int, ...], derivatives: tuple[int, ...]) -> float:
        if any(node[axis] == 0 for axis in derivatives):
            return 0.0
        factors = []
        for axis, state in enumerate(node):
            if state == 0:
                continue
            operator = (self.axes[axis].lower, self.axes[axis].upper)[state - 1]
            factors.append(
                operator.beta * (-1 if state == 1 else 1)
                if axis in derivatives
                else operator.alpha
            )
        return float((-1) ** sum(state != 0 for state in node) * prod(factors))

    def _batch(
        self, spatial: list[torch.Tensor], time: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        fixed = {
            (axis, state): torch.full_like(
                x, self.axes[axis].bounds[state - 1], requires_grad=True
            )
            for axis, x in enumerate(spatial)
            for state in (1, 2)
        }
        columns = [
            torch.cat([x if node[axis] == 0 else fixed[axis, node[axis]] for node in self.nodes])
            for axis, x in enumerate(spatial)
        ]
        return columns, torch.cat([time] * len(self.nodes))

    def _weights(self, spatial: list[torch.Tensor], time: torch.Tensor) -> list[torch.Tensor]:
        bases = [axis.basis(x) for axis, x in zip(self.axes, spatial)]
        cache: dict[tuple[tuple[int, int], ...], torch.Tensor] = {(): torch.ones_like(time)}

        def weight(key: tuple[tuple[int, int], ...]) -> torch.Tensor:
            if key not in cache:
                axis, state = key[-1]
                cache[key] = weight(key[:-1]).mul(bases[axis][state - 1])
            return cache[key]

        return [
            weight(tuple((axis, state) for axis, state in enumerate(node) if state))
            for node in self.nodes
        ]

    def apply(
        self,
        outputs: TensorOutputs,
        field: NetworkField,
        spatial: list[torch.Tensor],
        time: torch.Tensor,
    ) -> TensorOutputs:
        tracking = torch.is_grad_enabled()
        with torch.set_grad_enabled(True):
            columns, clock = self._batch(spatial, time)
            boundary = field(columns, clock)
            weights = torch.stack(self._weights(spatial, time))
            coefficients = weights.new_tensor(self.coefficients)
            result: TensorOutputs = {}
            for name, value in outputs.items():
                jets: dict[tuple[int, ...], torch.Tensor] = {(): boundary[name]}
                correction = torch.zeros_like(value)
                for index, subset in enumerate(self.derivatives):
                    if subset:
                        parent = jets[subset[:-1]]
                        derivative = (
                            torch.autograd.grad(
                                parent,
                                columns[subset[-1]],
                                torch.ones_like(parent),
                                create_graph=True,
                                retain_graph=True,
                                allow_unused=True,
                            )[0]
                            if parent.requires_grad
                            else None
                        )
                        jets[subset] = (
                            columns[subset[-1]].mul(0) if derivative is None else derivative
                        )
                    # Coefficients are fixed physics; keep weights and all network jets differentiable.
                    coefficient = coefficients[index].reshape(len(self.nodes), 1, 1)
                    values = jets[subset].reshape(len(self.nodes), *value.shape)
                    correction = correction.add((values * weights * coefficient).sum(dim=0))
                result[name] = value.add(correction)
        return result if tracking else {name: value.detach() for name, value in result.items()}
