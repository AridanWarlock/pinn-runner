"""Sample initial operator compatibility and differentiated data at box intersections."""

from itertools import combinations

import torch

from pinnstorch.contracts import TensorOutputs
from pinnstorch.training.robin_box import BoxRobinInitial

from .dirichlet import require_matching
from .settings import TaskSettings


def validation_points(
    settings: TaskSettings, fixed: dict[int, int], initial: bool
) -> tuple[list[torch.Tensor], torch.Tensor]:
    columns = [
        torch.tensor([axis.interval[fixed[i]]], dtype=torch.float64)
        if i in fixed
        else torch.linspace(*axis.interval, axis.points, dtype=torch.float64)
        for i, axis in enumerate(settings.domain.space.values())
    ]
    clock = settings.domain.time
    columns.append(
        torch.tensor([clock.interval[0]], dtype=torch.float64)
        if initial
        else torch.linspace(*clock.interval, clock.points, dtype=torch.float64)
    )
    mesh = torch.meshgrid(*columns, indexing="ij")
    flattened = [column.reshape(-1, 1) for column in mesh]
    return flattened[:-1], flattened[-1]


def validate_box_robin(settings: TaskSettings, recipe: BoxRobinInitial) -> None:
    with torch.set_grad_enabled(True):
        for axis, targets in enumerate(recipe.targets):
            for side, target in enumerate(targets):
                spatial, time = validation_points(settings, {axis: side}, initial=True)
                actual = recipe.projection.traces(recipe.initial, spatial, time, axis)[side]
                require_matching(
                    actual[recipe.field], target(spatial, time), "initial Robin/Neumann operator"
                )
        for left, right in combinations(range(len(recipe.targets)), 2):
            for left_side in (0, 1):
                for right_side in (0, 1):
                    spatial, time = validation_points(
                        settings, {left: left_side, right: right_side}, initial=False
                    )

                    def left_data(
                        columns: list[torch.Tensor], clock: torch.Tensor
                    ) -> TensorOutputs:
                        return {recipe.field: recipe.targets[left][left_side](columns, clock)}

                    def right_data(
                        columns: list[torch.Tensor], clock: torch.Tensor
                    ) -> TensorOutputs:
                        return {recipe.field: recipe.targets[right][right_side](columns, clock)}

                    left_then_right = recipe.projection.traces(left_data, spatial, time, right)[
                        right_side
                    ]
                    right_then_left = recipe.projection.traces(right_data, spatial, time, left)[
                        left_side
                    ]
                    require_matching(
                        left_then_right[recipe.field],
                        right_then_left[recipe.field],
                        "Robin/Neumann edge operator compatibility",
                    )
