"""Keep physical coordinates and named model parameters separate in user callbacks."""

from collections.abc import Callable, Sequence

import torch

from pinnstorch.contracts import (
    ArrayOutputs,
    FieldFunction,
    FloatArray,
    TensorOutputs,
    UserFunctions,
)


def bind_parameters(
    functions: UserFunctions, spatial_dimension: int, names: Sequence[str]
) -> UserFunctions:
    """Adapt combined network features to callbacks with a parameters keyword.

    The sampler batch stores spatial and parameter columns together. This
    adapter exposes physical coordinates separately; parameters remain fixed
    inputs to a forward problem, never trainable unknown coefficients.
    """
    if not names:
        return functions

    def wrap_field(function: FieldFunction) -> FieldFunction:
        def wrapped(outputs: TensorOutputs, *columns: torch.Tensor) -> TensorOutputs:
            spatial = columns[:spatial_dimension]
            parameters = dict(zip(names, columns[spatial_dimension:-1]))
            return function(outputs, *spatial, columns[-1], parameters=parameters)

        return wrapped

    def wrap_boundary(function: FieldFunction) -> FieldFunction:
        def wrapped(
            outputs: TensorOutputs, columns: list[torch.Tensor], time: torch.Tensor
        ) -> TensorOutputs:
            parameters = dict(zip(names, columns[spatial_dimension:]))
            return function(outputs, columns[:spatial_dimension], time, parameters=parameters)

        return wrapped

    initial: Callable[..., ArrayOutputs] = functions["initial_conditions"]

    def wrapped_initial(columns: FloatArray) -> ArrayOutputs:
        parameters = {
            name: columns[:, spatial_dimension + i : spatial_dimension + i + 1]
            for i, name in enumerate(names)
        }
        return initial(columns[:, :spatial_dimension], parameters=parameters)

    result: UserFunctions = {
        "pde_fn": wrap_field(functions["pde_fn"]),
        "initial_conditions": wrapped_initial,
    }
    if "output_fn" in functions:
        result["output_fn"] = wrap_field(functions["output_fn"])
    if "boundary_functions" in functions:
        result["boundary_functions"] = {
            name: wrap_boundary(function)
            for name, function in functions["boundary_functions"].items()
        }
    if "boundary_values" in functions:

        def wrap_value(function: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
            def wrapped(columns: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
                parameters = dict(zip(names, columns[spatial_dimension:]))
                return function(columns[:spatial_dimension], time, parameters=parameters)

            return wrapped

        result["boundary_values"] = {
            name: wrap_value(function) for name, function in functions["boundary_values"].items()
        }
    if "plot_fn" in functions:
        result["plot_fn"] = functions["plot_fn"]
    return result
