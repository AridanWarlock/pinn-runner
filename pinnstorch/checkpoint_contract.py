"""Validate the named input domain of a checkpoint before using its weights."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

InputBounds = tuple[tuple[float, float], ...]


def validate_checkpoint_domain(
    checkpoint: Mapping[str, Any],
    input_names: tuple[str, ...],
    requested_bounds: Sequence[tuple[float, float]],
    allow_extrapolation: bool = False,
) -> InputBounds:
    """Return the original normalization bounds after validating a query.

    Older runner checkpoints can recover bounds from FCN buffers. Input names
    must still match: tensor width alone cannot prove semantic compatibility.
    """
    metadata = checkpoint.get("hyper_parameters", {})
    saved_names = tuple(metadata.get("input_names", ()))
    if saved_names != input_names:
        raise ValueError(
            f"Checkpoint input names {saved_names} do not match task inputs {input_names}"
        )
    bounds = metadata.get("input_bounds")
    if not bounds:
        state = checkpoint.get("state_dict", {})
        if "net.lb" not in state or "net.ub" not in state:
            raise ValueError("Checkpoint does not record input bounds")
        bounds = tuple(zip(state["net.lb"].cpu().tolist(), state["net.ub"].cpu().tolist()))
    array = np.asarray(bounds, dtype=np.float64)
    requested = np.asarray(requested_bounds, dtype=np.float64)
    if (
        array.shape != (len(input_names), 2)
        or not np.isfinite(array).all()
        or np.any(array[:, 0] >= array[:, 1])
    ):
        raise ValueError("Checkpoint input bounds must be finite increasing pairs for every input")
    if requested.shape != array.shape or not np.isfinite(requested).all():
        raise ValueError("Requested input bounds do not match the checkpoint dimensions")
    if not allow_extrapolation:
        for name, (lower, upper), (query_lower, query_upper) in zip(input_names, array, requested):
            # FCN stores float32 buffers; allow only their rounding at endpoints.
            tolerance = 4 * np.finfo(np.float32).eps * max(abs(lower), abs(upper), 1.0)
            if query_lower < lower - tolerance or query_upper > upper + tolerance:
                raise ValueError(
                    f"Requested '{name}' range [{query_lower}, {query_upper}] lies outside "
                    f"checkpoint range [{lower}, {upper}]; set prediction.allow_extrapolation "
                    "to true only for an intentional prediction outside the trained domain"
                )
    return tuple((float(lower), float(upper)) for lower, upper in array)
