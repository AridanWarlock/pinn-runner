"""Resolve a requested backward policy without changing the physical problem."""

from typing import Literal, cast

from .settings import Runtime, Training


def resolve_backward(training: Training, runtime: Runtime) -> Literal["all", "parameters"]:
    requested = training.backward
    if requested not in ("auto", "all", "parameters"):
        raise ValueError("training.backward must be auto, all or parameters")
    count = len(runtime.devices) if isinstance(runtime.devices, tuple) else runtime.devices
    supported = runtime.device in ("cpu", "cuda") and runtime.precision == "32-true" and count == 1
    if requested == "auto":
        # Automatic adoption is deliberately scoped to the validated transform.
        return (
            "parameters" if supported and training.constraints == "hard_initial_robin" else "all"
        )
    if requested == "parameters" and not supported:
        raise ValueError("Parameter backward requires 32-true precision and one CPU/CUDA device")
    return cast(Literal["all", "parameters"], requested)
