"""Synchronized L-BFGS accounting; restoration stays outside measured intervals."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class StepTiming:
    seconds: float
    closure_seconds: float
    closure_calls: int
    optimizer_evaluations: int
    optimizer_iterations: int

    @property
    def outside_closure_seconds(self) -> float:
        return self.seconds - self.closure_seconds

    def to_dict(self) -> dict[str, float | int]:
        return {**asdict(self), "outside_closure_seconds": self.outside_closure_seconds}


def _counters(optimizer: torch.optim.LBFGS) -> tuple[int, int]:
    # PyTorch stores the global L-BFGS counters on its first parameter.
    first = optimizer.param_groups[0]["params"][0]
    state = optimizer.state.get(first, {})
    return int(state.get("func_evals", 0)), int(state.get("n_iter", 0))


def measure_steps(
    optimizer: torch.optim.LBFGS,
    closure: Callable[[], torch.Tensor],
    steps: int,
    synchronize: Callable[[], None],
) -> list[StepTiming]:
    """Measure real updates, including line search; the closure owns zero/backward.

    Use a no-op synchronizer on CPU, and device-specific CUDA synchronization
    remotely. Synchronization before a closure belongs to the outside interval;
    its completion barrier belongs to that closure. This is an instrumented
    attribution, not a substitute for an uninstrumented end-to-end measurement.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("A positive integer step count is required")
    if optimizer.defaults.get("line_search_fn") != "strong_wolfe":
        raise ValueError("The registered stage uses strong_wolfe line search")
    result = []
    for _ in range(steps):
        closure_seconds = 0.0
        calls = 0

        def timed_closure() -> torch.Tensor:
            nonlocal closure_seconds, calls
            synchronize()
            started = time.perf_counter()
            loss = closure()
            if loss.numel() != 1 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("Expected a finite scalar closure loss")
            synchronize()
            closure_seconds += time.perf_counter() - started
            calls += 1
            return loss

        before_evaluations, before_iterations = _counters(optimizer)
        synchronize()
        started = time.perf_counter()
        optimizer.step(timed_closure)
        synchronize()
        seconds = time.perf_counter() - started
        evaluations, iterations = _counters(optimizer)
        row = StepTiming(
            seconds,
            closure_seconds,
            calls,
            evaluations - before_evaluations,
            iterations - before_iterations,
        )
        if calls < 1 or row.optimizer_evaluations != calls:
            raise ValueError("Closure counter differs from the optimizer")
        if not math.isfinite(seconds) or not 0 <= closure_seconds <= seconds:
            raise ValueError("Invalid nonoverlapping timing intervals")
        result.append(row)
    return result


def measure_total(
    optimizer: torch.optim.LBFGS,
    closure: Callable[[], torch.Tensor],
    steps: int,
    synchronize: Callable[[], None],
) -> dict[str, float | int]:
    """Separate timing without extra barriers or checks inside the closure."""
    if type(steps) is not int or steps < 1:
        raise ValueError("A positive integer step count is required")
    before_evaluations, before_iterations = _counters(optimizer)
    synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.step(closure)
    synchronize()
    seconds = time.perf_counter() - started
    evaluations, iterations = _counters(optimizer)
    return dict(
        steps=steps,
        seconds=seconds,
        closure_calls=evaluations - before_evaluations,
        optimizer_iterations=iterations - before_iterations,
    )


@contextmanager
def preserved_state(model: nn.Module, optimizer: torch.optim.LBFGS) -> Iterator[None]:
    """Restore weights, buffers, gradients, optimizer history, modes and RNG.

    Intended for a deterministic benchmark with fixed coordinates and pure
    callbacks. External callback state, input mutations and extra stage counters
    are not restored; create a fresh refinement stage for each measured trial.
    CUDA RNG state is captured only for devices used by the model.
    """
    parameters = tuple(model.parameters())
    known = {id(p) for p in parameters}
    if any(id(p) not in known for group in optimizer.param_groups for p in group["params"]):
        raise ValueError("Optimizer parameters must belong to the preserved model")
    weights = copy.deepcopy(model.state_dict())
    history = copy.deepcopy(optimizer.state_dict())
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    modes = [(module, module.training) for module in model.modules()]
    cpu_rng = torch.get_rng_state()
    devices = {t.device for t in (*parameters, *model.buffers()) if t.device.type == "cuda"}
    cuda_rng = {device: torch.cuda.get_rng_state(device) for device in devices}
    try:
        yield
    finally:
        model.load_state_dict(weights, strict=True)
        optimizer.load_state_dict(history)
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient
        for module, mode in modes:
            module.training = mode
        torch.set_rng_state(cpu_rng)
        for device, rng in cuda_rng.items():
            torch.cuda.set_rng_state(rng, device)


def summarize(rows: list[StepTiming]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty measurements")
    seconds = sum(row.seconds for row in rows)
    closure = sum(row.closure_seconds for row in rows)
    calls = sum(row.closure_calls for row in rows)
    return dict(
        steps=len(rows),
        seconds=seconds,
        closure_seconds=closure,
        outside_closure_seconds=seconds - closure,
        closure_calls=calls,
        optimizer_evaluations=sum(row.optimizer_evaluations for row in rows),
        optimizer_iterations=sum(row.optimizer_iterations for row in rows),
        seconds_per_closure=closure / calls,
        rows=[row.to_dict() for row in rows],
        instrumented=True,
    )
