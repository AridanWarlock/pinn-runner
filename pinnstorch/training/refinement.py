"""Fixed-objective L-BFGS stage, separated from stochastic Lightning training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import torch

from pinnstorch.contracts import SamplerBatch
from pinnstorch.data.pinn_datamodule import PINNDataModule
from pinnstorch.models.pinn_module import PINNModule


@dataclass(frozen=True)
class FixedObjective:
    batches: dict[str, SamplerBatch]

    @classmethod
    def capture(cls, data: PINNDataModule, device: torch.device) -> FixedObjective:
        batches: dict[str, SamplerBatch] = {}
        for name, dataset in zip(data.condition_names, data.train_datasets):
            spatial, clock, targets = dataset[:]
            batches[name] = (
                [column.detach().to(device).clone() for column in spatial],
                clock.detach().to(device).clone(),
                {key: value.detach().to(device).clone() for key, value in targets.items()},
            )
        return cls(batches)

    def fresh(self) -> dict[str, SamplerBatch]:
        # Values stay fixed, but coordinate leaf gradients must not accumulate.
        return {
            name: ([column.clone() for column in spatial], clock.clone(), targets)
            for name, (spatial, clock, targets) in self.batches.items()
        }

    def to(self, device: torch.device) -> FixedObjective:
        return FixedObjective(
            {
                name: (
                    [column.to(device) for column in spatial],
                    clock.to(device),
                    {key: value.to(device) for key, value in targets.items()},
                )
                for name, (spatial, clock, targets) in self.batches.items()
            }
        )


class LBFGSRefinement:
    def __init__(
        self,
        model: PINNModule,
        objective: FixedObjective,
        learning_rate: float,
        history_size: int,
        tolerance_grad: float,
        tolerance_change: float,
    ) -> None:
        self.model, self.objective = model, objective
        self.optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=learning_rate,
            history_size=history_size,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            max_iter=1,
            line_search_fn="strong_wolfe",
        )
        self.steps = 0
        self.closures = 0

    def closure(self) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            loss, _ = self.model.model_step(self.objective.fresh())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite L-BFGS objective")
            torch.autograd.backward(loss)
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in self.model.parameters()
        ):
            raise FloatingPointError("Nonfinite L-BFGS gradient")
        self.closures += 1
        return loss.detach()

    def run(self, steps: int) -> dict[str, int | float]:
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        started = time.perf_counter()
        for _ in range(steps):
            self.optimizer.step(self.closure)
            self.steps += 1
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        first: dict[str, Any] = next(iter(self.optimizer.state.values()), {})
        return {
            "lbfgs_seconds": time.perf_counter() - started,
            "lbfgs_steps": self.steps,
            "lbfgs_internal_iterations": int(first.get("n_iter", 0)),
            "lbfgs_closure_evaluations": self.closures,
            "lbfgs_condition_evaluations": self.closures * len(self.objective.batches),
            "lbfgs_residual_points": self.closures * len(self.objective.batches["equations"][1]),
        }

    def state_dict(self, metrics: dict[str, int | float]) -> dict[str, Any]:
        return {
            "kind": "lbfgs",
            "version": 1,
            "completed": True,
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics,
            "fixed_batches": self.objective.batches,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("kind") != "lbfgs" or state.get("version") != 1:
            raise ValueError("Unsupported L-BFGS stage state")
        self.optimizer.load_state_dict(state["optimizer"])
        self.objective = FixedObjective(cast(dict[str, SamplerBatch], state["fixed_batches"])).to(
            self.model.device
        )
        self.steps = int(state["metrics"]["lbfgs_steps"])
        self.closures = int(state["metrics"]["lbfgs_closure_evaluations"])
