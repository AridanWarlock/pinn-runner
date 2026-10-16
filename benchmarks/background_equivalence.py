"""Replay one physical objective at identical weights and reconstructed collocation points."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from pinnstorch.api import Problem
from pinnstorch.models.pinn_module import PINNModule
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.settings import TaskSettings


@dataclass(frozen=True)
class ObjectiveObservation:
    loss: torch.Tensor
    field: torch.Tensor
    residual: torch.Tensor
    gradient: torch.Tensor


class BackgroundEquivalence:
    def __init__(
        self, settings: TaskSettings, problem: Problem, output: Path, source: Path
    ) -> None:
        self.settings = settings
        self.fields = tuple(problem.fields)
        self.equations = tuple(problem.equations)
        self.models: dict[str, PINNModule] = {}
        self.collocation_hashes: dict[str, str] = {}
        self.points: torch.Tensor | None = None
        for background in ("general", "static"):
            selected = replace(
                settings, training=replace(settings.training, hard_background=background)
            )
            factory = SolverFactory(selected, problem, output / background, source)
            objects = factory.build(predicting=False)
            objects.data.setup("fit")
            objects.model.function_mapping = objects.data.function_mapping
            self.models[background] = objects.model
            index = objects.data.condition_names.index("equations")
            dataset = objects.data.train_datasets[index]
            points = torch.cat(
                [*dataset.spatial_domain_sampled, dataset.time_domain_sampled], dim=1
            )
            self.collocation_hashes[background] = hashlib.sha256(
                points.numpy().tobytes()
            ).hexdigest()
            if self.points is None:
                self.points = points
            elif not torch.equal(self.points, points):
                raise ValueError("The two backgrounds reconstructed different collocation points")
        self.initial_state = {
            name: value.detach().cpu().clone()
            for name, value in self.models["general"].state_dict().items()
        }

    def observe(self, model: PINNModule, points: torch.Tensor) -> ObjectiveObservation:
        model.zero_grad(set_to_none=True)
        columns = [points[:, i : i + 1].detach().requires_grad_() for i in range(points.shape[1])]
        loss, physical = model.model_step({"equations": (columns[:-1], columns[-1], {})})
        # Keep values before terminal backward frees the derivative graph.
        value = torch.cat([physical[name] for name in self.fields], dim=1).detach().double().cpu()
        residual = (
            torch.cat([physical[name] for name in self.equations], dim=1).detach().double().cpu()
        )
        model.backward(loss)
        gradients = []
        for parameter in model.parameters():
            if parameter.requires_grad:
                if parameter.grad is None:
                    raise ValueError(
                        "Objective did not produce every trainable parameter gradient"
                    )
                gradients.append(parameter.grad.detach().flatten().double().cpu())
        return ObjectiveObservation(
            loss.detach().double().cpu(), value, residual, torch.cat(gradients)
        )

    def compare(
        self, count: int, dtype: torch.dtype, state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self.points is None or count < 1 or count > len(self.points):
            raise ValueError("Probe count must fit the reconstructed objective")
        weights = self.initial_state if state is None else state
        points = self.points[:count].to(device=self.settings.runtime.device, dtype=dtype)
        for model in self.models.values():
            model.to(device=points.device, dtype=dtype).eval()
            model.load_state_dict(weights)
        for _ in range(2):
            for model in self.models.values():
                self.observe(model, points)
        reference = self.observe(self.models["general"], points)
        reference_repeat = self.observe(self.models["general"], points)
        candidate = self.observe(self.models["static"], points)
        candidate_repeat = self.observe(self.models["static"], points)
        difference = candidate.gradient - reference.gradient
        norm = float(reference.gradient.norm())
        return {
            "points": count,
            "dtype": str(dtype),
            "general_loss": float(reference.loss),
            "static_loss": float(candidate.loss),
            "loss_absolute_difference": float((candidate.loss - reference.loss).abs()),
            "field_max_absolute_difference": float(
                (candidate.field - reference.field).abs().max()
            ),
            "residual_max_absolute_difference": float(
                (candidate.residual - reference.residual).abs().max()
            ),
            "gradient_max_absolute_difference": float(difference.abs().max()),
            "gradient_relative_l2_difference": float(difference.norm()) / norm if norm else None,
            "gradients_exact": torch.equal(reference.gradient, candidate.gradient),
            "general_self_repeat_exact": torch.equal(
                reference.gradient, reference_repeat.gradient
            ),
            "static_self_repeat_exact": torch.equal(candidate.gradient, candidate_repeat.gradient),
        }
