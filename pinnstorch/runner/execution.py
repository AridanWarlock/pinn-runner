"""Short execution paths for train, fine-tune and inference."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import cast

import torch

from pinnstorch.checkpoint_contract import InputBounds, validate_checkpoint_domain
from pinnstorch.contracts import TensorOutputs
from pinnstorch.training.refinement import FixedObjective, LBFGSRefinement
from pinnstorch.utils.module_fn import fix_predictions

from .artifacts import ArtifactWriter
from .factory import ExecutionObjects
from .settings import TaskSettings


class SolverExecution:
    def __init__(
        self,
        settings: TaskSettings,
        objects: ExecutionObjects,
        artifacts: ArtifactWriter,
        log: logging.Logger,
    ) -> None:
        self.settings, self.objects, self.artifacts, self.log = settings, objects, artifacts, log

    def train(self, checkpoint: Path | None = None) -> None:
        if checkpoint is not None:
            self._load(checkpoint, allow_extrapolation=False)
        self.log.info("Starting training!")
        objects = self.objects
        if self.settings.runtime.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        objects.trainer.fit(objects.model, datamodule=objects.data)
        performance: dict[str, int | float] = {
            "adam_steps": objects.trainer.global_step,
            "adam_seconds": time.perf_counter() - started,
            "adam_model_evaluations": objects.model.adam_evaluations,
            "adam_residual_points": objects.model.adam_residual_points,
        }
        refinement = self.settings.training.refinement
        if refinement.steps:
            objects.trainer.save_checkpoint(
                str(self.artifacts.directory / "checkpoints/adam.ckpt")
            )
            stage = LBFGSRefinement(
                objects.model,
                FixedObjective.capture(objects.data, objects.model.device),
                refinement.learning_rate,
                refinement.history_size,
                refinement.tolerance_grad,
                refinement.tolerance_change,
            )
            performance.update(stage.run(refinement.steps))
            objects.model.refinement_state = stage.state_dict(performance.copy())
        if self.settings.runtime.device == "cuda":
            performance.update(
                {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
        if performance:
            self.artifacts.save_training_performance(performance)
        objects.trainer.save_checkpoint(str(self.artifacts.final_checkpoint))
        if self.settings.validation is not None:
            self.log.info("Starting validation!")
            objects.trainer.validate(objects.model, datamodule=objects.data, verbose=False)
        if self.settings.runtime.save_training_predictions:
            best = (
                str(self.artifacts.final_checkpoint)
                if refinement.steps
                else (objects.checkpoint.best_model_path or str(self.artifacts.final_checkpoint))
            )
            self.artifacts.save_training_predictions(self._predict(best))

    def predict(self, checkpoint: Path) -> None:
        bounds = self._load(checkpoint, self.settings.prediction.allow_extrapolation)
        self.objects.model.eval()
        self.artifacts.save_prediction(self._predict(), self.objects.mesh, bounds)

    def _predict(self, checkpoint: str | None = None) -> TensorOutputs:
        results = self.objects.trainer.predict(
            self.objects.model, datamodule=self.objects.data, ckpt_path=checkpoint
        )
        if results is None:
            raise RuntimeError("Trainer did not produce predictions")
        return fix_predictions(cast(list[TensorOutputs], results))

    def _load(self, path: Path, allow_extrapolation: bool) -> InputBounds:
        checkpoint = torch.load(path, map_location="cpu")
        constraint = checkpoint.get("hyper_parameters", {}).get("constraint_kind", "soft")
        saved_signature = checkpoint.get("hyper_parameters", {}).get("constraint_signature", "")
        if saved_signature != self.objects.model.hparams.get("constraint_signature", ""):
            raise ValueError("Checkpoint hard constraint definition differs from task")
        if constraint != self.settings.training.constraints:
            raise ValueError(
                "Checkpoint constraint transform differs from task; preserve physical fields"
            )
        signature = checkpoint.get("hyper_parameters", {}).get("coefficient_signature") or {}
        current = self.objects.model.hparams.get("coefficient_signature") or {}
        if signature != current:
            raise ValueError(
                "Checkpoint coefficient definitions differ from this task; train a new model"
            )
        mesh = self.objects.mesh
        requested = [(float(column.min()), float(column.max())) for column in mesh.spatial_domain]
        requested.append((float(mesh.time_domain.min()), float(mesh.time_domain.max())))
        bounds = validate_checkpoint_domain(
            checkpoint, self.settings.domain.input_names, requested, allow_extrapolation
        )
        self.objects.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.objects.model.hparams["input_bounds"] = bounds
        return bounds
