"""Build backend objects from typed problem settings, once before training."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from pinnstorch.api import Problem
from pinnstorch.data import (
    BoundaryConditions,
    InitialCondition,
    MeshSampler,
    PINNDataModule,
    PointCloud,
)
from pinnstorch.data.sampler.sampler_base import SamplerBase
from pinnstorch.models import FCN, PINNModule
from pinnstorch.training.constraints import HardBoxInitial
from pinnstorch.training.sampling import ResampledLHS

from .coefficients import CoefficientBank
from .problem import ProblemAdapter
from .settings import Sampling, TaskSettings


@dataclass
class ExecutionObjects:
    mesh: PointCloud
    data: PINNDataModule
    model: PINNModule
    trainer: L.Trainer
    checkpoint: ModelCheckpoint


class SolverFactory:
    """Translate domain choices to vectorized samplers and a Lightning backend."""

    def __init__(
        self,
        settings: TaskSettings,
        problem: Problem,
        output: Path,
        input_directory: Path | None = None,
    ) -> None:
        self.settings, self.problem, self.output = settings, problem, output
        directory = input_directory if input_directory is not None else output
        bank = CoefficientBank.build(
            problem.coefficients, settings.coefficients, directory, settings.domain
        )
        source = (directory / "functions.py").read_bytes() if bank.formulas else None
        self.coefficient_signature = bank.signature(source)
        self.adapter = ProblemAdapter(problem, settings.domain, bank)
        self._validate_boundaries()
        self._validate_constraints()
        self.constraint_signature = (
            hashlib.sha256((directory / "functions.py").read_bytes()).hexdigest()
            if settings.training.constraints != "soft"
            else ""
        )

    def _validate_boundaries(self) -> None:
        for boundary in self.settings.boundaries:
            coefficient = boundary.conductivity
            if (
                boundary.kind == "heat_flux"
                and isinstance(coefficient, str)
                and coefficient
                not in {*self.settings.domain.parameters, *self.adapter.coefficients.names}
            ):
                raise ValueError(f"Unknown conductivity parameter/coefficient '{coefficient}'")
            if boundary.field not in self.problem.fields:
                raise ValueError(f"Unknown boundary field '{boundary.field}'")
            if (
                boundary.value_function is not None
                and boundary.value_function not in self.problem.boundary_values
            ):
                raise ValueError(f"Unknown boundary value function '{boundary.value_function}'")

    def _validate_constraints(self) -> None:
        if self.settings.training.constraints == "soft":
            return
        if self.problem.transform is not None or tuple(self.problem.outputs) != tuple(
            self.problem.fields
        ):
            raise ValueError("Hard constraints do not support custom transforms/network fields")
        if self.settings.domain.parameters:
            raise ValueError("Hard constraints with variable parameters are not yet supported")
        required = {
            (field, axis, side)
            for field in self.problem.fields
            for axis in self.settings.domain.space
            for side in ("lower", "upper")
        }
        provided: set[tuple[str, str, str]] = set()
        for boundary in self.settings.boundaries:
            if (
                boundary.kind != "dirichlet"
                or boundary.value != 0
                or boundary.value_function is not None
            ):
                raise ValueError("Hard constraints require homogeneous constant Dirichlet faces")
            sides = ("lower", "upper") if boundary.side == "both" else (boundary.side,)
            provided.update((boundary.field, boundary.axis, side) for side in sides)
        if provided != required:
            raise ValueError("Hard constraints require all faces for every physical field")
        grid = self.settings.domain.grid()
        mask = np.zeros(len(grid.spatial[0]), dtype=bool)
        for index, axis in enumerate(self.settings.domain.space.values()):
            column = grid.spatial[index][:, 0]
            mask |= np.isclose(column, axis.interval[0]) | np.isclose(column, axis.interval[1])
        values = self.adapter.initial(np.column_stack([column[:, 0] for column in grid.spatial]))
        if any(not np.allclose(value[mask], 0, atol=1e-6, rtol=0) for value in values.values()):
            raise ValueError("Initial values incompatible with homogeneous Dirichlet faces")

    def build(self, predicting: bool) -> ExecutionObjects:
        seed = self.settings.training.seed
        if seed is not None:
            L.seed_everything(seed, workers=True)
        query = self.settings.prediction.parameters if predicting else None
        grid = self.settings.domain.grid(query)
        mesh = PointCloud(lambda: grid)
        names, datasets = self._training_datasets(mesh) if not predicting else ([], [])
        validation = (
            self._sampler(mesh, self.settings.validation)
            if not predicting and self.settings.validation is not None
            else None
        )
        prediction = MeshSampler(mesh, use_lhs=False)
        data = PINNDataModule(
            datasets, validation, prediction, self.settings.training.batch_size, names
        )
        model = self._model(mesh)
        checkpoint = self._checkpoint()
        trainer = self._trainer(checkpoint)
        return ExecutionObjects(mesh, data, model, trainer, checkpoint)

    def _training_datasets(self, mesh: PointCloud) -> tuple[list[str], list[SamplerBase]]:
        if self.settings.training.constraints != "soft":
            return ["equations"], [self._sampler(mesh, self.settings.training.collocation)]
        names = ["initial", "equations"]
        datasets: list[SamplerBase] = [
            InitialCondition(
                mesh,
                solution=list(self.problem.fields),
                initial_fun=self.adapter.initial,
                num_sample=self.settings.training.initial_points,
            ),
            self._sampler(mesh, self.settings.training.collocation),
        ]
        if self.settings.boundaries:
            values = {
                name: partial(self.adapter.boundary_value, name)
                for name in self.problem.boundary_values
            }
            datasets.append(
                BoundaryConditions(
                    mesh,
                    [boundary.compile() for boundary in self.settings.boundaries],
                    values,
                    {
                        name: partial(self.adapter.coefficient_value, name)
                        for name in self.adapter.coefficients.names
                    },
                )
            )
            names.append("boundaries")
        return names, datasets

    def _sampler(self, mesh: PointCloud, sampling: Sampling) -> MeshSampler:
        policy = None
        if sampling.resample_every_epochs:
            if self.settings.training.seed is None or sampling.points is None:
                raise ValueError(
                    "Resampling requires explicit training.seed and collocation.points"
                )
            if sampling is self.settings.validation:
                raise ValueError("Validation sampling cannot refresh; keep holdout fixed")
            policy = ResampledLHS(
                sampling.points,
                mesh.lb,
                mesh.ub,
                self.settings.training.seed,
                sampling.resample_every_epochs,
            )
        return MeshSampler(
            mesh,
            collection_points=list(self.problem.equations),
            use_lhs=sampling.sampling == "lhs",
            num_sample=sampling.points,
            sampling_policy=policy,
        )

    def _model(self, mesh: PointCloud) -> PINNModule:
        settings = self.settings
        layers = [
            len(settings.domain.input_names),
            *settings.network.hidden_layers,
            len(self.problem.outputs),
        ]
        net = FCN(layers, mesh.lb, mesh.ub, self.problem.outputs)
        optimizer = settings.training.optimizer
        optimizer_class = torch.optim.AdamW if optimizer.name == "adamw" else torch.optim.Adam
        output = self.adapter.transform if self.problem.transform else None
        if settings.training.constraints != "soft":
            output = HardBoxInitial(
                [axis.interval for axis in settings.domain.space.values()],
                settings.domain.time.interval,
                self.adapter.initial_tensors,
            )
        return PINNModule(
            net,
            self.adapter.residuals,
            output_fn=output,
            constraint_kind=settings.training.constraints,
            constraint_signature=self.constraint_signature,
            optimizer=partial(
                optimizer_class, lr=optimizer.learning_rate, weight_decay=optimizer.weight_decay
            ),
            coefficients=self.adapter.coefficients,
            coefficient_signature=self.coefficient_signature,
            loss_settings=asdict(settings.training.loss),
            rate_settings={
                **asdict(settings.training.schedule),
                "steps": settings.training.steps or 1,
                "version": 1,
            },
            input_names=settings.domain.input_names,
            input_bounds=tuple(
                (float(lower), float(upper)) for lower, upper in zip(mesh.lb, mesh.ub)
            ),
        )

    def _checkpoint(self) -> ModelCheckpoint:
        return ModelCheckpoint(
            dirpath=self.output / "checkpoints",
            filename="best",
            monitor="val/loss" if self.settings.validation else "train/loss",
            mode="min",
            save_last=True,
            auto_insert_metric_name=False,
        )

    def _trainer(self, checkpoint: ModelCheckpoint) -> L.Trainer:
        training, runtime = self.settings.training, self.settings.runtime
        logger = CSVLogger(str(self.output / "csv"), name="metrics") if runtime.metrics else False
        devices = list(runtime.devices) if isinstance(runtime.devices, tuple) else runtime.devices
        return L.Trainer(
            accelerator="gpu" if runtime.device == "cuda" else runtime.device,
            devices=devices,
            precision=runtime.precision,
            max_epochs=training.epochs,
            max_steps=training.steps if training.steps is not None else -1,
            reload_dataloaders_every_n_epochs=1
            if training.collocation.resample_every_epochs
            else 0,
            num_sanity_val_steps=0,
            inference_mode=False,
            check_val_every_n_epoch=1,
            callbacks=[checkpoint],
            logger=logger,
            default_root_dir=self.output,
            enable_progress_bar=runtime.progress,
            enable_model_summary=False,
        )
