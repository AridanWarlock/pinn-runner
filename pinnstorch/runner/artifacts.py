"""Result persistence, separated from fitting and checkpoint selection."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.io import savemat

from pinnstorch.api import Problem
from pinnstorch.contracts import FloatArray, TensorOutputs
from pinnstorch.data import PointCloud

from .settings import TaskSettings


class ArtifactWriter:
    def __init__(self, directory: Path, settings: TaskSettings, problem: Problem) -> None:
        self.directory, self.settings, self.problem = directory, settings, problem
        self.final_checkpoint = directory / "checkpoints/final.ckpt"

    def save_definition(self) -> None:
        self.settings.save(self.directory / "config.resolved.yaml")
        definition = {
            "fields": list(self.problem.fields),
            "network_fields": list(self.problem.outputs),
            "equations": list(self.problem.equations),
            "boundary_values": list(self.problem.boundary_values),
        }
        self._json("problem.json", definition)

    def save_training_predictions(self, predictions: TensorOutputs) -> None:
        if self.settings.runtime.save_training_predictions:
            with (self.directory / "predictions.pkl").open("wb") as stream:
                portable = {
                    name: value.detach().float().cpu() for name, value in predictions.items()
                }
                pickle.dump(portable, stream)

    def save_prediction(
        self,
        predictions: TensorOutputs,
        mesh: PointCloud,
        training_bounds: tuple[tuple[float, float], ...],
    ) -> None:
        coordinates = self._coordinates(mesh)
        shape = (len(mesh.spatial_domain[0]), len(mesh.time_domain))
        for name, value in predictions.items():
            if name in coordinates:
                raise ValueError(f"Output '{name}' conflicts with a coordinate or parameter")
            array = value.detach().float().cpu().numpy()
            if array.shape != (shape[0] * shape[1], 1) or not np.isfinite(array).all():
                raise ValueError(f"Prediction '{name}' must be a finite (grid_rows, 1) column")
            coordinates[name] = array.reshape(shape)
        savemat(self.directory / "predictions.mat", coordinates)
        self._json("prediction.json", self._metadata(mesh, shape, training_bounds))

    def _coordinates(self, mesh: PointCloud) -> dict[str, FloatArray]:
        names = self.settings.domain.input_names[:-1]
        coordinates = {
            name: mesh.spatial_domain_mesh[:, 0, i : i + 1] for i, name in enumerate(names)
        }
        coordinates["t"] = np.asarray(mesh.time_domain).reshape(-1, 1)
        return coordinates

    def _metadata(
        self, mesh: PointCloud, shape: tuple[int, int], bounds: tuple[tuple[float, float], ...]
    ) -> dict[str, object]:
        requested = [(float(column.min()), float(column.max())) for column in mesh.spatial_domain]
        requested.append(
            (float(np.asarray(mesh.time_domain).min()), float(np.asarray(mesh.time_domain).max()))
        )
        return {
            "input_names": list(self.settings.domain.input_names),
            "training_bounds": bounds,
            "requested_bounds": requested,
            "allow_extrapolation": self.settings.prediction.allow_extrapolation,
            "spatial_dimension": len(self.settings.domain.space),
            "parameter_names": list(self.settings.domain.parameters),
            "field_shape": shape,
            "grid_order": "ij/C; parameters after space, time fastest in batches",
            "dtype": "float32",
            "batch_size": self.settings.training.batch_size,
        }

    def _json(self, name: str, value: Mapping[str, object]) -> None:
        (self.directory / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
