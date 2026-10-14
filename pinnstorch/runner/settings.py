"""Typed task settings. Public YAML contains choices, never Python import paths."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

from pinnstorch.data.domains.point_cloud import ParameterSpec, PointCloudData, create_interval_data


class Reader:
    """Read a strict mapping and attach its location to validation errors."""

    def __init__(self, value: object, path: str) -> None:
        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} must be a mapping with string keys")
        self.values = cast(Mapping[str, object], value)
        self.path = path

    def only(self, *names: str) -> None:
        unknown = set(self.values) - set(names)
        if unknown:
            raise ValueError(f"{self.path}: unknown options {sorted(unknown)}; see docs/API.md")

    def section(self, name: str) -> Reader:
        return Reader(self.values.get(name, {}), f"{self.path}.{name}")

    def integer(self, name: str, default: int | None, minimum: int = 1) -> int | None:
        value = self.values.get(name, default)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{self.path}.{name} must be an integer >= {minimum}")
        return value

    def number(self, name: str, default: float) -> float:
        value = self.values.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{self.path}.{name} must be a finite number")
        return float(value)

    def boolean(self, name: str, default: bool) -> bool:
        value = self.values.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"{self.path}.{name} must be true or false")
        return value

    def choice(self, name: str, default: str, choices: tuple[str, ...]) -> str:
        value = self.values.get(name, default)
        if value not in choices:
            raise ValueError(f"{self.path}.{name} must be one of {choices}")
        return cast(str, value)


def identifier(name: str) -> None:
    if not name.isascii() or not name.isidentifier() or name.startswith("_"):
        raise ValueError(
            f"Invalid name '{name}': use an ASCII identifier without a leading underscore"
        )


@dataclass(frozen=True)
class Axis:
    interval: tuple[float, float]
    points: int

    @classmethod
    def read(cls, value: object, path: str) -> Axis:
        reader = Reader(value, path)
        reader.only("interval", "points")
        bounds = reader.values.get("interval")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"{path}.interval must contain two increasing finite numbers")
        numbers = [Reader({"bound": bound}, path).number("bound", 0) for bound in bounds]
        if numbers[0] >= numbers[1]:
            raise ValueError(f"{path}.interval must be increasing")
        points = reader.integer("points", None, 2)
        if points is None:
            raise ValueError(f"{path}.points is required")
        return cls((numbers[0], numbers[1]), points)


@dataclass(frozen=True)
class Domain:
    space: Mapping[str, Axis]
    time: Axis
    parameters: Mapping[str, Axis]

    @classmethod
    def read(cls, reader: Reader) -> Domain:
        reader.only("space", "time", "parameters")
        space = reader.section("space")
        space.only("x", "y", "z")
        if "x" not in space.values or ("z" in space.values and "y" not in space.values):
            raise ValueError("domain.space requires x; z also requires y")
        axes = {
            name: Axis.read(space.values[name], f"domain.space.{name}")
            for name in ("x", "y", "z")
            if name in space.values
        }
        parameters = reader.section("parameters")
        for name in parameters.values:
            identifier(name)
            if name in ("x", "y", "z", "t"):
                raise ValueError(f"Parameter '{name}' conflicts with a coordinate")
        return cls(
            axes,
            Axis.read(reader.values.get("time"), "domain.time"),
            {
                name: Axis.read(value, f"domain.parameters.{name}")
                for name, value in parameters.values.items()
            },
        )

    @property
    def input_names(self) -> tuple[str, ...]:
        return (*self.space, *self.parameters, "t")

    def grid(self, query: Mapping[str, tuple[float, ...]] | None = None) -> PointCloudData:
        x, y, z = self.space["x"], self.space.get("y"), self.space.get("z")
        parameters: dict[str, ParameterSpec] = {
            name: {"interval": axis.interval, "points": axis.points}
            for name, axis in self.parameters.items()
        }
        return create_interval_data(
            x.interval,
            self.time.interval,
            x.points,
            self.time.points,
            y.interval if y else None,
            y.points if y else None,
            z.interval if z else None,
            z.points if z else None,
            parameters,
            parameter_values=query,
        )


@dataclass(frozen=True)
class Sampling:
    resample_every_epochs: int = 0
    sampling: Literal["grid", "lhs"] = "grid"
    points: int | None = None

    @classmethod
    def read(cls, reader: Reader) -> Sampling:
        reader.only("sampling", "points", "resample_every_epochs")
        kind = cast(Literal["grid", "lhs"], reader.choice("sampling", "grid", ("grid", "lhs")))
        points = reader.integer("points", None)
        if (kind == "lhs") != (points is not None):
            raise ValueError(f"{reader.path}: lhs requires points; grid does not accept points")
        every = reader.integer("resample_every_epochs", 0, 0)
        if every is None or (every and kind != "lhs"):
            raise ValueError("resampling requires lhs and a nonnegative interval")
        return cls(every, kind, points)


@dataclass(frozen=True)
class Optimizer:
    name: Literal["adam", "adamw"] = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0

    @classmethod
    def read(cls, reader: Reader) -> Optimizer:
        reader.only("name", "learning_rate", "weight_decay")
        name = cast(Literal["adam", "adamw"], reader.choice("name", "adam", ("adam", "adamw")))
        rate, decay = reader.number("learning_rate", 0.001), reader.number("weight_decay", 0)
        if rate <= 0 or decay < 0:
            raise ValueError(
                "Optimizer learning_rate must be positive and weight_decay nonnegative"
            )
        return cls(name, rate, decay)


@dataclass(frozen=True)
class LossSettings:
    kind: str = "equal"
    initial: float = 1.0
    equations: float = 1.0
    boundaries: float = 1.0
    gradient_norm_every: int = 0

    @classmethod
    def read(cls, reader: Reader) -> LossSettings:
        reader.only("kind", "initial", "equations", "boundaries", "gradient_norm_every")
        kind = reader.choice("kind", "equal", ("equal", "fixed"))
        weights = {
            name: reader.number(name, 1.0) for name in ("initial", "equations", "boundaries")
        }
        if any(value <= 0 for value in weights.values()):
            raise ValueError(f"{reader.path}: loss weights must be positive")
        if kind == "equal" and any(value != 1 for value in weights.values()):
            raise ValueError(f"{reader.path}: unequal weights require kind: fixed")
        every = reader.integer("gradient_norm_every", 0, 0)
        if every is None:
            raise ValueError(f"{reader.path}.gradient_norm_every must be an integer")
        return cls(kind, **weights, gradient_norm_every=every)


@dataclass(frozen=True)
class ScheduleSettings:
    kind: str = "constant"
    final_ratio: float = 1.0

    @classmethod
    def read(cls, reader: Reader) -> ScheduleSettings:
        reader.only("kind", "final_ratio")
        kind = reader.choice("kind", "constant", ("constant", "exponential"))
        ratio = reader.number("final_ratio", 1.0)
        if not 0 < ratio <= 1 or (kind == "constant" and ratio != 1):
            raise ValueError(f"{reader.path}: final_ratio must be in (0,1]; constant requires 1")
        return cls(kind, ratio)


@dataclass(frozen=True)
class Refinement:
    steps: int = 0
    learning_rate: float = 1.0
    history_size: int = 50
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9

    @classmethod
    def read(cls, reader: Reader) -> Refinement:
        reader.only("steps", "learning_rate", "history_size", "tolerance_grad", "tolerance_change")
        steps = reader.integer("steps", 0, 0)
        history = reader.integer("history_size", 50)
        rate = reader.number("learning_rate", 1)
        grad = reader.number("tolerance_grad", 1e-7)
        change = reader.number("tolerance_change", 1e-9)
        if steps is None or history is None or min(rate, grad, change) <= 0:
            raise ValueError(f"{reader.path}: positive rate/history/tolerances required")
        return cls(steps, rate, history, grad, change)


@dataclass(frozen=True)
class Training:
    epochs: int
    steps: int | None
    batch_size: int | None
    initial_points: int | None
    seed: int | None
    optimizer: Optimizer
    collocation: Sampling
    loss: LossSettings = LossSettings()
    schedule: ScheduleSettings = ScheduleSettings()
    constraints: str = "soft"
    refinement: Refinement = Refinement()
    hard_execution: str = "eager"

    @classmethod
    def read(cls, reader: Reader) -> Training:
        reader.only(
            "epochs",
            "steps",
            "batch_size",
            "initial_points",
            "seed",
            "optimizer",
            "collocation",
            "loss",
            "schedule",
            "constraints",
            "refinement",
            "hard_execution",
        )
        if (
            ScheduleSettings.read(reader.section("schedule")).kind != "constant"
            and reader.integer("steps", None) is None
        ):
            raise ValueError("training.schedule requires explicit training.steps")
        epochs = reader.integer("epochs", 100)
        refinement = Refinement.read(reader.section("refinement"))
        if refinement.steps and (
            reader.integer("steps", None) is None
            or Optimizer.read(reader.section("optimizer")).weight_decay != 0
        ):
            raise ValueError(
                "training.refinement requires explicit Adam steps and zero weight_decay"
            )
        if epochs is None:
            raise ValueError("training.epochs must be a positive integer")
        constraints = reader.choice(
            "constraints", "soft", ("soft", "hard_initial_dirichlet", "hard_initial_robin")
        )
        hard_execution = reader.choice(
            "hard_execution", "eager", ("eager", "fused_predict", "product_rule")
        )
        if constraints == "hard_initial_robin" and hard_execution != "eager":
            raise ValueError("hard_initial_robin currently requires eager execution")
        if constraints == "soft" and hard_execution != "eager":
            raise ValueError("training.hard_execution requires hard constraints")
        return cls(
            epochs,
            reader.integer("steps", None),
            reader.integer("batch_size", None),
            reader.integer("initial_points", None),
            reader.integer("seed", 1234, 0),
            Optimizer.read(reader.section("optimizer")),
            Sampling.read(reader.section("collocation")),
            LossSettings.read(reader.section("loss")),
            ScheduleSettings.read(reader.section("schedule")),
            constraints,
            refinement,
            hard_execution,
        )


@dataclass(frozen=True)
class Network:
    hidden_layers: tuple[int, ...]

    @classmethod
    def read(cls, reader: Reader) -> Network:
        reader.only("hidden_layers")
        layers = reader.values.get("hidden_layers", [32, 32])
        if (
            not isinstance(layers, list)
            or not layers
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in layers
            )
        ):
            raise ValueError("network.hidden_layers must be a nonempty list of positive integers")
        return cls(tuple(cast(list[int], layers)))


@dataclass(frozen=True)
class Boundary:
    kind: str
    field: str
    axis: str
    side: str
    weight: float
    value: float | None
    value_function: str | None
    alpha: float
    beta: float
    conductivity: float | str

    @classmethod
    def read(cls, value: object, index: int) -> Boundary:
        reader = Reader(value, f"boundaries[{index}]")
        reader.only(
            "kind",
            "field",
            "axis",
            "side",
            "weight",
            "value",
            "value_function",
            "alpha",
            "beta",
            "conductivity",
        )
        kind = reader.choice("kind", "", ("dirichlet", "neumann", "robin", "heat_flux"))
        axis = reader.choice("axis", "", ("x", "y", "z"))
        side = reader.choice("side", "both", ("lower", "upper", "both"))
        field_name, value_name = reader.values.get("field"), reader.values.get("value_function")
        if not isinstance(field_name, str) or (
            value_name is not None and not isinstance(value_name, str)
        ):
            raise ValueError(f"{reader.path}: field and value_function must be names")
        if value_name is not None and "value" in reader.values:
            raise ValueError(f"{reader.path}: use value or value_function, not both")
        conductivity = reader.values.get("conductivity", 1.0)
        if not isinstance(conductivity, str):
            conductivity = reader.number("conductivity", 1)
        result = cls(
            kind,
            field_name,
            axis,
            side,
            reader.number("weight", 1),
            None if value_name is not None else reader.number("value", 0),
            value_name,
            reader.number("alpha", 1),
            reader.number("beta", 1),
            conductivity,
        )
        result._validate_options(reader)
        return result

    def _validate_options(self, reader: Reader) -> None:
        if self.weight <= 0:
            raise ValueError(f"{reader.path}.weight must be positive")
        if self.kind != "robin" and ({"alpha", "beta"} & set(reader.values)):
            raise ValueError("alpha/beta are only valid for Robin boundaries")
        if self.kind == "robin" and self.alpha == self.beta == 0:
            raise ValueError("Robin alpha and beta cannot both be zero")
        if self.kind != "heat_flux" and "conductivity" in reader.values:
            raise ValueError("conductivity is only valid for heat_flux boundaries")
        if isinstance(self.conductivity, float) and self.conductivity <= 0:
            raise ValueError("Conductivity must be positive")

    def compile(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": self.kind,
            "field": self.field,
            "axis": self.axis,
            "side": self.side,
            "weight": self.weight,
        }
        if self.value_function:
            result["value_fn"] = self.value_function
        else:
            result["value"] = self.value
        if self.kind == "robin":
            result.update(alpha=self.alpha, beta=self.beta)
        if self.kind == "heat_flux":
            result["conductivity"] = self.conductivity
        return result


@dataclass(frozen=True)
class Prediction:
    parameters: Mapping[str, tuple[float, ...]]
    allow_extrapolation: bool

    @classmethod
    def read(cls, reader: Reader) -> Prediction:
        reader.only("parameters", "allow_extrapolation")
        parameters = reader.section("parameters")
        values: dict[str, tuple[float, ...]] = {}
        for name, raw in parameters.values.items():
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"prediction.parameters.{name} must be a nonempty numeric list")
            numbers = tuple(Reader({"value": value}, name).number("value", 0) for value in raw)
            if len(set(numbers)) != len(numbers):
                raise ValueError(f"prediction.parameters.{name} values must be distinct")
            values[name] = numbers
        return cls(values, reader.boolean("allow_extrapolation", False))


@dataclass(frozen=True)
class Runtime:
    device: str
    devices: int | tuple[int, ...]
    precision: Literal["32-true", "16-mixed", "bf16-mixed"]
    progress: bool
    metrics: bool
    save_training_predictions: bool

    @classmethod
    def read(cls, reader: Reader) -> Runtime:
        reader.only(
            "device", "devices", "precision", "progress", "metrics", "save_training_predictions"
        )
        device = reader.choice("device", "cpu", ("cpu", "cuda", "mps", "auto"))
        devices = reader.values.get("devices", 1)
        if isinstance(devices, list):
            if not devices or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in devices
            ):
                raise ValueError("runtime.devices must be a count or list of device indices")
            devices = tuple(cast(list[int], devices))
        elif isinstance(devices, bool) or not isinstance(devices, int) or devices < 1:
            raise ValueError("runtime.devices must be a positive count or list of indices")
        precision = cast(
            Literal["32-true", "16-mixed", "bf16-mixed"],
            reader.choice("precision", "32-true", ("32-true", "16-mixed", "bf16-mixed")),
        )
        return cls(
            device,
            devices,
            precision,
            reader.boolean("progress", False),
            reader.boolean("metrics", True),
            reader.boolean("save_training_predictions", True),
        )


@dataclass(frozen=True)
class TableCoefficient:
    file: str

    @classmethod
    def read(cls, value: object, name: str) -> TableCoefficient:
        identifier(name)
        reader = Reader(value, f"coefficients.{name}")
        reader.only("file")
        path = reader.values.get("file")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or Path(path).suffix != ".npz"
        ):
            raise ValueError(f"{reader.path}.file must be a relative NPZ path inside input-dir")
        return cls(path)


@dataclass(frozen=True)
class TaskSettings:
    domain: Domain
    network: Network
    training: Training
    validation: Sampling | None
    boundaries: tuple[Boundary, ...]
    prediction: Prediction
    runtime: Runtime
    coefficients: Mapping[str, TableCoefficient]
    version: int = 1

    @classmethod
    def load(cls, path: Path) -> TaskSettings:
        return cls.read(yaml.safe_load(path.read_text(encoding="utf-8")))

    @classmethod
    def read(cls, value: object) -> TaskSettings:
        reader = Reader(value, "config")
        reader.only(
            "version",
            "domain",
            "network",
            "training",
            "validation",
            "boundaries",
            "prediction",
            "runtime",
            "coefficients",
        )
        if type(reader.values.get("version")) is not int or reader.values.get("version") != 1:
            raise ValueError("config.version must be 1; old _target_ configs must be migrated")
        validation = reader.values.get("validation", {"sampling": "grid"})
        boundaries = reader.values.get("boundaries", [])
        if not isinstance(boundaries, list):
            raise ValueError("boundaries must be a list")
        result = cls(
            Domain.read(reader.section("domain")),
            Network.read(reader.section("network")),
            Training.read(reader.section("training")),
            None
            if validation is None or validation is False
            else Sampling.read(Reader(validation, "validation")),
            tuple(Boundary.read(value, index) for index, value in enumerate(boundaries)),
            Prediction.read(reader.section("prediction")),
            Runtime.read(reader.section("runtime")),
            {
                name: TableCoefficient.read(value, name)
                for name, value in reader.section("coefficients").values.items()
            },
        )
        result._validate_links()
        return result

    def _validate_links(self) -> None:
        if set(self.coefficients) & set(self.domain.input_names):
            raise ValueError("Coefficient names must differ from coordinates and parameters")
        unknown = set(self.prediction.parameters) - set(self.domain.parameters)
        if unknown:
            raise ValueError(f"Unknown prediction parameters: {sorted(unknown)}")
        for boundary in self.boundaries:
            if boundary.axis not in self.domain.space:
                raise ValueError(f"Boundary axis '{boundary.axis}' is absent from domain.space")
            if boundary.kind == "heat_flux" and isinstance(boundary.conductivity, str):
                coefficient = self.domain.parameters.get(boundary.conductivity)
                if coefficient is not None and coefficient.interval[0] <= 0:
                    raise ValueError("Heat flux conductivity parameter range must be positive")

    def save(self, path: Path) -> None:
        mapping = asdict(self)
        mapping["boundaries"] = [boundary.compile() for boundary in self.boundaries]
        for boundary in mapping["boundaries"]:
            if "value_fn" in boundary:
                boundary["value_function"] = boundary.pop("value_fn")
        path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
