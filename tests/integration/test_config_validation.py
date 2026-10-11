"""Public YAML validation should report actionable errors before training."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from pinnstorch.config_validation import validate_task_settings

CONFIG = yaml.safe_load(
    (Path(__file__).parents[2] / "examples/heat_parameterized/config.yaml").read_text()
)


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("net", "layers"), [4, 16, 2], "output layer"),
        (("net", "layers"), [3, 16, 1], "input layer"),
        (("net", "layers"), [4, True, 1], "positive integers"),
        (("net", "output_names"), ["x"], "distinct"),
        (("net", "output_names"), ["conductivity"], "distinct"),
        (("net", "output_names"), ["T", "T"], "distinct"),
        (("net", "output_names"), [], "nonempty"),
        (("data", "batch_size"), 0, "batch_size"),
        (("trainer", "inference_mode"), True, "inference_mode"),
        (("pred_dataset",), None, "full-grid"),
        (("val_dataset",), None, "Monitoring val/loss"),
    ],
)
def test_invalid_settings(path: tuple[str, ...], value: object, message: str) -> None:
    source = deepcopy(CONFIG)
    source.setdefault("data", {})
    source.setdefault("callbacks", {}).setdefault("model_checkpoint", {})["monitor"] = "val/loss"
    target = source
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_task_settings(OmegaConf.create(source), 2, ["conductivity"])


def test_prediction_must_use_grid_and_validation_must_have_residual() -> None:
    for group, update, message in [
        ("pred_dataset", {"use_lhs": True}, "use_lhs"),
        ("pred_dataset", {"collection_points": ["residual"]}, "collection_points"),
        ("val_dataset", {"collection_points": []}, "collection_points"),
        ("val_dataset", {"use_lhs": True}, "num_sample"),
    ]:
        source = deepcopy(CONFIG)
        source["data"] = {}
        source[group][0]["data"].update(update)
        with pytest.raises(ValueError, match=message):
            validate_task_settings(OmegaConf.create(source), 2, ["conductivity"])


def test_lhs_requires_sample_count() -> None:
    source = deepcopy(CONFIG)
    source["data"] = {}
    source["train_datasets"][1]["pde"]["use_lhs"] = True
    with pytest.raises(ValueError, match="num_sample"):
        validate_task_settings(OmegaConf.create(source), 2, ["conductivity"])
    source["train_datasets"][1]["pde"]["num_sample"] = 32
    validate_task_settings(OmegaConf.create(source), 2, ["conductivity"])
