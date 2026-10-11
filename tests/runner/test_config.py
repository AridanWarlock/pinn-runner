"""Device and seed choices survive public configuration parsing."""

from pathlib import Path

import pytest
import yaml

from pinnstorch.runner.settings import TaskSettings

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("device,devices", [("cpu", 1), ("cuda", [0]), ("cuda", [1]), ("mps", 1)])
def test_preserves_requested_device(device: str, devices: int | list[int]) -> None:
    config = yaml.safe_load((ROOT / "examples/heat2d/config.yaml").read_text())
    config["runtime"].update(device=device, devices=devices)
    settings = TaskSettings.read(config)
    assert settings.runtime.device == device
    assert settings.runtime.devices == (tuple(devices) if isinstance(devices, list) else devices)


@pytest.mark.parametrize("seed", [0, 42, None])
def test_preserves_requested_seed(seed: int | None) -> None:
    config = yaml.safe_load((ROOT / "examples/heat2d/config.yaml").read_text())
    config["training"]["seed"] = seed
    assert TaskSettings.read(config).training.seed == seed
