"""Configuration tests without PyTorch: pytest --noconftest tests/runner."""

import importlib.util
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("runner_config_under_test", ROOT / "run.py")
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def config() -> DictConfig:
    """Use actual package defaults for the device and random seed."""
    trainer = OmegaConf.load(ROOT / "pinnstorch/conf/trainer/default.yaml")
    defaults = OmegaConf.load(ROOT / "pinnstorch/conf/train.yaml")
    return OmegaConf.create(
        {
            "trainer": trainer,
            "seed": defaults.seed,
            "domain": {"x_interval": [0, 1], "t_interval": [0, 1], "x_points": 8, "t_points": 3},
            "train_datasets": [{"initial": {"_target_": "pinnstorch.data.InitialCondition"}}],
            "model": {"optimizer": {"_target_": "torch.optim.Adam"}, "scheduler": None},
            "net": {"_target_": "pinnstorch.models.FCN"},
        }
    )


@pytest.mark.parametrize(
    ("accelerator", "devices"),
    [("cpu", 1), ("gpu", [0]), ("gpu", [1]), ("mps", 1)],
)
def test_preserves_requested_device(
    config: DictConfig, accelerator: str, devices: int | list[int]
) -> None:
    config.trainer.accelerator = accelerator
    config.trainer.devices = devices

    prepared = runner.setup_config(config)

    assert prepared.trainer.accelerator == accelerator
    assert prepared.trainer.devices == devices
    assert "devices" not in prepared


@pytest.mark.parametrize("seed", [0, 42, None])
def test_preserves_requested_seed(config: DictConfig, seed: int | None) -> None:
    config.seed = seed

    prepared = runner.setup_config(config)

    assert prepared.seed == seed


def test_preserves_package_defaults(config: DictConfig) -> None:
    prepared = runner.setup_config(config)

    assert prepared.trainer.accelerator == "gpu"
    assert prepared.trainer.devices == [0]
    assert prepared.seed == 1234
