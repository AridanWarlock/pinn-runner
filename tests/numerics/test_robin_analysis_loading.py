"""A saved CUDA task can be analyzed on CPU without allocating a CUDA trainer."""

import json
from pathlib import Path

import torch
import yaml

from benchmarks.cases import CASES
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from scripts.benchmark_neumann_projection import analyze
from scripts.benchmark_robin_execution import load_model


def test_cpu_analysis_of_cuda_config_and_independent_quadrature(tmp_path: Path) -> None:
    case = CASES["heat_neumann2d"]
    task = tmp_path / "task"
    task.mkdir()
    (task / "functions.py").write_bytes(case.source.read_bytes())
    config = case.config("cpu", 3, 19)
    config["network"]["hidden_layers"] = [4]
    config["training"].update(constraints="hard_initial_robin", backward="parameters")
    factory = SolverFactory(
        TaskSettings.read(config), load_problem(task / "functions.py"), tmp_path, task
    )
    original = factory.build(predicting=True).model
    checkpoint = tmp_path / "training/checkpoints/final.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {"state_dict": original.state_dict(), "hyper_parameters": dict(original.hparams)},
        checkpoint,
    )
    config["runtime"]["device"] = "cuda"
    (task / "config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "result.json").write_text(
        json.dumps({"case": case.name, "seed": 19, "hard_lifting": "recursive"})
    )
    loaded_factory, model = load_model(tmp_path, "robin_grouped", device="cpu")
    assert loaded_factory.settings.runtime.device == "cpu"
    assert model.device.type == "cpu"
    for name, value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    report = analyze(tmp_path, [5], 3, "cpu")
    measurement = report["measurements"][0]
    assert measurement["points"] == 75
    assert measurement["prediction_seconds"] > 0
    assert measurement["gram_defect"] < 1e-6
    for row in measurement["rows"]:
        if row["error_energy"] > 0:
            total = sum(row[k] for k in row if k.endswith("_fraction"))
            assert abs(total - 1) < 1e-6
