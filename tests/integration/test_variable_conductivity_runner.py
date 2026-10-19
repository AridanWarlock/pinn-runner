"""Public variable-conductivity tasks preserve weights and reject different materials."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import loadmat

from benchmarks.cases import CASES
from scripts.audit_variable_conductivity import make_factory
from tests.integration.test_runner_files import launch

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("representation", ["formula", "table"])
@pytest.mark.parametrize("beta", [0, 1])
def test_variable_conductivity_train_predict_retrain_and_identity(
    tmp_path: Path, representation: str, beta: int
) -> None:
    task = tmp_path / "input"
    make_factory(task, 2, beta, representation, 19)
    functions = task / "functions.py"
    original_source = functions.read_text()
    config = CASES["heat_neumann_polynomial2d"].config("cpu", 3, 19)
    config["domain"]["time"].update(interval=[0, 1], points=3)
    for axis in config["domain"]["space"].values():
        axis["points"] = 3
    config["network"].update(hidden_layers=[4, 4], spatial_encoding="neumann_polynomial")
    config["training"].update(
        constraints="hard_initial_robin",
        hard_execution="eager",
        hard_background="general",
        refinement={"steps": 2},
        batch_size=7,
        collocation={"sampling": "lhs", "points": 7},
    )
    path = task / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = launch(task, tmp_path / "train")
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = tmp_path / "train/checkpoints/final.ckpt"
    state = torch.load(checkpoint, map_location="cpu")
    signature = state["hyper_parameters"]["coefficient_signature"]
    assert set(signature) == {"conductivity"}
    assert signature["conductivity"].startswith(
        "grid:" if representation == "table" else "functions.py:"
    )

    probe = deepcopy(config)
    probe["training"].update(steps=1, refinement={"steps": 0})
    probe["training"]["optimizer"].update(learning_rate=1e-300, weight_decay=0)
    path.write_text(yaml.safe_dump(probe))
    result = launch(task, tmp_path / "load-probe", "retrain", str(checkpoint))
    assert result.returncode == 0, result.stdout + result.stderr
    loaded = torch.load(tmp_path / "load-probe/checkpoints/final.ckpt", map_location="cpu")
    assert loaded["state_dict"].keys() == state["state_dict"].keys()
    for name, expected in state["state_dict"].items():
        torch.testing.assert_close(loaded["state_dict"][name], expected, rtol=0, atol=0)

    for axis in config["domain"]["space"].values():
        axis["points"] = 4
    config["domain"]["time"]["points"] = 5
    path.write_text(yaml.safe_dump(config))
    result = launch(task, tmp_path / "predict", "predict", str(checkpoint))
    assert result.returncode == 0, result.stdout + result.stderr
    saved = loadmat(tmp_path / "predict/predictions.mat")
    assert saved["u"].shape == (16, 5)
    assert np.isfinite(saved["u"]).all()
    q = np.ones_like(saved["x"])
    for axis in ("x", "y"):
        q *= 3 * saved[axis] ** 2 - 2 * saved[axis] ** 3
    np.testing.assert_allclose(saved["u"][:, :1], q, rtol=0, atol=2e-6)
    observed = json.loads((tmp_path / "predict/coefficients.json").read_text())
    assert observed["signatures"] == signature

    result = launch(task, tmp_path / "retrain", "retrain", str(checkpoint))
    assert result.returncode == 0, result.stdout + result.stderr
    refined = torch.load(tmp_path / "retrain/checkpoints/final.ckpt", map_location="cpu")
    assert any(
        not torch.equal(value, refined["state_dict"][name])
        for name, value in state["state_dict"].items()
        if name.startswith("net.")
    )
    for name, expected in state["state_dict"].items():
        if name.startswith("coefficients."):
            torch.testing.assert_close(refined["state_dict"][name], expected, rtol=0, atol=0)
    assert refined["hyper_parameters"]["coefficient_signature"] == signature

    for mutation, source in (
        ("material", original_source.replace(f"BETA = {beta}", f"BETA = {1 - beta}")),
        (
            "representation",
            original_source.replace(
                f'REPRESENTATION = "{representation}"',
                f'REPRESENTATION = "{"table" if representation == "formula" else "formula"}"',
            ),
        ),
    ):
        assert source != original_source
        functions.write_text(source)
        for mode in ("predict", "retrain"):
            output = tmp_path / f"rejected-{mutation}-{mode}"
            result = launch(task, output, mode, str(checkpoint))
            assert result.returncode != 0
            # functions.py is also the hard-constraint identity. This guard
            # intentionally runs before the narrower coefficient signature.
            assert "hard constraint definition differs" in result.stdout + result.stderr
            assert not (output / "predictions.mat").exists()


def test_external_material_change_reaches_coefficient_guard(tmp_path: Path) -> None:
    task = tmp_path / "input"
    make_factory(task, 2, 1, "table", 19)
    functions = task / "functions.py"
    declaration = (
        'coefficients={"conductivity": conductivity if REPRESENTATION == "formula" else table()},'
    )
    text = functions.read_text()
    assert text.count(declaration) == 1
    functions.write_text(text.replace(declaration, "coefficients={},"))
    config = CASES["heat_neumann_polynomial2d"].config("cpu", 1, 19)
    config["domain"]["time"].update(interval=[0, 1], points=3)
    for axis in config["domain"]["space"].values():
        axis["points"] = 3
    config["network"].update(hidden_layers=[4, 4], spatial_encoding="neumann_polynomial")
    config["training"].update(
        constraints="hard_initial_robin",
        hard_execution="eager",
        collocation={"sampling": "lhs", "points": 7},
        batch_size=7,
    )
    config["coefficients"] = {"conductivity": {"file": "conductivity.npz"}}
    (task / "config.yaml").write_text(yaml.safe_dump(config))
    x, y = np.array([0, 0.125, 0.5, 1]), np.array([0, 0.25, 0.75, 1])
    values = (1 + (x[:, None] + y[None, :]) / 2) / 8
    material = task / "conductivity.npz"
    np.savez(material, x=x, y=y, values=values)
    result = launch(task, tmp_path / "train")
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = tmp_path / "train/checkpoints/final.ckpt"
    # The Python source, initial data and constraints stay byte-identical.
    source_before = functions.read_bytes()
    np.savez(material, x=x, y=y, values=values * 2)
    for mode in ("predict", "retrain"):
        output = tmp_path / f"rejected-{mode}"
        result = launch(task, output, mode, str(checkpoint))
        assert result.returncode != 0
        assert "coefficient definitions differ" in result.stdout + result.stderr
        assert not (output / "predictions.mat").exists()
    assert functions.read_bytes() == source_before
