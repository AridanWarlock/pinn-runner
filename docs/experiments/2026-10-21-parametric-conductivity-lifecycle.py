"""Small CPU-only public CLI lifecycle, explicitly not a quality experiment."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pinnstorch.checkpoint_contract import validate_checkpoint_domain  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402

PREFIX = ROOT / "docs/experiments/2026-10-21-parametric-conductivity"
spec = importlib.util.spec_from_file_location("network_audit", f"{PREFIX}-network.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    folder = ROOT / "numerical-reports/parametric038-lifecycle"
    folder.mkdir(exist_ok=False)
    input_dir = folder / "input"
    input_dir.mkdir()
    original = Path(f"{PREFIX}-problem.py").read_bytes()
    (input_dir / "functions.py").write_bytes(original)
    cfg = module.config(11)
    cfg["training"].update(epochs=3, steps=3)
    cfg["runtime"]["save_training_predictions"] = False
    calls = []

    def save():
        (input_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    def call(name, mode, checkpoint=None):
        command = [
            sys.executable,
            "-B",
            str(ROOT / "run.py"),
            "--mode",
            mode,
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(folder / name),
        ]
        if checkpoint:
            command.extend(["--checkpoint", str(checkpoint)])
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=90,
            env=dict(
                os.environ, MPLCONFIGDIR=str(folder / "matplotlib"), PYTHONDONTWRITEBYTECODE="1"
            ),
        )
        (folder / f"{name}.stdout.txt").write_text(result.stdout)
        (folder / f"{name}.stderr.txt").write_text(result.stderr)
        calls.append(dict(name=name, mode=mode, exit_code=result.returncode))
        print(calls[-1], flush=True)
        return result

    save()
    result = call("train", "train")
    assert result.returncode == 0, result.stderr[-2000:]
    checkpoint = folder / "train/checkpoints/final.ckpt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, map_location="cpu")
    cfg["domain"]["space"]["x"]["points"] = 9
    cfg["domain"]["time"]["points"] = 7
    cfg["prediction"] = {"parameters": {"a": [-0.43, 0.17], "b": [0.19, -0.51]}}
    save()
    result = call("predict", "predict", checkpoint)
    assert result.returncode == 0, result.stderr[-2000:]
    predicted = loadmat(folder / "predict/predictions.mat")
    settings = TaskSettings.read(cfg)
    direct_dir = folder / "direct"
    direct_dir.mkdir()
    factory = SolverFactory(
        settings, load_problem(input_dir / "functions.py"), direct_dir, input_dir
    )
    objects = factory.build(predicting=True)
    model = objects.model
    # Direct weight replay independently checks the CLI result and original normalization.
    requested = [(float(v.min()), float(v.max())) for v in objects.mesh.spatial_domain]
    requested.append((0.0, 1.0))
    validate_checkpoint_domain(saved, settings.domain.input_names, requested)
    model.load_state_dict(saved["state_dict"], strict=True)
    grid = settings.domain.grid(settings.prediction.parameters)
    mesh = objects.mesh
    columns, time = mesh.collection_points(None, use_lhs=False)
    with torch.no_grad():
        direct = model(
            [
                torch.tensor(columns[:, i : i + 1], dtype=torch.float32)
                for i in range(columns.shape[1])
            ],
            torch.tensor(time, dtype=torch.float32),
        )["u"].numpy()
    actual = predicted["u"].reshape(-1, 1)
    # MAT data use (space-and-parameters, time); mesh uses the same row-major layout.
    max_difference = float(np.abs(actual - direct).max())
    prediction_passed = bool(np.allclose(actual, direct, atol=1e-6, rtol=1e-5))
    parameter_passed = all(
        np.array_equal(predicted[name], grid.spatial[i]) for i, name in ((1, "a"), (2, "b"))
    )
    cfg["training"].update(epochs=1, steps=1)
    cfg["training"]["optimizer"]["learning_rate"] = 1e-300
    save()
    result = call("retrain", "retrain", checkpoint)
    assert result.returncode == 0, result.stderr[-2000:]
    retrained = torch.load(folder / "retrain/checkpoints/final.ckpt", map_location="cpu")
    restored = all(
        torch.equal(value, retrained["state_dict"][key])
        for key, value in saved["state_dict"].items()
    )
    cfg["prediction"]["parameters"]["a"] = [0.8]
    save()
    outside = call("reject-outside", "predict", checkpoint)
    cfg["prediction"]["parameters"]["a"] = [0.17]
    save()
    (input_dir / "functions.py").write_bytes(original.replace(b"/ 8", b"/ 7"))
    changed = call("reject-coefficient", "predict", checkpoint)
    (input_dir / "functions.py").write_bytes(original)
    summary = dict(
        protocol_sha256=hashlib.sha256(
            (ROOT / "docs/parametric-conductivity-lifecycle-protocol-2026-10-21.md").read_bytes()
        ).hexdigest(),
        source_sha256=hashlib.sha256(original).hexdigest(),
        device="cpu",
        quality_training=False,
        initial_adam_steps=3,
        retrain_adam_steps=1,
        lbfgs_steps=0,
        calls=calls,
        prediction_shape=list(predicted["u"].shape),
        prediction_direct_max_difference=max_difference,
        prediction_matches=prediction_passed,
        parameters_preserved=parameter_passed,
        retrain_tensors_unchanged=restored,
        checkpoint_file_unchanged=digest == hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        reject_outside_passed=outside.returncode == 1,
        reject_changed_coefficient_passed=changed.returncode == 1,
        input_names=list(saved["hyper_parameters"]["input_names"]),
        normalization_bounds=saved["hyper_parameters"]["input_bounds"],
    )
    summary["all_passed"] = bool(
        prediction_passed
        and parameter_passed
        and restored
        and summary["checkpoint_file_unchanged"]
        and outside.returncode == 1
        and changed.returncode == 1
        and np.isfinite(actual).all()
    )
    Path(f"{PREFIX}-lifecycle.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))
    assert summary["all_passed"], "Preserve unsuccessful lifecycle"


if __name__ == "__main__":
    main()
