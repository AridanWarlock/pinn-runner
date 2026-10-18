"""Registered fresh-process public prediction, without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path(r"C:\Projects\pinn-runner")
    if (
        sys.platform != "win32"
        or Path.cwd() != root
        or a.output.exists()
        or not sys.dont_write_bytecode
    ):
        p.error("Guarded Windows root and fresh output required")
    import numpy as np
    import torch
    import yaml
    from scipy.io import loadmat

    from benchmarks.cases import CASES
    from benchmarks.fingerprints import tensor_fingerprint
    from pinnstorch.runner.checkpoint_io import load_checkpoint
    from pinnstorch.runner.factory import SolverFactory
    from pinnstorch.runner.problem import load_problem
    from pinnstorch.runner.settings import TaskSettings
    from research_tools.public_prediction import compare, inventory, validate
    from research_tools.queue import registered_manifest, verify_snapshot
    from research_tools.schema import digest, inside

    verify_snapshot(a.manifest.parent)
    m = validate(registered_manifest(a.manifest.parent))
    if os.environ.get("PINN_PINNED_REVISION") != m["revision"]:
        raise ValueError("Pinned prediction revision differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    environment = dict(
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        default_dtype=str(torch.get_default_dtype()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=False,
        cudnn_allow_tf32=False,
    )
    if environment != m["environment_requirements"]:
        raise ValueError("Public runtime differs")
    source = inside(root, m["trial"])
    weights = inside(source, m["weights"])
    configuration = source / "config.yaml"
    if digest(weights) != m["weights_sha256"] or digest(configuration) != m["config_sha256"]:
        raise ValueError("Public source differs")
    a.output.mkdir(parents=True)
    task = a.output / "input"
    task.mkdir()
    shutil.copyfile(CASES[m["case"]].source, task / "functions.py")
    config = yaml.safe_load(configuration.read_text())
    config["training"]["hard_execution"] = "eager"
    config["training"]["batch_size"] = 128
    for axis in config["domain"]["space"].values():
        axis["points"] = 17
    config["domain"]["time"]["points"] = 31
    (task / "config.yaml").write_text(yaml.safe_dump(config))
    settings = TaskSettings.load(task / "config.yaml")
    if settings.runtime.device != "cuda" or settings.runtime.precision != "32-true":
        raise ValueError("Saved public task must use CUDA FP32")
    objects = SolverFactory(settings, load_problem(task / "functions.py"), a.output, task).build(
        predicting=True
    )
    state = load_checkpoint(weights, map_location="cpu")
    objects.model.load_state_dict(state, strict=True)
    fingerprint = tensor_fingerprint(objects.model.state_dict())
    checkpoint = a.output / "prediction.ckpt"
    torch.save(
        dict(
            state_dict=objects.model.state_dict(),
            hyper_parameters=dict(objects.model.hparams),
            research_conversion=dict(
                source_sha256=m["weights_sha256"], optimizer_state_restored=False
            ),
        ),
        checkpoint,
    )
    loaded = load_checkpoint(checkpoint)["state_dict"]
    if loaded.keys() != state.keys() or not all(
        torch.equal(loaded[k], v) for k, v in state.items()
    ):
        raise ValueError("Conversion changed source state")
    env = dict(os.environ)
    for name in ("APPDATA", "LOCALAPPDATA", "MPLCONFIGDIR"):
        target = a.output / name.lower()
        target.mkdir()
        env[name] = str(target)
    bootstrap = """import sys,runpy,atexit,json,os
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).parent))
import torch
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
def audit():
    environment=dict(torch=torch.__version__,cuda_runtime=torch.version.cuda,
        default_dtype=str(torch.get_default_dtype()),float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    result=dict(environment=environment,gpu=torch.cuda.get_device_name(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    Path(os.environ['PINN_PUBLIC_RUNTIME']).write_text(json.dumps(result))
atexit.register(audit)
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    rows = []
    artifacts = {"prediction.ckpt": digest(checkpoint)}
    control = None
    max_difference = 0.0
    for repeat, batch, rule in inventory():
        name = f"r{repeat}-b{batch}-{rule}"
        config["training"].update(hard_execution=rule, batch_size=batch)
        (task / "config.yaml").write_text(yaml.safe_dump(config))
        target = a.output / name
        runtime_path = a.output / (name + "-runtime.json")
        env["PINN_PUBLIC_RUNTIME"] = str(runtime_path)
        command = [
            sys.executable,
            "-B",
            "-c",
            bootstrap,
            str(ROOT / "run.py"),
            "--input-dir",
            str(task),
            "--output-dir",
            str(target),
            "--mode",
            "predict",
            "--checkpoint",
            str(checkpoint),
        ]
        with (a.output / (name + ".log")).open("wb") as log:
            started = time.perf_counter()
            result = subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=90,
                check=False,
            )
            seconds = time.perf_counter() - started
        if result.returncode:
            raise RuntimeError(f"Public process {name} exited {result.returncode}")
        runtime = json.loads(runtime_path.read_text())
        field = loadmat(target / "predictions.mat")["u"]
        if control is None:
            control = field.copy()
        np.testing.assert_allclose(field, control, rtol=2e-6, atol=2e-6)
        max_difference = max(max_difference, float(np.max(np.abs(field - control))))
        rows.append(
            dict(
                **runtime,
                repeat=repeat,
                batch=batch,
                rule=rule,
                seconds=seconds,
                exit_code=result.returncode,
                field_shape=list(field.shape),
                finite=bool(np.isfinite(field).all()),
                field_sha256=hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest(),
            )
        )
        artifacts[name + "-runtime.json"] = digest(runtime_path)
        artifacts[name + ".log"] = digest(a.output / (name + ".log"))
        artifacts[name + "/predictions.mat"] = digest(target / "predictions.mat")
    if (
        digest(weights) != m["weights_sha256"]
        or digest(configuration) != m["config_sha256"]
        or tensor_fingerprint(objects.model.state_dict()) != fingerprint
    ):
        raise ValueError("Source changed during public prediction")
    e = dict(
        contract=1,
        kind="public-prediction-evidence",
        manifest=m,
        revision=m["revision"],
        environment=environment,
        gpu=torch.cuda.get_device_name(),
        source_unchanged=True,
        converted_state_equal=True,
        cross_batch_close=True,
        cross_batch_max_difference=max_difference,
        trials=rows,
        artifact_sha256=artifacts,
    )
    compare(e)
    (a.output / "public-prediction.json").write_text(json.dumps(e, indent=2) + "\n")


if __name__ == "__main__":
    main()
