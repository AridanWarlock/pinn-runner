"""CPU-only final-state residual audit; no optimizer or acceptance changes."""

import argparse
import base64
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.cases import CASES  # noqa: E402
from benchmarks.convergence import ConvergenceArm, ConvergencePlan  # noqa: E402
from benchmarks.execution import holdout_inputs  # noqa: E402
from benchmarks.fingerprints import objective_fingerprint, tensor_fingerprint  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402
from pinnstorch.training.refinement import FixedObjective  # noqa: E402
from research_tools.evidence import read_bundle  # noqa: E402


def download(job, folder, trials, final_step):
    commands = []
    for trial in trials:
        name = f"eager-recursive-all-{trial['seed']}"
        relative = (
            f".gpu-jobs\\{job}\\r\\{trial['case']}\\processes\\{name}\\{name}\\lb-{final_step}.pt"
        )
        commands.append(f"$paths += '{relative}'")
    command = r"""Set-Location -LiteralPath 'C:\Projects\pinn-runner' -ErrorAction Stop
if ((Get-Location).Path -ne 'C:\Projects\pinn-runner') { exit 1 }
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$paths = @()
__PATHS__
$rows = @()
foreach ($path in $paths) {
  $bytes = [IO.File]::ReadAllBytes((Join-Path (Get-Location).Path $path))
  $rows += [pscustomobject]@{ path=$path; bytes=[Convert]::ToBase64String($bytes); sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() }
}
ConvertTo-Json -InputObject $rows -Depth 3 -Compress
""".replace("__PATHS__", "\n".join(commands))
    encoded = base64.b64encode(command.encode("utf-16le")).decode("ascii")
    data = json.loads(
        subprocess.check_output(
            [
                "ssh",
                "aridanPC",
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            text=True,
            timeout=60,
        )
    )
    assert len(data) == len(trials)
    for index, row in enumerate(data):
        payload = base64.b64decode(row.pop("bytes"), validate=True)
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        (folder / f"state-{index}.pt").write_bytes(payload)
    (folder / "origin.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-id", choices=("nonheat032", "nonheat033", "nonheat034"), required=True
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    evidence = read_bundle(ROOT / f"numerical-reports/{args.job_id}-compact")
    m = evidence["manifest"]
    folder = ROOT / f"numerical-reports/{args.job_id}-objective-audit"
    folder.mkdir(exist_ok=False)
    origin = download(
        args.job_id, folder, evidence["trials"], m["report_parameters"]["lbfgs_snapshots"][-1]
    )
    rows = []
    for index, trial in enumerate(evidence["trials"]):
        case = CASES[trial["case"]]
        assert (
            hashlib.sha256(case.source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            == m["problem_sha256"][case.name]
        )
        plan = ConvergencePlan(
            case,
            500,
            tuple(m["report_parameters"]["lbfgs_snapshots"]),
            m["case_limits"][case.name],
            "hard_initial_dirichlet",
            m["report_parameters"]["points_override"],
            "cpu",
            1e-12,
        )
        settings = TaskSettings.read(plan.config(trial["seed"], ConvergenceArm()))
        factory = SolverFactory(settings, load_problem(case.source), folder, case.source.parent)
        objects = factory.build(predicting=False)
        objective = FixedObjective.capture(objects.data, torch.device("cpu"))
        assert objective_fingerprint(objective.batches) == trial["fixed_objective_sha256"]
        assert tensor_fingerprint(objects.model.state_dict()) == trial["initial_state_sha256"]
        state = torch.load(folder / f"state-{index}.pt", map_location="cpu", weights_only=True)
        objects.model.load_state_dict(state, strict=True)
        objects.model.eval()
        before = tensor_fingerprint(objects.model.state_dict())

        def residual(points):
            columns = [
                torch.tensor(points[:, j : j + 1], dtype=torch.float32, requires_grad=True)
                for j in (0, 1)
            ]
            physical = dict(objects.model(columns[:1], columns[1]))
            return (
                factory.adapter.residuals(physical, *columns)["burgers"]
                .detach()
                .numpy()
                .flatten()
                .astype(np.float64)
            )

        spatial, clock, _ = objective.batches["equations"]
        train = residual(np.column_stack((spatial[0].numpy().flatten(), clock.numpy().flatten())))
        points = holdout_inputs(settings)
        control = residual(points)
        regional = np.random.default_rng(20261014).uniform(size=(2048, 2))
        regional[:, 0] = 0.2 * regional[:, 0] - 0.1
        regional[:, 1] = 0.15 + 0.85 * regional[:, 1]
        region = residual(regional)

        def rms(r):
            return float(np.sqrt(np.mean(r**2)))

        metrics = trial["snapshots"][-1]["metrics"]
        # Different CPU/GPU arithmetic is diagnostic, never replacement evidence.
        assert abs(rms(control) - metrics["pde_rmse"]) < 5e-5
        assert abs(rms(region) - metrics["regional_pde_rmse"]) < 5e-5
        row = dict(
            job_id=args.job_id,
            case=case.name,
            seed=trial["seed"],
            points=plan.points,
            optimizer_steps=0,
            checkpoint_sha256=origin[index]["sha256"],
            final_state_sha256=before,
            training_rms=rms(train),
            control_rms=rms(control),
            regional_rms=rms(region),
            holdout_to_training_rms=rms(control) / rms(train),
        )
        for label, mask in (
            ("central", np.abs(points[:, 0]) <= 0.1),
            ("late", points[:, 1] >= 0.8),
        ):
            row[label + "_holdout_count"] = int(mask.sum())
            row[label + "_squared_residual_share"] = float(
                np.sum(control[mask] ** 2) / np.sum(control**2)
            )
        assert tensor_fingerprint(objects.model.state_dict()) == before
        rows.append(row)
    date = "2026-10-21" if args.job_id == "nonheat034" else "2026-10-20"
    target = Path(__file__).with_name(f"{date}-{args.job_id}-unforced-objective.csv")
    assert not target.exists()
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            dict(job_id=args.job_id, states=len(rows), all_hashes_matched=True, optimizer_steps=0)
        )
    )


if __name__ == "__main__":
    main()
