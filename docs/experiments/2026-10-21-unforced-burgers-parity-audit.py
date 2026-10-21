"""Fixed-state parity decomposition and independent projected-PDE controls on CPU."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.burgers import grid_reference  # noqa: E402
from benchmarks.burgers_quality import field_metrics  # noqa: E402
from benchmarks.cases import CASES  # noqa: E402
from benchmarks.convergence import ConvergenceArm, ConvergencePlan  # noqa: E402
from benchmarks.execution import holdout_inputs  # noqa: E402
from benchmarks.fingerprints import tensor_fingerprint  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402
from research_tools.evidence import read_bundle  # noqa: E402


def main():
    torch.set_num_threads(1)
    output = ROOT / "numerical-reports/unforced037-parity"
    output.mkdir(exist_ok=False)
    x, t = np.linspace(-1, 1, 1601), np.linspace(0, 1, 401)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    fx, ft = xx.reshape(-1), tt.reshape(-1)
    expected = grid_reference(tuple(x), tuple(t), 0.01)
    rows = []

    def integral(values):
        return float(np.trapz(np.trapz(values, x, axis=0), t))

    for job in ("nonheat033", "nonheat034"):
        e = read_bundle(ROOT / f"numerical-reports/{job}-compact")
        m = e["manifest"]
        folder = ROOT / f"numerical-reports/{job}-objective-audit"
        origin = json.loads((folder / "origin.json").read_text())
        for index, trial in enumerate(e["trials"]):
            case = CASES[trial["case"]]
            checkpoint = folder / f"state-{index}.pt"
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == origin[index]["sha256"]
            plan = ConvergencePlan(
                case,
                500,
                tuple(m["report_parameters"]["lbfgs_snapshots"]),
                m["case_limits"][case.name],
                "hard_initial_dirichlet",
                4096,
                "cpu",
                1e-12,
            )
            directory = output / f"{job}-{trial['seed']}"
            directory.mkdir()
            settings = TaskSettings.read(plan.config(trial["seed"], ConvergenceArm()))
            factory = SolverFactory(
                settings, load_problem(case.source), directory, case.source.parent
            )
            model = factory.build(predicting=False).model
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
            )
            model.double().eval()
            before = tensor_fingerprint(model.state_dict())
            prediction = np.empty(xx.size)
            with torch.no_grad():
                for start in range(0, xx.size, 4096):
                    xc, tc = [
                        torch.tensor(a[start : start + 4096, None], dtype=torch.float64)
                        for a in (fx, ft)
                    ]
                    prediction[start : start + 4096] = model([xc], tc)["u"].numpy().flatten()
            prediction = prediction.reshape(xx.shape)
            odd = (prediction - prediction[::-1]) / 2
            even = (prediction + prediction[::-1]) / 2
            total = integral((prediction - expected) ** 2)
            removed = integral(even**2)
            retained = integral((odd - expected) ** 2)
            defect = abs(total - removed - retained) / total
            row = dict(
                job_id=job,
                seed=trial["seed"],
                even_squared_error_fraction=removed / total,
                orthogonality_relative_defect=defect,
                orthogonality_passed=defect <= 1e-8,
                optimizer_steps=0,
            )
            for label, field in (("raw", prediction), ("projected", odd)):
                row.update(
                    {
                        f"{label}_{key}": value
                        for key, value in field_metrics(field, expected).items()
                    }
                )
            regional = np.random.default_rng(20261014).uniform(size=(2048, 2))
            regional[:, 0] = 0.2 * regional[:, 0] - 0.1
            regional[:, 1] = 0.15 + 0.85 * regional[:, 1]
            for label, points in (("global", holdout_inputs(settings)), ("regional", regional)):
                xc, tc = [
                    torch.tensor(points[:, j : j + 1], dtype=torch.float64, requires_grad=True)
                    for j in (0, 1)
                ]
                forward = dict(model([xc], tc))
                reflected_x = -xc
                reflected = dict(model([reflected_x], tc))

                def residual(value):
                    ux, ut = torch.autograd.grad(
                        value,
                        (xc, tc),
                        torch.ones_like(value),
                        create_graph=True,
                        retain_graph=True,
                    )
                    uxx = torch.autograd.grad(
                        ux, xc, torch.ones_like(ux), create_graph=True, retain_graph=True
                    )[0]
                    return ut + value * ux - 0.01 * uxx

                u, mirror = forward["u"], reflected["u"]
                vo = (u - mirror) / 2
                ve = (u + mirror) / 2
                direct = residual(vo)
                r = residual(u)
                mirror_r = factory.adapter.residuals(reflected, reflected_x, tc)["burgers"]
                vex = torch.autograd.grad(
                    ve, xc, torch.ones_like(ve), create_graph=True, retain_graph=True
                )[0]
                controlled = (r - mirror_r) / 2 - ve * vex
                delta = float((direct - controlled).abs().max().detach())
                scale = float(direct.abs().max().detach())
                row[f"{label}_projection_control_error"] = delta
                row[f"{label}_projection_control_passed"] = delta <= 1e-10 + 1e-9 * scale
                row[f"{label}_raw_pde_rms"] = float(r.detach().square().mean().sqrt())
                row[f"{label}_projected_pde_rms"] = float(direct.detach().square().mean().sqrt())
            assert tensor_fingerprint(model.state_dict()) == before
            row["state_unchanged"] = True
            rows.append(row)
    assert len(rows) == 10
    base = Path(__file__).with_name("2026-10-21-unforced-burgers-parity")
    assert not base.with_suffix(".csv").exists() and not base.with_suffix(".json").exists()
    with base.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        states=10,
        optimizer_steps=0,
        device="cpu",
        dtype="float64",
        grid=[1601, 401],
        orthogonality_passed=sum(r["orthogonality_passed"] for r in rows),
        projection_controls_passed=sum(
            r[f"{label}_projection_control_passed"]
            for r in rows
            for label in ("global", "regional")
        ),
        states_unchanged=True,
        protocol_sha256=hashlib.sha256(
            (ROOT / "docs/unforced-burgers-parity-protocol-2026-10-21.md").read_bytes()
        ).hexdigest(),
        scope="Post-hoc symmetry diagnostics; projected predictions are a different method and do not replace registered quality outcomes.",
    )
    base.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
