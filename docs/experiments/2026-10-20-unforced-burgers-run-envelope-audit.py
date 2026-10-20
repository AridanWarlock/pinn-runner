"""Discrete L1 stability diagnostic on fixed trained states; CPU, no fitting."""

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
from benchmarks.cases import CASES  # noqa: E402
from benchmarks.catalog import UNFORCED_VISCOSITIES  # noqa: E402
from benchmarks.convergence import ConvergenceArm, ConvergencePlan  # noqa: E402
from benchmarks.fingerprints import tensor_fingerprint  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402
from research_tools.evidence import read_bundle  # noqa: E402

BASE = Path(__file__).parent


def integrals(prediction, residual, reference, x, t):
    error = prediction - reference
    e1 = np.trapz(np.abs(error), x, axis=0)
    r1 = np.trapz(np.abs(residual), x, axis=0)
    bound = e1[0] + np.concatenate(([0.0], np.cumsum((r1[1:] + r1[:-1]) * np.diff(t) / 2)))
    e2 = np.sqrt(np.trapz(error**2, x, axis=0))
    norm = np.sqrt(np.trapz(reference**2, x, axis=0))
    diagnostic_l2_bound = np.sqrt((np.max(np.abs(prediction), axis=0) + 1) * bound) / norm
    return dict(
        e1=e1, bound=bound, relative_l2=e2 / norm, diagnostic_relative_l2_bound=diagnostic_l2_bound
    )


def main():
    torch.set_num_threads(1)
    output = ROOT / "numerical-reports/unforced036-run-envelope"
    output.mkdir(exist_ok=False)
    rows, snapshots = [], []
    x, t = np.linspace(-1, 1, 801), np.linspace(0, 1, 401)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    flat_x, flat_t = xx.reshape(-1), tt.reshape(-1)
    for job in ("nonheat032", "nonheat033"):
        evidence = read_bundle(ROOT / f"numerical-reports/{job}-compact")
        manifest = evidence["manifest"]
        folder = ROOT / f"numerical-reports/{job}-objective-audit"
        origin = json.loads((folder / "origin.json").read_text())
        with (BASE / f"2026-10-20-{job}-unforced-objective.csv").open() as stream:
            controls = list(csv.DictReader(stream))
        for index, trial in enumerate(evidence["trials"]):
            case = CASES[trial["case"]]
            viscosity = UNFORCED_VISCOSITIES[case.name]
            checkpoint = folder / f"state-{index}.pt"
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == origin[index]["sha256"]
            control = next(
                r for r in controls if r["case"] == case.name and int(r["seed"]) == trial["seed"]
            )
            plan = ConvergencePlan(
                case,
                500,
                (0, 100, 250, 500),
                manifest["case_limits"][case.name],
                "hard_initial_dirichlet",
                manifest["report_parameters"]["points_override"],
                "cpu",
                1e-12,
            )
            directory = output / f"{job}-{case.name}-{trial['seed']}"
            directory.mkdir()
            factory = SolverFactory(
                TaskSettings.read(plan.config(trial["seed"], ConvergenceArm())),
                load_problem(case.source),
                directory,
                case.source.parent,
            )
            model = factory.build(predicting=False).model
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
            )
            assert tensor_fingerprint(model.state_dict()) == control["final_state_sha256"]
            model.double().eval()
            before = tensor_fingerprint(model.state_dict())
            prediction, residual = np.empty(xx.size), np.empty(xx.size)
            for start in range(0, xx.size, 2048):
                stop = min(start + 2048, xx.size)
                xc, tc = [
                    torch.tensor(a[start:stop, None], dtype=torch.float64, requires_grad=True)
                    for a in (flat_x, flat_t)
                ]
                u = model([xc], tc)["u"]
                ux, ut = torch.autograd.grad(u, (xc, tc), torch.ones_like(u), create_graph=True)
                uxx = torch.autograd.grad(ux, xc, torch.ones_like(ux))[0]
                prediction[start:stop] = u.detach().numpy().flatten()
                residual[start:stop] = (ut + u * ux - viscosity * uxx).detach().numpy().flatten()
            prediction, residual = prediction.reshape(xx.shape), residual.reshape(xx.shape)
            reference = grid_reference(tuple(x), tuple(t), viscosity)
            fine = integrals(prediction, residual, reference, x, t)
            coarse = integrals(
                prediction[::2, ::2], residual[::2, ::2], reference[::2, ::2], x[::2], t[::2]
            )
            assert all(np.isfinite(a).all() for a in (*fine.values(), *coarse.values()))
            assert tensor_fingerprint(model.state_dict()) == before
            identity = dict(
                job_id=job,
                case=case.name,
                seed=trial["seed"],
                checkpoint_sha256=origin[index]["sha256"],
            )
            row = identity | dict(
                final_e1=fine["e1"][-1],
                final_b1=fine["bound"][-1],
                final_b1_over_e1=fine["bound"][-1] / fine["e1"][-1],
                max_e1_over_b1=float(np.max(fine["e1"][1:] / fine["bound"][1:])),
                max_discrete_violation=float(max(0, np.max(fine["e1"] - fine["bound"]))),
                e1_grid_change=float(
                    np.max(np.abs(fine["e1"][::2] - coarse["e1"])) / np.max(fine["e1"])
                ),
                b1_grid_change=float(
                    np.max(np.abs(fine["bound"][::2] - coarse["bound"])) / np.max(fine["bound"])
                ),
                final_relative_l2=fine["relative_l2"][-1],
                final_diagnostic_relative_l2_bound=fine["diagnostic_relative_l2_bound"][-1],
                optimizer_steps=0,
                promoted_state_unchanged=True,
            )
            rows.append(row)
            for j in (0, 100, 200, 300, 400):
                snapshots.append(
                    identity | dict(time=t[j], **{key: values[j] for key, values in fine.items()})
                )
    assert len(rows) == 15 and len(snapshots) == 75
    for suffix, data in (("-final.csv", rows), ("-snapshots.csv", snapshots)):
        target = BASE / ("2026-10-20-unforced-burgers-run-envelope" + suffix)
        assert not target.exists()
        with target.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    summary = dict(
        states=15,
        snapshots=75,
        optimizer_steps=0,
        device="cpu",
        dtype="torch.float64",
        coarse_grid=[401, 201],
        fine_grid=[801, 401],
        max_discrete_violation=max(r["max_discrete_violation"] for r in rows),
        max_e1_grid_change=max(r["e1_grid_change"] for r in rows),
        max_b1_grid_change=max(r["b1_grid_change"] for r in rows),
        max_e1_over_b1=max(r["max_e1_over_b1"] for r in rows),
        final_b1_over_e1_min=min(r["final_b1_over_e1"] for r in rows),
        final_b1_over_e1_max=max(r["final_b1_over_e1"] for r in rows),
        protocol_sha256=hashlib.sha256(
            (ROOT / "docs/unforced-burgers-observed-error-envelope-2026-10-20.md").read_bytes()
        ).hexdigest(),
        scope="FP64 interpretation of fixed FP32 weights; finite-grid diagnostics, no continuous certificate and no replacement of CUDA acceptance.",
    )
    target = BASE / "2026-10-20-unforced-burgers-run-envelope.json"
    assert not target.exists()
    target.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
