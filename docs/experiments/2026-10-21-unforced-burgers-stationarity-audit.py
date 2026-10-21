"""Actual-objective gradients and restored directional controls, without fitting."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.cases import CASES  # noqa: E402
from benchmarks.convergence import ConvergenceArm, ConvergencePlan  # noqa: E402
from benchmarks.fingerprints import objective_fingerprint, tensor_fingerprint  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402
from pinnstorch.training.refinement import FixedObjective  # noqa: E402
from research_tools.evidence import read_bundle  # noqa: E402


def main():
    torch.set_num_threads(1)
    output = ROOT / "numerical-reports/unforced037-stationarity"
    output.mkdir(exist_ok=False)
    rows = []
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
            factory = SolverFactory(
                TaskSettings.read(plan.config(trial["seed"], ConvergenceArm())),
                load_problem(case.source),
                directory,
                case.source.parent,
            )
            objects = factory.build(predicting=False)
            objective = FixedObjective.capture(objects.data, torch.device("cpu"))
            assert objective_fingerprint(objective.batches) == trial["fixed_objective_sha256"]
            assert tensor_fingerprint(objects.model.state_dict()) == trial["initial_state_sha256"]
            objects.model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
            )
            objects.data.setup("fit")
            model = objects.model.eval()
            model.function_mapping = objects.data.function_mapping
            row = dict(
                job_id=job,
                seed=trial["seed"],
                checkpoint_sha256=origin[index]["sha256"],
                optimizer_steps=0,
            )
            for dtype in (torch.float32, torch.float64):
                model.to(dtype=dtype)
                before = tensor_fingerprint(model.state_dict())
                batch = FixedObjective(
                    {
                        name: (
                            [x.to(dtype=dtype) for x in spatial],
                            t.to(dtype=dtype),
                            {k: v.to(dtype=dtype) for k, v in target.items()},
                        )
                        for name, (spatial, t, target) in objective.batches.items()
                    }
                )

                def loss():
                    value, _ = model.model_step(batch.fresh())
                    return value

                value = loss()
                spatial, clock, _ = batch.fresh()["equations"]
                coordinates = [x.requires_grad_(True) for x in (*spatial, clock)]
                physical = dict(model(coordinates[:1], coordinates[1]))
                explicit = (
                    factory.adapter.residuals(physical, *coordinates)["burgers"].square().mean()
                )
                assert torch.allclose(value, explicit, rtol=1e-6, atol=1e-12)
                parameters = tuple(model.parameters())
                gradient = torch.cat([g.flatten() for g in torch.autograd.grad(value, parameters)])
                assert gradient.numel() == 2241 and torch.isfinite(gradient).all()
                label = "fp32" if dtype == torch.float32 else "fp64"
                row.update(
                    {
                        f"{label}_objective": float(value.detach()),
                        f"{label}_gradient_l2": float(gradient.norm()),
                        f"{label}_gradient_inf": float(gradient.abs().max()),
                    }
                )
                if dtype == torch.float64:
                    direction = (
                        gradient / gradient.norm()
                        if gradient.norm() > 0
                        else torch.ones_like(gradient) / gradient.numel() ** 0.5
                    )
                    derivative = float(torch.dot(gradient, direction))
                    original = torch.cat([p.detach().flatten() for p in parameters])

                    def assign(vector):
                        start = 0
                        with torch.no_grad():
                            for p in parameters:
                                p.copy_(vector[start : start + p.numel()].reshape(p.shape))
                                start += p.numel()

                    for h, key in ((1e-5, "h1"), (5e-6, "h2")):
                        try:
                            assign(original + h * direction)
                            plus = float(loss().detach())
                            assign(original - h * direction)
                            minus = float(loss().detach())
                        finally:
                            assign(original)
                        observed = (plus - minus) / (2 * h)
                        row[f"{key}_fd_absolute_error"] = abs(observed - derivative)
                        row[f"{key}_fd_passed"] = abs(observed - derivative) <= 1e-9 + 1e-3 * abs(
                            derivative
                        )
                    row["fp64_directional_derivative"] = derivative
                assert tensor_fingerprint(model.state_dict()) == before
            row["states_unchanged"] = True
            rows.append(row)
    assert len(rows) == 10
    base = Path(__file__).with_name("2026-10-21-unforced-burgers-stationarity")
    assert not base.with_suffix(".csv").exists() and not base.with_suffix(".json").exists()
    with base.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        states=10,
        optimizer_steps=0,
        parameters=2241,
        finite_difference_controls=20,
        passed_controls=sum(r[f"{k}_fd_passed"] for r in rows for k in ("h1", "h2")),
        all_states_unchanged=True,
        protocol_sha256=hashlib.sha256(
            (ROOT / "docs/unforced-burgers-stationarity-protocol-2026-10-21.md").read_bytes()
        ).hexdigest(),
        scope="CPU FP32/FP64 diagnostics; no exact CUDA stopping reconstruction or capacity proof.",
    )
    base.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
