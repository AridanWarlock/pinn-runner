"""Fixed-state diagnostic of reflection asymmetry in the unpaired training objective."""

import csv
import hashlib
import json
import statistics
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
    directory = ROOT / "numerical-reports/unforced037-sampling-symmetry"
    directory.mkdir(exist_ok=False)
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
            output = directory / f"{job}-{trial['seed']}"
            output.mkdir()
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
            factory = SolverFactory(
                TaskSettings.read(plan.config(trial["seed"], ConvergenceArm())),
                load_problem(case.source),
                output,
                case.source.parent,
            )
            objects = factory.build(predicting=False)
            batch = FixedObjective.capture(objects.data, torch.device("cpu"))
            assert objective_fingerprint(batch.batches) == trial["fixed_objective_sha256"]
            assert tensor_fingerprint(objects.model.state_dict()) == trial["initial_state_sha256"]
            objects.model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
            )
            model = objects.model.double().eval()
            before = tensor_fingerprint(model.state_dict())
            spatial, clock, _ = batch.batches["equations"]

            def residual(sign):
                x = (sign * spatial[0].double()).clone().requires_grad_(True)
                t = clock.double().clone().requires_grad_(True)
                physical = dict(model([x], t))
                return factory.adapter.residuals(physical, x, t)["burgers"].detach().flatten()

            raw, mirror = residual(1), residual(-1)
            odd, even = (raw - mirror) / 2, (raw + mirror) / 2
            original = float(raw.square().mean())
            reflected = float(mirror.square().mean())
            paired = (original + reflected) / 2
            odd_energy = float(odd.square().mean())
            even_energy = float(even.square().mean())
            cross = float(2 * (odd * even).mean())
            defect = max(abs(paired - odd_energy - even_energy), abs(original - paired - cross))
            assert torch.isfinite(raw).all() and torch.isfinite(mirror).all()
            assert tensor_fingerprint(model.state_dict()) == before
            rows.append(
                dict(
                    job_id=job,
                    seed=trial["seed"],
                    points=4096,
                    optimizer_steps=0,
                    original_objective=original,
                    reflected_objective=reflected,
                    reflected_over_original=reflected / original,
                    reflected_over_original_rms=(reflected / original) ** 0.5,
                    paired_objective=paired,
                    residual_even_energy_fraction=even_energy / paired,
                    cross_over_paired=cross / paired,
                    identity_absolute_defect=defect,
                    identity_passed=defect <= 1e-12 * max(1, paired),
                    state_unchanged=True,
                )
            )
    assert len(rows) == 10
    base = Path(__file__).with_name("2026-10-21-unforced-burgers-sampling-symmetry")
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
        identities_passed=sum(r["identity_passed"] for r in rows),
        all_states_unchanged=True,
        groups={},
    )
    for job in ("nonheat033", "nonheat034"):
        selected = [r for r in rows if r["job_id"] == job]
        ratios = [r["reflected_over_original"] for r in selected]
        summary["groups"][job] = dict(
            reflected_objective_ratio_minimum=min(ratios),
            reflected_objective_ratio_median=statistics.median(ratios),
            reflected_objective_ratio_maximum=max(ratios),
            reflected_higher=sum(x > 1 for x in ratios),
        )
    summary["scope"] = (
        "Post-hoc finite-point diagnostic; no training or causal attribution; residual odd part is not generally the residual of the odd field."
    )
    base.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
