"""Count actual factory-created fixed points without fitting any model."""

import csv
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


def main():
    torch.set_num_threads(1)
    protocol = json.loads(
        (
            ROOT / "docs/experiments/2026-10-20-nonheat032-unforced-quality-protocol.json"
        ).read_text()
    )
    case = CASES["burgers_unforced_nu001"]
    output = ROOT / "numerical-reports/unforced036-sampling"
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for count in (1024, 4096):
        plan = ConvergencePlan(
            case,
            500,
            (0, 100, 250, 500),
            protocol["case_limits"][case.name],
            "hard_initial_dirichlet",
            count,
            "cpu",
            1e-12,
        )
        for seed in (11, 29, 83, 271, 997):
            folder = output / f"{count}-{seed}"
            folder.mkdir()
            objects = SolverFactory(
                TaskSettings.read(plan.config(seed, ConvergenceArm())),
                load_problem(case.source),
                folder,
                case.source.parent,
            ).build(predicting=False)
            batch = FixedObjective.capture(objects.data, torch.device("cpu"))
            spatial, clock, _ = batch.batches["equations"]
            x, t = spatial[0].flatten(), clock.flatten()
            assert x.numel() == count and t.numel() == count
            rows.append(
                dict(
                    points=count,
                    seed=seed,
                    optimizer_steps=0,
                    regional_count=int(((x.abs() <= 0.1) & (t >= 0.15)).sum()),
                    narrow_count=int(((x.abs() <= 0.02) & (t >= 0.3) & (t <= 0.8)).sum()),
                    peak_window_count=int(((x.abs() <= 0.02) & (t >= 0.45) & (t <= 0.6)).sum()),
                    fixed_objective_sha256=objective_fingerprint(batch.batches),
                    initial_state_sha256=tensor_fingerprint(objects.model.state_dict()),
                )
            )
    target = Path(__file__).with_name("2026-10-20-unforced-burgers-run-sampling.csv")
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
