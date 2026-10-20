"""Preserve every paired density outcome, including regressions and added cost."""

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research_tools.comparison import compare  # noqa: E402
from research_tools.evidence import read_bundle  # noqa: E402

BASE = Path(__file__).parent


def read_csv(name):
    with (BASE / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    left, right = [
        read_bundle(ROOT / f"numerical-reports/{job}-compact")
        for job in ("nonheat032", "nonheat033")
    ]
    for e in (left, right):
        assert compare(e)["passed"] == 0
    lm, rm = left["manifest"], right["manifest"]
    case = "burgers_unforced_nu001"
    assert left["environment"] == right["environment"]
    for key in ("revision", "network", "layer_evaluation", "physical_limits", "absolute_quality"):
        assert lm[key] == rm[key]
    assert lm["case_limits"][case] == rm["case_limits"][case]
    assert lm["problem_sha256"][case] == rm["problem_sha256"][case]
    assert {k: v for k, v in lm["report_parameters"].items() if k != "points_override"} == {
        k: v for k, v in rm["report_parameters"].items() if k != "points_override"
    }
    limits = lm["case_limits"][case] | lm["physical_limits"]
    sampling = read_csv("2026-10-20-unforced-burgers-run-sampling.csv")
    rows, pairs = [], []
    for seed in (11, 29, 83, 271, 997):
        a, b = [
            next(t for t in e["trials"] if t["case"] == case and t["seed"] == seed)
            for e in (left, right)
        ]
        assert a["initial_state_sha256"] == b["initial_state_sha256"]
        assert a["initial_parameters_sha256"] == b["initial_parameters_sha256"]
        assert a["fixed_objective_sha256"] != b["fixed_objective_sha256"]
        for trial, count in ((a, 1024), (b, 4096)):
            audit = next(
                r for r in sampling if int(r["seed"]) == seed and int(r["points"]) == count
            )
            assert audit["initial_state_sha256"] == trial["initial_state_sha256"]
            assert audit["fixed_objective_sha256"] == trial["fixed_objective_sha256"]
        for metric, limit in limits.items():
            values = [
                t["snapshots"][-1]["metrics"].get(metric, t["snapshots"][-1].get(metric))
                for t in (a, b)
            ]
            rows.append(
                dict(
                    seed=seed,
                    metric=metric,
                    limit=limit,
                    baseline=values[0],
                    dense=values[1],
                    ratio=values[1] / values[0],
                    baseline_passed=values[0] <= limit,
                    dense_passed=values[1] <= limit,
                )
            )
        pairs.append(
            dict(
                seed=seed,
                initial_state_equal=True,
                initial_parameters_equal=True,
                samples_changed=True,
            )
        )
    target = BASE / "2026-10-20-unforced-burgers-density-pairs.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(pairs=pairs, metrics={}, cost={}, objective={})
    for metric in limits:
        ratios = [r["ratio"] for r in rows if r["metric"] == metric]
        summary["metrics"][metric] = dict(
            minimum_ratio=min(ratios),
            median_ratio=statistics.median(ratios),
            maximum_ratio=max(ratios),
            improved=sum(x < 1 for x in ratios),
            regressed=sum(x > 1 for x in ratios),
        )
    for job, evidence in (("nonheat032", left), ("nonheat033", right)):
        trials = [t for t in evidence["trials"] if t["case"] == case]
        summary["cost"][job] = dict(
            trials=5,
            training_seconds=sum(t["training_seconds_with_snapshots"] for t in trials),
            process_seconds=sum(t["process_seconds"] for t in trials),
            closures=sum(t["snapshots"][-1]["closures"] for t in trials),
            adam_residual_points=sum(t["snapshots"][-1]["adam_residual_points"] for t in trials),
            lbfgs_residual_points=sum(t["snapshots"][-1]["lbfgs_residual_points"] for t in trials),
        )
        objective = read_csv(f"2026-10-20-{job}-unforced-objective.csv")
        for name in sorted({r["case"] for r in objective}):
            selected = [r for r in objective if r["case"] == name]
            summary["objective"][f"{job}:{name}"] = {
                key: dict(
                    minimum=min(float(r[key]) for r in selected),
                    maximum=max(float(r[key]) for r in selected),
                )
                for key in (
                    "training_rms",
                    "holdout_to_training_rms",
                    "central_squared_residual_share",
                    "late_squared_residual_share",
                )
            }
        summary["cost"][job]["passed"] = sum(t["snapshots"][-1]["passed"] for t in trials)
    summary["scope"] = (
        "Post-hoc descriptive pairs on five fixed states; different objectives and costs, no speedup or causal attribution. CPU residual audit does not replace CUDA acceptance."
    )
    (BASE / "2026-10-20-unforced-burgers-density-pairs.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
