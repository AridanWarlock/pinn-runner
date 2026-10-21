"""Compare every fixed-state budget pair and verify the actual shared 500-step states."""

import csv
import hashlib
import json
import runpy
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.fingerprints import tensor_fingerprint  # noqa: E402
from research_tools.comparison import compare  # noqa: E402
from research_tools.evidence import read_bundle  # noqa: E402


def main():
    left, right = [
        read_bundle(ROOT / f"numerical-reports/{job}-compact")
        for job in ("nonheat033", "nonheat034")
    ]
    lm, rm = left["manifest"], right["manifest"]
    normalized = json.loads(json.dumps(lm))
    normalized["report_parameters"]["lbfgs_snapshots"] = rm["report_parameters"]["lbfgs_snapshots"]
    assert normalized == rm and left["environment"] == right["environment"]
    assert len(left["trials"]) == len(right["trials"]) == 5
    prefix = ROOT / "numerical-reports/nonheat034-prefix-audit"
    prefix.mkdir(exist_ok=False)
    helper = runpy.run_path(
        str(ROOT / "docs/experiments/2026-10-20-unforced-burgers-objective-audit.py")
    )
    origin = helper["download"]("nonheat034", prefix, right["trials"], 500)
    old_folder = ROOT / "numerical-reports/nonheat033-objective-audit"
    old_origin = json.loads((old_folder / "origin.json").read_text())
    goals = rm["case_limits"]["burgers_unforced_nu001"] | rm["physical_limits"]
    rows, pairs, shared = [], [], []
    for index, b in enumerate(right["trials"]):
        ai = next(i for i, t in enumerate(left["trials"]) if t["seed"] == b["seed"])
        a = left["trials"][ai]
        for key in ("initial_state_sha256", "initial_parameters_sha256", "fixed_objective_sha256"):
            assert a[key] == b[key]
        old_state = old_folder / f"state-{ai}.pt"
        new_state = prefix / f"state-{index}.pt"
        for path, checksum in (
            (old_state, old_origin[ai]["sha256"]),
            (new_state, origin[index]["sha256"]),
        ):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
        old_hash, new_hash = [
            tensor_fingerprint(torch.load(p, map_location="cpu", weights_only=True))
            for p in (old_state, new_state)
        ]
        pairs.append(
            dict(
                seed=b["seed"],
                initial_state_equal=True,
                initial_parameters_equal=True,
                samples_equal=True,
                shared_500_state_equal=old_hash == new_hash,
                baseline_500_sha256=old_hash,
                new_500_sha256=new_hash,
            )
        )
        for metric, limit in goals.items():
            values = [
                t["snapshots"][-1]["metrics"].get(metric, t["snapshots"][-1].get(metric))
                for t in (a, b)
            ]
            rows.append(
                dict(
                    seed=b["seed"],
                    metric=metric,
                    limit=limit,
                    baseline_500=values[0],
                    extended_2000=values[1],
                    ratio=values[1] / values[0],
                    baseline_passed=values[0] <= limit,
                    extended_passed=values[1] <= limit,
                )
            )
        for step in (0, 250, 500):
            old, new = [next(s for s in t["snapshots"] if s["lb_steps"] == step) for t in (a, b)]
            errors = []
            for metric, limit in goals.items():
                values = [s["metrics"].get(metric, s.get(metric)) for s in (old, new)]
                errors.append(abs(values[0] - values[1]) / limit)
            shared.append(
                dict(
                    seed=b["seed"],
                    lb_steps=step,
                    maximum_difference_over_goal=max(errors),
                    all_nine_metrics_equal=all(x == 0 for x in errors),
                )
            )
    summary = dict(pairs=pairs, metrics={}, cost={}, shared_snapshots=shared)
    for metric in goals:
        ratios = [r["ratio"] for r in rows if r["metric"] == metric]
        summary["metrics"][metric] = dict(
            minimum_ratio=min(ratios),
            median_ratio=statistics.median(ratios),
            maximum_ratio=max(ratios),
            improved=sum(x < 1 for x in ratios),
            regressed=sum(x > 1 for x in ratios),
        )
    for job, e in (("nonheat033", left), ("nonheat034", right)):
        trials = e["trials"]
        summary["cost"][job] = dict(
            passed=compare(e)["passed"],
            training_seconds=sum(t["training_seconds_with_snapshots"] for t in trials),
            process_seconds=sum(t["process_seconds"] for t in trials),
            closures=sum(t["snapshots"][-1]["closures"] for t in trials),
            adam_residual_points=sum(t["snapshots"][-1]["adam_residual_points"] for t in trials),
            lbfgs_residual_points=sum(t["snapshots"][-1]["lbfgs_residual_points"] for t in trials),
            confirmed_crossings=sum(t["crossing"]["status"] == "observed" for t in trials),
        )
    summary["scope"] = (
        "Five fixed pairs; changed budget and snapshot schedule; all failures retained, no speedup or universal probability."
    )
    base = Path(__file__).with_name("2026-10-21-unforced-burgers-budget-pairs")
    for suffix, data in ((".csv", rows), ("-shared.csv", shared)):
        target = Path(str(base) + suffix)
        assert not target.exists()
        with target.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    assert not base.with_suffix(".json").exists()
    base.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                shared_500_states_equal=sum(p["shared_500_state_equal"] for p in pairs),
                shared_snapshots_equal=sum(p["all_nine_metrics_equal"] for p in shared),
                cost=summary["cost"],
            )
        )
    )


if __name__ == "__main__":
    main()
