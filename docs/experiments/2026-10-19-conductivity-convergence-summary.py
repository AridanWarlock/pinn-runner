"""Summarize an intact completed coefficient study; negative accuracy results are retained."""

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research_tools.comparison import compare  # noqa: E402
from research_tools.evidence import read_bundle  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--output-prefix", type=Path, required=True)
args = parser.parse_args()
evidence = read_bundle(args.bundle)
comparison = compare(evidence)
manifest = evidence["manifest"]
if "physics" not in manifest:
    raise ValueError("An explicitly registered coefficient study is required")
paths = [args.output_prefix.with_suffix(suffix) for suffix in (".csv", ".json")]
if any(path.exists() for path in paths):
    raise FileExistsError("Summary outputs already exist")
rows = []
for trial in evidence["trials"]:
    limits = manifest["case_limits"][trial["case"]]
    snapshot = trial["snapshots"][-1]
    metrics = snapshot["metrics"]
    failed = [key for key, threshold in limits.items() if metrics[key] > threshold]
    initial_max = max(s["metrics"]["initial_rmse"] for s in trial["snapshots"])
    boundary_max = max(s["boundary_max"] for s in trial["snapshots"])
    if initial_max > manifest["physical_limits"]["initial_rmse"]:
        failed.append("initial_rmse")
    if boundary_max > manifest["physical_limits"]["boundary_max"]:
        failed.append("boundary_max")
    rows.append(
        dict(
            case=trial["case"],
            coefficient_representation=trial["coefficient_representation"],
            seed=trial["seed"],
            **metrics,
            initial_max=initial_max,
            boundary_max=boundary_max,
            absolute_passed=not failed,
            failed_metrics=";".join(failed),
            training_seconds=trial["training_seconds_with_snapshots"],
            process_seconds=trial["process_seconds"],
            closures=snapshot["closures"],
            adam_residual_points=snapshot["adam_residual_points"],
            lbfgs_residual_points=snapshot["lbfgs_residual_points"],
            peak_allocated_bytes=trial["peak_allocated_bytes"],
            peak_reserved_bytes=trial["peak_reserved_bytes"],
            initial_parameters_sha256=trial["initial_parameters_sha256"],
            crossing=trial["crossing"]["status"],
        )
    )
with paths[0].open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
groups = []
for case in manifest["case_limits"]:
    for representation in ("formula", "table"):
        subset = [
            r
            for r in rows
            if r["case"] == case and r["coefficient_representation"] == representation
        ]
        metric_names = [
            *manifest["case_limits"][case],
            "mean_heat_max_error",
            "training_seconds",
            "closures",
        ]
        groups.append(
            dict(
                case=case,
                coefficient_representation=representation,
                trials=len(subset),
                passed=sum(r["absolute_passed"] for r in subset),
                metrics={
                    name: dict(
                        min=min(r[name] for r in subset),
                        median=statistics.median(r[name] for r in subset),
                        max=max(r[name] for r in subset),
                    )
                    for name in metric_names
                },
            )
        )
summary = dict(
    source_revision=manifest["revision"],
    provenance=evidence.get("provenance"),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    evidence_sha256=hashlib.sha256((args.bundle / "evidence.json").read_bytes()).hexdigest(),
    trials=len(rows),
    absolute_passed=sum(r["absolute_passed"] for r in rows),
    failed_by_metric={
        name: sum(name in r["failed_metrics"].split(";") for r in rows)
        for name in {*next(iter(manifest["case_limits"].values())), "initial_rmse", "boundary_max"}
    },
    distinct_initial_states=len({r["initial_parameters_sha256"] for r in rows}),
    training_seconds_total=sum(r["training_seconds"] for r in rows),
    process_seconds_total=sum(r["process_seconds"] for r in rows),
    max_peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in rows),
    max_peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in rows),
    comparison_passed=comparison["comparison_passed"],
    groups=groups,
    limitations="Registered completed matrix only. Timing is diagnostic. Two states do not replace five-state confirmation; cancelled kappa021 is not pooled.",
)
paths[1].write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(
    json.dumps(
        {
            k: summary[k]
            for k in ("trials", "absolute_passed", "failed_by_metric", "comparison_passed")
        }
    )
)
