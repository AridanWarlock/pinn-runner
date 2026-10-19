"""Verify the preregistered 8+12 union without manufacturing a third bundle."""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research_tools.comparison import compare  # noqa: E402
from research_tools.evidence import read_bundle  # noqa: E402
from research_tools.schema import identity  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pilot", type=Path, required=True)
parser.add_argument("--confirmation", type=Path, required=True)
parser.add_argument("--output-prefix", type=Path, required=True)
args = parser.parse_args()
outputs = [args.output_prefix.with_suffix(s) for s in (".csv", ".json")]
if any(p.exists() for p in outputs):
    raise FileExistsError("Union outputs already exist")
source = "fbcc9a29c6027eec4ffb89499fe8479ff774dbcc"
registrations = [
    (
        "kappa022",
        args.pilot,
        "439b1ec98ee88dd8dacffbcebb766369a2171d41",
        "docs/experiments/2026-10-19-kappa022-convergence-pilot-protocol.json",
        {19, 73},
    ),
    (
        "kappa023",
        args.confirmation,
        "b27e8746b59a5860ffdbfbc0505cebbb646e5178",
        "docs/experiments/2026-10-19-kappa023-convergence-confirmation-protocol.json",
        {1234, 2026, 4321},
    ),
]
all_trials, provenance, comparisons = [], [], []
reference_manifest = None
reference_environment = None
for stage, bundle, commit, protocol_path, seeds in registrations:
    evidence = read_bundle(bundle)
    comparison = compare(evidence)
    manifest = evidence["manifest"]
    assert manifest["revision"] == source
    assert {t["seed"] for t in evidence["trials"]} == seeds
    assert len(evidence["trials"]) == 4 * len(seeds)
    registered_bytes = subprocess.check_output(
        ["git", "show", f"{commit}:{protocol_path}"], cwd=ROOT
    )
    assert json.loads(registered_bytes) == manifest
    assert evidence["provenance"] == dict(
        protocol_commit=commit, protocol_sha256=hashlib.sha256(registered_bytes).hexdigest()
    )
    if reference_manifest is None:
        reference_manifest, reference_environment = manifest, evidence["environment"]
    else:
        for key in (
            "report_parameters",
            "case_limits",
            "physical_limits",
            "case_faces",
            "network",
            "physics",
            "comparison",
            "environment_requirements",
        ):
            assert manifest[key] == reference_manifest[key], key
        assert evidence["environment"] == reference_environment
    all_trials.extend(dict(t, stage=stage) for t in evidence["trials"])
    provenance.append(
        dict(
            stage=stage,
            protocol_commit=commit,
            protocol_sha256=hashlib.sha256(registered_bytes).hexdigest(),
            evidence_sha256=hashlib.sha256((bundle / "evidence.json").read_bytes()).hexdigest(),
        )
    )
    comparisons.append(comparison)
assert reference_manifest is not None
expected = {
    (case, seed, "eager", "recursive", "all", "general", "neumann_polynomial", representation)
    for case in ("heat_conductivity_beta0", "heat_conductivity_beta1")
    for seed in (19, 73, 1234, 2026, 4321)
    for representation in ("formula", "table")
}
actual = [identity(t) for t in all_trials]
assert len(actual) == len(set(actual)) == 20 and set(actual) == expected
assert len({t["initial_parameters_sha256"] for t in all_trials}) == 5
rows = []
for trial in all_trials:
    snap = trial["snapshots"][-1]
    metrics = snap["metrics"]
    failed = [
        k for k, v in reference_manifest["case_limits"][trial["case"]].items() if metrics[k] > v
    ]
    if (
        max(s["metrics"]["initial_rmse"] for s in trial["snapshots"])
        > reference_manifest["physical_limits"]["initial_rmse"]
    ):
        failed.append("initial_rmse")
    if (
        max(s["boundary_max"] for s in trial["snapshots"])
        > reference_manifest["physical_limits"]["boundary_max"]
    ):
        failed.append("boundary_max")
    rows.append(
        dict(
            stage=trial["stage"],
            case=trial["case"],
            seed=trial["seed"],
            coefficient_representation=trial["coefficient_representation"],
            **metrics,
            absolute_passed=not failed,
            failed_metrics=";".join(failed),
            initial_max=max(s["metrics"]["initial_rmse"] for s in trial["snapshots"]),
            boundary_max=max(s["boundary_max"] for s in trial["snapshots"]),
            training_seconds=trial["training_seconds_with_snapshots"],
            process_seconds=trial["process_seconds"],
            closures=snap["closures"],
            adam_residual_points=snap["adam_residual_points"],
            lbfgs_residual_points=snap["lbfgs_residual_points"],
            peak_allocated_bytes=trial["peak_allocated_bytes"],
            peak_reserved_bytes=trial["peak_reserved_bytes"],
            crossing=trial["crossing"]["status"],
        )
    )
with outputs[0].open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
groups = []
for case in ("heat_conductivity_beta0", "heat_conductivity_beta1"):
    for representation in ("formula", "table"):
        subset = [
            r
            for r in rows
            if r["case"] == case and r["coefficient_representation"] == representation
        ]
        names = [
            *reference_manifest["case_limits"][case],
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
                    for name in names
                },
            )
        )
summary = dict(
    source_revision=source,
    stages=provenance,
    complete_unique_trials=20,
    distinct_initial_states=5,
    all_absolute_targets_passed=all(r["absolute_passed"] for r in rows),
    absolute_passed=sum(r["absolute_passed"] for r in rows),
    failed_by_metric={
        name: sum(name in r["failed_metrics"].split(";") for r in rows)
        for name in {
            *next(iter(reference_manifest["case_limits"].values())),
            "initial_rmse",
            "boundary_max",
        }
    },
    stage_comparisons_passed=[c["comparison_passed"] for c in comparisons],
    snapshots=sum(len(t["snapshots"]) for t in all_trials),
    confirmed_crossings=sum(r["crossing"] == "observed" for r in rows),
    initial_max=max(r["initial_max"] for r in rows),
    boundary_max=max(r["boundary_max"] for r in rows),
    training_seconds_total=sum(r["training_seconds"] for r in rows),
    process_seconds_total=sum(r["process_seconds"] for r in rows),
    max_peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in rows),
    max_peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in rows),
    environment=reference_environment,
    groups=groups,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limitations="Sequential pilot/confirmation, fixed finite five-state matrix. Separate original bundles; cancelled kappa021 excluded. No global material/seed guarantee or speedup claim.",
)
outputs[1].write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(
    json.dumps(
        {
            k: summary[k]
            for k in (
                "complete_unique_trials",
                "distinct_initial_states",
                "all_absolute_targets_passed",
                "snapshots",
                "confirmed_crossings",
            )
        }
    )
)
