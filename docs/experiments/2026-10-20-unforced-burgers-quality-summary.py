"""Audit all nine goals of the preregistered unforced Burgers series."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research_tools.comparison import compare  # noqa: E402
from research_tools.evidence import read_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--job-id", choices=("nonheat032", "nonheat033"), default="nonheat032")
    args = parser.parse_args()
    cases = (
        ("burgers_unforced_nu010", "burgers_unforced_nu001")
        if args.job_id == "nonheat032"
        else ("burgers_unforced_nu001",)
    )
    count = 5 * len(cases)
    e = read_bundle(args.bundle)
    m = e["manifest"]
    result = compare(e)
    assert result["kind"] == "absolute-quality-comparison"
    expected = {(case, seed) for case in cases for seed in (11, 29, 83, 271, 997)}
    assert {(t["case"], t["seed"]) for t in e["trials"]} == expected
    assert len(e["trials"]) == count
    assert m["report_parameters"]["points_override"] == (
        1024 if args.job_id == "nonheat032" else 4096
    )
    protocol_commit = e["provenance"]["protocol_commit"]
    protocol_path = f"docs/experiments/2026-10-20-{args.job_id}-unforced-quality-protocol.json"
    blob = subprocess.check_output(["git", "show", f"{protocol_commit}:{protocol_path}"], cwd=ROOT)
    assert json.loads(blob) == m
    assert hashlib.sha256(blob).hexdigest() == e["provenance"]["protocol_sha256"]
    for case, checksum in m["problem_sha256"].items():
        source = subprocess.check_output(
            ["git", "show", f"{m['revision']}:benchmarks/problems/{case}/functions.py"], cwd=ROOT
        )
        assert hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() == checksum
    reference = subprocess.check_output(
        ["git", "show", f"{m['revision']}:benchmarks/burgers.py"], cwd=ROOT
    )
    assert (
        hashlib.sha256(reference.replace(b"\r\n", b"\n")).hexdigest()
        == m["layer_evaluation"]["reference_sha256"]
    )
    assert m["layer_evaluation"]["coarse_grid"] == [801, 201]
    assert m["layer_evaluation"]["fine_grid"] == [1601, 401]
    final_rows = []
    snapshot_rows = []
    summaries = {}
    for case in cases:
        trials = [t for t in e["trials"] if t["case"] == case]
        assert len({t["initial_state_sha256"] for t in trials}) == 5
        goals = m["case_limits"][case] | m["physical_limits"]
        failures = Counter()
        for trial in trials:
            assert trial["adam_steps"] == 500
            assert trial["constraints"] == "hard_initial_dirichlet"
            assert trial["layer_evaluation"] == m["layer_evaluation"]
            assert trial["network"] == m["network"]
            assert [s["lb_steps"] for s in trial["snapshots"]] == [0, 100, 250, 500]
            assert trial["problem_source_sha256"] == m["problem_sha256"][case]
            for snap in trial["snapshots"]:
                metrics = snap["metrics"] | dict(boundary_max=snap["boundary_max"])
                failed = [k for k, v in goals.items() if metrics[k] > v]
                assert bool(snap["passed"]) == (not failed)
                row = dict(
                    case=case,
                    seed=trial["seed"],
                    lb_steps=snap["lb_steps"],
                    passed=not failed,
                    failed_metrics=";".join(failed),
                    **metrics,
                    closures=snap["closures"],
                    seconds=snap["seconds"],
                    adam_residual_points=snap["adam_residual_points"],
                    lbfgs_residual_points=snap["lbfgs_residual_points"],
                )
                snapshot_rows.append(row)
            last = snapshot_rows[-1]
            failures.update(last["failed_metrics"].split(";") if last["failed_metrics"] else [])
            final_rows.append(
                last
                | dict(
                    warm_prediction_seconds=trial["warm_prediction_seconds"],
                    prediction_points=trial["prediction_points"],
                    prediction_batch_size=trial["prediction_batch_size"],
                    prediction_warmup_passes=trial["prediction_warmup_passes"],
                    prediction_measurement_passes=trial["prediction_measurement_passes"],
                    training_seconds=trial["training_seconds_with_snapshots"],
                    process_seconds=trial["process_seconds"],
                    peak_allocated_bytes=trial["peak_allocated_bytes"],
                    peak_reserved_bytes=trial["peak_reserved_bytes"],
                    crossing_status=trial["crossing"]["status"],
                    initial_state_sha256=trial["initial_state_sha256"],
                    initial_parameters_sha256=trial["initial_parameters_sha256"],
                    fixed_objective_sha256=trial["fixed_objective_sha256"],
                )
            )
        selected = [r for r in final_rows if r["case"] == case]
        summaries[case] = dict(
            trials=5,
            passed=sum(r["passed"] for r in selected),
            failed_metrics=dict(failures),
            maxima={k: max(r[k] for r in selected) for k in goals},
            minima={k: min(r[k] for r in selected) for k in goals},
            snapshots={
                str(step): dict(
                    passed=sum(
                        r["passed"]
                        for r in snapshot_rows
                        if r["case"] == case and r["lb_steps"] == step
                    ),
                    trials=5,
                )
                for step in (0, 100, 250, 500)
            },
            confirmed_crossings=sum(t["crossing"]["status"] == "observed" for t in trials),
        )
    for suffix, rows in (("-final.csv", final_rows), ("-snapshots.csv", snapshot_rows)):
        with Path(str(args.output_prefix) + suffix).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(dict.fromkeys(k for row in rows for k in row)),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
    initial_pairs = []
    for seed in (11, 29, 83, 271, 997) if len(cases) == 2 else ():
        pair = [t for t in e["trials"] if t["seed"] == seed]
        initial_pairs.append(
            dict(
                seed=seed,
                same_initial_network_state=pair[0]["initial_state_sha256"]
                == pair[1]["initial_state_sha256"],
                same_initial_parameters=pair[0]["initial_parameters_sha256"]
                == pair[1]["initial_parameters_sha256"],
                same_objective_sample_hash=pair[0]["fixed_objective_sha256"]
                == pair[1]["fixed_objective_sha256"],
            )
        )
    summary = dict(
        initial_pairs=initial_pairs,
        layer_evaluation=m["layer_evaluation"],
        job_id=args.job_id,
        revision=m["revision"],
        provenance=e["provenance"],
        evidence_sha256=hashlib.sha256((args.bundle / "evidence.json").read_bytes()).hexdigest(),
        environment=e["environment"],
        trials=count,
        passed=result["passed"],
        comparison_passed=result["comparison_passed"],
        cases=summaries,
        total_snapshots=len(snapshot_rows),
        failed_snapshots=sum(not r["passed"] for r in snapshot_rows),
        training_seconds=sum(r["training_seconds"] for r in final_rows),
        process_seconds=sum(r["process_seconds"] for r in final_rows),
        lbfgs_closures=sum(r["closures"] for r in final_rows),
        adam_residual_points=sum(r["adam_residual_points"] for r in final_rows),
        lbfgs_residual_points=sum(r["lbfgs_residual_points"] for r in final_rows),
        scope="Fixed matrix only; failed quality retained; no speedup or continuous certificate",
    )
    Path(str(args.output_prefix) + ".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
