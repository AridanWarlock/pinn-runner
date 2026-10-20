"""Keep the five exact-constraint trajectories separate from the earlier soft hypothesis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research_tools.comparison import compare  # noqa: E402
from research_tools.evidence import read_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--soft-bundle", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    e = read_bundle(args.bundle)
    soft = read_bundle(args.soft_bundle)
    m = e["manifest"]
    sm = soft["manifest"]
    result = compare(e)
    assert len(e["trials"]) == 5
    assert {(t["case"], t["seed"]) for t in e["trials"]} == {
        ("burgers", s) for s in (19, 73, 1234, 2026, 4321)
    }
    assert m["report_parameters"] == sm["report_parameters"] | dict(
        constraints="hard_initial_dirichlet"
    )
    for key in ("network", "physical_limits", "absolute_quality"):
        assert m[key] == sm[key]
    for key in ("case_limits", "case_faces", "problem_sha256"):
        assert m[key] == {"burgers": sm[key]["burgers"]}
    assert {k: v for k, v in e["environment"].items() if k != "revision"} == {
        k: v for k, v in soft["environment"].items() if k != "revision"
    }
    path = "docs/experiments/2026-10-20-nonheat028-hard-quality-protocol.json"
    commit = e["provenance"]["protocol_commit"]
    blob = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    assert json.loads(blob) == m
    assert hashlib.sha256(blob).hexdigest() == e["provenance"]["protocol_sha256"]
    source = subprocess.check_output(
        ["git", "show", f"{m['revision']}:benchmarks/problems/burgers/functions.py"], cwd=ROOT
    )
    assert (
        hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest()
        == m["problem_sha256"]["burgers"]
    )
    goals = m["case_limits"]["burgers"] | m["physical_limits"]
    snapshots = []
    final = []
    soft_by_seed = {t["seed"]: t for t in soft["trials"] if t["case"] == "burgers"}
    for t in e["trials"]:
        assert t["constraints"] == "hard_initial_dirichlet"
        for s in t["snapshots"]:
            metrics = s["metrics"] | dict(boundary_max=s["boundary_max"])
            failed = [k for k, v in goals.items() if metrics[k] > v]
            assert bool(s["passed"]) == (not failed)
            snapshots.append(
                dict(
                    seed=t["seed"],
                    lb_steps=s["lb_steps"],
                    passed=not failed,
                    failed_metrics=";".join(failed),
                    **metrics,
                    seconds=s["seconds"],
                    closures=s["closures"],
                    adam_residual_points=s["adam_residual_points"],
                    lbfgs_residual_points=s["lbfgs_residual_points"],
                )
            )
        old = soft_by_seed[t["seed"]]
        final.append(
            snapshots[-1]
            | dict(
                initial_parameters_sha256=t["initial_parameters_sha256"],
                initial_state_sha256=t["initial_state_sha256"],
                initial_network_state_matches_soft=old["initial_state_sha256"]
                == t["initial_state_sha256"],
                training_seconds=t["training_seconds_with_snapshots"],
                process_seconds=t["process_seconds"],
                peak_allocated_bytes=t["peak_allocated_bytes"],
                peak_reserved_bytes=t["peak_reserved_bytes"],
                crossing_status=t["crossing"]["status"],
                soft_boundary_max=old["snapshots"][-1]["boundary_max"],
            )
        )
    assert len({r["initial_parameters_sha256"] for r in final}) == 5
    for suffix, rows in (("-final.csv", final), ("-snapshots.csv", snapshots)):
        with Path(str(args.output_prefix) + suffix).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(
        job_id="nonheat028",
        revision=m["revision"],
        provenance=e["provenance"],
        evidence_sha256=hashlib.sha256((args.bundle / "evidence.json").read_bytes()).hexdigest(),
        environment=e["environment"],
        trials=5,
        passed=result["passed"],
        comparison_passed=result["comparison_passed"],
        maxima={k: max(r[k] for r in final) for k in goals},
        initial_network_state_matches_soft=sum(
            r["initial_network_state_matches_soft"] for r in final
        ),
        snapshots={
            str(step): dict(
                passed=sum(r["passed"] for r in snapshots if r["lb_steps"] == step), trials=5
            )
            for step in (0, 100, 250, 500)
        },
        confirmed_crossings=sum(r["crossing_status"] == "observed" for r in final),
        training_seconds=sum(r["training_seconds"] for r in final),
        process_seconds=sum(r["process_seconds"] for r in final),
        lbfgs_closures=sum(r["closures"] for r in final),
        adam_residual_points=sum(r["adam_residual_points"] for r in final),
        lbfgs_residual_points=sum(r["lbfgs_residual_points"] for r in final),
        scope="Prospective follow-up after soft failure; changed function class and objective; not pooled with soft and no speedup claim",
    )
    Path(str(args.output_prefix) + ".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
