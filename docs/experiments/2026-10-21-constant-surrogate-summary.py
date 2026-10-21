"""Reproduce compact tables from the validated shared-model bundle."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research_tools.comparison import compare  # noqa: E402
from research_tools.evidence import read_bundle  # noqa: E402

PREFIX = ROOT / "docs/experiments/2026-10-21-constant-surrogate"


def write_csv(label, rows):
    with Path(f"{PREFIX}-{label}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    evidence = read_bundle(args.bundle)
    result = compare(evidence)
    rows, trials = [], []
    for trial in evidence["trials"]:
        final = trial["snapshots"][-1]
        for snapshot in trial["snapshots"]:
            for hypothesis in snapshot["hypotheses"]:
                rows.append(dict(seed=trial["seed"], lb_steps=snapshot["lb_steps"], **hypothesis))
        trials.append(
            dict(
                seed=trial["seed"],
                **final["metrics"],
                boundary_max=final["boundary_max"],
                training_seconds=trial["training_seconds_with_snapshots"],
                setup_seconds=trial["setup_seconds"],
                evaluation_seconds=trial["evaluation_seconds"],
                trial_wall_seconds=trial["trial_wall_seconds"],
                process_seconds=trial.get("process_seconds"),
                closures=final["closures"],
                warm_prediction_seconds=trial["warm_prediction_seconds"],
                prediction_points=trial["prediction_points"],
                prediction_batch_size=trial["prediction_batch_size"],
                initial_state_sha256=trial["initial_state_sha256"],
                final_passed=final["passed"],
            )
        )
    write_csv("hypotheses", rows)
    write_csv("trials", trials)
    summary = dict(
        code_revision=evidence["manifest"]["revision"],
        comparison=result,
        trials=trials,
        hypotheses_per_snapshot=9,
        independently_initialized_models=len({t["initial_state_sha256"] for t in trials}),
        evaluation_state_unchanged=all(
            s["evaluation_state_unchanged"] for t in evidence["trials"] for s in t["snapshots"]
        ),
    )
    Path(f"{PREFIX}-result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
