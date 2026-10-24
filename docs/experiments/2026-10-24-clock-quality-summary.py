"""Компактные таблицы качества; время при общей нагрузке не сравнивается."""

import csv
import hashlib
import json
import sys
from pathlib import Path

bundle = Path(sys.argv[1])
tag = sys.argv[2] if len(sys.argv) > 2 else ""
assert all(c.isalnum() or c == "-" for c in tag)
prefix = "2026-10-24-clock-quality" + ("-" + tag if tag else "")
marker = json.loads((bundle / "bundle.json").read_text())
for name, digest in marker["sha256"].items():
    assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest, name
value = json.loads((bundle / "evidence.json").read_text())
manifest = value["manifest"]
name = "heat_quadratic_clock"
limits = {**manifest["case_limits"][name], **manifest["physical_limits"]}
rows, pairs, trials = [], [], []
for trial in value["trials"]:
    seed = trial["seed"]
    for snapshot in trial["snapshots"]:
        for row in snapshot["hypotheses"]:
            rows.append(dict(seed=seed, lb_steps=snapshot["lb_steps"], **row))
        for pair in snapshot["responses"]:
            left = snapshot["hypotheses"][pair["left"]]
            right = snapshot["hypotheses"][pair["right"]]
            pairs.append(
                dict(
                    seed=seed,
                    lb_steps=snapshot["lb_steps"],
                    kind="shape" if left["a"] == right["a"] else "scale",
                    left_a=left["a"],
                    left_b=left["b"],
                    right_a=right["a"],
                    right_b=right["b"],
                    relative_l2=pair["relative_l2"],
                )
            )
    final = trial["snapshots"][-1]
    metrics = {**final["metrics"], "boundary_max": final["boundary_max"]}
    failed = [key for key, threshold in limits.items() if metrics[key] > threshold]
    worst = {}
    for key in (
        "relative_l2",
        "worst_time_relative_l2",
        "dx_relative_l2",
        "flux_relative_l2",
        "pde_rmse",
    ):
        row = max(final["hypotheses"], key=lambda r: r[key])
        worst[key] = dict(
            a=row["a"],
            b=row["b"],
            split=row["split"],
            value=row[key],
            worst_time=row["worst_time"],
        )
    trials.append(
        dict(
            seed=seed,
            passed=not failed,
            failed_metrics=failed,
            final_metrics=metrics,
            worst_hypotheses=worst,
        )
    )
output = Path(__file__).resolve().parent
for suffix, data in (("hypotheses", rows), ("responses", pairs)):
    with (output / f"{prefix}-{suffix}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
summary = dict(
    kind="fixed_step_quality_shared_pc",
    revision=manifest["revision"],
    trials=len(trials),
    passed=sum(t["passed"] for t in trials),
    timing_comparable=False,
    timing_reason="Пользователь параллельно использует Windows ПК; время только диагностическое",
    limits=limits,
    results=trials,
)
(output / f"{prefix}-result.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
)
print(
    json.dumps(
        dict(
            trials=len(trials),
            passed=summary["passed"],
            results=[
                dict(
                    seed=t["seed"],
                    failed_metrics=t["failed_metrics"],
                    final_metrics=t["final_metrics"],
                )
                for t in trials
            ],
        ),
        indent=2,
    )
)
