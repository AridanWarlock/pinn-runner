"""Сверка начальных весов/точек и качества двух отдельных серий, без времени."""

import json
import sys
from pathlib import Path

baseline = json.loads((Path(sys.argv[1]) / "evidence.json").read_text())
candidate = json.loads((Path(sys.argv[2]) / "evidence.json").read_text())
left = {trial["seed"]: trial for trial in baseline["trials"]}
right = {trial["seed"]: trial for trial in candidate["trials"]}
assert set(left) == set(right) == {11, 29, 83}
assert baseline["manifest"]["case_limits"] == candidate["manifest"]["case_limits"]
assert (
    baseline["manifest"]["hypothesis_evaluation"] == candidate["manifest"]["hypothesis_evaluation"]
)
protocol_differences = [
    key for key in baseline["manifest"] if baseline["manifest"][key] != candidate["manifest"][key]
]
assert set(protocol_differences) == {"revision", "problem_sha256"}
rows = []
for seed in sorted(left):
    a, b = left[seed], right[seed]
    old, new = a["snapshots"][-1]["metrics"], b["snapshots"][-1]["metrics"]
    rows.append(
        dict(
            seed=seed,
            equal_controls={
                key: a[key] == b[key]
                for key in (
                    "initial_state_sha256",
                    "initial_parameters_sha256",
                    "fixed_objective_sha256",
                )
            },
            final_metric_ratios={key: new[key] / old[key] if old[key] else None for key in old},
        )
    )
value = dict(
    kind="separate_representation_studies",
    protocol_differences=sorted(protocol_differences),
    physical_initial_fields_equal=False,
    physical_initial_note="Same raw weights; gain changes the physical initial correction",
    timing_comparable=False,
    results=rows,
)
(Path(__file__).parent / "2026-10-21-quadratic-gain8-controls.json").write_text(
    json.dumps(value, indent=2) + "\n"
)
print(json.dumps(value, indent=2))
