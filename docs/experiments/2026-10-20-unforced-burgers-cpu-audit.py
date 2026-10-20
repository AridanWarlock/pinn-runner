"""Publish compact every-weight controls for the two new unforced cases."""

import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.audit_nonheat import audit  # noqa: E402


def main():
    torch.set_num_threads(1)
    output = ROOT / "numerical-reports/unforced036-cpu"
    output.mkdir(parents=True, exist_ok=False)
    rows = [
        audit(
            output / f"{case}-{dtype}-{seed}", case, dtype, seed, "cpu", "hard_initial_dirichlet"
        )
        for case, dtype, seed in itertools.product(
            ("burgers_unforced_nu010", "burgers_unforced_nu001"),
            (torch.float32, torch.float64),
            (11, 29),
        )
    ]
    base = Path(__file__).with_name("2026-10-20-unforced-burgers-cpu")
    with base.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        models=len(rows),
        optimizer_steps=0,
        parameters_per_model=2241,
        all_states_unchanged=all(r["state_unchanged"] for r in rows),
        finite_difference_checks=sum(r["finite_difference_error"] is not None for r in rows),
        finite_difference_error_max=max(r["finite_difference_error"] or 0 for r in rows),
        lifting_control_max=max(r["lifting_control_max"] for r in rows),
        gradient_relative_l2_max=max(r["gradient_relative_l2"] for r in rows),
        helper_sha256=hashlib.sha256(
            (ROOT / "scripts/audit_nonheat.py").read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        reference_sha256=hashlib.sha256(
            (ROOT / "benchmarks/burgers.py").read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
    )
    base.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
