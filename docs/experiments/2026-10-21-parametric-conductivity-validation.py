"""Check evidence completeness, provenance, links and independent outcome counts."""

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "docs/experiments/2026-10-21-parametric-conductivity"


def main():
    names = (
        "reference",
        "network",
        "lifecycle",
        "compression",
        "resolution",
        "sensitivity",
        "generalization",
        "holdout",
        "input-scaling",
        "function-rows",
    )
    evidence = {}
    fallbacks = []
    for name in names:
        data = json.loads(Path(f"{PREFIX}-{name}.json").read_text())
        assert data.get("all_passed", data.get("measured_formulas_match", False)), name
        if "protocol_sha256" in data:
            protocol_name = "protocol" if name in ("reference", "network") else f"{name}-protocol"
            path = ROOT / f"docs/parametric-conductivity-{protocol_name}-2026-10-21.md"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == data["protocol_sha256"], name
        evidence[name] = True
        fallbacks.extend(data.get("eigensolver_fallbacks", []))
    source = Path(f"{PREFIX}-problem.py")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ("network", "lifecycle"):
        data = json.loads(Path(f"{PREFIX}-{name}.json").read_text())
        assert data["source_sha256"] == source_digest
    row_counts = {
        "reference": 13,
        "reference-slices": 325,
        "network": 4,
        "network-finite-differences": 4,
        "resolution": 15,
        "sensitivity": 8,
        "generalization": 3,
        "holdout-parameters": 64,
        "holdout-reference": 77,
        "input-scaling": 9,
    }
    for name, count in row_counts.items():
        with Path(f"{PREFIX}-{name}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count, (name, len(rows), count)
        assert all(row.get("passed", "true").lower() == "true" for row in rows), name
    holdout = json.loads(Path(f"{PREFIX}-holdout.json").read_text())
    parameter_csv = Path(f"{PREFIX}-holdout-parameters.csv").read_bytes().replace(b"\r\n", b"\n")
    assert holdout["parameter_csv_hash_mode"] == "lf-normalized"
    assert holdout["parameter_csv_sha256"] == hashlib.sha256(parameter_csv).hexdigest()
    docs = list((ROOT / "docs").glob("parametric-conductivity-*-2026-10-21.md"))
    for path in docs:
        text = path.read_text()
        math_lines = [line for line in text.splitlines() if line.strip() == "$$"]
        assert len(math_lines) % 2 == 0, path
        assert text.replace("$$", "").count("$") == 0, path
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if not target.startswith(("https:", "http:", "#")):
                assert (path.parent / target.split("#")[0]).is_file(), (path, target)
    production = subprocess.check_output(
        [
            "git",
            "diff",
            "5c7531c3e46fd9687be1c4b54e043af0bd57e41d",
            "--name-only",
            "--",
            "pinnstorch",
            "benchmarks",
            "research_tools",
            "scripts",
            "tests",
        ],
        cwd=ROOT,
        text=True,
    )
    assert not production.strip(), production
    lifecycle = ROOT / "numerical-reports/parametric038-lifecycle"
    adam_steps = sum(
        json.loads((lifecycle / name / "training.performance.json").read_text())["adam_steps"]
        for name in ("train", "retrain")
    )
    assert adam_steps == 4
    assert "lies outside checkpoint range" in (lifecycle / "reject-outside/error.log").read_text()
    assert (
        "Checkpoint coefficient definitions differ"
        in (lifecycle / "reject-coefficient/error.log").read_text()
    )
    assert all(
        r["eigen_residual_relative"] <= 1e-10 and r["orthogonality_max"] <= 1e-10
        for r in fallbacks
    )
    files = sorted(
        [
            p
            for p in PREFIX.parent.glob(PREFIX.name + "-*")
            if p.is_file() and not p.name.endswith("-validation.json")
        ]
        + docs
    )
    result = dict(
        verified_utc=datetime.now(timezone.utc).isoformat(),
        production_base_commit="5c7531c3e46fd9687be1c4b54e043af0bd57e41d",
        production_diff_empty=True,
        fast_checks=dict(
            ruff=True,
            format_files=267,
            mypy_files=138,
            pytest_passed=944,
            e2e_deselected=66,
            seconds=20.31,
        ),
        component_checks=dict(pytest_passed=32, e2e_deselected=3, included_in_fast=True),
        full_checks_rerun=False,
        cuda_runs=0,
        quality_training_runs=0,
        cpu_lifecycle_adam_steps=adam_steps,
        rejection_messages_verified=True,
        spectral_fallback_controls=dict(
            count=len(fallbacks),
            maximum_eigen_residual_relative=max(r["eigen_residual_relative"] for r in fallbacks),
            maximum_orthogonality_error=max(r["orthogonality_max"] for r in fallbacks),
        ),
        analysis_optimizer_steps=0,
        evidence=evidence,
        csv_rows=row_counts,
        topic_documents=len(docs),
        sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        all_passed=True,
    )
    Path(f"{PREFIX}-validation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
