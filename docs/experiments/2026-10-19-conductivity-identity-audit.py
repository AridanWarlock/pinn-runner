"""CPU counterexample: batch/weight fingerprints do not identify the PDE."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.fingerprints import objective_fingerprint, tensor_fingerprint  # noqa: E402
from pinnstorch.training.refinement import FixedObjective  # noqa: E402
from scripts.audit_variable_conductivity import make_factory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows, residuals = [], {}
    for beta in (0, 1):
        for kind in ("formula", "table"):
            directory = args.output / f"beta{beta}-{kind}"
            factory = make_factory(directory, 2, beta, kind, 19)
            objects = factory.build(predicting=False)
            model = objects.model.eval()
            state = tensor_fingerprint(model.state_dict())
            objective = FixedObjective.capture(objects.data, model.device)
            assert tuple(objective.batches) == ("equations",)
            spatial, time, _ = objective.fresh()["equations"]
            columns = [x.requires_grad_() for x in (*spatial, time)]
            field = model(columns[:-1], columns[-1])["u"]
            residual = factory.adapter.residuals({"u": field}, *columns)["heat"].detach()
            residuals[beta, kind] = residual
            assert tensor_fingerprint(model.state_dict()) == state
            rows.append(
                dict(
                    beta=beta,
                    representation=kind,
                    initial_parameters_sha256=tensor_fingerprint(dict(model.named_parameters())),
                    initial_state_sha256=state,
                    fixed_objective_sha256=objective_fingerprint(objective.batches),
                    field_sha256=tensor_fingerprint({"u": field}),
                    source_sha256=hashlib.sha256(
                        (directory / "functions.py").read_bytes()
                    ).hexdigest(),
                    coefficient_signature=json.dumps(
                        factory.coefficient_signature, sort_keys=True
                    ),
                    residual_rms=float(residual.square().mean().sqrt()),
                    state_unchanged=True,
                )
            )
    assert len({r["initial_parameters_sha256"] for r in rows}) == 1
    assert len({r["fixed_objective_sha256"] for r in rows}) == 1
    assert len({r["field_sha256"] for r in rows}) == 1
    formulas = [r for r in rows if r["representation"] == "formula"]
    assert formulas[0]["initial_state_sha256"] == formulas[1]["initial_state_sha256"]
    assert len({r["source_sha256"] for r in rows}) == 4
    differences = {}
    for kind in ("formula", "table"):
        delta = residuals[1, kind] - residuals[0, kind]
        assert not torch.allclose(residuals[1, kind], residuals[0, kind], rtol=2e-4, atol=5e-6)
        differences[kind] = float(delta.square().mean().sqrt())
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        device="cpu",
        beta_residual_difference_rms=differences,
        batch_hashes_equal=True,
        parameter_hashes_equal=True,
        formula_full_state_hashes_equal=True,
        training_runs=0,
        interpretation="Batch fingerprints cover stored tensors, not callbacks or PDE identity",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
