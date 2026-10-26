"""Фиксированный контроль вместимости 32 мод; без оптимизатора и полей-учителей."""

import csv
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.quadratic_modal_case import SOURCE, modal_config
from benchmarks.quadratic_modal_objective import ModalResidualObjective
from pinnstorch.models.net.quadratic_modal import QuadraticModalFCN
from pinnstorch.runner.problem import ProblemAdapter, load_problem
from pinnstorch.runner.settings import TaskSettings

OUT = Path(__file__).parent
FORMS = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0]


class InitialProjectionOracle(QuadraticModalFCN):
    def __init__(self, adapter, targets, ignore_shape=False):
        super().__init__(
            [32, 32, 32],
            ((0, 1), (0.05, 0.15), (1, 2), (0, 1)),
            adapter.initial_tensors,
            hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        )
        self.register_buffer("query_forms", torch.tensor(FORMS, dtype=torch.float64))
        self.register_buffer("initial_projections", targets.clone())
        self.ignore_shape = ignore_shape
        self.requires_grad_(False)

    def coefficients(self, b):
        distance = (b - self.query_forms[None, :]).abs()
        minimum, indices = distance.min(dim=1)
        assert minimum.max() < 1e-6
        return self.initial_projections[indices]

    def forward(self, features, time):
        if self.ignore_shape:
            features = [features[0], features[1], torch.full_like(features[2], 1.5)]
        return super().forward(features, time)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    adapter = ProblemAdapter(
        load_problem(SOURCE), TaskSettings.read(modal_config("cpu", 1, 11)).domain
    )
    prepared = ModalResidualObjective(adapter.initial_tensors, FORMS)
    refined = ModalResidualObjective(adapter.initial_tensors, FORMS, quadrature=512)
    discrepancy = float((prepared.targets - refined.targets).abs().max())
    assert discrepancy < 2e-13
    models = {
        "initial_projection_oracle": InitialProjectionOracle(adapter, prepared.targets).float(),
        "ignored_shape": InitialProjectionOracle(adapter, prepared.targets, True).float(),
    }
    # Таблицы готовы и заморожены; только теперь импортируем независимую оценку решения.
    from benchmarks.quadratic_surrogate_quality import evaluate
    from tests.numerics.test_quadratic_clock_quality import NAME, clock_manifest

    manifest = clock_manifest()
    limits = {**manifest["case_limits"][NAME], **manifest["physical_limits"]}
    result = {}
    for name, model in models.items():
        metrics, details = evaluate(model, adapter, "cpu", NAME)
        metrics["boundary_max"] = max(row["boundary_max"] for row in details["hypotheses"])
        failed = [key for key, limit in limits.items() if metrics[key] > limit]
        result[name] = dict(
            metrics=metrics, failed=failed, state_unchanged=details["evaluation_state_unchanged"]
        )
        for kind in ("hypotheses", "responses"):
            rows = details[kind]
            with (OUT / f"2026-10-26-modal-capacity-{name}-{kind}.csv").open(
                "w", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    assert not result["initial_projection_oracle"]["failed"]
    assert result["ignored_shape"]["failed"]
    summary = dict(
        kind="per_query_initial_projection_capacity_control",
        modes=32,
        quadrature_orders=[256, 512],
        projection_disagreement=discrepancy,
        teacher_at_positive_time=False,
        trained_neural_surrogate=False,
        interval_certificate=False,
        limits=limits,
        results=result,
    )
    (OUT / "2026-10-26-modal-capacity-result.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
