"""Retain and validate every registered constant-k hypothesis in compact evidence."""

from __future__ import annotations

from .schema import Json, number, object_value

FIELDS = {
    "k",
    "relative_l2",
    "final_relative_l2",
    "worst_time_relative_l2",
    "pde_rmse",
    "initial_rmse",
    "boundary_max",
    "ignored_k_relative_l2",
}
AGGREGATES = {
    "relative_l2",
    "final_relative_l2",
    "worst_time_relative_l2",
    "pde_rmse",
    "initial_rmse",
}


def validate_snapshots(raw: Json, compact: list[Json], manifest: Json) -> None:
    identity = manifest["hypothesis_evaluation"]
    for original, snapshot in zip(raw["snapshots"], compact):
        if (
            original.get("hypothesis_evaluation") != identity
            or original.get("evaluation_state_unchanged") is not True
        ):
            raise ValueError("Shared-model evaluation identity or frozen state differs")
        rows = original.get("hypotheses")
        if (
            not isinstance(rows, list)
            or [object_value(r, "hypothesis").get("k") for r in rows] != identity["hypotheses"]
        ):
            raise ValueError("Missing, reordered or duplicate hypotheses")
        for row in rows:
            if set(row) != FIELDS:
                raise ValueError("Incomplete per-hypothesis metrics")
            for key, value in row.items():
                number(value, key)
        for key in AGGREGATES:
            if snapshot["metrics"][key] != max(row[key] for row in rows):
                raise ValueError("Surrogate aggregate must be worst hypothesis")
        if snapshot["boundary_max"] != max(row["boundary_max"] for row in rows):
            raise ValueError("Boundary aggregate must be worst hypothesis")
        snapshot["hypotheses"] = rows
        snapshot["hypothesis_evaluation"] = identity
        snapshot["evaluation_state_unchanged"] = True
