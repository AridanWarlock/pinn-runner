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
    if identity.get("parameters") == ["a", "b"]:
        validate_whole_codes(raw, compact, manifest)
        return
    parameter = identity.get("parameter", "k")
    fields = (FIELDS - {"k", "ignored_k_relative_l2"}) | {
        parameter,
        f"ignored_{parameter}_relative_l2",
    }
    for original, snapshot in zip(raw["snapshots"], compact):
        if (
            original.get("hypothesis_evaluation") != identity
            or original.get("evaluation_state_unchanged") is not True
        ):
            raise ValueError("Shared-model evaluation identity or frozen state differs")
        rows = original.get("hypotheses")
        if (
            not isinstance(rows, list)
            or [object_value(r, "hypothesis").get(parameter) for r in rows]
            != identity["hypotheses"]
        ):
            raise ValueError("Missing, reordered or duplicate hypotheses")
        for row in rows:
            if set(row) != fields:
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


def validate_whole_codes(raw: Json, compact: list[Json], manifest: Json) -> None:
    identity = manifest["hypothesis_evaluation"]
    aggregates = AGGREGATES | {"dx_relative_l2", "flux_relative_l2"}
    fields = (FIELDS - {"k", "ignored_k_relative_l2"}) | {
        "a",
        "b",
        "split",
        "worst_time",
        "dx_relative_l2",
        "flux_relative_l2",
        "ignored_a_relative_l2",
        "ignored_b_relative_l2",
    }
    for original, snapshot in zip(raw["snapshots"], compact):
        if (
            original.get("hypothesis_evaluation") != identity
            or original.get("evaluation_state_unchanged") is not True
        ):
            raise ValueError("Whole-code evaluation identity or frozen state differs")
        rows = original.get("hypotheses")
        if not isinstance(rows, list) or len(rows) != len(identity["hypotheses"]):
            raise ValueError("Incomplete whole-code hypotheses")
        for row, code, split in zip(rows, identity["hypotheses"], identity["splits"]):
            row = object_value(row, "whole-code row")
            if set(row) != fields or [row["a"], row["b"]] != code or row["split"] != split:
                raise ValueError("Missing, reordered or changed whole codes or splits")
            for key, value in row.items():
                if key != "split":
                    number(value, key)
            if row["worst_time"] not in [i / 100 for i in range(101)]:
                # np.linspace can differ by one ulp: validate grid membership with rounding.
                if (
                    abs(row["worst_time"] * 100 - round(row["worst_time"] * 100)) > 1e-10
                    or not 0 <= row["worst_time"] <= 1
                ):
                    raise ValueError("Worst-time location outside registered grid")
        for key in aggregates:
            if snapshot["metrics"][key] != max(row[key] for row in rows):
                raise ValueError("Whole-code aggregate must be worst hypothesis")
        if snapshot["boundary_max"] != max(row["boundary_max"] for row in rows):
            raise ValueError("Whole-code boundary aggregate differs")
        responses = original.get("responses")
        if not isinstance(responses, list) or len(responses) != len(identity["response_pairs"]):
            raise ValueError("Incomplete response inventory")
        for response, pair in zip(responses, identity["response_pairs"]):
            response = object_value(response, "response")
            if (
                set(response) != {"left", "right", "relative_l2"}
                or [response["left"], response["right"]] != pair
            ):
                raise ValueError("Missing, reordered or changed response pair")
            for key, value in response.items():
                number(value, key)
        if snapshot["metrics"]["coefficient_response_relative_l2"] != max(
            row["relative_l2"] for row in responses
        ):
            raise ValueError("Response aggregate differs")
        snapshot.update(
            hypotheses=rows,
            responses=responses,
            hypothesis_evaluation=identity,
            evaluation_state_unchanged=True,
        )
