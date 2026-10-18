"""Read-only service-time budget for the existing polynomial-2D Nsight trace."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

from .schema import Json, digest
from .trace_analysis import PHASES, analyze, interval_union

MATRIX_NAMES = (
    "ampere_sgemm_",
    "gemmk1_kernel",
    "gemvNSP_kernel",
    "internal::gemvx::kernel",
)


def matrix_kernel(name: str) -> bool:
    """Use full names: this export calls one cuBLAS GEMV simply `kernel`."""
    return any(part in name for part in MATRIX_NAMES)


def partition_field(events: list[Json]) -> list[Json]:
    """Require the observed 4-Linear/3-Tanh signature; never guess a new model."""
    matrices = [i for i, e in enumerate(events) if matrix_kernel(e["full_name"])]
    tanhs = [i for i, e in enumerate(events) if "tanh_kernel_cuda" in e["full_name"]]
    if (
        len(events) != 57
        or matrices != [29, 32, 35, 38]
        or tanhs != [30, 33, 36]
        or "CatArrayBatchedCopy" not in events[22]["full_name"]
        or any("direct_copy_kernel_cuda" not in events[i]["full_name"] for i in (28, 31, 34, 37))
    ):
        raise ValueError("Unrecognized polynomial-2D field signature")
    if len({e["stream_id"] for e in events}) != 1 or any(
        a["end"] > b["start"] for a, b in zip(events, events[1:])
    ):
        raise ValueError("Field kernels overlap or use multiple streams")
    return [
        dict(
            component=name,
            first_kernel=first,
            last_kernel=last - 1,
            kernel_count=last - first,
            service_nanoseconds=sum(e["end"] - e["start"] for e in events[first:last]),
        )
        for name, first, last in (
            ("features_and_normalization", 0, 28),
            ("mlp_including_bias_copies", 28, 39),
            ("initial_background_and_tau_reconstruction", 39, 57),
        )
    ]


def service_budget(events: list[Json]) -> Json:
    intervals = [(e["start"], e["end"]) for e in events]
    service = sum(b - a for a, b in intervals)
    union = interval_union(intervals)
    if union != service:
        raise ValueError("Overlapping kernels invalidate additive service budget")
    matrix = sum(e["end"] - e["start"] for e in events if matrix_kernel(e["full_name"]))
    if matrix <= 0:
        raise ValueError("Matrix service floor is absent")
    return dict(
        kernel_count=len(events),
        matrix_kernel_count=sum(matrix_kernel(e["full_name"]) for e in events),
        service_nanoseconds=service,
        matrix_service_nanoseconds=matrix,
        nonmatrix_service_fraction=1 - matrix / service,
        conditional_service_speedup_ceiling=service / matrix,
    )


def analyze_budget(path: Path, expected_sha256: str) -> Json:
    baseline = analyze(path, expected_sha256)
    grouped: dict[tuple[int, str], list[Json]] = defaultdict(list)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        names = dict(connection.execute("SELECT id,value FROM StringIds"))
        counters: dict[str, int] = defaultdict(int)
        stages = []
        for row in connection.execute(
            "SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start"
        ):
            label = row["text"] or names.get(row["textId"])
            if label in PHASES:
                stages.append(dict(row) | {"stage": label, "call_index": counters[label]})
                counters[label] += 1
        launches: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            launches[row["correlationId"]].append(row)
        for event in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            candidates = launches.get(event["correlationId"], [])
            if len(candidates) != 1:
                continue
            launch = candidates[0]
            if event["globalPid"] != (launch["globalTid"] & ~0xFFFFFF):
                continue
            matches = [
                s
                for s in stages
                if s["start"] <= launch["start"] <= launch["end"] <= s["end"]
                and s["globalTid"] >> 24 == launch["globalTid"] >> 24
            ]
            if len(matches) != 1:
                continue
            stage = matches[0]
            grouped[(stage["call_index"], stage["stage"])].append(
                dict(
                    start=event["start"],
                    end=event["end"],
                    host_start=launch["start"],
                    stream_id=event["streamId"],
                    full_name=names[event["demangledName"]],
                )
            )
    if sum(map(len, grouped.values())) != baseline["attributed_kernel_count"]:
        raise ValueError("Budget attribution differs from validated phase attribution")
    fields: list[Json] = []
    budgets: list[Json] = []
    for (index, stage_name), events in sorted(grouped.items()):
        events.sort(key=lambda event: event["host_start"])
        if index < 4:
            continue
        budgets.append(dict(call_index=index, stage=stage_name, **service_budget(events)))
        if stage_name == "physical_field":
            fields.extend(dict(call_index=index, **part) for part in partition_field(events))
    combined = [
        dict(
            call_index=index,
            **service_budget([event for phase in PHASES for event in grouped[(index, phase)]]),
        )
        for index in range(4, 13)
    ]
    if len(fields) != 27 or len(budgets) != 27:
        raise ValueError("Expected nine measured calls with three phases")
    if digest(path) != expected_sha256:
        raise ValueError("Trace changed during budget analysis")
    return dict(
        trace_sha256=expected_sha256,
        fields=fields,
        budgets=budgets,
        combined=combined,
        timing_is_profiled=True,
        end_to_end_speedup_claim=False,
        scope="Kernel service only; conditional fixed matrix service, no host/gaps/copy time",
    )
