"""Attribute existing Nsight events by host launch, without running a GPU workload."""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .schema import Json, digest

PHASES = ("physical_field", "physical_residual", "parameter_gradient")


def interval_union(intervals: list[tuple[int, int]]) -> int:
    """Length of a union: overlapping events must not be double counted."""
    total = 0
    right = -1
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Negative event duration")
        total += max(0, end - max(start, right))
        right = max(right, end)
    return total


def analyze(path: Path, expected_sha256: str) -> Json:
    if digest(path) != expected_sha256:
        raise ValueError("Trace checksum differs before analysis")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        names = dict(connection.execute("SELECT id,value FROM StringIds"))
        ranges = [
            dict(row)
            for row in connection.execute(
                "SELECT start,end,text,globalTid,textId FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start"
            )
        ]
        stages = []
        counters: Counter[str] = Counter()
        for row in ranges:
            label = row["text"] or names.get(row["textId"])
            if label in PHASES:
                index = counters[label]
                counters[label] += 1
                stages.append(
                    dict(
                        call_index=index,
                        stage=label,
                        start=row["start"],
                        end=row["end"],
                        globalTid=row["globalTid"],
                        kernel_intervals=[],
                        copy_intervals=[],
                        copy_bytes=0,
                        kernels=Counter(),
                    )
                )
        if set(counters) != set(PHASES) or len(set(counters.values())) != 1:
            raise ValueError("Incomplete phase sequence")
        for i in range(counters[PHASES[0]]):
            ordered = [s for s in stages if s["call_index"] == i]
            if [s["stage"] for s in ordered] != list(PHASES) or any(
                a["end"] > b["start"] for a, b in zip(ordered, ordered[1:])
            ):
                raise ValueError("Overlapping or reordered host phases")
        launches: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for row in connection.execute(
            "SELECT start,end,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        ):
            launches[row["correlationId"]].append(row)
        unattributed: Counter[str] = Counter()
        kernel_families: dict[tuple[int, str, str], list[int]] = defaultdict(list)
        all_kernels = 0
        for kind, table in (
            ("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
            ("copy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
        ):
            for event in connection.execute('SELECT * FROM "' + table + '"'):
                if kind == "kernel":
                    all_kernels += 1
                candidates = launches.get(event["correlationId"], [])
                if len(candidates) != 1:
                    unattributed[kind + "_missing_or_ambiguous_runtime"] += 1
                    continue
                launch = candidates[0]
                if event["globalPid"] != (launch["globalTid"] & ~0xFFFFFF):
                    unattributed[kind + "_process_mismatch"] += 1
                    continue
                matches = [
                    s
                    for s in stages
                    if (s["globalTid"] >> 24) == (launch["globalTid"] >> 24)
                    and s["start"] <= launch["start"]
                    and launch["end"] <= s["end"]
                ]
                if len(matches) != 1:
                    unattributed[kind + "_outside_phase"] += 1
                    continue
                stage = matches[0]
                start, end = event["start"], event["end"]
                if end < start:
                    raise ValueError("Negative GPU event duration")
                if kind == "kernel":
                    stage["kernel_intervals"].append((start, end))
                    name = names.get(event["shortName"], str(event["shortName"]))
                    stage["kernels"][name] += 1
                    kernel_families[(stage["call_index"], stage["stage"], name)].append(
                        end - start
                    )
                else:
                    stage["copy_intervals"].append((start, end))
                    stage["copy_bytes"] += event["bytes"]
        summary = []
        for stage in stages:
            intervals = stage["kernel_intervals"]
            summary.append(
                dict(
                    call_index=stage["call_index"],
                    stage=stage["stage"],
                    host_nanoseconds=stage["end"] - stage["start"],
                    kernel_count=len(intervals),
                    kernel_summed_nanoseconds=sum(b - a for a, b in intervals),
                    kernel_union_nanoseconds=interval_union(intervals),
                    kernel_span_nanoseconds=max((b for _, b in intervals), default=0)
                    - min((a for a, _ in intervals), default=0),
                    copy_count=len(stage["copy_intervals"]),
                    copy_bytes=stage["copy_bytes"],
                )
            )
        families = [
            dict(
                call_index=i,
                stage=stage,
                kernel=name,
                count=len(durations),
                summed_nanoseconds=sum(durations),
            )
            for (i, stage, name), durations in sorted(kernel_families.items())
        ]
    if digest(path) != expected_sha256:
        raise ValueError("Trace changed during analysis")
    return dict(
        trace_sha256=expected_sha256,
        phase_calls=dict(counters),
        all_kernel_count=all_kernels,
        attributed_kernel_count=sum(s["kernel_count"] for s in summary),
        unattributed=dict(unattributed),
        stages=summary,
        kernel_families=families,
        attribution="unique runtime correlation + same encoded process + contained host launch; includes autograd worker threads; GPU event may finish outside host range",
        timing_is_profiled=True,
        performance_claim=False,
    )
