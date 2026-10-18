"""Allocator metadata validation does not simulate numerical acceptance."""

from copy import deepcopy

import pytest

from benchmarks.allocation_profile import reduce_memory_events
from research_tools.allocation_profile import STAGES
from research_tools.comparison import compare
from research_tools.evidence import read_bundle, write_bundle
from research_tools.planning import plan
from research_tools.schema import validate_manifest
from tests.runner.test_frozen_profile import protocol as native_protocol


def protocol():
    return native_protocol() | dict(contract=7, kind="allocation-profile")


def evidence():
    m = protocol()
    stages = [
        dict(
            stage=s,
            before_bytes=10,
            after_bytes=15,
            peak_bytes=20,
            reserved_before_bytes=30,
            reserved_after_bytes=30,
            peak_reserved_bytes=30,
            allocation_events=2,
            free_events=1,
            allocated_bytes=10,
            freed_bytes=5,
        )
        for s in STAGES
    ]
    return dict(
        contract=1,
        kind="allocation-profile-evidence",
        manifest=m,
        environment=m["environment_requirements"],
        artifact_sha256={},
        trials=[
            dict(
                gpu="NVIDIA GeForce RTX 4070 Ti",
                environment=m["environment_requirements"],
                rule=r,
                revision=m["revision"],
                weights_sha256=m["weights_sha256"],
                source_unchanged=True,
                decomposed_gradient_passed=True,
                state_sha256="d" * 64,
                independent_autograd_control=dict(
                    loss_absolute_difference=0,
                    gradient_relative_l2_difference=0,
                    gradient_max_absolute_difference=0,
                ),
                profiling=dict(
                    stages=deepcopy(stages),
                    cuda_memory_events=15,
                    profiled=True,
                    cuda_kernel_timing=False,
                    performance_claim=False,
                ),
            )
            for r in ("autograd", "analytic")
        ],
    )


def test_allocator_contract_and_compact_roundtrip(tmp_path):
    m = protocol()
    assert validate_manifest(m, tmp_path) == m
    assert plan(m)["optimizer_updates"] == 0
    e = evidence()
    assert compare(e)["cuda_allocator_events_captured"] is True
    write_bundle(tmp_path / "bundle", e)
    assert compare(read_bundle(tmp_path / "bundle"))["comparison_passed"] is True


@pytest.mark.parametrize(
    "change", ["environment", "state", "gradient", "peak", "stage", "negative", "kind"]
)
def test_allocator_provenance_and_counter_failures(change):
    e = evidence()
    t = e["trials"][0]
    if change == "environment":
        e["environment"] = {}
    elif change == "state":
        t["state_sha256"] = "e" * 64
    elif change == "gradient":
        t["independent_autograd_control"]["loss_absolute_difference"] = 1
    elif change == "peak":
        t["profiling"]["stages"][0]["peak_bytes"] = 0
    elif change == "stage":
        t["profiling"]["stages"].pop()
    elif change == "negative":
        t["profiling"]["stages"][0]["allocated_bytes"] = -1
    else:
        e["kind"] = "other"
    with pytest.raises(ValueError):
        compare(e)


def test_unsupported_cuda_memory_capture_is_a_preserved_negative_result():
    e = evidence()
    e["trials"][0]["profiling"]["cuda_memory_events"] = 0
    for row in e["trials"][0]["profiling"]["stages"]:
        row.update(allocation_events=0, free_events=0, allocated_bytes=0, freed_bytes=0)
    assert compare(e)["comparison_passed"] is False


def test_memory_events_include_worker_thread_and_exclude_host_and_foreign_process():
    events = [dict(name=s, ph="X", ts=i * 10, dur=8, pid=1, tid=2) for i, s in enumerate(STAGES)]
    for pid, device, ts, size in [
        (1, 1, 1, 100),
        (1, 1, 2, -30),
        (1, 0, 3, 500),
        (2, 1, 4, 700),
        (1, 1, 8, 900),
    ]:
        events.append(
            dict(
                name="[memory]",
                ph="i",
                ts=ts,
                pid=pid,
                tid=3,
                args={"Device Type": device, "Bytes": size},
            )
        )
    rows = reduce_memory_events({"traceEvents": events})
    assert rows[0] == dict(
        stage=STAGES[0], allocation_events=1, free_events=1, allocated_bytes=100, freed_bytes=30
    )
    assert sum(r["allocation_events"] for r in rows) == 1
    events[1]["ts"] = 5
    with pytest.raises(ValueError, match="nonoverlapping"):
        reduce_memory_events({"traceEvents": events})
