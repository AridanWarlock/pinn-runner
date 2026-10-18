"""Bounds apply to serial GPU kernel service, never to host or file latency."""

from copy import deepcopy

import pytest

from research_tools.trace_budget import matrix_kernel, partition_field, service_budget


def field_events():
    events = [
        dict(start=2 * i, end=2 * i + 1, host_start=i, stream_id=7, full_name="pointwise")
        for i in range(57)
    ]
    events[22]["full_name"] = "CatArrayBatchedCopy_aligned16_contig"
    for i in (28, 31, 34, 37):
        events[i]["full_name"] = "direct_copy_kernel_cuda"
    for i in (29, 32, 35):
        events[i]["full_name"] = "ampere_sgemm_128x32_tn"
    events[38]["full_name"] = "std::enable_if internal::gemvx::kernel<int,float>"
    for i in (30, 33, 36):
        events[i]["full_name"] = "tanh_kernel_cuda"
    return events


def test_full_name_recovers_generic_gemv_but_not_arbitrary_kernel():
    assert matrix_kernel("internal::gemvx::kernel<int,float>")
    assert not matrix_kernel("kernel")
    assert not matrix_kernel("vectorized_elementwise_kernel")


def test_field_partition_includes_linear_bias_copies_and_closes_budget():
    events = field_events()
    parts = partition_field(events)
    assert [(p["first_kernel"], p["last_kernel"], p["kernel_count"]) for p in parts] == [
        (0, 27, 28),
        (28, 38, 11),
        (39, 56, 18),
    ]
    assert sum(p["service_nanoseconds"] for p in parts) == 57
    budget = service_budget(events)
    assert budget["matrix_kernel_count"] == 4
    assert budget["conditional_service_speedup_ceiling"] == 57 / 4
    assert budget["nonmatrix_service_fraction"] == 1 - 4 / 57


@pytest.mark.parametrize("change", ["matrix", "tanh", "copy", "cat", "extra", "stream", "overlap"])
def test_unknown_architecture_or_nonadditive_service_is_rejected(change):
    events = deepcopy(field_events())
    if change in ("matrix", "tanh", "copy", "cat"):
        index = {"matrix": 38, "tanh": 33, "copy": 28, "cat": 22}[change]
        events[index]["full_name"] = "unknown"
    elif change == "extra":
        events.append(events[-1])
    elif change == "stream":
        events[1]["stream_id"] = 8
    else:
        events[0]["end"] = events[1]["start"] + 1
    with pytest.raises(ValueError):
        partition_field(events)


def test_service_ceiling_rejects_overlap_or_missing_fixed_floor():
    events = field_events()
    events[0]["end"] = events[1]["start"] + 1
    with pytest.raises(ValueError, match="Overlapping"):
        service_budget(events)
    with pytest.raises(ValueError, match="floor"):
        service_budget([dict(start=1, end=2, full_name="pointwise")])
