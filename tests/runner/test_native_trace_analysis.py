"""Read-only event attribution, including asynchronous kernels and ambiguous launches."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from research_tools.schema import digest
from research_tools.trace_analysis import analyze, interval_union


def trace(path: Path) -> None:
    with sqlite3.connect(path) as c:
        c.executescript("""
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        INSERT INTO StringIds VALUES(1,'matrix'),(2,'physical_field');
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER,textId INTEGER);
        INSERT INTO NVTX_EVENTS VALUES(10,20,'physical_field',7,NULL),
            (21,30,'physical_residual',7,NULL),(31,40,'parameter_gradient',7,NULL);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(12,15,7,1),(22,25,7,2),(32,35,7,3),
            (32,35,7,4),(32,35,8,4),(0,1,7,5);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,shortName INTEGER,globalPid INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(50,60,1,1,0),(55,65,2,1,0),(60,70,3,1,0),(60,70,4,1,0),(1,2,5,1,0);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,correlationId INTEGER,bytes INTEGER,globalPid INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(70,80,2,16,0);
        """)


def test_launch_attribution_handles_gpu_finishing_after_host_range(tmp_path: Path) -> None:
    path = tmp_path / "trace.sqlite"
    trace(path)
    before = digest(path)
    result = analyze(path, before)
    assert result["all_kernel_count"] == 5
    assert result["attributed_kernel_count"] == 3
    assert result["stages"][0]["kernel_count"] == 1
    assert result["stages"][1]["copy_bytes"] == 16
    assert result["unattributed"] == {
        "kernel_missing_or_ambiguous_runtime": 1,
        "kernel_outside_phase": 1,
    }
    assert digest(path) == before


def test_union_does_not_double_count_overlap() -> None:
    assert interval_union([(10, 30), (20, 40), (45, 50)]) == 35
    with pytest.raises(ValueError, match="Negative"):
        interval_union([(3, 2)])


def test_trace_source_checksum_must_match(tmp_path: Path) -> None:
    path = tmp_path / "trace.sqlite"
    trace(path)
    with pytest.raises(ValueError, match="checksum"):
        analyze(path, "a" * 64)


def test_incomplete_nvtx_sequence_rejected(tmp_path: Path) -> None:
    path = tmp_path / "trace.sqlite"
    trace(path)
    with sqlite3.connect(path) as c:
        c.execute("DELETE FROM NVTX_EVENTS WHERE text='physical_residual'")
    with pytest.raises(ValueError, match="Incomplete"):
        analyze(path, digest(path))
