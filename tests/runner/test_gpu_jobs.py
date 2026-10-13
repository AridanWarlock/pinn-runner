"""Control-plane validation: no network or GPU needed."""

import json
import stat
import zipfile
from pathlib import Path

import pytest

from scripts.gpu_jobs import json_output
from scripts.gpu_worker import extract_snapshot, valid_id, validate_request, write_json


@pytest.mark.parametrize("identifier", ["../escape", "a/b", "", "a'", "a b"])
def test_job_ids_are_safe(identifier: str) -> None:
    with pytest.raises(ValueError):
        valid_id(identifier)


def test_request_validation() -> None:
    request = {
        "job_id": "heat3d-001",
        "commit": "a" * 40,
        "case": "heat3d",
        "steps": 6000,
        "seeds": [1234],
        "timeout_seconds": 3600,
    }
    validate_request(request)
    for key, value in (
        ("commit", "HEAD"),
        ("steps", -1),
        ("steps", True),
        ("seeds", [1, 1]),
        ("case", "unknown"),
    ):
        with pytest.raises(ValueError):
            validate_request({**request, key: value})


@pytest.mark.parametrize("name", ["../escape", "..\\escape", "C:/escape", "/escape"])
def test_snapshot_paths_cannot_escape(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr(name, "unsafe")
    with pytest.raises(ValueError):
        extract_snapshot(archive, tmp_path / "checkout")


def test_snapshot_symlinks_are_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    entry = zipfile.ZipInfo("link")
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr(entry, "../outside")
    with pytest.raises(ValueError):
        extract_snapshot(archive, tmp_path / "checkout")


def test_atomic_state_and_json_response(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    write_json(path, {"state": "queued"})
    write_json(path, {"state": "passed"})
    assert json.loads(path.read_text()) == {"state": "passed"}
    assert not path.with_suffix(".tmp").exists()
    assert json_output('notice\n{"state":"running"}\n') == {"state": "running"}
