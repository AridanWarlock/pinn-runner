"""Bounded board discovery without leaking task implementation histories."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from research_tools import board


def install_board(monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]) -> list[list[str]]:
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(response))

    monkeypatch.setattr(board.subprocess, "run", run)
    monkeypatch.setattr(
        board,
        "brief",
        lambda _, task_id: dict(
            id=task_id,
            title="Текущая задача",
            status="To Do",
            priority="high",
            documentation=["docs/requirements.md"],
            dependencies=["TASK-1"],
        ),
    )
    return commands


def response() -> dict[str, Any]:
    return dict(
        schemaVersion=1,
        kind="task-list",
        tasks=[
            dict(
                id="TASK-2",
                title="Текущая задача",
                status="To Do",
                priority="high",
                isReady=False,
                implementationNotes="SECRET HISTORY",
            )
        ],
        total=3,
        nextSkip=1,
    )


def test_page_preserves_navigation_and_requirements(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = install_board(monkeypatch, response())
    result = board.task_page(Path("."), limit=1)
    assert result["next_skip"] == 1 and result["total"] == 3
    assert result["tasks"][0]["documentation"] == ["docs/requirements.md"]
    assert result["tasks"][0]["board_ready"] is False
    assert "SECRET" not in json.dumps(result)
    assert commands[0][-4:] == ["--max-count", "1", "--skip", "0"]
    assert result["default_excluded_statuses"] == ["Done", "Deferred"]


def test_explicit_status_and_final_page(monkeypatch: pytest.MonkeyPatch) -> None:
    value = response()
    value.pop("total")
    value.pop("nextSkip")
    commands = install_board(monkeypatch, value)
    result = board.task_page(Path("."), status="To Do", skip=2)
    assert result["next_skip"] is None
    assert result["total"] is None  # No fabricated count when the CLI omits it.
    assert "--exclude-status" not in commands[0]
    assert commands[0][commands[0].index("--status") + 1] == "To Do"


@pytest.mark.parametrize(
    "kwargs", [dict(limit=0), dict(limit=21), dict(skip=-1), dict(status="x")]
)
def test_bad_options_never_read_board(
    kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = install_board(monkeypatch, response())
    with pytest.raises(ValueError):
        board.task_page(Path("."), **kwargs)
    assert not commands


@pytest.mark.parametrize(
    "damage", ["schema", "duplicate", "readiness", "next", "count", "changed"]
)
def test_bad_or_changed_board_is_rejected(damage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    value = response()
    if damage == "schema":
        value["schemaVersion"] = 2
    elif damage == "duplicate":
        value["tasks"] *= 2
    elif damage == "readiness":
        del value["tasks"][0]["isReady"]
    elif damage == "next":
        value["nextSkip"] = 0
    elif damage == "count":
        value["total"] = -1
    else:
        value["tasks"][0]["title"] = "Changed"
    install_board(monkeypatch, value)
    with pytest.raises(ValueError):
        board.task_page(Path("."))
