"""Bound Backlog output before it reaches an agent's context."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .schema import Json, object_value


def brief(board: Path, task_id: str, *, criteria: bool = False) -> Json:
    if not re.fullmatch(r"TASK-[0-9]+", task_id):
        raise ValueError("Expected TASK-number")
    result = subprocess.run(
        ["backlog", "task", "view", task_id, "--json"],
        cwd=board,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=30,
    )
    response = object_value(json.loads(result.stdout), "backlog response")
    if response.get("schemaVersion") != 1 or response.get("kind") != "task-view":
        raise ValueError("Unsupported Backlog JSON contract")
    task = object_value(response.get("task"), "task")
    if task.get("id") != task_id:
        raise ValueError("Backlog returned a different task")
    output: Json = {
        "contract": 1,
        "kind": "task-brief",
        **{
            key: task.get(key)
            for key in (
                "id",
                "title",
                "status",
                "priority",
                "dependencies",
                "documentation",
                "acceptanceCriteriaCompleted",
                "acceptanceCriteriaCount",
            )
        },
    }
    if criteria:
        output["acceptanceCriteria"] = task.get("acceptanceCriteria")
    return output
