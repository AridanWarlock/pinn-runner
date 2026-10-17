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


def task_page(board: Path, *, status: str | None = None, limit: int = 5, skip: int = 0) -> Json:
    """Read a bounded board page; readiness is advisory, never execution authority."""
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("Task page limit must be between 1 and 20")
    if type(skip) is not int or skip < 0:
        raise ValueError("Task page skip must be a nonnegative integer")
    statuses = ("To Do", "In Progress", "Waiting CUDA", "Review", "Deferred", "Done")
    if status is not None and status not in statuses:
        raise ValueError("Unknown task status")
    command = ["backlog", "task", "list", "--json", "--sort", "priority"]
    command += (
        ["--status", status]
        if status is not None
        else ["--exclude-status", "Done", "--exclude-status", "Deferred"]
    )
    command += ["--max-count", str(limit), "--skip", str(skip)]
    result = subprocess.run(
        command,
        cwd=board,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=30,
    )
    response = object_value(json.loads(result.stdout), "backlog response")
    if response.get("schemaVersion") != 1 or response.get("kind") != "task-list":
        raise ValueError("Unsupported Backlog list contract")
    tasks = response.get("tasks")
    if not isinstance(tasks, list) or len(tasks) > limit:
        raise ValueError("Backlog returned an invalid task page")
    identifiers: set[str] = set()
    rows: list[Json] = []
    for raw in tasks:
        task = object_value(raw, "task")
        identifier = task.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"TASK-[0-9]+", identifier):
            raise ValueError("Invalid task identity")
        if identifier in identifiers:
            raise ValueError("Duplicate task in board page")
        identifiers.add(identifier)
        row = brief(board, identifier)
        # Two CLI reads are not an atomic snapshot. Refuse mixed board states.
        for key in ("title", "status", "priority"):
            if row[key] != task.get(key):
                raise ValueError("Board changed while reading; retry tasks")
        ready = task.get("isReady")
        if type(ready) is not bool:
            raise ValueError("Backlog readiness must be explicit")
        rows.append(row | {"board_ready": ready})
    total, next_skip = response.get("total"), response.get("nextSkip")
    if total is not None and (type(total) is not int or total < len(rows)):
        raise ValueError("Invalid task count")
    if next_skip is not None and (
        type(next_skip) is not int or not rows or next_skip != skip + len(rows)
    ):
        raise ValueError("Invalid next task page")
    return {
        "contract": 1,
        "kind": "task-page",
        "status_filter": status,
        "default_excluded_statuses": ["Done", "Deferred"] if status is None else [],
        "skip": skip,
        "limit": limit,
        "total": total,
        "next_skip": next_skip,
        "tasks": rows,
        "limitations": "Board readiness is advisory; inspect dependencies and requirements before starting. No task is selected or executed automatically.",
    }
