"""Read only a bounded Windows study status using the existing SSH transport."""

from __future__ import annotations

import json
import re

from scripts.gpu_jobs import WINDOWS_ROOT, remote

from .evidence import compact_status
from .schema import Json, object_value


def remote_status(relative: str) -> Json:
    if not re.fullmatch(r"numerical-reports/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", relative):
        raise ValueError("Remote root must be numerical-reports/name with safe path components")
    path = WINDOWS_ROOT + "\\" + relative.replace("/", "\\") + "\\status.json"
    output = remote(f"Get-Content -Raw -LiteralPath '{path}' -ErrorAction Stop\n")
    return compact_status(object_value(json.loads(output), "remote status"))
