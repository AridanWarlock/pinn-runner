"""Постановка приведённого времени; регистрация качества выполняется отдельно."""

from pathlib import Path
from typing import Any

from .cases import CASES
from .catalog import QUADRATIC_TRAINING_ROWS

SOURCE = Path(__file__).parent / "problems/heat_quadratic_clock/functions.py"


def clock_config(device: str, steps: int, seed: int) -> dict[str, Any]:
    config = CASES["heat_quadratic_shapes"].config(device, steps, seed)
    config["network"]["time_scale"] = {"parameter": "a", "reference": 0.15}
    config["domain"]["parameter_rows"] = [
        [0.15, b] for b in sorted({row[1] for row in QUADRATIC_TRAINING_ROWS})
    ]
    return config
