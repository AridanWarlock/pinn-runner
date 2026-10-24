"""Постановка приведённого времени; регистрация качества выполняется отдельно."""

from pathlib import Path
from typing import Any

from .cases import CASES

SOURCE = Path(__file__).parent / "problems/heat_quadratic_clock/functions.py"


def clock_config(device: str, steps: int, seed: int) -> dict[str, Any]:
    return CASES["heat_quadratic_clock"].config(device, steps, seed)
