"""Generate a prescribed conductivity field, not a reference temperature solution."""

from pathlib import Path

import numpy as np


def main() -> None:
    x = np.linspace(0, 1, 9)
    y = np.linspace(0, 1, 7)
    conductivity = 0.1 + 0.2 * x[:, None] + 0.2 * y[None, :]
    path = Path(__file__).with_name("conductivity.npz")
    np.savez(path, x=x, y=y, values=conductivity)
    print(f"Saved conductivity field: {path}")


if __name__ == "__main__":
    main()
