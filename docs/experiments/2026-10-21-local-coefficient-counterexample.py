"""CPU check: the same local k does not specify the same heat evolution."""

import csv
import math
from pathlib import Path

import torch


def main():
    rows = []
    for a in (0.05, 0.1, 0.2):
        for epsilon in (0.1, 0.2, 0.5):
            rates = []
            local = []
            for variable in (False, True):
                x = torch.tensor([[0.25]], dtype=torch.float64, requires_grad=True)
                field = torch.sin(torch.pi * x)
                gradient = torch.autograd.grad(
                    field, x, torch.ones_like(field), create_graph=True
                )[0]
                conductivity = (
                    a * (1 + epsilon * torch.cos(2 * torch.pi * x)) if variable else a + 0 * x
                )
                operator = torch.autograd.grad(conductivity * gradient, x, torch.ones_like(field))[
                    0
                ]
                rates.append(float(operator.detach()))
                local.append(float(conductivity.detach()))
            exact = -math.sqrt(2) * a * epsilon * math.pi**2
            difference = rates[1] - rates[0]
            assert abs(local[1] - local[0]) < 1e-15
            assert abs(difference - exact) < 1e-14
            rows.append(
                dict(
                    a=a,
                    epsilon=epsilon,
                    constant_local_k=local[0],
                    variable_local_k=local[1],
                    constant_initial_rate=rates[0],
                    variable_initial_rate=rates[1],
                    rate_difference=difference,
                    analytical_difference=exact,
                    absolute_error=abs(difference - exact),
                )
            )
    path = Path(__file__).with_suffix(".csv")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        dict(
            controls=len(rows),
            optimizer_updates=0,
            device="cpu",
            max_error=max(r["absolute_error"] for r in rows),
        )
    )


if __name__ == "__main__":
    main()
