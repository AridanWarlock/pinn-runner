"""CPU-only integration-grid diagnostics for the proposed Burgers reference."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

source = Path(__file__).with_name("2026-10-20-unforced-burgers-reference-audit.py")
spec = importlib.util.spec_from_file_location("burgers_reference", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows = []
energy_rows = []
for nu in (0.1, 0.01):
    for nx, nt in ((201, 101), (801, 101), (1601, 101), (1601, 201)):
        x = np.linspace(-1, 1, nx)
        t = np.linspace(0, 1, nt)
        xx, tt = np.meshgrid(x, t, indexing="ij")
        values = np.empty(xx.size)
        slopes = np.empty(xx.size)
        for start in range(0, xx.size, 256):
            indices = slice(start, start + 256)
            values[indices] = module.reference(
                xx.reshape(-1)[indices], tt.reshape(-1)[indices], nu
            )
            slopes[indices] = module.reference(
                xx.reshape(-1)[indices], tt.reshape(-1)[indices], nu, derivative=True
            )
        values = values.reshape(nx, nt)
        slopes = slopes.reshape(nx, nt)
        field_squared = np.trapz(values**2, x, axis=0)
        slope_squared = np.trapz(slopes**2, x, axis=0)
        energy = 0.5 * field_squared
        dt = t[1] - t[0]
        derivative = (energy[:-4] - 8 * energy[1:-3] + 8 * energy[3:-1] - energy[4:]) / (12 * dt)
        balance = derivative + nu * slope_squared[2:-2]
        if nx == 1601 and nt == 201:
            for instant in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.99):
                i = int(round(instant / dt))
                energy_rows.append(
                    dict(
                        nu=nu,
                        t=t[i],
                        energy=energy[i],
                        energy_time_derivative=derivative[i - 2],
                        viscous_dissipation=nu * slope_squared[i],
                        balance_defect=balance[i - 2],
                    )
                )
        rows.append(
            dict(
                energy_balance_max=float(abs(balance).max()),
                energy_balance_relative_max=float(
                    (abs(balance) / (nu * slope_squared[2:-2])).max()
                ),
                nu=nu,
                nx=nx,
                nt=nt,
                field_spacetime_l2=float(np.sqrt(np.trapz(field_squared, t))),
                field_final_l2=float(np.sqrt(field_squared[-1])),
                slope_spacetime_l2=float(np.sqrt(np.trapz(slope_squared, t))),
                slope_final_l2=float(np.sqrt(slope_squared[-1])),
            )
        )
base = Path(__file__).with_name("2026-10-20-unforced-burgers-grid")
with base.with_suffix(".csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
with base.with_name("2026-10-20-unforced-burgers-energy.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(energy_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(energy_rows)
summary = dict(
    networks=0,
    optimizer_steps=0,
    quadrature_order=512,
    rows=rows,
    scope="Reference norm integration only, not a convergence certificate for future PINN errors",
)
base.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
