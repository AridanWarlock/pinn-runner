# Parameterized 2D heat equation

```sh
python run.py --mode train --input-dir examples/heat_parameterized --output-dir /tmp/pinn-parameterized
python run.py --mode predict --input-dir examples/heat_parameterized --output-dir /tmp/pinn-parameterized-predict --checkpoint /tmp/pinn-parameterized/checkpoints/final.ckpt
```

Conductivity is a network input in [0.1, 0.5]. Initial temperature and boundary
values are independent of it, but their callbacks still receive parameters.
The model represents T(x, y, t, conductivity); conductivity is supplied, not
inferred. The three CPU steps only exercise the pipeline, not convergence.
MAT output includes the conductivity of each spatial row. Prediction may use
a different parameter grid with the same names and input ordering.

BoundaryConditions applies zero Dirichlet temperature to a union of all faces;
shared corner points are included once.

For a specific coefficient add this to config.yaml, keeping domain.parameters unchanged:

```yaml
prediction:
  parameters:
    conductivity: [0.25]
  allow_extrapolation: false
```

The query is used only by predict. Training still covers the original interval.
A prediction query can have one value or multiple distinct values in the desired
order. The runner checks the query against checkpoint bounds and writes
prediction.json with domain and shape metadata. See [the API](../../docs/API.md).
