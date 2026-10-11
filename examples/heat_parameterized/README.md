# Parameterized 2D heat equation

```sh
python run.py --mode train --input-dir examples/heat_parameterized --output-dir /tmp/pinn-parameterized
python run.py --mode predict --input-dir examples/heat_parameterized --output-dir /tmp/pinn-parameterized-predict --checkpoint /tmp/pinn-parameterized/checkpoints/best.ckpt
```

Conductivity is a network input in [0.1, 0.5]. Initial temperature and boundary
values are independent of it, but their callbacks still receive parameters.
The model represents T(x, y, t, conductivity); conductivity is supplied, not
inferred. The three CPU steps only exercise the pipeline, not convergence.
MAT output includes the conductivity of each spatial row. Prediction may use
a different parameter grid with the same names and input ordering.

BoundaryConditions applies zero Dirichlet temperature to a union of all faces;
shared corner points are included once.
