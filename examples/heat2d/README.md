# 2D heat equation on a square

```sh
python run.py --mode train --input-dir examples/heat2d --output-dir /tmp/pinn-heat2d
python run.py --mode predict --input-dir examples/heat2d --output-dir /tmp/pinn-heat2d-predict --checkpoint /tmp/pinn-heat2d/checkpoints/final.ckpt
```

Install the integration dependencies first. The supplied configuration uses CPU
and only three training steps to exercise the pipeline, not to demonstrate
convergence. Initial temperature is sin(pi*x)*sin(pi*y), conductivity is 0.3,
and all four faces have zero temperature. No reference solution file is needed.
For 3D add z_interval/z_points, use four network inputs, constrain z faces, and
include the z derivative in the PDE. See docs/task-inputs.md for grid ordering.

BoundaryConditions applies zero Dirichlet temperature to a union of all faces;
shared corner points are included once.
