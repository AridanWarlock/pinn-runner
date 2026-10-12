# 2D heat diffusion with a spatial conductivity field

```sh
python examples/heat_spatial_coefficient/prepare.py
python run.py --mode train --input-dir examples/heat_spatial_coefficient --output-dir /tmp/pinn-spatial
python run.py --mode predict --input-dir examples/heat_spatial_coefficient --output-dir /tmp/pinn-spatial-predict --checkpoint /tmp/pinn-spatial/checkpoints/final.ckpt
```

The generated NPZ contains conductivity c(x,y)=0.1+0.2x+0.2y, ranging from 0.1
to 0.5, sampled on a 9×7 grid. It is an input coefficient, not a temperature
solution. The model solves T_t=div(c·grad(T)); c·laplacian(T) would omit grad(c).
The table resolution is independent of collocation and prediction grids.

The example runs three CPU steps to verify the pipeline, not convergence.
Use more training steps and validation appropriate to the physical problem.
A checkpoint is tied to this field: replacing the NPZ values requires new
training. Scalar inputs can control a family of fields through a formula
coefficient, for example amplitude×c(x,y).

See [coefficient API](../../docs/coefficients.md). Generated NPZ is not versioned.
