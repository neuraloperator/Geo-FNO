# Geometry-Aware Fourier Neural Operator (Geo-FNO)
Deep learning surrogate models have shown promise in solving partial differential equations (PDEs). Among them, the Fourier neural operator (FNO) achieves good accuracy, and is significantly faster compared to numerical solvers,  on a variety of   PDEs, such as fluid flows. However, the FNO uses the Fast Fourier transform  (FFT), which is limited to rectangular domains with uniform grids. In this work, we propose a new framework, viz., geo-FNO, to solve PDEs on arbitrary geometries. Geo-FNO learns to deform the input (physical) domain, which may be irregular, into a latent space with a uniform grid. The FNO model with the FFT is applied in the latent space. The resulting geo-FNO model has both the computation efficiency of FFT and the flexibility of handling arbitrary geometries. Our geo-FNO is also flexible in terms of its input formats, viz.,  point clouds, meshes, and design parameters are all valid inputs. We consider a variety of PDEs such as the Elasticity, Plasticity, Euler's, and Navier-Stokes equations, and both forward modeling and inverse design problems. Geo-FNO is $10^5$ times faster than the standard numerical solvers and twice more accurate compared to direct interpolation on existing ML-based PDE solvers such as the standard FNO.

https://arxiv.org/abs/2207.05209

## Requirements
- The code only depends on pytorch (>=1.8.0) [PyTorch](https://pytorch.org/). 


## Datasets
We provide the elasticity, plasticity, airfoil flows (naca) and pipe flows (ns) datasets we used in the paper. 
The data generation configuration can be found in the paper.
- [Geo-PDE datasets](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8?usp=sharing)

## Examples
```bash
python3 elasticity/elas_geofno.py
```
