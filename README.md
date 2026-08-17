# Native VG-Mapping & RecurGS SE(3) Pose Alignment Library

This standalone Python package provides a complete implementation of:
1. **VG-Mapping** (*Variation-aware Density Control for Online 3D Gaussian Mapping in Semi-static Scenes*, arXiv:2510.09962)
2. **RecurGS Lie Algebra SE(3) Refinement** (*RecurGS: Interactive Scene Modeling via Discrete-State Recurrent Gaussian Fusion*, arXiv:2512.18386)

---

## Features
- **TSDF Voxel Map** (`tsdf.py`): Vectorized frustum depth integration with dynamic negative weight updates ($W_t = -5$) for modified regions.
- **Morton Coding**: 3D Morton spatial indexing ($V = \text{interleave\_bits}(i, j, k)$) for per-voxel Gaussian primitive tracking and fast raycast pruning without point-in-box checks.
- **Variation-aware Density Control (VDC)** (`vdc.py`):
  - **AVD**: SSIM patch-level appearance variation detection.
  - **GVD**: TSDF weight query ($W(\mathbf{p}_c) > 1$) to prevent redundant initializations.
  - **Surface-Normal Guidance**: $\nabla S(\mathbf{p}_c)$ TSDF central finite-difference calculation for flat surface-aligned Gaussian initializations.
  - **Morton Raycast Pruning**: Ray casting $r_{\mathbf{u}}$ from $z=n_p$ to $z=D[\mathbf{u}] - s$, pruning Gaussians matching Morton codes of deleted objects ($F < \tau_p$) or floaters ($F > 0.95$).
- **RecurGS SE(3) Pose Alignment** (`recurgs_se3.py`):
  - Exponential map $\text{Exp}: \mathfrak{se}(3) \to SE(3)$ over 6D Lie algebra vector $\xi = (\boldsymbol{\omega}, \mathbf{v})$.
  - Coarse ICP point-cloud alignment followed by PyTorch autograd optimization over photometric and geometric loss $\mathcal{L}_{\text{align}}(\xi)$.
- **Solid Surface Mesh Extraction**: Marching Cubes algorithm producing 3D mesh vertices and faces for PyBullet simulation engines.

---

## Quickstart & Verification

Run the standalone verification test suite:
```bash
python3 vg_mapping_recurgs_native/test_standalone.py
```
# vgmapping_drema
