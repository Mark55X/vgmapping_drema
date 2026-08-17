import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional

def hat_so3(w: torch.Tensor) -> torch.Tensor:
    """
    Skew-symmetric matrix operator [w]_x for 3D vector w = (w_x, w_y, w_z).
    """
    zero = torch.zeros_like(w[0])
    return torch.stack([
        torch.stack([zero, -w[2], w[1]]),
        torch.stack([w[2], zero, -w[0]]),
        torch.stack([-w[1], w[0], zero])
    ])

def exp_so3(w: torch.Tensor) -> torch.Tensor:
    """
    Exponential map Exp: so(3) -> SO(3) using Rodrigues' formula.
    w: (3,) rotation vector (axis-angle).
    Returns (3, 3) rotation matrix R.
    """
    theta = torch.norm(w)
    if theta < 1e-6:
        return torch.eye(3, device=w.device) + hat_so3(w)

    w_hat = hat_so3(w / theta)
    R = torch.eye(3, device=w.device) + torch.sin(theta) * w_hat + (1.0 - torch.cos(theta)) * (w_hat @ w_hat)
    return R

def exp_se3(xi: torch.Tensor) -> torch.Tensor:
    """
    Exponential map Exp: se(3) -> SE(3).
    xi: (6,) vector [omega (3,), v (3,)].
    Returns (4, 4) homogeneous transformation matrix T.
    """
    w = xi[:3]
    v = xi[3:]
    R = exp_so3(w)
    
    T = torch.eye(4, device=xi.device)
    T[:3, :3] = R
    T[:3, 3] = v
    return T

def icp_coarse_alignment(source_points: torch.Tensor, target_points: torch.Tensor, max_iters: int = 50) -> torch.Tensor:
    """
    Computes coarse SE(3) transformation T_coarse bringing source_points (N, 3) to target_points (M, 3) via SVD/ICP.
    Returns (4, 4) T_coarse matrix.
    """
    device = source_points.device
    if len(source_points) < 3 or len(target_points) < 3:
        return torch.eye(4, device=device)

    src = source_points.clone()
    tgt = target_points.clone()

    T_accum = torch.eye(4, device=device)

    for _ in range(max_iters):
        # Find nearest neighbors
        dists = torch.cdist(src, tgt) # (N, M)
        min_idx = torch.argmin(dists, dim=1)
        matched_tgt = tgt[min_idx]

        # Compute centroids
        src_mean = src.mean(dim=0, keepdim=True)
        tgt_mean = matched_tgt.mean(dim=0, keepdim=True)

        src_centered = src - src_mean
        tgt_centered = matched_tgt - tgt_mean

        # SVD of covariance matrix
        H = src_centered.T @ tgt_centered
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T

        # Reflection correction
        if torch.det(R) < 0:
            Vt_copy = Vt.clone()
            Vt_copy[2, :] *= -1
            R = Vt_copy.T @ U.T

        t = tgt_mean.squeeze() - src_mean.squeeze() @ R.T

        # Update transformed src
        src = src @ R.T + t

        # Accumulate T
        T_step = torch.eye(4, device=device)
        T_step[:3, :3] = R
        T_step[:3, 3] = t
        T_accum = T_step @ T_accum

    return T_accum


class RecurGSLieAlgebraAligner(nn.Module):
    """
    RecurGS SE(3) Lie Algebra Pose Refinement Module (arXiv:2512.18386).
    
    Refines coarse object displacement by optimizing Lie algebra parameter xi in se(3)
    to minimize photometric and geometric rendering loss L_align(xi) (Eq. 7-8).
    """
    def __init__(self, lambda_ssim: float = 0.2, lambda_depth: float = 0.5, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth
        self.device = device

    def optimize_se3_pose(
        self,
        object_gaussians: Dict[str, torch.Tensor],
        gt_rgb: torch.Tensor,
        gt_depth: torch.Tensor,
        intrinsic: torch.Tensor,
        camera_pose: torch.Tensor,
        initial_T_coarse: Optional[torch.Tensor] = None,
        num_iterations: int = 300,
        lr: float = 1e-3
    ) -> torch.Tensor:
        """
        Optimizes Lie algebra vector xi in se(3) over object Gaussians.
        
        object_gaussians: Dict with 'xyz' (N, 3), 'rgb' (N, 3), 'scale' (N, 3)
        gt_rgb: (3, H, W) target RGB observation at frame t
        gt_depth: (1, H, W) target depth observation at frame t
        intrinsic: (3, 3) camera K
        camera_pose: (4, 4) camera pose T_c2w
        
        Returns T_fine in SE(3) (4, 4).
        """
        device = self.device
        if initial_T_coarse is None:
            initial_T_coarse = torch.eye(4, device=device)

        # Initialize Lie algebra parameter xi = [w (3,), v (3,)]
        xi = nn.Parameter(torch.zeros(6, device=device, dtype=torch.float32))
        optimizer = torch.optim.Adam([xi], lr=lr)

        src_xyz = object_gaussians['xyz'].to(device)
        src_rgb = object_gaussians['rgb'].to(device)

        if len(src_xyz) == 0:
            return initial_T_coarse

        if src_xyz.ndim == 1:
            src_xyz = src_xyz.unsqueeze(0)
        if src_rgb.ndim == 1:
            src_rgb = src_rgb.unsqueeze(0)

        # Apply initial coarse transformation
        src_xyz_coarse = src_xyz @ initial_T_coarse[:3, :3].T + initial_T_coarse[:3, 3]

        H, W = gt_rgb.shape[1], gt_rgb.shape[2]
        fx, fy = intrinsic[0, 0].item(), intrinsic[1, 1].item()
        cx, cy = intrinsic[0, 2].item(), intrinsic[1, 2].item()

        w2c = torch.inverse(camera_pose.to(device))
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]

        for it in range(num_iterations):
            optimizer.zero_grad()

            # T(xi) = Exp(xi)
            T_xi = exp_se3(xi)
            R_xi = T_xi[:3, :3]
            t_xi = T_xi[:3, 3]

            # Transform points
            transformed_xyz = src_xyz_coarse @ R_xi.T + t_xi

            # Simple differentiable point splatting for pose optimization
            p_cam = transformed_xyz @ R_w2c.T + t_w2c
            z_cam = p_cam[:, 2]

            valid = z_cam > 0.1
            if not torch.any(valid):
                break

            u_proj = (p_cam[valid, 0] * fx / z_cam[valid]) + cx
            v_proj = (p_cam[valid, 1] * fy / z_cam[valid]) + cy

            in_bounds = (u_proj >= 0) & (u_proj < W - 1) & (v_proj >= 0) & (v_proj < H - 1)
            if not torch.any(in_bounds):
                break

            valid_u = u_proj[in_bounds].long()
            valid_v = v_proj[in_bounds].long()
            valid_z = z_cam[valid][in_bounds]
            valid_rgb = src_rgb[valid][in_bounds]

            # Bilinear/Nearest depth & color loss against GT
            gt_rgb_vals = gt_rgb[:, valid_v, valid_u].T # (M, 3)
            gt_depth_vals = gt_depth[0, valid_v, valid_u] # (M,)

            loss_rgb = F.l1_loss(valid_rgb, gt_rgb_vals)
            loss_depth = F.l1_loss(valid_z, gt_depth_vals)

            total_loss = loss_rgb + self.lambda_depth * loss_depth
            total_loss.backward()

            optimizer.step()

        with torch.no_grad():
            T_fine_opt = exp_se3(xi)
            T_fine = T_fine_opt @ initial_T_coarse

        return T_fine
