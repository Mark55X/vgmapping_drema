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

def batch_exp_se3(xi: torch.Tensor) -> torch.Tensor:
    """
    Batched Exponential map Exp: se(3)^K -> SE(3)^K.
    xi: (K, 6) tensor of independent Lie algebra parameters [omega (3,), v (3,)].
    Returns (K, 4, 4) homogeneous transformation matrices.
    """
    K = xi.shape[0]
    w = xi[:, :3]
    v = xi[:, 3:]
    
    theta = torch.norm(w, dim=1, keepdim=True) # (K, 1)
    
    w_hat = torch.zeros((K, 3, 3), device=xi.device, dtype=xi.dtype)
    w_hat[:, 0, 1] = -w[:, 2]
    w_hat[:, 0, 2] = w[:, 1]
    w_hat[:, 1, 0] = w[:, 2]
    w_hat[:, 1, 2] = -w[:, 0]
    w_hat[:, 2, 0] = -w[:, 1]
    w_hat[:, 2, 1] = w[:, 0]
    
    eye3 = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(K, 3, 3)
    
    safe_theta = torch.clamp(theta, min=1e-6).unsqueeze(-1)
    w_hat_norm = w_hat / safe_theta
    
    sin_theta = torch.sin(theta).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    
    R = eye3 + sin_theta * w_hat_norm + (1.0 - cos_theta) * torch.bmm(w_hat_norm, w_hat_norm)
    
    small_mask = (theta.squeeze(1) < 1e-6)
    if torch.any(small_mask):
        R[small_mask] = eye3[small_mask] + w_hat[small_mask]
        
    T = torch.eye(4, device=xi.device, dtype=xi.dtype).unsqueeze(0).repeat(K, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = v
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
    RecurGS SE(3) Lie Algebra Multi-Object Pose Refinement Module (arXiv:2512.18386).
    
    Optimizes independent Lie algebra parameters xi_k in se(3) for multiple dynamic objects
    in parallel on GPU with adaptive early-stopping.
    """
    def __init__(self, lambda_ssim: float = 0.2, lambda_depth: float = 0.5, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth
        self.device = device

    def optimize_multi_object_se3_pose(
        self,
        objects_gaussians: Dict[int, Dict[str, torch.Tensor]],
        gt_rgb: torch.Tensor,
        gt_depth: torch.Tensor,
        intrinsic: torch.Tensor,
        camera_pose: torch.Tensor,
        initial_T_coarse_dict: Optional[Dict[int, torch.Tensor]] = None,
        num_iterations: int = 50,
        lr: float = 2e-3,
        tol: float = 1e-4
    ) -> Dict[int, torch.Tensor]:
        """
        Simultaneously optimizes independent Lie algebra parameters xi_k in se(3)
        for all K dynamic objects in parallel on GPU using a single batched loss graph.
        
        objects_gaussians: Dict mapping obj_id -> {'xyz': (N_k, 3), 'rgb': (N_k, 3)}
        Returns Dict mapping obj_id -> T_fine (4, 4)
        """
        device = self.device
        obj_ids = list(objects_gaussians.keys())
        K = len(obj_ids)
        if K == 0:
            return {}

        initial_T_coarse = []
        for oid in obj_ids:
            if initial_T_coarse_dict and oid in initial_T_coarse_dict:
                initial_T_coarse.append(initial_T_coarse_dict[oid].to(device))
            else:
                initial_T_coarse.append(torch.eye(4, device=device))

        # Initialize K independent se(3) Lie algebra parameters xi in R^(K x 6)
        xi = nn.Parameter(torch.zeros((K, 6), device=device, dtype=torch.float32))
        optimizer = torch.optim.Adam([xi], lr=lr)

        H, W = gt_rgb.shape[1], gt_rgb.shape[2]
        fx, fy = intrinsic[0, 0].item(), intrinsic[1, 1].item()
        cx, cy = intrinsic[0, 2].item(), intrinsic[1, 2].item()

        w2c = torch.inverse(camera_pose.to(device))
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]

        src_data = []
        for k, oid in enumerate(obj_ids):
            g = objects_gaussians[oid]
            src_xyz = g['xyz'].to(device)
            src_rgb = g['rgb'].to(device)
            if src_xyz.ndim == 1:
                src_xyz = src_xyz.unsqueeze(0)
            if src_rgb.ndim == 1:
                src_rgb = src_rgb.unsqueeze(0)
            
            T_c = initial_T_coarse[k]
            src_xyz_coarse = src_xyz @ T_c[:3, :3].T + T_c[:3, 3]
            src_data.append((src_xyz_coarse, src_rgb))

        prev_loss = None
        consecutive_small_changes = 0

        for it in range(num_iterations):
            optimizer.zero_grad()

            T_xi = batch_exp_se3(xi) # (K, 4, 4)
            total_loss = 0.0
            valid_objects_count = 0

            for k in range(K):
                src_xyz_coarse, src_rgb = src_data[k]
                if len(src_xyz_coarse) == 0:
                    continue

                R_xi = T_xi[k, :3, :3]
                t_xi = T_xi[k, :3, 3]

                # Independent rigid transformation for object k
                transformed_xyz = src_xyz_coarse @ R_xi.T + t_xi

                # Projection to camera
                p_cam = transformed_xyz @ R_w2c.T + t_w2c
                z_cam = p_cam[:, 2]

                valid = z_cam > 0.1
                if not torch.any(valid):
                    continue

                u_proj = (p_cam[valid, 0] * fx / z_cam[valid]) + cx
                v_proj = (p_cam[valid, 1] * fy / z_cam[valid]) + cy

                in_bounds = (u_proj >= 0) & (u_proj < W - 1) & (v_proj >= 0) & (v_proj < H - 1)
                if not torch.any(in_bounds):
                    continue

                valid_u = u_proj[in_bounds].long()
                valid_v = v_proj[in_bounds].long()
                valid_z = z_cam[valid][in_bounds]
                valid_rgb = src_rgb[valid][in_bounds]

                gt_rgb_vals = gt_rgb[:, valid_v, valid_u].T
                gt_depth_vals = gt_depth[0, valid_v, valid_u]

                loss_rgb = F.l1_loss(valid_rgb, gt_rgb_vals)
                loss_depth = F.l1_loss(valid_z, gt_depth_vals)

                total_loss = total_loss + (loss_rgb + self.lambda_depth * loss_depth)
                valid_objects_count += 1

            if valid_objects_count == 0 or not isinstance(total_loss, torch.Tensor):
                break

            total_loss.backward()
            optimizer.step()

            # Early stopping check on loss convergence
            current_loss_val = total_loss.item()
            if prev_loss is not None:
                rel_change = abs(prev_loss - current_loss_val) / (prev_loss + 1e-6)
                if rel_change < tol:
                    consecutive_small_changes += 1
                    if consecutive_small_changes >= 3:
                        break
                else:
                    consecutive_small_changes = 0
            prev_loss = current_loss_val

        results = {}
        with torch.no_grad():
            T_final = batch_exp_se3(xi)
            for k, oid in enumerate(obj_ids):
                results[oid] = T_final[k] @ initial_T_coarse[k]

        return results

    def optimize_se3_pose(
        self,
        object_gaussians: Dict[str, torch.Tensor],
        gt_rgb: torch.Tensor,
        gt_depth: torch.Tensor,
        intrinsic: torch.Tensor,
        camera_pose: torch.Tensor,
        initial_T_coarse: Optional[torch.Tensor] = None,
        num_iterations: int = 50,
        lr: float = 2e-3
    ) -> torch.Tensor:
        """
        Optimizes Lie algebra vector xi in se(3) for a single object (backward compatible).
        """
        initial_dict = {0: initial_T_coarse} if initial_T_coarse is not None else None
        res = self.optimize_multi_object_se3_pose(
            objects_gaussians={0: object_gaussians},
            gt_rgb=gt_rgb,
            gt_depth=gt_depth,
            intrinsic=intrinsic,
            camera_pose=camera_pose,
            initial_T_coarse_dict=initial_dict,
            num_iterations=num_iterations,
            lr=lr
        )
        return res.get(0, torch.eye(4, device=self.device))
