import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List

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
        return torch.eye(3, device=w.device, dtype=w.dtype) + hat_so3(w)

    w_hat = hat_so3(w / theta)
    R = torch.eye(3, device=w.device, dtype=w.dtype) + torch.sin(theta) * w_hat + (1.0 - torch.cos(theta)) * (w_hat @ w_hat)
    return R

def left_jacobian_so3(w: torch.Tensor) -> torch.Tensor:
    """
    Computes the left Jacobian V(w) of SO(3) so that t = V(w) * v under the SE(3) exponential map.
    """
    theta = torch.norm(w)
    eye3 = torch.eye(3, device=w.device, dtype=w.dtype)
    w_hat = hat_so3(w)
    if theta < 1e-6:
        return eye3 + 0.5 * w_hat + (1.0 / 6.0) * (w_hat @ w_hat)
    
    w_hat_norm = w_hat / theta
    term1 = ((1.0 - torch.cos(theta)) / (theta * theta)) * (theta * w_hat_norm)
    term2 = ((theta - torch.sin(theta)) / (theta * theta * theta)) * (theta * theta * (w_hat_norm @ w_hat_norm))
    return eye3 + (1.0 - torch.cos(theta)) / (theta) * w_hat_norm + ((theta - torch.sin(theta)) / theta) * (w_hat_norm @ w_hat_norm)

def exp_se3(xi: torch.Tensor) -> torch.Tensor:
    """
    Exact Exponential map Exp: se(3) -> SE(3).
    xi: (6,) vector [omega (3,), v (3,)].
    Returns (4, 4) homogeneous transformation matrix T.
    """
    w = xi[:3]
    v = xi[3:]
    R = exp_so3(w)
    V = left_jacobian_so3(w)
    t = V @ v
    
    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def batch_exp_se3(xi: torch.Tensor) -> torch.Tensor:
    """
    Batched Exponential map Exp: se(3)^K -> SE(3)^K with left Jacobian.
    xi: (K, 6) tensor of Lie algebra parameters [omega (3,), v (3,)].
    Returns (K, 4, 4) homogeneous transformation matrices.
    """
    K = xi.shape[0]
    if K == 0:
        return torch.empty((0, 4, 4), device=xi.device, dtype=xi.dtype)

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
    
    safe_theta = torch.clamp(theta, min=1e-6).unsqueeze(-1) # (K, 1, 1)
    w_hat_norm = w_hat / safe_theta
    
    sin_theta = torch.sin(theta).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    
    w_hat_sq = torch.bmm(w_hat_norm, w_hat_norm)
    R = eye3 + sin_theta * w_hat_norm + (1.0 - cos_theta) * w_hat_sq
    
    # Left Jacobian V: I + (1 - cos theta)/theta * w_hat_norm + (theta - sin theta)/theta * w_hat_sq
    V = eye3 + ((1.0 - cos_theta) / safe_theta) * (safe_theta * w_hat_norm) + ((safe_theta - sin_theta) / safe_theta) * w_hat_sq
    
    small_mask = (theta.squeeze(1) < 1e-6)
    if torch.any(small_mask):
        R[small_mask] = eye3[small_mask] + w_hat[small_mask]
        V[small_mask] = eye3[small_mask] + 0.5 * w_hat[small_mask]
        
    t = torch.bmm(V, v.unsqueeze(-1)).squeeze(-1) # (K, 3)

    T = torch.eye(4, device=xi.device, dtype=xi.dtype).unsqueeze(0).repeat(K, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T

def rotation_matrix_to_quaternion(R: torch.Tensor) -> Tuple[float, float, float, float]:
    """
    Converts 3x3 rotation matrix to quaternion (qx, qy, qz, qw) using stable Shepperd's algorithm.
    """
    R_cpu = R.detach().cpu().numpy()
    tr = R_cpu[0, 0] + R_cpu[1, 1] + R_cpu[2, 2]
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R_cpu[2, 1] - R_cpu[1, 2]) / S
        qy = (R_cpu[0, 2] - R_cpu[2, 0]) / S
        qz = (R_cpu[1, 0] - R_cpu[0, 1]) / S
    elif (R_cpu[0, 0] > R_cpu[1, 1]) and (R_cpu[0, 0] > R_cpu[2, 2]):
        S = np.sqrt(1.0 + R_cpu[0, 0] - R_cpu[1, 1] - R_cpu[2, 2]) * 2.0
        qw = (R_cpu[2, 1] - R_cpu[1, 2]) / S
        qx = 0.25 * S
        qy = (R_cpu[0, 1] + R_cpu[1, 0]) / S
        qz = (R_cpu[0, 2] + R_cpu[2, 0]) / S
    elif R_cpu[1, 1] > R_cpu[2, 2]:
        S = np.sqrt(1.0 + R_cpu[1, 1] - R_cpu[0, 0] - R_cpu[2, 2]) * 2.0
        qw = (R_cpu[0, 2] - R_cpu[2, 0]) / S
        qx = (R_cpu[0, 1] + R_cpu[1, 0]) / S
        qy = 0.25 * S
        qz = (R_cpu[1, 2] + R_cpu[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R_cpu[2, 2] - R_cpu[0, 0] - R_cpu[1, 1]) * 2.0
        qw = (R_cpu[1, 0] - R_cpu[0, 1]) / S
        qx = (R_cpu[0, 2] + R_cpu[2, 0]) / S
        qy = (R_cpu[1, 2] + R_cpu[2, 1]) / S
        qz = 0.25 * S

    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm > 1e-8:
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    else:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

    return (float(qx), float(qy), float(qz), float(qw))


def icp_coarse_alignment(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    max_iters: int = 40,
    max_dist_thresh: float = 0.15,
    initial_T: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes robust coarse SE(3) transformation T_coarse bringing source_points (N, 3)
    to target_points (M, 3) using Kabsch SVD with distance outlier rejection and centroid centering.
    
    Returns (4, 4) homogeneous transformation matrix T_coarse.
    """
    device = source_points.device
    dtype = source_points.dtype
    if len(source_points) < 4 or len(target_points) < 4:
        return torch.eye(4, device=device, dtype=dtype) if initial_T is None else initial_T.clone()

    src = source_points.clone()
    tgt = target_points.clone()

    T_accum = torch.eye(4, device=device, dtype=dtype)
    if initial_T is not None:
        T_init = initial_T.to(device=device, dtype=dtype)
        src = src @ T_init[:3, :3].T + T_init[:3, 3]
        T_accum = T_init.clone()

    for _ in range(max_iters):
        # 1. Nearest neighbor search
        dists = torch.cdist(src, tgt) # (N, M)
        min_dists, min_idx = torch.min(dists, dim=1)
        matched_tgt = tgt[min_idx]

        # 2. Outlier rejection: threshold by max_dist_thresh or 85th percentile
        thresh = min(max_dist_thresh, torch.quantile(min_dists, 0.90).item() * 1.5 + 1e-4)
        valid_mask = min_dists <= thresh
        
        if valid_mask.sum() < 4:
            valid_mask = min_dists <= torch.quantile(min_dists, 0.50)

        src_valid = src[valid_mask]
        tgt_valid = matched_tgt[valid_mask]

        # 3. Compute Centroids
        src_mean = src_valid.mean(dim=0, keepdim=True)
        tgt_mean = tgt_valid.mean(dim=0, keepdim=True)

        src_centered = src_valid - src_mean
        tgt_centered = tgt_valid - tgt_mean

        # 4. SVD of Covariance Matrix (Kabsch algorithm)
        H = src_centered.T @ tgt_centered
        try:
            U, S, Vh = torch.linalg.svd(H)
            R = Vh.T @ U.T

            # Reflection correction
            if torch.det(R) < 0:
                Vh_corr = Vh.clone()
                Vh_corr[2, :] *= -1
                R = Vh_corr.T @ U.T

            t = tgt_mean.squeeze() - src_mean.squeeze() @ R.T

            # Update transformed source
            src = src @ R.T + t

            # Accumulate transformation
            T_step = torch.eye(4, device=device, dtype=dtype)
            T_step[:3, :3] = R
            T_step[:3, 3] = t
            T_accum = T_step @ T_accum

            # Early stopping check on step magnitude
            angle = torch.norm(torch.tensor([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], device=device)) * 0.5
            trans_norm = torch.norm(t)
            if angle < 1e-4 and trans_norm < 1e-4:
                break
        except Exception:
            break

    return T_accum


class RecurGSLieAlgebraAligner(nn.Module):
    """
    RecurGS SE(3) Lie Algebra Multi-Object Pose Refinement Module (arXiv:2512.18386).
    
    Key Features:
    1. Centroid-Centered Lie Algebra Optimization: Optimizes rotation around the object's centroid,
       completely uncoupling rotation from translation.
    2. Two-Stage Pipeline: Robust outlier-trimmed ICP coarse alignment + fine Lie-algebra refinement.
    3. Differentiable Multimodal Loss: Combines 3D Chamfer/Huber geometric loss with
       sub-pixel bilinear grid-sampled photometric + depth loss.
    4. Table Support Constraint: Prevents objects from penetrating below the physical table surface.
    """
    def __init__(
        self,
        lambda_geom: float = 1.0,
        lambda_photo: float = 0.2,
        lambda_depth: float = 0.5,
        lambda_table: float = 100.0,
        lambda_upright: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.lambda_geom = lambda_geom
        self.lambda_photo = lambda_photo
        self.lambda_depth = lambda_depth
        self.lambda_table = lambda_table
        self.lambda_upright = lambda_upright
        self.device = device

    def optimize_multi_object_se3_pose(
        self,
        objects_source: Dict[int, Dict[str, torch.Tensor]],
        objects_target: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        gt_rgb: Optional[torch.Tensor] = None,
        gt_depth: Optional[torch.Tensor] = None,
        intrinsic: Optional[torch.Tensor] = None,
        camera_pose: Optional[torch.Tensor] = None,
        initial_T_coarse_dict: Optional[Dict[int, torch.Tensor]] = None,
        z_table: Optional[float] = None,
        num_iterations: int = 15,
        icp_max_iters: int = 12,
        lr: float = 3e-3,
        tol: float = 1e-5
    ) -> Dict[int, torch.Tensor]:
        """
        Simultaneously refines SE(3) poses for K dynamic objects.
        
        objects_source: Dict mapping obj_id -> {'xyz': (N_k, 3), 'rgb': (N_k, 3)} (canonical or t-1 model)
        objects_target: Optional Dict mapping obj_id -> {'xyz': (M_k, 3), 'rgb': (M_k, 3)} (time t observations)
        Returns: Dict mapping obj_id -> T_fine (4, 4) homogeneous transformation matrix.
        """
        device = self.device
        obj_ids = list(objects_source.keys())
        K = len(obj_ids)
        if K == 0:
            return {}

        # 1. Coarse Alignment: Compute T_coarse via ICP if targets provided and no initial T given
        initial_T_coarse = {}
        for oid in obj_ids:
            if initial_T_coarse_dict and oid in initial_T_coarse_dict:
                initial_T_coarse[oid] = initial_T_coarse_dict[oid].to(device)
            elif objects_target and oid in objects_target and len(objects_target[oid].get('xyz', [])) >= 4:
                # Run robust ICP coarse alignment
                src_pts = objects_source[oid]['xyz'].to(device)
                tgt_pts = objects_target[oid]['xyz'].to(device)
                T_icp = icp_coarse_alignment(src_pts, tgt_pts, max_iters=icp_max_iters, max_dist_thresh=0.15)
                initial_T_coarse[oid] = T_icp
            else:
                initial_T_coarse[oid] = torch.eye(4, device=device)

        # 2. Prepare Source Object Models & Centroids
        src_data = []
        centroids = []
        for oid in obj_ids:
            g = objects_source[oid]
            src_xyz = g['xyz'].to(device)
            src_rgb = g.get('rgb', torch.zeros_like(src_xyz)).to(device)
            if src_xyz.ndim == 1:
                src_xyz = src_xyz.unsqueeze(0)
            if src_rgb.ndim == 1:
                src_rgb = src_rgb.unsqueeze(0)

            # Apply coarse transform
            T_c = initial_T_coarse[oid]
            src_coarse = src_xyz @ T_c[:3, :3].T + T_c[:3, 3]
            
            # Compute centroid of coarse model to decouple rotation from translation
            c_obj = src_coarse.mean(dim=0, keepdim=True) # (1, 3)
            src_centered = src_coarse - c_obj # (N, 3)

            src_data.append((src_centered, c_obj, src_rgb))
            centroids.append(c_obj)

        # 3. Initialize K independent Lie algebra parameters xi = [omega, v] in se(3)
        xi = nn.Parameter(torch.zeros((K, 6), device=device, dtype=torch.float32))
        optimizer = torch.optim.Adam([xi], lr=lr)

        # Camera calibration parameters if 2D supervision is provided
        has_camera = (gt_rgb is not None and gt_depth is not None and intrinsic is not None and camera_pose is not None)
        if has_camera:
            H, W = gt_rgb.shape[1], gt_rgb.shape[2]
            fx, fy = intrinsic[0, 0].item(), intrinsic[1, 1].item()
            cx, cy = intrinsic[0, 2].item(), intrinsic[1, 2].item()
            w2c = torch.inverse(camera_pose.to(device))
            R_w2c = w2c[:3, :3]
            t_w2c = w2c[:3, 3]

        prev_loss = None
        consecutive_small_changes = 0

        # 4. Lie-Algebra Optimization Loop
        for it in range(num_iterations):
            optimizer.zero_grad()

            T_xi = batch_exp_se3(xi) # (K, 4, 4)
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            valid_objects_count = 0

            for k, oid in enumerate(obj_ids):
                src_centered, c_obj, src_rgb = src_data[k]
                if len(src_centered) == 0:
                    continue

                R_xi = T_xi[k, :3, :3]
                t_xi = T_xi[k, :3, 3]

                # Centroid-centered rigid transformation:
                # p' = R_xi * (p - c_obj) + c_obj + t_xi
                P_trans = src_centered @ R_xi.T + c_obj + t_xi

                obj_loss = torch.tensor(0.0, device=device)

                # A. 3D Geometric Chamfer/Huber consistency with target point cloud
                if objects_target and oid in objects_target:
                    tgt_pts = objects_target[oid]['xyz'].to(device)
                    if len(tgt_pts) >= 4:
                        # Pairwise distances
                        dists = torch.cdist(P_trans, tgt_pts) # (N, M)
                        min_src_to_tgt, _ = torch.min(dists, dim=1) # (N,)
                        min_tgt_to_src, _ = torch.min(dists, dim=0) # (M,)

                        # Smooth L1 / Huber loss
                        loss_geom_src = F.smooth_l1_loss(min_src_to_tgt, torch.zeros_like(min_src_to_tgt), beta=0.01)
                        loss_geom_tgt = F.smooth_l1_loss(min_tgt_to_src, torch.zeros_like(min_tgt_to_src), beta=0.01)
                        loss_geom = 0.5 * (loss_geom_src + loss_geom_tgt)
                        obj_loss = obj_loss + self.lambda_geom * loss_geom

                # B. Differentiable Sub-pixel Photometric & Depth Loss via grid_sample
                if has_camera:
                    p_cam = P_trans @ R_w2c.T + t_w2c
                    z_cam = p_cam[:, 2]

                    valid_depth = z_cam > 0.05
                    if torch.any(valid_depth):
                        # Normalized device coordinates in [-1, 1] for bilinear grid_sample
                        u_px = (p_cam[valid_depth, 0] * fx / z_cam[valid_depth]) + cx
                        v_px = (p_cam[valid_depth, 1] * fy / z_cam[valid_depth]) + cy

                        u_norm = (u_px / (W - 1.0)) * 2.0 - 1.0
                        v_norm = (v_px / (H - 1.0)) * 2.0 - 1.0

                        in_bounds = (u_norm >= -1.0) & (u_norm <= 1.0) & (v_norm >= -1.0) & (v_norm <= 1.0)
                        if torch.any(in_bounds):
                            grid = torch.stack([u_norm[in_bounds], v_norm[in_bounds]], dim=-1).view(1, 1, -1, 2)
                            
                            # Bilinear interpolation provides exact sub-pixel gradients dI/du, dI/dv
                            sampled_rgb = F.grid_sample(gt_rgb.unsqueeze(0), grid, mode='bilinear', align_corners=True).squeeze(0).squeeze(1).T
                            sampled_depth = F.grid_sample(gt_depth.unsqueeze(0), grid, mode='bilinear', align_corners=True).squeeze(0).squeeze(0).squeeze(0)

                            v_rgb = src_rgb[valid_depth][in_bounds]
                            v_z = z_cam[valid_depth][in_bounds]

                            loss_photo = F.l1_loss(sampled_rgb, v_rgb)
                            loss_depth = F.smooth_l1_loss(v_z, sampled_depth, beta=0.01)

                            obj_loss = obj_loss + (self.lambda_photo * loss_photo + self.lambda_depth * loss_depth)

                # C. Physical Table Support Barrier Constraint
                if z_table is not None:
                    min_z = torch.min(P_trans[:, 2])
                    if min_z < z_table:
                        loss_table = F.relu(z_table - min_z) ** 2
                        obj_loss = obj_loss + self.lambda_table * loss_table

                # D. Upright Regularization (penalizes tilt away from table normal Z)
                T_c = initial_T_coarse[oid]
                R_composite = R_xi @ T_c[:3, :3]
                # Measure deviation from upright (dot product of local Z with world Z)
                tilt_loss = 1.0 - R_composite[2, 2]
                obj_loss = obj_loss + self.lambda_upright * F.relu(tilt_loss)

                total_loss = total_loss + obj_loss
                valid_objects_count += 1

            if valid_objects_count == 0 or not total_loss.requires_grad:
                break

            total_loss.backward()
            optimizer.step()

            # Early stopping check
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

        # 5. Extract Final Refined Homogeneous SE(3) Matrices
        results = {}
        with torch.no_grad():
            T_final = batch_exp_se3(xi)
            for k, oid in enumerate(obj_ids):
                R_xi = T_final[k, :3, :3]
                t_xi = T_final[k, :3, 3]
                T_c = initial_T_coarse[oid]
                c_obj = centroids[k].squeeze(0) # (3,)

                # Mathematically:
                # P_final = R_xi * (R_c * P_src + t_c - c_obj) + c_obj + t_xi
                #         = (R_xi * R_c) * P_src + [ R_xi * (t_c - c_obj) + c_obj + t_xi ]
                R_fine = R_xi @ T_c[:3, :3]
                t_fine = R_xi @ (T_c[:3, 3] - c_obj) + c_obj + t_xi

                T_res = torch.eye(4, device=device, dtype=torch.float32)
                T_res[:3, :3] = R_fine
                T_res[:3, 3] = t_fine
                results[oid] = T_res

        return results

    def optimize_se3_pose(
        self,
        object_gaussians: Dict[str, torch.Tensor],
        gt_rgb: Optional[torch.Tensor] = None,
        gt_depth: Optional[torch.Tensor] = None,
        intrinsic: Optional[torch.Tensor] = None,
        camera_pose: Optional[torch.Tensor] = None,
        initial_T_coarse: Optional[torch.Tensor] = None,
        target_gaussians: Optional[Dict[str, torch.Tensor]] = None,
        z_table: Optional[float] = None,
        num_iterations: int = 50,
        lr: float = 3e-3
    ) -> torch.Tensor:
        """
        Optimizes Lie algebra vector xi in se(3) for a single object (backward compatible).
        """
        initial_dict = {0: initial_T_coarse} if initial_T_coarse is not None else None
        target_dict = {0: target_gaussians} if target_gaussians is not None else None
        res = self.optimize_multi_object_se3_pose(
            objects_source={0: object_gaussians},
            objects_target=target_dict,
            gt_rgb=gt_rgb,
            gt_depth=gt_depth,
            intrinsic=intrinsic,
            camera_pose=camera_pose,
            initial_T_coarse_dict=initial_dict,
            z_table=z_table,
            num_iterations=num_iterations,
            lr=lr
        )
        return res.get(0, torch.eye(4, device=self.device))
