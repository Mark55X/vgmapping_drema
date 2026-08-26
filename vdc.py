import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from .tsdf import TSDFVoxelMap

def compute_ssim_map(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 5, C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    """
    Computes local SSIM map between img1 and img2 over a window_size x window_size patch (Eq. 11-14).
    img1, img2: (3, H, W) or (1, H, W) tensors in range [0, 1].
    Returns (1, H, W) SSIM map.
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    channels = img1.shape[1]
    kernel = torch.ones((channels, 1, window_size, window_size), device=img1.device) / (window_size * window_size)

    mu1 = F.conv2d(img1, kernel, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=window_size//2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size//2, groups=channels) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, kernel, padding=window_size//2, groups=channels) - mu1_mu2

    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = ssim_n / (ssim_d + 1e-8)

    return ssim_map.mean(dim=1, keepdim=True) # (1, 1, H, W)


def quadtree_segmentation_vectorized(
    img: torch.Tensor,
    min_size: int = 4,
    max_size: int = 16,
    threshold: float = 0.005
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized GPU Quadtree segmentation using fast avg_pool2d variance maps.
    Returns (u_c, v_c, patch_sizes) directly on GPU without Python recursion or GPU-CPU sync stalls.
    """
    device = img.device
    C, H, W = img.shape
    img_4d = img.unsqueeze(0)
    img_sq_4d = img_4d.pow(2)

    # 1. Level 1: max_size blocks (e.g. 16x16)
    mean_max = F.avg_pool2d(img_4d, max_size, stride=max_size)
    sq_mean_max = F.avg_pool2d(img_sq_4d, max_size, stride=max_size)
    var_max = (sq_mean_max - mean_max.pow(2)).mean(dim=1).squeeze(0)

    leaf_mask_max = (var_max < threshold)

    v_m, u_m = torch.where(leaf_mask_max)
    u_leaves = [u_m * max_size + max_size // 2]
    v_leaves = [v_m * max_size + max_size // 2]
    sz_leaves = [torch.full_like(u_m, max_size, dtype=torch.float32)]

    # 2. Level 2: Sub-blocks needing split (e.g. 8x8)
    split_v, split_u = torch.where(~leaf_mask_max)
    if len(split_v) > 0:
        half_sz = max_size // 2
        mean_half = F.avg_pool2d(img_4d, half_sz, stride=half_sz)
        sq_mean_half = F.avg_pool2d(img_sq_4d, half_sz, stride=half_sz)
        var_half = (sq_mean_half - mean_half.pow(2)).mean(dim=1).squeeze(0)

        sub_v = torch.stack([split_v * 2, split_v * 2, split_v * 2 + 1, split_v * 2 + 1], dim=1).flatten()
        sub_u = torch.stack([split_u * 2, split_u * 2 + 1, split_u * 2, split_u * 2 + 1], dim=1).flatten()

        valid_sub = (sub_v < var_half.shape[0]) & (sub_u < var_half.shape[1])
        sub_v = sub_v[valid_sub]
        sub_u = sub_u[valid_sub]

        var_sub = var_half[sub_v, sub_u]
        leaf_mask_sub = (var_sub < threshold) | (half_sz <= min_size)

        u_leaves.append(sub_u[leaf_mask_sub] * half_sz + half_sz // 2)
        v_leaves.append(sub_v[leaf_mask_sub] * half_sz + half_sz // 2)
        sz_leaves.append(torch.full((leaf_mask_sub.sum(),), half_sz, dtype=torch.float32, device=device))

        # 3. Level 3: Fine sub-blocks (e.g. 4x4)
        if half_sz > min_size:
            q_sz = half_sz // 2
            split_sub_v = sub_v[~leaf_mask_sub]
            split_sub_u = sub_u[~leaf_mask_sub]
            if len(split_sub_v) > 0:
                q_v = torch.stack([split_sub_v * 2, split_sub_v * 2, split_sub_v * 2 + 1, split_sub_v * 2 + 1], dim=1).flatten()
                q_u = torch.stack([split_sub_u * 2, split_sub_u * 2 + 1, split_sub_u * 2, split_sub_u * 2 + 1], dim=1).flatten()
                mean_q = F.avg_pool2d(img_4d, q_sz, stride=q_sz)
                valid_q = (q_v < mean_q.shape[2]) & (q_u < mean_q.shape[3])
                q_v = q_v[valid_q]
                q_u = q_u[valid_q]

                u_leaves.append(q_u * q_sz + q_sz // 2)
                v_leaves.append(q_v * q_sz + q_sz // 2)
                sz_leaves.append(torch.full((len(q_u),), q_sz, dtype=torch.float32, device=device))

    all_u = torch.cat(u_leaves)
    all_v = torch.cat(v_leaves)
    all_sz = torch.cat(sz_leaves)
    return all_u, all_v, all_sz

# Backward compatibility alias
quadtree_segmentation = quadtree_segmentation_vectorized


class VariationAwareDensityController:
    """
    Variation-aware Density Control (VDC) mechanism for VG-Mapping (arXiv:2510.09962).
    
    Provides:
    - Appearance-based Variation Detection (AVD) using SSIM (Eq. 11-14)
    - Geometry-based Variation Detection (GVD) using TSDF queries
    - Surface-normal guided Gaussian primitive initialization (Eq. 15-16)
    - Morton-code based frustum raycast pruning for deleted objects & floaters (Eq. 17)
    """
    def __init__(
        self,
        tau_s: float = 0.6,
        tau_p: float = 0.2,
        near_plane: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.tau_s = tau_s
        self.tau_p = tau_p
        self.n_p = near_plane
        self.device = device

    def detect_and_initialize_gaussians(
        self,
        rgb_obs: torch.Tensor,
        depth_obs: torch.Tensor,
        rendered_rgb: torch.Tensor,
        rendered_depth: torch.Tensor,
        intrinsic: torch.Tensor,
        pose: torch.Tensor,
        tsdf_map: TSDFVoxelMap,
        mask_obs: Optional[torch.Tensor] = None,
        workspace_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_initial_timestep: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Runs vectorized AVD and GVD passes over quadtree image patches to initialize new Gaussian primitives.
        """
        u_c, v_c, patch_sizes = quadtree_segmentation_vectorized(rgb_obs, min_size=4, max_size=16, threshold=0.005)
        if len(u_c) == 0:
            return {
                'xyz': torch.empty((0, 3), device=self.device),
                'rgb': torch.empty((0, 3), device=self.device),
                'scale': torch.empty((0, 3), device=self.device),
                'morton': torch.empty((0,), dtype=torch.int64, device=self.device),
                'obj_id': torch.empty((0,), dtype=torch.int32, device=self.device)
            }

        ssim_map = compute_ssim_map(rendered_rgb, rgb_obs).squeeze() # (H, W)
        H, W = rgb_obs.shape[1], rgb_obs.shape[2]

        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        R_c2w = pose[:3, :3].to(self.device)
        t_c2w = pose[:3, 3].to(self.device)

        valid_bounds = (u_c < W) & (v_c < H)
        u_c = u_c[valid_bounds]
        v_c = v_c[valid_bounds]
        patch_sizes = patch_sizes[valid_bounds]

        if len(u_c) == 0:
            return {
                'xyz': torch.empty((0, 3), device=self.device),
                'rgb': torch.empty((0, 3), device=self.device),
                'scale': torch.empty((0, 3), device=self.device),
                'morton': torch.empty((0,), dtype=torch.int64, device=self.device),
                'obj_id': torch.empty((0,), dtype=torch.int32, device=self.device)
            }

        d_vals = depth_obs[0, v_c, u_c]
        valid_depth = (d_vals > 0.1) & (d_vals < 5.0)

        u_c = u_c[valid_depth]
        v_c = v_c[valid_depth]
        patch_sizes = patch_sizes[valid_depth]
        d_vals = d_vals[valid_depth]

        if len(u_c) == 0:
            return {
                'xyz': torch.empty((0, 3), device=self.device),
                'rgb': torch.empty((0, 3), device=self.device),
                'scale': torch.empty((0, 3), device=self.device),
                'morton': torch.empty((0,), dtype=torch.int64, device=self.device),
                'obj_id': torch.empty((0,), dtype=torch.int32, device=self.device)
            }

        # 1. Appearance-based Variation Detection (AVD)
        patch_ssim = ssim_map[v_c, u_c]
        avd_flag = patch_ssim < self.tau_s

        # Back-project to 3D world points
        x_cam = (u_c.float() - cx) * d_vals / fx
        y_cam = (v_c.float() - cy) * d_vals / fy
        p_cam = torch.stack([x_cam, y_cam, d_vals], dim=-1) # (N, 3)
        p_world = p_cam @ R_c2w.T + t_c2w # (N, 3)

        # Filter to robot workspace table ROI
        if workspace_bounds is not None:
            min_b, max_b = workspace_bounds
            in_workspace = (
                (p_world[:, 0] >= min_b[0]) & (p_world[:, 0] <= max_b[0]) &
                (p_world[:, 1] >= min_b[1]) & (p_world[:, 1] <= max_b[1]) &
                (p_world[:, 2] >= min_b[2]) & (p_world[:, 2] <= max_b[2])
            )
        else:
            in_workspace = tsdf_map.is_inside_grid(p_world)

        # 2. Geometry-based Variation Detection (GVD) in batch
        _, w_vals = tsdf_map.query_tsdf_and_weight(p_world)
        gvd_flag = w_vals <= 1.0

        if is_initial_timestep:
            init_mask = in_workspace
        else:
            init_mask = in_workspace & (avd_flag | gvd_flag)

        if not torch.any(init_mask):
            return {
                'xyz': torch.empty((0, 3), device=self.device),
                'rgb': torch.empty((0, 3), device=self.device),
                'scale': torch.empty((0, 3), device=self.device),
                'morton': torch.empty((0,), dtype=torch.int64, device=self.device),
                'obj_id': torch.empty((0,), dtype=torch.int32, device=self.device)
            }

        p_world_init = p_world[init_mask]
        u_init = u_c[init_mask]
        v_init = v_c[init_mask]
        patch_sizes_init = patch_sizes[init_mask]
        d_vals_init = d_vals[init_mask]

        # Compute surface normals in batch
        grad_s = tsdf_map.compute_surface_normal(p_world_init) # (N_init, 3)
        grad_norm = torch.norm(grad_s, dim=-1, keepdim=True)
        
        # Exact Equations (15) and (16) from VG-Mapping paper (arXiv:2510.09962)
        L = patch_sizes_init.unsqueeze(-1) / 2.0
        d = (L * d_vals_init.unsqueeze(-1)) / torch.abs(fx)

        # n = 1_3 ⊘ (1_3 + abs(∇S(pc)))
        n = torch.where(grad_norm > 1e-5, 1.0 / (1.0 + torch.abs(grad_s)), torch.ones_like(grad_s))
        n_norm = n / (torch.norm(n, dim=-1, keepdim=True) + 1e-6)

        # S = d · diag(n / ||n||_2)
        S_diag = d * n_norm
        S_diag = torch.clamp(S_diag, min=1e-4)

        rgb_vals = rgb_obs[:, v_init, u_init].T # (N_init, 3)
        morton_vals = tsdf_map.point_to_morton(p_world_init) # (N_init,)

        if mask_obs is not None:
            if mask_obs.ndim == 3:
                mask_obs = mask_obs[:, :, 0]
            obj_id_vals = mask_obs[v_init, u_init].to(torch.int32).squeeze()
            if obj_id_vals.ndim > 1:
                obj_id_vals = obj_id_vals[:, 0]
            elif obj_id_vals.ndim == 0:
                obj_id_vals = obj_id_vals.unsqueeze(0)
        else:
            obj_id_vals = torch.zeros(len(p_world_init), dtype=torch.int32, device=self.device)

        return {
            'xyz': p_world_init,
            'rgb': rgb_vals,
            'scale': S_diag,
            'morton': morton_vals,
            'obj_id': obj_id_vals
        }

    def prune_gaussians_via_morton(
        self,
        depth_obs: torch.Tensor,
        intrinsic: torch.Tensor,
        pose: torch.Tensor,
        tsdf_map: TSDFVoxelMap,
        gaussian_morton_codes: torch.Tensor,
        stride: int = 12
    ) -> torch.Tensor:
        """
        Vectorized frustum ray-casting (Eq. 17) in GPU batch.
        """
        if len(gaussian_morton_codes) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        H, W = depth_obs.shape[1], depth_obs.shape[2]
        s = tsdf_map.voxel_size

        u_coords = torch.arange(0, W, stride, device=self.device)
        v_coords = torch.arange(0, H, stride, device=self.device)
        grid_v, grid_u = torch.meshgrid(v_coords, u_coords, indexing="ij")
        grid_u = grid_u.flatten()
        grid_v = grid_v.flatten()

        depth_vals = depth_obs[0, grid_v, grid_u]
        valid_mask = depth_vals > self.n_p

        grid_u = grid_u[valid_mask]
        grid_v = grid_v[valid_mask]
        depth_vals = depth_vals[valid_mask]

        if len(grid_u) == 0:
            return torch.zeros(len(gaussian_morton_codes), dtype=torch.bool, device=self.device)

        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        R_c2w = pose[:3, :3].to(self.device)
        t_c2w = pose[:3, 3].to(self.device)

        # Batch Ray Sampling: rays from near plane to current observed depth minus 1.5cm margin
        num_steps = 20
        step_fractions = torch.linspace(0.0, 1.0, num_steps, device=self.device).view(1, -1) # (1, S)
        z_start = self.n_p
        safety_margin = max(0.015, 1.5 * s)
        z_end = torch.clamp(depth_vals - safety_margin, min=self.n_p).unsqueeze(1) # (R, 1)
        z_samples = z_start + step_fractions * (z_end - z_start) # (R, S)

        u_exp = grid_u.unsqueeze(1).expand(-1, num_steps) # (R, S)
        v_exp = grid_v.unsqueeze(1).expand(-1, num_steps) # (R, S)

        x_cam = (u_exp.float() - cx) * z_samples / fx
        y_cam = (v_exp.float() - cy) * z_samples / fy
        p_cam = torch.stack([x_cam, y_cam, z_samples], dim=-1) # (R, S, 3)

        p_cam_flat = p_cam.reshape(-1, 3) # (R*S, 3)
        p_w_flat = p_cam_flat @ R_c2w.T + t_c2w # (R*S, 3)

        # Only ray samples strictly inside the TSDF grid boundary are evaluated
        inside_rays = tsdf_map.is_inside_grid(p_w_flat)
        if not torch.any(inside_rays):
            return torch.zeros(len(gaussian_morton_codes), dtype=torch.bool, device=self.device)

        p_w_valid_rays = p_w_flat[inside_rays]
        bad_mortons = tsdf_map.point_to_morton(p_w_valid_rays).unique()
        bad_mortons = bad_mortons[bad_mortons >= 0] # exclude invalid sentinel -1

        prune_mask = torch.zeros(len(gaussian_morton_codes), dtype=torch.bool, device=self.device)
        valid_g_mask = (gaussian_morton_codes >= 0)
        if len(bad_mortons) > 0 and torch.any(valid_g_mask):
            prune_mask[valid_g_mask] = torch.isin(gaussian_morton_codes[valid_g_mask], bad_mortons)

        return prune_mask
