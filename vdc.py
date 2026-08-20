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


class QuadtreeNode:
    def __init__(self, x: int, y: int, size: int):
        self.x = x
        self.y = y
        self.size = size
        self.children: List['QuadtreeNode'] = []
        self.is_leaf = True

def quadtree_segmentation(img: torch.Tensor, min_size: int = 4, max_size: int = 32, threshold: float = 0.02) -> List[QuadtreeNode]:
    """
    Quadtree image segmentation based on MSE variance.
    Returns leaf nodes corresponding to square image patches.
    """
    _, H, W = img.shape
    device = img.device
    leaves = []

    def build_tree(x: int, y: int, size: int):
        node = QuadtreeNode(x, y, size)
        if size <= min_size:
            leaves.append(node)
            return node

        patch = img[:, y:y+size, x:x+size]
        if patch.numel() == 0:
            return node

        mse = torch.var(patch, dim=(1, 2)).mean().item()
        if mse < threshold or size // 2 < min_size:
            leaves.append(node)
            return node

        node.is_leaf = False
        half = size // 2
        for dy in [0, half]:
            for dx in [0, half]:
                if x + dx < W and y + dy < H:
                    child = build_tree(x + dx, y + dy, half)
                    node.children.append(child)
        return node

    for y in range(0, H, max_size):
        for x in range(0, W, max_size):
            h_s = min(max_size, H - y)
            w_s = min(max_size, W - x)
            sz = min(h_s, w_s)
            if sz >= min_size:
                build_tree(x, y, sz)

    return leaves


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
        ssim_threshold: float = 0.6,   # tau_s
        prune_threshold: float = 0.2,  # tau_p
        keyframe_threshold: int = 200, # tau_k
        min_ray_dist: float = 0.1,     # n_p
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.tau_s = ssim_threshold
        self.tau_p = prune_threshold
        self.tau_k = keyframe_threshold
        self.n_p = min_ray_dist
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
        mask_obs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Runs vectorized AVD and GVD passes over quadtree image patches to initialize new Gaussian primitives.
        """
        leaves = quadtree_segmentation(rgb_obs, min_size=4, max_size=32)
        if len(leaves) == 0:
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

        # Vectorize patch properties
        u_c = torch.tensor([p.x + p.size // 2 for p in leaves], dtype=torch.long, device=self.device)
        v_c = torch.tensor([p.y + p.size // 2 for p in leaves], dtype=torch.long, device=self.device)
        patch_sizes = torch.tensor([p.size for p in leaves], dtype=torch.float32, device=self.device)

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

        # 2. Geometry-based Variation Detection (GVD) in batch
        _, w_vals = tsdf_map.query_tsdf_and_weight(p_world)
        gvd_flag = w_vals <= 1.0

        init_mask = avd_flag | gvd_flag
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
        
        n = torch.where(grad_norm > 1e-5, 1.0 / (1.0 + torch.abs(grad_s)), torch.ones_like(grad_s))
        n_norm = n / (torch.norm(n, dim=-1, keepdim=True) + 1e-6)

        L = patch_sizes_init.unsqueeze(-1) / 2.0
        d_scale = (L * d_vals_init.unsqueeze(-1)) / torch.abs(fx)
        S_diag = torch.abs(d_scale * n_norm)

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
        stride: int = 8
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

        # Batch Ray Sampling with safety margin to prevent pruning valid surface geometry
        num_steps = 40
        step_fractions = torch.linspace(0.0, 1.0, num_steps, device=self.device).view(1, -1) # (1, S)
        z_start = self.n_p
        safety_margin = max(0.035, 3.5 * s)
        z_end = torch.clamp(depth_vals - safety_margin, min=self.n_p).unsqueeze(1) # (R, 1)
        z_samples = z_start + step_fractions * (z_end - z_start) # (R, S)

        u_exp = grid_u.unsqueeze(1).expand(-1, num_steps) # (R, S)
        v_exp = grid_v.unsqueeze(1).expand(-1, num_steps) # (R, S)

        x_cam = (u_exp.float() - cx) * z_samples / fx
        y_cam = (v_exp.float() - cy) * z_samples / fy
        p_cam = torch.stack([x_cam, y_cam, z_samples], dim=-1) # (R, S, 3)

        p_cam_flat = p_cam.reshape(-1, 3) # (R*S, 3)
        p_w_flat = p_cam_flat @ R_c2w.T + t_c2w # (R*S, 3)

        f_vals, w_vals = tsdf_map.query_tsdf_and_weight(p_w_flat)

        prune_condition = (w_vals > 0) & ((f_vals < self.tau_p) | (f_vals > 0.95))
        if not torch.any(prune_condition):
            return torch.zeros(len(gaussian_morton_codes), dtype=torch.bool, device=self.device)

        bad_points = p_w_flat[prune_condition]
        bad_mortons = tsdf_map.point_to_morton(bad_points).unique()

        prune_mask = torch.isin(gaussian_morton_codes, bad_mortons)
        return prune_mask
