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
        tsdf_map: TSDFVoxelMap
    ) -> Dict[str, torch.Tensor]:
        """
        Runs AVD and GVD passes over quadtree image patches to initialize new Gaussian primitives.
        
        Returns dictionary of Gaussian parameters:
        - 'xyz': (N, 3)
        - 'rgb': (N, 3)
        - 'scale': (N, 3)
        - 'morton': (N,) int64 Morton codes
        """
        leaves = quadtree_segmentation(rgb_obs, min_size=4, max_size=32)
        ssim_map = compute_ssim_map(rendered_rgb, rgb_obs).squeeze() # (H, W)

        fx = intrinsic[0, 0].item()
        fy = intrinsic[1, 1].item()
        cx = intrinsic[0, 2].item()
        cy = intrinsic[1, 2].item()

        R_c2w = pose[:3, :3].to(self.device)
        t_c2w = pose[:3, 3].to(self.device)

        new_xyz_list = []
        new_rgb_list = []
        new_scale_list = []
        new_morton_list = []

        for patch in leaves:
            u_c = patch.x + patch.size // 2
            v_c = patch.y + patch.size // 2

            H, W = rgb_obs.shape[1], rgb_obs.shape[2]
            if u_c >= W or v_c >= H:
                continue

            # 1. Appearance-based Variation Detection (AVD)
            patch_ssim = ssim_map[patch.y:patch.y+patch.size, patch.x:patch.x+patch.size].mean().item()
            avd_flag = patch_ssim < self.tau_s

            # Back-project patch center to 3D point p_c
            d_val = depth_obs[0, v_c, u_c].item()
            if d_val <= 0.1 or d_val > 5.0:
                continue

            x_cam = (u_c - cx) * d_val / fx
            y_cam = (v_c - cy) * d_val / fy
            p_cam = torch.tensor([x_cam, y_cam, d_val], dtype=torch.float32, device=self.device)
            p_world = R_c2w @ p_cam + t_c2w

            # 2. Geometry-based Variation Detection (GVD)
            _, w_val = tsdf_map.query_tsdf_and_weight(p_world.unsqueeze(0))
            gvd_flag = (w_val.item() <= 1.0) # Changed or uninitialized region

            if avd_flag or gvd_flag:
                # Initialize Gaussian primitive (Eq. 15-16)
                # Compute gradient grad S(p_c) for normal n
                grad_s = tsdf_map.compute_surface_normal(p_world.unsqueeze(0)).squeeze(0) # (3,)
                
                if torch.norm(grad_s) > 1e-5:
                    n = 1.0 / (1.0 + torch.abs(grad_s))
                else:
                    n = torch.ones(3, device=self.device)

                n_norm = n / (torch.norm(n) + 1e-6)

                L = patch.size / 2.0
                d_scale = (L * d_val) / fx
                S_diag = d_scale * n_norm

                rgb_val = rgb_obs[:, v_c, u_c]
                morton_val = tsdf_map.point_to_morton(p_world.unsqueeze(0)).squeeze(0)

                new_xyz_list.append(p_world)
                new_rgb_list.append(rgb_val)
                new_scale_list.append(S_diag)
                new_morton_list.append(morton_val)

        if len(new_xyz_list) == 0:
            return {
                'xyz': torch.empty((0, 3), device=self.device),
                'rgb': torch.empty((0, 3), device=self.device),
                'scale': torch.empty((0, 3), device=self.device),
                'morton': torch.empty((0,), dtype=torch.int64, device=self.device)
            }

        return {
            'xyz': torch.stack(new_xyz_list, dim=0),
            'rgb': torch.stack(new_rgb_list, dim=0),
            'scale': torch.stack(new_scale_list, dim=0),
            'morton': torch.stack(new_morton_list, dim=0)
        }

    def prune_gaussians_via_morton(
        self,
        depth_obs: torch.Tensor,
        intrinsic: torch.Tensor,
        pose: torch.Tensor,
        tsdf_map: TSDFVoxelMap,
        gaussian_morton_codes: torch.Tensor,
        stride: int = 4
    ) -> torch.Tensor:
        """
        Performs frustum ray-casting (Eq. 17) to identify deleted objects (TSDF < tau_p)
        and floaters (TSDF > 0.95), matching Morton codes to create a boolean prune mask.
        
        gaussian_morton_codes: (N,) int64 Morton codes stored for current Gaussians.
        Returns boolean prune mask (N,) where True indicates Gaussian to be pruned.
        """
        if len(gaussian_morton_codes) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        H, W = depth_obs.shape[1], depth_obs.shape[2]
        s = tsdf_map.voxel_size

        u_coords = torch.arange(0, W, stride, device=self.device)
        v_coords = torch.arange(0, H, stride, device=self.device)
        grid_u, grid_v = torch.meshgrid(u_coords, v_coords, indexing="ij")
        grid_u = grid_u.flatten()
        grid_v = grid_v.flatten()

        depth_vals = depth_obs[0, grid_v, grid_u]
        valid_mask = depth_vals > self.n_p

        grid_u = grid_u[valid_mask]
        grid_v = grid_v[valid_mask]
        depth_vals = depth_vals[valid_mask]

        fx = intrinsic[0, 0].item()
        fy = intrinsic[1, 1].item()
        cx = intrinsic[0, 2].item()
        cy = intrinsic[1, 2].item()

        R_c2w = pose[:3, :3].to(self.device)
        t_c2w = pose[:3, 3].to(self.device)

        prune_morton_set: Set[int] = set()

        # Perform ray marching along sampled rays (Eq. 17)
        for i in range(len(grid_u)):
            u_i = grid_u[i].item()
            v_i = grid_v[i].item()
            d_max = depth_vals[i].item() - s

            z_steps = torch.arange(self.n_p, max(self.n_p + s, d_max), step=s, device=self.device)
            if len(z_steps) == 0:
                continue

            x_c = (u_i - cx) * z_steps / fx
            y_c = (v_i - cy) * z_steps / fy
            p_c = torch.stack([x_c, y_c, z_steps], dim=-1) # (Z, 3)

            p_w = p_c @ R_c2w.T + t_c2w # (Z, 3)

            f_vals, _ = tsdf_map.query_tsdf_and_weight(p_w)
            
            # Condition: Deleted object (F < tau_p) OR Floater noise (F > 0.95)
            prune_condition = (f_vals < self.tau_p) | (f_vals > 0.95)
            if torch.any(prune_condition):
                bad_points = p_w[prune_condition]
                bad_mortons = tsdf_map.point_to_morton(bad_points).cpu().tolist()
                prune_morton_set.update(bad_mortons)

        if len(prune_morton_set) == 0:
            return torch.zeros(len(gaussian_morton_codes), dtype=torch.bool, device=self.device)

        # Match Morton codes
        prune_morton_tensor = torch.tensor(list(prune_morton_set), dtype=torch.int64, device=self.device)
        prune_mask = torch.isin(gaussian_morton_codes, prune_morton_tensor)

        return prune_mask
