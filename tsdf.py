import torch
import numpy as np
from typing import Tuple, Optional, Dict, List

def interleave_bits_3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Computes 3D Morton code (z-order curve) for integer coordinates (x, y, z).
    x, y, z: 1D or ND int64 tensors with values in range [0, 1023] (10 bits each).
    Returns uint64/int64 tensor of Morton codes (30 bits total).
    """
    def expand_bits(v: torch.Tensor) -> torch.Tensor:
        v = v.to(torch.int64) & 0x3FF  # 10 bits
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8))  & 0x0300F00F
        v = (v | (v << 4))  & 0x030C30C3
        v = (v | (v << 2))  & 0x09249249
        return v

    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)

def deinterleave_bits_3d(morton: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decodes 3D Morton code back into integer coordinates (x, y, z).
    """
    def compact_bits(v: torch.Tensor) -> torch.Tensor:
        v = v.to(torch.int64) & 0x09249249
        v = (v | (v >> 2))  & 0x030C30C3
        v = (v | (v >> 4))  & 0x0300F00F
        v = (v | (v >> 8))  & 0x030000FF
        v = (v | (v >> 16)) & 0x000003FF
        return v

    x = compact_bits(morton)
    y = compact_bits(morton >> 1)
    z = compact_bits(morton >> 2)
    return x, y, z


class TSDFVoxelMap:
    """
    Truncated Signed Distance Function (TSDF) Voxel Map with Morton Code indexing,
    as specified in VG-Mapping (arXiv:2510.09962).
    
    Supports:
    - Dynamic TSDF update with responsive negative weights for changed regions (Eq. 2-6)
    - Morton code spatial indexing for 3D Gaussian assignment and pruning
    - Central finite-difference surface normal estimation grad S(p)
    - Marching Cubes solid surface mesh extraction for PyBullet
    """
    def __init__(
        self,
        voxel_size: float = 0.01,
        truncation_margin: float = 0.04,
        noise_threshold: float = 0.05,
        grid_dim: Tuple[int, int, int] = (256, 256, 256),
        origin: Tuple[float, float, float] = (-1.28, -1.28, -1.28),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.voxel_size = voxel_size
        self.truncation_margin = truncation_margin
        self.epsilon_F = noise_threshold
        self.grid_dim = grid_dim
        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)
        self.device = device

        # TSDF storage: F(p) in [-1, 1], W(p) >= 0
        self.F = torch.ones(grid_dim, dtype=torch.float32, device=device)
        self.W = torch.zeros(grid_dim, dtype=torch.float32, device=device)

        # Precompute voxel 3D center positions
        nx, ny, nz = grid_dim
        ix = torch.arange(nx, device=device)
        iy = torch.arange(ny, device=device)
        iz = torch.arange(nz, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(ix, iy, iz, indexing="ij")
        
        self.grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1) # (nx, ny, nz, 3)
        self.voxel_centers = self.origin + (self.grid_coords.float() + 0.5) * self.voxel_size

        # Precompute Morton codes for all voxels
        self.morton_codes = interleave_bits_3d(grid_x, grid_y, grid_z) # (nx, ny, nz)

    def is_inside_grid(self, points: torch.Tensor) -> torch.Tensor:
        """
        Returns boolean mask (N,) of points strictly inside voxel grid bounding box.
        """
        if len(points) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        max_bound = self.origin + torch.tensor(self.grid_dim, dtype=torch.float32, device=self.device) * self.voxel_size
        inside = (points >= self.origin) & (points < max_bound)
        return inside.all(dim=-1)

    def point_to_voxel_index(self, points: torch.Tensor) -> torch.Tensor:
        """
        Converts 3D world points (N, 3) into voxel grid integer indices (N, 3).
        """
        indices = torch.floor((points - self.origin) / self.voxel_size).long()
        indices = torch.clamp(indices, min=torch.tensor([0, 0, 0], device=self.device),
                                       max=torch.tensor([self.grid_dim[0]-1, self.grid_dim[1]-1, self.grid_dim[2]-1], device=self.device))
        return indices

    def point_to_morton(self, points: torch.Tensor) -> torch.Tensor:
        """
        Computes Morton code V for 3D world points p. Points outside the grid get -1.
        """
        if len(points) == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        inside = self.is_inside_grid(points)
        indices = self.point_to_voxel_index(points)
        morton = interleave_bits_3d(indices[:, 0], indices[:, 1], indices[:, 2])
        morton = torch.where(inside, morton, torch.tensor(-1, dtype=torch.int64, device=self.device))
        return morton

    def integrate_depth_frame(
        self,
        depth: torch.Tensor,
        intrinsic: torch.Tensor,
        pose: torch.Tensor,
        depth_scale: float = 1.0,
        max_depth: float = 3.0
    ):
        """
        Integrates per-frame depth observation into global TSDF grid following Eq. (2)-(6).
        
        depth: (H, W) depth map in meters (or divided by depth_scale)
        intrinsic: (3, 3) camera intrinsic matrix K
        pose: (4, 4) camera pose T_t = [R_t | t_t] in world frame
        """
        if depth.dim() == 2:
            depth = depth.unsqueeze(0) # (1, H, W)
        
        H, W_img = depth.shape[1], depth.shape[2]
        c2w = pose.to(self.device)
        w2c = torch.inverse(c2w)
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]

        K = intrinsic.to(self.device)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Transform voxel centers into camera frame
        # p_cam = R_w2c * p_world + t_w2c
        flat_centers = self.voxel_centers.reshape(-1, 3) # (V, 3)
        p_cam = flat_centers @ R_w2c.T + t_w2c # (V, 3)

        # Project to image space u = [u, v]
        z_cam = p_cam[:, 2]
        valid_mask = (z_cam > 0.1) & (z_cam < max_depth)
        
        u_proj = torch.round((p_cam[:, 0] * fx / z_cam) + cx).long()
        v_proj = torch.round((p_cam[:, 1] * fy / z_cam) + cy).long()

        in_frustum = valid_mask & (u_proj >= 0) & (u_proj < W_img) & (v_proj >= 0) & (v_proj < H)
        
        indices_in_frustum = torch.nonzero(in_frustum).squeeze(-1)
        if len(indices_in_frustum) == 0:
            return

        u_valid = u_proj[indices_in_frustum]
        v_valid = v_proj[indices_in_frustum]
        z_valid = z_cam[indices_in_frustum]

        obs_depth = depth[0, v_valid, u_valid] / depth_scale
        valid_depth = obs_depth > 0.0

        indices_obs = indices_in_frustum[valid_depth]
        z_obs = z_valid[valid_depth]
        d_obs = obs_depth[valid_depth]

        # Signed distance F_t(p) = phi(D[u] - ||t_t - p|| / ...)
        diff = d_obs - z_obs
        valid_trunc = diff > -self.truncation_margin

        indices_update = indices_obs[valid_trunc]
        diff_update = diff[valid_trunc]
        
        # phi(x) = max(-1, min(1, x / mu))
        F_t = torch.clamp(diff_update / self.truncation_margin, min=-1.0, max=1.0)

        # Get existing TSDF values
        flat_F = self.F.view(-1)
        flat_W = self.W.view(-1)
        
        F_old = flat_F[indices_update]
        W_old = flat_W[indices_update]

        # Eq. (4): W_t(p) = 1 if |F_t(p) - F(p)| < epsilon_F else -5
        change_mask = torch.abs(F_t - F_old) >= self.epsilon_F
        W_t = torch.where(change_mask, torch.tensor(-5.0, device=self.device), torch.tensor(1.0, device=self.device))

        # Eq. (5) & (6): Global TSDF update with responsive negative weight
        # F(p) = (|W_t| * F_t + W * F) / (|W_t| + W)
        abs_W_t = torch.abs(W_t)
        F_new = (abs_W_t * F_t + W_old * F_old) / (abs_W_t + W_old + 1e-6)
        W_new = torch.clamp(W_old + W_t, min=1.0)

        flat_F[indices_update] = F_new
        flat_W[indices_update] = W_new

    def query_tsdf_and_weight(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Queries F(p) and W(p) at 3D world points (N, 3).
        """
        indices = self.point_to_voxel_index(points)
        f_vals = self.F[indices[:, 0], indices[:, 1], indices[:, 2]]
        w_vals = self.W[indices[:, 0], indices[:, 1], indices[:, 2]]
        return f_vals, w_vals

    def compute_surface_normal(self, points: torch.Tensor) -> torch.Tensor:
        """
        Computes surface normal n(p) using central finite-difference approximation
        grad S(p) over TSDF voxel grid. Eq. (16).
        """
        indices = self.point_to_voxel_index(points)
        x, y, z = indices[:, 0], indices[:, 1], indices[:, 2]

        x_prev = torch.clamp(x - 1, min=0)
        x_next = torch.clamp(x + 1, max=self.grid_dim[0] - 1)
        y_prev = torch.clamp(y - 1, min=0)
        y_next = torch.clamp(y + 1, max=self.grid_dim[1] - 1)
        z_prev = torch.clamp(z - 1, min=0)
        z_next = torch.clamp(z + 1, max=self.grid_dim[2] - 1)

        dx = (self.F[x_next, y, z] - self.F[x_prev, y, z]) / (2.0 * self.voxel_size)
        dy = (self.F[x, y_next, z] - self.F[x, y_prev, z]) / (2.0 * self.voxel_size)
        dz = (self.F[x, y, z_next] - self.F[x, y, z_prev]) / (2.0 * self.voxel_size)

        grad = torch.stack([dx, dy, dz], dim=-1) # (N, 3)
        return grad

    def extract_mesh(self, level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts solid surface mesh using Marching Cubes algorithm on TSDF grid F(p)=level.
        Returns vertices (V, 3) and faces (F, 3).
        """
        try:
            import mcubes
            F_np = self.F.cpu().numpy()
            W_np = self.W.cpu().numpy()
            
            # Mask out unobserved voxels (W <= 0.5)
            F_np[W_np <= 0.5] = 1.0

            vertices, triangles = mcubes.marching_cubes(F_np, level)
            
            # Convert grid coordinates to world coordinates
            origin_np = self.origin.cpu().numpy()
            vertices = origin_np + (vertices + 0.5) * self.voxel_size
            return vertices, triangles
        except ImportError:
            # Fallback using skimage marching cubes
            try:
                from skimage.measure import marching_cubes
                F_np = self.F.cpu().numpy()
                W_np = self.W.cpu().numpy()
                F_np[W_np <= 0.5] = 1.0
                if F_np.min() > level or F_np.max() < level:
                    return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
                
                verts, faces, normals, values = marching_cubes(F_np, level=level)
                origin_np = self.origin.cpu().numpy()
                verts = origin_np + (verts + 0.5) * self.voxel_size
                return verts, faces
            except Exception:
                return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
