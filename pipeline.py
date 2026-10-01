import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from .tsdf import TSDFVoxelMap
from .vdc import VariationAwareDensityController
from .recurgs_se3 import RecurGSLieAlgebraAligner, icp_coarse_alignment

class NativeVGMappingRecurGSPipeline:
    """
    Unified Standalone Native Implementation of VG-Mapping & RecurGS SE(3) Pose Alignment.
    
    Combines:
    - TSDF Voxel Grid Mapping with Morton code spatial indexing
    - Variation-aware Density Control (AVD + GVD + Morton raycast pruning)
    - RecurGS Lie algebra se(3) pose refinement for moving objects
    - Solid surface mesh extraction
    """
    def __init__(
        self,
        voxel_size: float = 0.01,
        grid_dim: Tuple[int, int, int] = (256, 256, 256),
        origin: Tuple[float, float, float] = (-1.28, -1.28, -1.28),
        max_weight: float = 15.0,
        tau_p: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.tsdf_map = TSDFVoxelMap(voxel_size=voxel_size, grid_dim=grid_dim, origin=origin, max_weight=max_weight, device=device)
        self.vdc = VariationAwareDensityController(tau_p=tau_p, device=device)
        self.se3_aligner = RecurGSLieAlgebraAligner(device=device)

        # Gaussian scene representation storage
        self.gaussians = {
            'xyz': torch.empty((0, 3), dtype=torch.float32, device=device),
            'rgb': torch.empty((0, 3), dtype=torch.float32, device=device),
            'scale': torch.empty((0, 3), dtype=torch.float32, device=device),
            'morton': torch.empty((0,), dtype=torch.int64, device=device)
        }

    def add_gaussians(self, new_gaussians: Dict[str, torch.Tensor]):
        """
        Adds newly initialized Gaussians to current scene representation,
        preventing redundant initialization on already occupied voxels (GVD Paper Sec. III-B.1).
        """
        if len(new_gaussians['xyz']) == 0:
            return

        added_xyz = new_gaussians['xyz']
        added_rgb = new_gaussians['rgb']
        added_scale = new_gaussians['scale']
        added_morton = new_gaussians['morton']

        if len(self.gaussians['morton']) > 0 and len(added_morton) > 0:
            unoccupied = ~torch.isin(added_morton, self.gaussians['morton'])
            added_xyz = added_xyz[unoccupied]
            added_rgb = added_rgb[unoccupied]
            added_scale = added_scale[unoccupied]
            added_morton = added_morton[unoccupied]

        if len(added_xyz) > 0:
            self.gaussians['xyz'] = torch.cat([self.gaussians['xyz'], added_xyz], dim=0)
            self.gaussians['rgb'] = torch.cat([self.gaussians['rgb'], added_rgb], dim=0)
            self.gaussians['scale'] = torch.cat([self.gaussians['scale'], added_scale], dim=0)
            self.gaussians['morton'] = torch.cat([self.gaussians['morton'], added_morton], dim=0)

    def process_frame(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        intrinsic: torch.Tensor,
        pose: torch.Tensor,
        rendered_rgb: torch.Tensor,
        rendered_depth: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Processes incoming RGB-D frame at timestamp t:
        1. VDC Morton-code raycast pruning of deleted objects & floaters against prior TSDF map
        2. TSDF frustum depth integration
        3. AVD & GVD variation detection + surface-normal guided Gaussian initialization
        """
        # Step 1: Morton raycast pruning (evaluates ray free-space against prior TSDF map)
        prune_mask = self.vdc.prune_gaussians_via_morton(
            depth_obs=depth,
            intrinsic=intrinsic,
            pose=pose,
            tsdf_map=self.tsdf_map,
            gaussian_morton_codes=self.gaussians['morton']
        )

        if len(prune_mask) > 0 and torch.any(prune_mask):
            keep_mask = ~prune_mask
            self.gaussians['xyz'] = self.gaussians['xyz'][keep_mask]
            self.gaussians['rgb'] = self.gaussians['rgb'][keep_mask]
            self.gaussians['scale'] = self.gaussians['scale'][keep_mask]
            self.gaussians['morton'] = self.gaussians['morton'][keep_mask]

        # Step 2: Integrate TSDF depth frame
        self.tsdf_map.integrate_depth_frame(depth, intrinsic, pose)

        # Step 3: AVD & GVD initialization using updated TSDF geometry
        new_gaussians = self.vdc.detect_and_initialize_gaussians(
            rgb_obs=rgb,
            depth_obs=depth,
            rendered_rgb=rendered_rgb,
            rendered_depth=rendered_depth,
            intrinsic=intrinsic,
            pose=pose,
            tsdf_map=self.tsdf_map
        )
        self.add_gaussians(new_gaussians)

        return {
            'num_gaussians': len(self.gaussians['xyz']),
            'num_pruned': torch.sum(prune_mask).item() if len(prune_mask) > 0 else 0,
            'num_added': len(new_gaussians['xyz'])
        }

    def estimate_object_se3_motion(
        self,
        object_mask_before: torch.Tensor,
        object_mask_after: torch.Tensor,
        gt_rgb_after: torch.Tensor,
        gt_depth_after: torch.Tensor,
        intrinsic: torch.Tensor,
        pose: torch.Tensor
    ) -> torch.Tensor:
        """
        Executes RecurGS Lie algebra SE(3) pose refinement module:
        1. Coarse ICP alignment over object point clusters
        2. Lie algebra se(3) optimization
        3. Returns T_fine in SE(3)
        """
        # Segment object Gaussians
        obj_gaussians_before = {
            'xyz': self.gaussians['xyz'][object_mask_before],
            'rgb': self.gaussians['rgb'][object_mask_before],
            'scale': self.gaussians['scale'][object_mask_before]
        }
        
        obj_xyz_after = self.gaussians['xyz'][object_mask_after]

        # Coarse ICP
        T_coarse = icp_coarse_alignment(obj_gaussians_before['xyz'], obj_xyz_after)

        # Fine Lie algebra optimization with 3D geometric + photometric alignment
        T_fine = self.se3_aligner.optimize_se3_pose(
            object_gaussians=obj_gaussians_before,
            target_gaussians={'xyz': obj_xyz_after},
            gt_rgb=gt_rgb_after,
            gt_depth=gt_depth_after,
            intrinsic=intrinsic,
            camera_pose=pose,
            initial_T_coarse=T_coarse,
            num_iterations=100
        )

        return T_fine

    def extract_solid_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts solid surface mesh from TSDF grid for PyBullet/simulation engine.
        """
        return self.tsdf_map.extract_mesh(level=0.0)
