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
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.tsdf_map = TSDFVoxelMap(voxel_size=voxel_size, grid_dim=grid_dim, origin=origin, device=device)
        self.vdc = VariationAwareDensityController(device=device)
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
        Adds newly initialized Gaussians to current scene representation.
        """
        if len(new_gaussians['xyz']) == 0:
            return

        self.gaussians['xyz'] = torch.cat([self.gaussians['xyz'], new_gaussians['xyz']], dim=0)
        self.gaussians['rgb'] = torch.cat([self.gaussians['rgb'], new_gaussians['rgb']], dim=0)
        self.gaussians['scale'] = torch.cat([self.gaussians['scale'], new_gaussians['scale']], dim=0)
        self.gaussians['morton'] = torch.cat([self.gaussians['morton'], new_gaussians['morton']], dim=0)

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
        1. TSDF frustum depth integration
        2. VDC Morton-code raycast pruning of deleted objects & floaters
        3. AVD & GVD variation detection + surface-normal guided Gaussian initialization
        """
        # Step 1: Integrate TSDF
        self.tsdf_map.integrate_depth_frame(depth, intrinsic, pose)

        # Step 2: Morton raycast pruning
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

        # Step 3: AVD & GVD initialization
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

        # Fine Lie algebra optimization
        T_fine = self.se3_aligner.optimize_se3_pose(
            object_gaussians=obj_gaussians_before,
            gt_rgb=gt_rgb_after,
            gt_depth=gt_depth_after,
            intrinsic=intrinsic,
            camera_pose=pose,
            initial_T_coarse=T_coarse,
            num_iterations=200
        )

        return T_fine

    def extract_solid_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts solid surface mesh from TSDF grid for PyBullet/simulation engine.
        """
        return self.tsdf_map.extract_mesh(level=0.0)
