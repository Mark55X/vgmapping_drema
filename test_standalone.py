import torch
import numpy as np
import sys
import os

# Ensure package path is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgmapping_drema.tsdf import TSDFVoxelMap, interleave_bits_3d, deinterleave_bits_3d
from vgmapping_drema.vdc import VariationAwareDensityController, compute_ssim_map
from vgmapping_drema.recurgs_se3 import exp_se3, exp_so3, icp_coarse_alignment, RecurGSLieAlgebraAligner
from vgmapping_drema.pipeline import NativeVGMappingRecurGSPipeline

def test_morton_encoding():
    print("--- Test 1: Morton Code Encoding/Decoding ---")
    x = torch.tensor([1, 10, 255, 512, 1023], dtype=torch.int64)
    y = torch.tensor([2, 20, 128, 256, 512], dtype=torch.int64)
    z = torch.tensor([3, 30, 64, 128, 256], dtype=torch.int64)

    morton = interleave_bits_3d(x, y, z)
    x_rec, y_rec, z_rec = deinterleave_bits_3d(morton)

    assert torch.equal(x, x_rec), f"X mismatch: {x} vs {x_rec}"
    assert torch.equal(y, y_rec), f"Y mismatch: {y} vs {y_rec}"
    assert torch.equal(z, z_rec), f"Z mismatch: {z} vs {z_rec}"
    print("✓ Morton code 3D interleave/deinterleave roundtrip PASSED.")

def test_tsdf_and_mesh():
    print("\n--- Test 2: TSDF Integration & Mesh Extraction ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tsdf = TSDFVoxelMap(voxel_size=0.01, grid_dim=(64, 64, 64), origin=(-0.32, -0.32, -0.32), device=device)

    # Create synthetic camera intrinsic & pose
    K = torch.tensor([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], device=device)
    pose = torch.eye(4, device=device)
    pose[2, 3] = -0.5 # Camera at z = -0.5, looking towards z = 0.0

    depth = torch.ones((1, 64, 64), dtype=torch.float32, device=device) * 0.5

    tsdf.integrate_depth_frame(depth, K, pose)

    # Verify query
    query_pts = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    f_val, w_val = tsdf.query_tsdf_and_weight(query_pts)
    assert w_val.item() > 0, "TSDF integration weight should be > 0"
    print(f"✓ TSDF depth integration PASSED. Sample F: {f_val.item():.4f}, W: {w_val.item():.4f}")

    verts, faces = tsdf.extract_mesh()
    print(f"✓ Marching Cubes Mesh extracted: {len(verts)} vertices, {len(faces)} faces.")

def test_recurgs_se3_alignment():
    print("\n--- Test 3: RecurGS SE(3) Lie Algebra Pose Alignment ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Synthetic object point cloud
    num_pts = 100
    src_pts = torch.randn((num_pts, 3), device=device) * 0.1

    # Ground truth transformation T_gt in SE(3)
    gt_xi = torch.tensor([0.05, -0.05, 0.1, 0.02, -0.01, 0.03], device=device) # (w, v)
    T_gt = exp_se3(gt_xi)

    tgt_pts = (T_gt[:3, :3] @ src_pts.T + T_gt[:3, 3:4]).T

    # Coarse ICP
    T_coarse = icp_coarse_alignment(src_pts, tgt_pts, max_iters=30)
    
    trans_err_coarse = torch.norm(T_coarse[:3, 3] - T_gt[:3, 3]).item()
    print(f"✓ Coarse ICP translation error: {trans_err_coarse:.4f} meters")

    # Fine SE(3) Lie algebra optimization
    aligner = RecurGSLieAlgebraAligner(device=device)

    gt_rgb = torch.rand((3, 64, 64), device=device)
    gt_depth = torch.ones((1, 64, 64), device=device) * 0.5
    K = torch.tensor([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], device=device)
    pose = torch.eye(4, device=device)

    obj_gaussians = {
        'xyz': src_pts,
        'rgb': torch.rand((num_pts, 3), device=device),
        'scale': torch.ones((num_pts, 3), device=device) * 0.01
    }

    T_fine = aligner.optimize_se3_pose(
        object_gaussians=obj_gaussians,
        gt_rgb=gt_rgb,
        gt_depth=gt_depth,
        intrinsic=K,
        camera_pose=pose,
        initial_T_coarse=T_coarse,
        num_iterations=50
    )

    print(f"✓ RecurGS SE(3) Lie algebra pose alignment completed. Estimated T_fine shape: {T_fine.shape}")

def test_full_pipeline():
    print("\n--- Test 4: Full Standalone Native Pipeline ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = NativeVGMappingRecurGSPipeline(grid_dim=(64, 64, 64), device=device)

    rgb = torch.rand((3, 64, 64), device=device)
    depth = torch.ones((1, 64, 64), device=device) * 0.5
    rendered_rgb = torch.rand((3, 64, 64), device=device)
    rendered_depth = torch.ones((1, 64, 64), device=device) * 0.5

    K = torch.tensor([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], device=device)
    pose = torch.eye(4, device=device)

    res = pipeline.process_frame(rgb, depth, K, pose, rendered_rgb, rendered_depth)
    print(f"✓ Frame processed successfully: {res}")

if __name__ == "__main__":
    print("==================================================================")
    print(" Running Standalone Native VG-Mapping & RecurGS SE(3) Test Suite ")
    print("==================================================================")
    test_morton_encoding()
    test_tsdf_and_mesh()
    test_recurgs_se3_alignment()
    test_full_pipeline()
    print("\n✓ ALL TESTS PASSED SUCCESSFULLY!")
