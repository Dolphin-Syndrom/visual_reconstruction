"""
Phase 4B: Dense Reconstruction (Multi-View Stereo)
===================================================
This script runs after run_reconstruction.py (SfM).
It takes the sparse camera positions as input and runs:
  1. Patch-Match Stereo  - computes a dense depth map for every image
  2. Stereo Fusion       - merges all depth maps into a single dense point cloud

Result: outputs/dense/fused.ply with millions of colored points (vs 4K sparse).

NOTE: Without a CUDA GPU, this uses CPU which is very slow (~4-8 hours for 38 images).
      The script will print progress every few minutes so you can track it.
"""

import pycolmap
from pathlib import Path

def run_dense():
    print("=" * 60)
    print("Phase 4B: Dense Multi-View Stereo Reconstruction")
    print("=" * 60)
    
    # 1. Paths
    script_dir = Path(__file__).resolve().parent
    image_dir = script_dir / "../data/scene"
    sparse_dir = script_dir / "../outputs/sparse/0" # COLMAP saves the best model in the '0' subfolder
    if not sparse_dir.exists():
        sparse_dir = script_dir / "../outputs/sparse"
    
    dense_dir = script_dir / "../outputs/dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    
    if not sparse_dir.exists():
        print("ERROR: Sparse reconstruction not found. Run run_reconstruction.py first!")
        return
    
    # 2. Undistort images
    # Before dense matching, we must undistort (remove lens distortion from) every image.
    # This creates a mathematically ideal "pinhole camera" image from every photo.
    print("\n--- STEP 1: Image Undistortion ---")
    print("Removing lens distortion from all 38 images...")
    print("(This creates perfect pinhole images for stereo matching)")
    
    pycolmap.undistort_images(
        output_path=dense_dir,
        input_path=sparse_dir,
        image_path=image_dir,
    )
    print("Undistortion complete!")
    
    # 3. Patch-Match Stereo
    print("\n--- STEP 2: Patch-Match Stereo ---")
    print("Calculating depth at every pixel for all 38 images...")
    
    import subprocess, shutil
    colmap_bin = shutil.which("colmap")
    
    try:
        pycolmap.patch_match_stereo(
            workspace_path=dense_dir,
        )
        print("Patch-Match Stereo complete via pycolmap!")
    except Exception as e:
        print(f"\n[NOTICE] pycolmap Python wheel reported: {e}")
        if colmap_bin:
            print(f"Found system COLMAP CLI binary at '{colmap_bin}'. Attempting CUDA Dense Reconstruction via system COLMAP...")
            cmd = [colmap_bin, "patch_match_stereo", "--workspace_path", str(dense_dir)]
            subprocess.run(cmd, check=True)
            print("Patch-Match Stereo complete via system COLMAP CLI!")
        else:
            print("\n[ERROR] Neither pycolmap wheel nor system 'colmap' CLI binary has CUDA enabled.")
            print("To fix this on Linux:")
            print("  1. Check if NVIDIA driver & CUDA are active: nvidia-smi")
            print("  2. If system colmap is installed, check: colmap -h")
            print("  3. Or run dense stereo using system colmap binary with CUDA.")
            return

    # 4. Stereo Fusion
    print("\n--- STEP 3: Stereo Fusion ---")
    print("Merging all depth maps into a single dense point cloud...")
    
    fused_ply_path = dense_dir / "fused.ply"
    try:
        pycolmap.stereo_fusion(
            output_path=fused_ply_path,
            workspace_path=dense_dir,
        )
        print("Stereo Fusion complete via pycolmap!")
    except Exception as e:
        if colmap_bin:
            print(f"Attempting Stereo Fusion via system COLMAP CLI...")
            cmd = [
                colmap_bin, "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--output_path", str(fused_ply_path)
            ]
            subprocess.run(cmd, check=True)
            print("Stereo Fusion complete via system COLMAP CLI!")
        else:
            print(f"[ERROR] Stereo fusion failed: {e}")
            return
    
    # Count points in the final cloud
    with open(dense_dir / "fused.ply", 'rb') as f:
        header = ""
        for line in f:
            header += line.decode('utf-8', errors='ignore')
            if "end_header" in header:
                break
    
    import re
    match = re.search(r'element vertex (\d+)', header)
    num_points = int(match.group(1)) if match else "unknown"
    
    print(f"\n{'=' * 60}")
    print(f"[SUCCESS] Dense reconstruction complete!")
    print(f"Dense point cloud saved to: {dense_dir}/fused.ply")
    print(f"Total dense points: {num_points:,} (vs 4,164 sparse)")
    print(f"\nView with:")
    print(f"  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 visualization/view_dense.py")

if __name__ == "__main__":
    run_dense()
