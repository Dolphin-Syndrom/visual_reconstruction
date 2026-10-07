"""
Diagnose & View 3D Reconstruction
===================================
Checks all reconstruction outputs, reports quality metrics,
and visualizes with proper camera framing + dark background.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Resolve all paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DENSE_PLY   = PROJECT_DIR / "outputs" / "dense" / "fused.ply"
SPARSE_PLY  = PROJECT_DIR / "outputs" / "export" / "scene_scaled.ply"
DENSE_DIR   = PROJECT_DIR / "outputs" / "dense"
SPARSE_DIR  = PROJECT_DIR / "outputs" / "sparse"


def check_ply_header(ply_path):
    """Read PLY header and return vertex count."""
    if not ply_path.exists():
        return -1
    with open(ply_path, 'rb') as f:
        header = ""
        for line in f:
            decoded = line.decode('utf-8', errors='ignore')
            header += decoded
            if "end_header" in decoded:
                break
    import re
    match = re.search(r'element vertex (\d+)', header)
    return int(match.group(1)) if match else 0


def diagnose():
    """Run full diagnostic on reconstruction outputs."""
    print("=" * 60)
    print("  3D RECONSTRUCTION DIAGNOSTIC REPORT")
    print("=" * 60)

    # 1. Check sparse reconstruction
    print("\n[1] SPARSE RECONSTRUCTION (SfM)")
    sparse_bin = SPARSE_DIR / "0"
    if not sparse_bin.exists():
        sparse_bin = SPARSE_DIR
    
    cameras_bin = sparse_bin / "cameras.bin"
    images_bin  = sparse_bin / "images.bin"
    points_bin  = sparse_bin / "points3D.bin"

    if cameras_bin.exists() and images_bin.exists() and points_bin.exists():
        print(f"    ✓ Sparse model found at: {sparse_bin}")
        print(f"    ✓ cameras.bin : {cameras_bin.stat().st_size / 1024:.1f} KB")
        print(f"    ✓ images.bin  : {images_bin.stat().st_size / 1024:.1f} KB")
        print(f"    ✓ points3D.bin: {points_bin.stat().st_size / 1024:.1f} KB")
    else:
        print(f"    ✗ Sparse model NOT found at {sparse_bin}")
        print(f"      Run: python3 reconstruction/run_reconstruction.py")
        return None

    # 2. Check dense reconstruction workspace
    print("\n[2] DENSE RECONSTRUCTION (MVS)")
    
    # Check undistorted images
    undistorted_dir = DENSE_DIR / "images"
    if undistorted_dir.exists():
        undist_images = list(undistorted_dir.glob("*"))
        print(f"    ✓ Undistorted images: {len(undist_images)} files")
    else:
        print(f"    ✗ No undistorted images directory at {undistorted_dir}")

    # Check depth maps
    stereo_dir = DENSE_DIR / "stereo" / "depth_maps"
    if stereo_dir.exists():
        depth_maps = list(stereo_dir.glob("*.bin")) + list(stereo_dir.glob("*.geometric.bin"))
        photometric = [f for f in stereo_dir.iterdir() if "photometric" in f.name]
        geometric   = [f for f in stereo_dir.iterdir() if "geometric" in f.name]
        print(f"    ✓ Depth map files: {len(list(stereo_dir.iterdir()))} total")
        print(f"      - Photometric: {len(photometric)}")
        print(f"      - Geometric:   {len(geometric)}")
        
        # Check if depth maps are non-empty
        if depth_maps:
            sizes = [f.stat().st_size for f in depth_maps[:5]]
            avg_size = sum(sizes) / len(sizes)
            print(f"      - Avg depth map size: {avg_size / 1024:.1f} KB")
            if avg_size < 100:
                print(f"    ⚠ WARNING: Depth maps are suspiciously small!")
                print(f"      This means patch_match_stereo may not have run correctly.")
    else:
        print(f"    ✗ No depth maps found at {stereo_dir}")
        print(f"      Patch-Match Stereo did not run or failed silently.")

    # Check fused.ply
    print("\n[3] FUSED POINT CLOUD")
    if DENSE_PLY.exists():
        file_size = DENSE_PLY.stat().st_size
        vertex_count = check_ply_header(DENSE_PLY)
        print(f"    File: {DENSE_PLY}")
        print(f"    Size: {file_size / (1024*1024):.2f} MB")
        print(f"    Vertices: {vertex_count:,}")
        
        if vertex_count == 0:
            print(f"    ✗ EMPTY! The fused.ply has 0 points.")
            print(f"      Stereo fusion produced no output.")
            print(f"      This likely means depth maps were empty.")
        elif vertex_count < 100:
            print(f"    ⚠ Very few points. Reconstruction quality is poor.")
        else:
            print(f"    ✓ Dense cloud looks good!")
            return "dense"
    else:
        print(f"    ✗ fused.ply NOT found at {DENSE_PLY}")

    # 4. Check sparse PLY export
    print("\n[4] SPARSE PLY EXPORT")
    if SPARSE_PLY.exists():
        vertex_count = check_ply_header(SPARSE_PLY)
        print(f"    File: {SPARSE_PLY}")
        print(f"    Vertices: {vertex_count:,}")
        if vertex_count > 0:
            return "sparse_export"
    else:
        print(f"    ✗ Sparse PLY not exported yet.")
        print(f"      Run: python3 geometry/scale_scene.py")
        print(f"      Then: python3 visualization/export_scene.py")

    # 5. Fall back to loading sparse binary directly
    print("\n[5] FALLBACK: Loading sparse model via pycolmap")
    return "sparse_binary"


def visualize(cloud_type):
    """Visualize the best available point cloud."""
    try:
        import open3d as o3d
    except ImportError:
        print("\n[ERROR] open3d not installed. Install with: pip install open3d")
        return

    pcd = None

    if cloud_type == "dense":
        print(f"\nLoading dense cloud: {DENSE_PLY}")
        pcd = o3d.io.read_point_cloud(str(DENSE_PLY))

    elif cloud_type == "sparse_export":
        print(f"\nLoading sparse exported cloud: {SPARSE_PLY}")
        pcd = o3d.io.read_point_cloud(str(SPARSE_PLY))

    elif cloud_type == "sparse_binary":
        print("\nLoading sparse model from COLMAP binary files...")
        try:
            import pycolmap
            sparse_bin = SPARSE_DIR / "0"
            if not sparse_bin.exists():
                sparse_bin = SPARSE_DIR
            reconstruction = pycolmap.Reconstruction(sparse_bin)
            
            points = []
            colors = []
            for pt in reconstruction.points3D.values():
                points.append(pt.xyz)
                colors.append(np.array(pt.color) / 255.0)
            
            if len(points) == 0:
                print("No 3D points in sparse reconstruction!")
                return
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
            print(f"Loaded {len(points):,} sparse points from COLMAP binary.")
        except Exception as e:
            print(f"Failed to load sparse binary: {e}")
            return

    if pcd is None or len(pcd.points) == 0:
        print("\n[ERROR] No points to display!")
        return

    num_points = len(pcd.points)
    print(f"\n{'=' * 60}")
    print(f"  Displaying {num_points:,} points")
    print(f"{'=' * 60}")

    # If no colors or all white, assign a visible color
    colors_arr = np.asarray(pcd.colors)
    if len(colors_arr) == 0 or np.all(colors_arr > 0.95):
        print("  ⚠ Points have no color or are all white.")
        print("  → Assigning height-based color gradient for visibility.")
        pts = np.asarray(pcd.points)
        z = pts[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        gradient = np.zeros((len(z_norm), 3))
        gradient[:, 0] = z_norm          # Red increases with height
        gradient[:, 1] = 0.3             # Constant green tint
        gradient[:, 2] = 1.0 - z_norm    # Blue decreases with height
        pcd.colors = o3d.utility.Vector3dVector(gradient)

    # Compute point size based on density
    point_size = 1.0 if num_points > 50000 else (2.0 if num_points > 5000 else 4.0)

    print(f"\n  Controls:")
    print(f"    Left-drag   = Rotate")
    print(f"    Right-drag  = Pan")
    print(f"    Scroll      = Zoom")
    print(f"    +/-         = Point size")
    print(f"    R           = Reset view")
    print(f"    Q / Esc     = Close\n")

    # Use draw_geometries with proper settings
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"3D Reconstruction ({num_points:,} points)", width=1280, height=720)
    vis.add_geometry(pcd)

    # Dark background + proper rendering
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.1])  # Dark navy background
    opt.point_size = point_size
    opt.show_coordinate_frame = True

    # Auto-frame the camera to fit the point cloud
    vis.reset_view_point(True)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    result = diagnose()
    
    if result is None:
        print("\n[FATAL] No reconstruction data available.")
        sys.exit(1)
    
    print(f"\n→ Best available cloud: {result}")
    visualize(result)
