"""
Object Measurement from Dense Point Cloud
==========================================
Loads the metric-scaled fused.ply dense point cloud and computes:
  - Bounding box dimensions (Width x Height x Depth) in centimeters
  - Longest axis (corresponds to the real-world "length")
  - Volume estimate from the bounding box

The pipeline must have already run:
  1. run_reconstruction.py     (sparse SfM)
  2. geometry/scale_scene.py   (metric scaling)
  3. run_dense_reconstruction.py (dense MVS — produces fused.ply)

Usage:
    cd /home/intern2/visual_reconstruction
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 geometry/measure_object.py

NOTE: The fused.ply from run_dense_reconstruction.py IS already in metric cm.
The scale factor was applied to the sparse reconstruction before dense reconstruction
was seeded from it, so all depth estimates inherit the metric calibration.
"""

import open3d as o3d
import numpy as np
from pathlib import Path


def measure_object():
    script_dir = Path(__file__).resolve().parent
    dense_ply = (script_dir / "../outputs/dense/fused.ply").resolve()
    mesh_ply  = (script_dir / "../outputs/export/scene_mesh.ply").resolve()

    # -------------------------------------------------------------------------
    # 1. Load point cloud (prefer mesh if available, fall back to dense cloud)
    # -------------------------------------------------------------------------
    if mesh_ply.exists():
        print(f"Loading Poisson mesh from: {mesh_ply}")
        geo = o3d.io.read_triangle_mesh(str(mesh_ply))
        pcd = geo.sample_points_uniformly(number_of_points=200_000)
        source_label = "Poisson Mesh"
    elif dense_ply.exists():
        print(f"Loading dense point cloud from: {dense_ply}")
        pcd = o3d.io.read_point_cloud(str(dense_ply))
        geo = pcd
        source_label = "Dense Point Cloud"
    else:
        print("[ERROR] No reconstruction output found.")
        print("  Run run_dense_reconstruction.py first to produce outputs/dense/fused.ply")
        return

    print(f"Loaded {len(np.asarray(pcd.points)):,} points ({source_label})")

    # -------------------------------------------------------------------------
    # 2. Statistical Outlier Removal
    # Remove isolated noise points that inflate the bounding box.
    # -------------------------------------------------------------------------
    print("\n--- Removing statistical outliers ---")
    pcd_clean, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )
    removed = len(np.asarray(pcd.points)) - len(np.asarray(pcd_clean.points))
    print(f"Removed {removed:,} outlier points. Retained {len(np.asarray(pcd_clean.points)):,} points.")

    # -------------------------------------------------------------------------
    # 3. Compute Axis-Aligned Bounding Box (AABB)
    # -------------------------------------------------------------------------
    aabb = pcd_clean.get_axis_aligned_bounding_box()
    aabb.color = (0.0, 1.0, 0.0)  # Green box

    min_bound = np.asarray(aabb.min_bound)
    max_bound = np.asarray(aabb.max_bound)
    extents   = max_bound - min_bound   # [dx, dy, dz] in cm

    # -------------------------------------------------------------------------
    # 4. Compute Oriented Bounding Box (OBB) — tighter fit, better dimensions
    # -------------------------------------------------------------------------
    obb = pcd_clean.get_oriented_bounding_box()
    obb.color = (1.0, 0.5, 0.0)  # Orange box

    obb_extents = np.sort(obb.extent)[::-1]  # Sorted: longest → shortest axis

    # -------------------------------------------------------------------------
    # 5. Print the Measurement Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  OBJECT MEASUREMENT REPORT — CAT DOLL")
    print("=" * 60)
    print(f"\n  Source         : {source_label}")
    print(f"  Units          : centimeters (cm)  [metric-scaled from SfM]\n")

    print("  ── Axis-Aligned Bounding Box (AABB) ──────────────────")
    print(f"  Width  (X-axis): {extents[0]:.2f} cm")
    print(f"  Height (Y-axis): {extents[1]:.2f} cm")
    print(f"  Depth  (Z-axis): {extents[2]:.2f} cm")
    print(f"  Diagonal       : {np.linalg.norm(extents):.2f} cm")
    print(f"  Box Volume     : {extents[0]*extents[1]*extents[2]:.2f} cm³\n")

    print("  ── Oriented Bounding Box (OBB) — tighter fit ─────────")
    print(f"  Longest dim    : {obb_extents[0]:.2f} cm")
    print(f"  Middle  dim    : {obb_extents[1]:.2f} cm")
    print(f"  Shortest dim   : {obb_extents[2]:.2f} cm")
    print(f"  Box Volume     : {np.prod(obb_extents):.2f} cm³")
    print("=" * 60)
    print("\n  NOTE: OBB dimensions are more accurate for non-axis-aligned")
    print("  objects (e.g., a cat sitting at an angle).\n")

    # -------------------------------------------------------------------------
    # 6. Create measurement annotation lines for visualization
    # -------------------------------------------------------------------------
    # Draw thin lines along each axis of the AABB to visualize W × H × D

    def make_line(p1, p2, color):
        pts = np.array([p1, p2])
        lines = [[0, 1]]
        colors = [color]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines  = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(colors)
        return ls

    cx, cy, cz = min_bound
    ex, ey, ez = max_bound

    # Three measurement bars in X, Y, Z
    line_x = make_line([cx, cy - 0.5, cz - 0.5], [ex, cy - 0.5, cz - 0.5], [1, 0, 0])  # Red  = Width
    line_y = make_line([cx - 0.5, cy, cz - 0.5], [cx - 0.5, ey, cz - 0.5], [0, 1, 0])  # Green = Height
    line_z = make_line([cx - 0.5, cy - 0.5, cz], [cx - 0.5, cy - 0.5, ez], [0, 0.5, 1])  # Blue = Depth

    # -------------------------------------------------------------------------
    # 7. Visualise
    # -------------------------------------------------------------------------
    print("Opening 3D Viewer…")
    print("  RED   bar = Width  (X)")
    print("  GREEN bar = Height (Y)")
    print("  BLUE  bar = Depth  (Z)")
    print("  GREEN box = AABB   (axis-aligned)")
    print("  ORANGE box= OBB    (oriented bounding box)\n")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Object Measurement — Cat Doll", width=1440, height=900)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.08])
    opt.point_size = 2.0

    vis.add_geometry(pcd_clean)
    vis.add_geometry(aabb)
    vis.add_geometry(obb)
    vis.add_geometry(line_x)
    vis.add_geometry(line_y)
    vis.add_geometry(line_z)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    measure_object()
