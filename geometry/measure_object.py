"""
Object Measurement from Dense Point Cloud / Mesh
=================================================
Isolates the target object (Cat Doll) from the floor plane and background,
then computes accurate physical dimensions:
  - Bounding box dimensions (Length x Width x Height) in centimeters
  - Oriented Bounding Box (OBB) for tightest orientation fit
  - Axis-Aligned Bounding Box (AABB)
  - Estimated physical volume

Usage on Workstation:
    cd /home/intern2/visual_reconstruction
    # Automatic floor removal and object segmentation (Default):
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 geometry/measure_object.py

    # Interactive manual crop mode (if you want to hand-pick the object):
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 geometry/measure_object.py --interactive
"""

import sys
import argparse
from pathlib import Path
import open3d as o3d
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Measure reconstructed object dimensions.")
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Open interactive visual editor to manually crop around the object."
    )
    parser.add_argument(
        "--plane-dist", type=float, default=1.0,
        help="RANSAC distance threshold in cm to identify floor plane (default: 1.0 cm)."
    )
    parser.add_argument(
        "--min-height", type=float, default=0.5,
        help="Minimum height above floor in cm to consider object points (default: 0.5 cm)."
    )
    parser.add_argument(
        "--max-height", type=float, default=35.0,
        help="Maximum height above floor in cm (default: 35.0 cm)."
    )
    parser.add_argument(
        "--dbscan-eps", type=float, default=2.0,
        help="DBSCAN cluster distance threshold in cm (default: 2.0 cm)."
    )
    return parser.parse_args()


def load_reconstruction():
    script_dir = Path(__file__).resolve().parent
    dense_ply = (script_dir / "../outputs/dense/fused.ply").resolve()
    mesh_ply  = (script_dir / "../outputs/export/scene_mesh.ply").resolve()
    cropped_ply = (script_dir / "../outputs/export/cat_doll_cropped.ply").resolve()

    # If already cropped previously, allow using it directly
    if cropped_ply.exists():
        print(f"[INFO] Found previous cropped object: {cropped_ply}")
        use_cached = input("  Do you want to use the previously cropped object? [Y/n]: ").strip().lower()
        if use_cached in ("", "y", "yes"):
            pcd = o3d.io.read_point_cloud(str(cropped_ply))
            return pcd, None, "Previous Cropped Object"

    # Prefer dense point cloud for rich RGB textures and point segmentation
    if dense_ply.exists():
        print(f"Loading dense point cloud from: {dense_ply}")
        pcd = o3d.io.read_point_cloud(str(dense_ply))
        source_label = "Dense Point Cloud"
    elif mesh_ply.exists():
        print(f"Loading Poisson mesh from: {mesh_ply}")
        geo = o3d.io.read_triangle_mesh(str(mesh_ply))
        pcd = geo.sample_points_uniformly(number_of_points=300_000)
        source_label = "Poisson Mesh Samples"
    else:
        print("[ERROR] No reconstruction output found.")
        print("  Expected either outputs/dense/fused.ply or outputs/export/scene_mesh.ply")
        sys.exit(1)

    print(f"Loaded {len(np.asarray(pcd.points)):,} points ({source_label})")
    return pcd, cropped_ply, source_label


def interactive_crop(pcd, save_path=None):
    print("\n" + "=" * 60)
    print("  INTERACTIVE CROPPING MODE")
    print("=" * 60)
    print("  Instructions in Open3D Viewer:")
    print("    1. Rotate view so the cat doll is clearly centered.")
    print("    2. Press [Y] to start polygon selection.")
    print("    3. Click with mouse around the cat doll to draw a contour.")
    print("    4. Press [C] to crop out everything outside the polygon.")
    print("    5. Press [Q] or [Esc] to finish and calculate measurements.\n")

    # Launch visual editing window
    o3d.visualization.draw_geometries_with_editing([pcd])

    # Note: open3d draw_geometries_with_editing allows saving cropped geometry
    print("[INFO] If you saved the cropped geometry or want to proceed with filtered points, continuing...")
    return pcd


def segment_object_from_floor(pcd, plane_dist=1.0, min_height=0.5, max_height=35.0, eps=2.0):
    print("\n--- Step 1: Statistical Outlier Removal ---")
    pcd_clean, inliers = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
    print(f"Retained {len(pcd_clean.points):,} clean points.")

    print("\n--- Step 2: RANSAC Floor Plane Detection ---")
    # Dominant plane in this scene is the floor / tiled surface
    plane_model, floor_inliers = pcd_clean.segment_plane(
        distance_threshold=plane_dist,
        ransac_n=3,
        num_iterations=2000
    )
    [a, b, c, d] = plane_model
    norm = np.sqrt(a**2 + b**2 + c**2)
    print(f"Detected Floor Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

    floor_cloud = pcd_clean.select_by_index(floor_inliers)
    floor_cloud.paint_uniform_color([0.25, 0.25, 0.25])  # Dim grey floor

    pts = np.asarray(pcd_clean.points)
    # Signed orthogonal distance from each point to the floor plane
    signed_dists = (a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / norm

    # Determine which direction points "up" (away from the floor into the cat doll)
    # The cat doll has a cluster of points protruding on one side of the plane
    pos_count = np.sum(signed_dists > plane_dist)
    neg_count = np.sum(signed_dists < -plane_dist)

    sign = 1.0 if pos_count >= neg_count else -1.0
    heights = signed_dists * sign

    # Filter points located between min_height (above floor roughness) and max_height
    object_height_mask = (heights >= min_height) & (heights <= max_height)
    object_indices = np.where(object_height_mask)[0]

    if len(object_indices) == 0:
        print("[WARNING] Could not separate object by height. Using all non-floor points.")
        non_floor_cloud = pcd_clean.select_by_index(floor_inliers, invert=True)
    else:
        non_floor_cloud = pcd_clean.select_by_index(object_indices)

    print(f"Extracted {len(non_floor_cloud.points):,} candidate points above floor.")

    print("\n--- Step 3: DBSCAN Clustering to Isolate Cat Doll ---")
    # Group remaining non-floor points into clusters to discard walls or distant background
    labels = np.array(non_floor_cloud.cluster_dbscan(eps=eps, min_points=40, print_progress=False))
    unique_labels = [l for l in np.unique(labels) if l != -1]

    if len(unique_labels) == 0:
        print("[WARNING] DBSCAN found no distinct clusters. Retaining candidate points.")
        cat_pcd = non_floor_cloud
    else:
        # Find the cluster with the highest point density near the scene center
        scene_center = np.mean(pts, axis=0)
        best_label = None
        best_score = -1

        for l in unique_labels:
            cluster_idx = np.where(labels == l)[0]
            cluster_pts = np.asarray(non_floor_cloud.select_by_index(cluster_idx).points)
            cluster_size = len(cluster_pts)
            # Distance from cluster center to scene center
            dist_to_center = np.linalg.norm(np.mean(cluster_pts, axis=0) - scene_center)
            # Score favors large clusters closer to the capture center
            score = cluster_size / (1.0 + 0.05 * dist_to_center)

            if score > best_score and cluster_size > 500:
                best_score = score
                best_label = l

        if best_label is not None:
            cat_indices = np.where(labels == best_label)[0]
            cat_pcd = non_floor_cloud.select_by_index(cat_indices)
            print(f"Isolated Cat Doll cluster (ID {best_label}) with {len(cat_pcd.points):,} points.")
        else:
            cat_pcd = non_floor_cloud

    # Secondary outlier cleanup on the isolated object
    cat_pcd, _ = cat_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    return cat_pcd, floor_cloud


def make_line(p1, p2, color):
    pts = np.array([p1, p2])
    lines = [[0, 1]]
    colors = [color]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def measure_object():
    args = parse_args()

    # 1. Load data
    pcd, save_path, source_label = load_reconstruction()

    # 2. Segment object or run interactive mode
    if args.interactive:
        cat_pcd = interactive_crop(pcd, save_path)
        floor_cloud = None
    else:
        cat_pcd, floor_cloud = segment_object_from_floor(
            pcd,
            plane_dist=args.plane_dist,
            min_height=args.min_height,
            max_height=args.max_height,
            eps=args.dbscan_eps
        )

    # Save isolated object so user can inspect or reuse
    script_dir = Path(__file__).resolve().parent
    out_cat_path = (script_dir / "../outputs/export/cat_doll_isolated.ply").resolve()
    out_cat_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_cat_path), cat_pcd)
    print(f"\n[SUCCESS] Saved isolated object point cloud to:\n  {out_cat_path}")

    # -------------------------------------------------------------------------
    # 3. Compute Bounding Boxes on ISOLATED OBJECT ONLY
    # -------------------------------------------------------------------------
    # Axis-Aligned Bounding Box (AABB)
    aabb = cat_pcd.get_axis_aligned_bounding_box()
    aabb.color = (0.0, 1.0, 0.2)  # Bright Green

    min_b = np.asarray(aabb.min_bound)
    max_b = np.asarray(aabb.max_bound)
    aabb_dims = max_b - min_b  # [dx, dy, dz] in cm

    # Oriented Bounding Box (OBB) - tightest bounding box aligned to object's principal axes
    obb = cat_pcd.get_oriented_bounding_box()
    obb.color = (1.0, 0.45, 0.0)  # Bright Orange
    obb_extents = np.sort(obb.extent)[::-1]  # [Longest, Middle, Shortest]

    # -------------------------------------------------------------------------
    # 4. Print Measurement Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("       OBJECT MEASUREMENT REPORT — CAT DOLL (ISOLATED)")
    print("=" * 65)
    print(f"  Source Model       : {source_label}")
    print(f"  Object Point Count : {len(cat_pcd.points):,} points")
    print(f"  Units              : centimeters (cm) [Calibrated Metric Scale]\n")

    print("  ── Oriented Bounding Box (OBB — Recommended True Dimensions) ──")
    print(f"  Length (Longest axis): {obb_extents[0]:.2f} cm")
    print(f"  Width  (Middle axis) : {obb_extents[1]:.2f} cm")
    print(f"  Height (Short axis)  : {obb_extents[2]:.2f} cm")
    print(f"  Oriented Box Volume  : {np.prod(obb_extents):.2f} cm³ ({np.prod(obb_extents)/1000.0:.3f} L)")
    print("  -------------------------------------------------------------")
    print("  Note: OBB aligns with the natural body angle (head-to-tail),")
    print("  providing true object dimensions independent of floor tilt.\n")

    print("  ── Axis-Aligned Bounding Box (AABB) ─────────────────────────")
    print(f"  X-Extent : {aabb_dims[0]:.2f} cm")
    print(f"  Y-Extent : {aabb_dims[1]:.2f} cm")
    print(f"  Z-Extent : {aabb_dims[2]:.2f} cm")
    print(f"  Diagonal : {np.linalg.norm(aabb_dims):.2f} cm")
    print(f"  Volume   : {np.prod(aabb_dims):.2f} cm³")
    print("=" * 65 + "\n")

    # -------------------------------------------------------------------------
    # 5. Build Measurement Visual Lines
    # -------------------------------------------------------------------------
    cx, cy, cz = min_b
    ex, ey, ez = max_b
    line_x = make_line([cx, cy - 0.5, cz], [ex, cy - 0.5, cz], [1, 0, 0])      # Red   = X Dimension
    line_y = make_line([cx - 0.5, cy, cz], [cx - 0.5, ey, cz], [0, 0.8, 1])    # Cyan  = Y Dimension
    line_z = make_line([cx - 0.5, cy - 0.5, cz], [cx - 0.5, cy - 0.5, ez], [1, 1, 0]) # Yellow = Z Dimension

    # -------------------------------------------------------------------------
    # 6. Interactive 3D Visualization
    # -------------------------------------------------------------------------
    print("Opening 3D Measurement Viewer...")
    print("  GREEN BOX  = Axis-Aligned Bounds (AABB)")
    print("  ORANGE BOX = Oriented Minimum Bounding Box (OBB)")
    print("  RED LINE   = X-extent")
    print("  CYAN LINE  = Y-extent")
    print("  YELLOW LINE= Z-extent")
    print("  GREY MESH  = Floor context (dimmed)\n")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Cat Doll — Isolated Object Measurement", width=1440, height=900)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.04, 0.04, 0.06])
    opt.point_size = 2.5

    vis.add_geometry(cat_pcd)
    vis.add_geometry(obb)
    vis.add_geometry(aabb)
    vis.add_geometry(line_x)
    vis.add_geometry(line_y)
    vis.add_geometry(line_z)

    if floor_cloud is not None:
        vis.add_geometry(floor_cloud)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    measure_object()
