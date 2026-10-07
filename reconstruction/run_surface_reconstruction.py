"""
Phase 4C: Poisson Surface Reconstruction
========================================
Converts a dense point cloud into a solid 3D triangle mesh.
"""
import open3d as o3d
import numpy as np
from pathlib import Path

def reconstruct_surface():
    script_dir = Path(__file__).resolve().parent
    dense_ply = (script_dir / "../outputs/dense/fused.ply").resolve()
    output_mesh = (script_dir / "../outputs/export/scene_mesh.ply").resolve()
    
    if not dense_ply.exists():
        print(f"Error: Dense point cloud not found at {dense_ply}")
        return
        
    print("--- STEP 1: Loading Dense Point Cloud ---")
    pcd = o3d.io.read_point_cloud(str(dense_ply))
    print(f"Loaded {len(pcd.points):,} points.")
    
    # 1. Estimate Normals
    print("\n--- STEP 2: Estimating Surface Normals ---")
    # Normals are perpendicular vectors. Poisson needs them to determine the "inside" vs "outside" of the object.
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
    # Try to align normals consistently
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # 2. Poisson Surface Reconstruction
    print("\n--- STEP 3: Poisson Surface Reconstruction ---")
    print("Wrapping a solid mesh 'skin' around the point cloud (this may take a moment)...")
    # depth defines the resolution of the mesh. 9 is high-detail.
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    print(f"Generated initial mesh with {len(mesh.vertices):,} vertices.")
    
    # 3. Clean up artifacts
    print("\n--- STEP 4: Cleaning up low-density artifacts ---")
    # Poisson mathematically creates a giant "bubble" that stretches to infinity.
    # We remove vertices that have low point density (i.e., fake extrapolated surfaces).
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.10) # Remove the lowest 10% density vertices
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Crop strictly to the bounding box of the original point cloud just in case
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    
    print(f"Cleaned mesh has {len(mesh.vertices):,} vertices and {len(mesh.triangles):,} triangles.")
    
    # 4. Export
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_mesh), mesh)
    print(f"\n[SUCCESS] Solid mesh exported to: {output_mesh}")
    
    # 5. Visualize
    print("\nOpening 3D Viewer for the solid mesh...")
    mesh.compute_vertex_normals()
    
    # Optional styling for visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Poisson Surface Mesh", width=1280, height=720)
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.15])
    opt.mesh_show_wireframe = False
    opt.mesh_show_back_face = True
    
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    reconstruct_surface()
