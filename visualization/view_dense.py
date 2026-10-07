import open3d as o3d
from pathlib import Path

def view_dense():
    print("--- Dense 3D Viewer ---")
    script_dir = Path(__file__).resolve().parent
    ply_path = (script_dir / "../outputs/dense/fused.ply").resolve()
    
    if not ply_path.exists():
        print(f"Dense point cloud not found at {ply_path}")
        print("Run run_dense_reconstruction.py first!")
        return
    
    print(f"Loading Dense Point Cloud: {ply_path.name}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    print(f"Successfully loaded {len(pcd.points):,} points!")
    print("\nA 3D window will now open. You can:")
    print("- Left Click & Drag: Rotate the model")
    print("- Right Click & Drag: Pan the model")
    print("- Scroll Wheel: Zoom in and out")
    print("\nClose the window to exit.")
    
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    view_dense()
