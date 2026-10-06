import open3d as o3d
from pathlib import Path

def view_cloud():
    print("--- 3D Viewer ---")
    script_dir = Path(__file__).resolve().parent
    ply_path = script_dir / "../outputs/export/scene_scaled.ply"
    if not ply_path.exists():
        ply_path = Path("outputs/export/scene_scaled.ply")
    
    if not ply_path.exists():
        print(f"File not found at: {ply_path.resolve()}")
        return

        
    print(f"Loading Point Cloud: {ply_path.name}")
    
    # Read the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    print(f"Successfully loaded {len(pcd.points)} points!")
    print("\nA 3D window will now open. You can:")
    print("- Left Click & Drag: Rotate the model")
    print("- Right Click & Drag: Pan the model")
    print("- Scroll Wheel: Zoom in and out")
    print("\nClose the window to exit the script.")
    
    # Launch the interactive 3D viewer
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    view_cloud()
