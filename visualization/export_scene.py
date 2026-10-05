import pycolmap
from pathlib import Path

def export_scene_to_simulator():
    print("--- Phase 7: Visualization & Export ---")
    
    # 1. Define paths
    sparse_scaled_dir = Path("../outputs/sparse_scaled")
    output_dir = Path("../outputs/export")
    output_dir.mkdir(exist_ok=True)
    
    ply_path = output_dir / "scene_scaled.ply"
    urdf_path = output_dir / "scene.urdf"
    
    if not sparse_scaled_dir.exists():
        print("Scaled reconstruction not found!")
        return
        
    # 2. Load the physically scaled reconstruction
    reconstruction = pycolmap.Reconstruction(sparse_scaled_dir)
    print(f"Loaded scaled reconstruction with {reconstruction.num_points3D()} points.")
    
    # 3. Export to PLY
    # A .ply file is a standard 3D file format that can be opened in Blender, MeshLab, or a game engine.
    reconstruction.export_PLY(str(ply_path))
    print(f"\n[SUCCESS] Exported 3D point cloud to: {ply_path}")
    
    # 4. Generate a URDF File
    # URDF (Unified Robot Description Format) is the XML standard used by ROS, PyBullet, and Mujoco.
    # We create a simple URDF that tells a physics simulator to load our .ply file as a physical object.
    
    urdf_content = f"""<?xml version="1.0" ?>
<robot name="photogrammetry_scene">
  <link name="base_link">
    <!-- Visual representation of the environment -->
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{ply_path.name}"/>
      </geometry>
    </visual>
    
    <!-- Physics collision box (uses the exact same 3D mesh) -->
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{ply_path.name}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    
    with open(urdf_path, "w") as f:
        f.write(urdf_content)
        
    print(f"[SUCCESS] Exported Robot Simulator file to: {urdf_path}")
    print("\nPhase 7 Complete! The environment is ready to be loaded into a robot simulation.")

if __name__ == "__main__":
    export_scene_to_simulator()
