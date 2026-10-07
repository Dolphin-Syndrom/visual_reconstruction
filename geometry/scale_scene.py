import pycolmap
import numpy as np
from pathlib import Path

def scale_reconstruction():
    # 1. Define paths
    script_dir = Path(__file__).resolve().parent
    sparse_dir = (script_dir / "../outputs/sparse/0").resolve()
    if not sparse_dir.exists():
        sparse_dir = (script_dir / "../outputs/sparse").resolve()
    measurement_file = (script_dir / "../data/scene/measurement.txt").resolve()
    scaled_dir = (script_dir / "../outputs/sparse_scaled").resolve()
    scaled_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- STEP 1: Loading Data ---")
    if not sparse_dir.exists():
        print("Error: No sparse reconstruction found. Run SfM first.")
        return
        
    # Load the COLMAP reconstruction
    reconstruction = pycolmap.Reconstruction(sparse_dir)
    print(f"Loaded reconstruction with {reconstruction.num_points3D()} points.")
    
    # 2. Read the real-world measurement
    print("\n--- STEP 2: Calculating Scale Factor ---")
    try:
        import re
        with open(measurement_file, "r") as f:
            text = f.read().strip()
            # Robust parser: find the first number followed by 'cm' in the file
            match = re.search(r'([\d.]+)\s*cm', text, re.IGNORECASE)
            if not match:
                print("Could not find a measurement like '6.5 cm' in the file.")
                return
            real_world_cm = float(match.group(1))
            print(f"Real world measurement read: {real_world_cm} cm")
    except Exception as e:
        print(f"Could not read measurement file. Error: {e}")
        return

    # 3. Find distance in COLMAP (Arbitrary space)
    # In a professional pipeline (like your final goal), we would use AprilTags.
    # The code would detect AprilTag corners in 2D, find their 3D points, and measure them.
    # Since we don't have AprilTags in this dataset, we will calculate the distance 
    # between the first two cameras as our "reference distance".
    
    camera_ids = list(reconstruction.images.keys())
    if len(camera_ids) < 2:
        print("Not enough cameras to scale.")
        return
        
    # Get the 3D position (projection center) of two cameras
    cam1_pos = reconstruction.images[camera_ids[0]].projection_center()
    cam2_pos = reconstruction.images[camera_ids[1]].projection_center()
    
    # Calculate Euclidean distance in COLMAP's arbitrary units
    colmap_distance = np.linalg.norm(cam1_pos - cam2_pos)
    print(f"Distance between Camera 1 and 2 in COLMAP units: {colmap_distance:.4f}")
    
    # 4. Calculate the scaling factor
    # Scale = Real World Distance / COLMAP Distance
    scale_factor = real_world_cm / colmap_distance
    print(f"Math: Scale Factor = {real_world_cm} / {colmap_distance:.4f} = {scale_factor:.4f}")
    
    # 5. Apply the scale to the entire 3D space
    print("\n--- STEP 3: Applying Metric Scale ---")
    
    # We create a 3D Similarity Transform (Sim3d)
    # Sim3d contains: Scale (our factor), Rotation (identity matrix), Translation (zero)
    transform = pycolmap.Sim3d()
    transform.scale = scale_factor
    
    # Mathematically scale all 3D points and Camera positions
    reconstruction.transform(transform)
    
    # Verify the scale applied correctly
    cam1_pos_new = reconstruction.images[camera_ids[0]].projection_center()
    cam2_pos_new = reconstruction.images[camera_ids[1]].projection_center()
    new_colmap_distance = np.linalg.norm(cam1_pos_new - cam2_pos_new)
    
    print(f"Verification: New distance is {new_colmap_distance:.4f} cm (Should match {real_world_cm})")
    
    # 6. Save the scaled model
    reconstruction.write(scaled_dir)
    print(f"\n[SUCCESS] Scaled model saved to {scaled_dir}/")

if __name__ == "__main__":
    scale_reconstruction()
