import os
import pycolmap
from pathlib import Path

def run_sfm():
    scene_dir = Path("../data/scene")
    output_dir = Path("../outputs")
    output_dir.mkdir(exist_ok=True)
    
    db_path = output_dir / "database.db"
    
    # Always start with a clean database for a new run
    if db_path.exists():
        db_path.unlink()
        
    print("--- STEP 1: Feature Extraction ---")
    print("Scanning images for unique pixels (corners, edges, textures)...")
    pycolmap.extract_features(db_path, scene_dir)
    
    print("\n--- STEP 2: Feature Matching ---")
    print("Comparing images to find matching features (Exhaustive Match)...")
    # Exhaustive matching compares every image to every other image. 
    # This is perfect for small datasets (like our 15 images).
    pycolmap.match_exhaustive(db_path)
    
    print("\n--- STEP 3: Incremental Mapping & Bundle Adjustment ---")
    print("Triangulating 3D points and estimating camera poses...")
    sparse_model_dir = output_dir / "sparse"
    sparse_model_dir.mkdir(exist_ok=True)
    
    # Run the core Structure from Motion algorithm
    reconstructions = pycolmap.incremental_mapping(db_path, scene_dir, sparse_model_dir)
    
    if len(reconstructions) == 0:
        print("\n[FATAL ERROR] COLMAP failed to reconstruct the scene.")
        print("Reason: The object lacked enough texture, the background was too blurry, or the overlap was too low to mathematically connect the images.")
        return
        
    print(f"\n[SUCCESS] Reconstructed {len(reconstructions)} independent models.")
    best_model = reconstructions[0] # The first reconstruction is usually the largest/best
    
    # 5. Output Metrics and Validation
    print("\n--- RECONSTRUCTION METRICS ---")
    print(f"Total Images Provided : {len(list(scene_dir.glob('*.png')) + list(scene_dir.glob('*.jpg')) + list(scene_dir.glob('*.jpeg')))}")
    print(f"Images Registered     : {best_model.num_reg_images()} (If this is low, SfM failed to connect photos)")
    print(f"3D Points Generated   : {best_model.num_points3D()}")
    
    # Calculate Mean Reprojection Error for the reconstruction
    # (How well the 3D points align mathematically with the 2D images)
    mean_error = 0
    for point3D in best_model.points3D.values():
        mean_error += point3D.error
    if best_model.num_points3D() > 0:
        mean_error /= best_model.num_points3D()
    
    print(f"Mean Reprojection Err : {mean_error:.4f} pixels")
    
    # 6. Export the reusable files
    best_model.write(sparse_model_dir) # Saves cameras, images, points in binary format
    print(f"\nExported raw COLMAP binary files to {sparse_model_dir}/")

if __name__ == "__main__":
    run_sfm()
