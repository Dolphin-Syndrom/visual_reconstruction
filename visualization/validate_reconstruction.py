import pycolmap
import numpy as np
import cv2
from pathlib import Path

def validate_reprojection():
    print("--- Phase 6: Reprojection Validation ---")
    
    # 1. Paths
    sparse_dir = Path("../outputs/sparse_scaled")
    image_dir = Path("../data/scene")
    output_image_path = Path("../outputs/reprojection_test.jpg")
    
    if not sparse_dir.exists():
        print("Scaled reconstruction not found!")
        return
        
    # 2. Load Reconstruction
    reconstruction = pycolmap.Reconstruction(sparse_dir)
    
    # Pick the first registered image
    image_id = list(reconstruction.images.keys())[0]
    img_data = reconstruction.images[image_id]
    camera = reconstruction.cameras[img_data.camera_id]
    
    print(f"Validating using Image: {img_data.name}")
    
    # 3. Get the Extrinsic Matrix (World to Camera)
    # R is the Rotation Matrix, t is the Translation Vector
    R = img_data.cam_from_world().rotation.matrix()
    t = img_data.cam_from_world().translation
    
    # 4. Get the Intrinsic Matrix (K)
    # K contains focal length (fx, fy) and principal point (cx, cy)
    K = camera.calibration_matrix()
    
    print("\n--- The Math Matrices ---")
    print("Rotation (R):\n", R)
    print("Translation (t):", t)
    print("Camera Matrix (K):\n", K)
    
    # 5. Read the raw 2D image using OpenCV
    raw_img_path = image_dir / img_data.name
    img_cv = cv2.imread(str(raw_img_path))
    if img_cv is None:
        print(f"Failed to read image at {raw_img_path}")
        return
        
    # 6. Reproject all 3D points visible in this image
    points_drawn = 0
    
    for point2D in img_data.points2D:
        if point2D.has_point3D():
            # Get the actual 3D point in world coordinates (X, Y, Z)
            point3D = reconstruction.points3D[point2D.point3D_id]
            P_world = point3D.xyz
            
            # THE CORE MATH: Project 3D World to 3D Camera (P_cam = R * P_world + t)
            P_cam = np.dot(R, P_world) + t
            
            # THE CORE MATH: Project 3D Camera to 2D Image Plane using Pinhole Math
            # Divide by Z to handle perspective (things further away appear smaller)
            x_prime = P_cam[0] / P_cam[2]
            y_prime = P_cam[1] / P_cam[2]
            
            # Apply Intrinsics (Focal Length & Principal Point) to get Pixel Coordinates (u, v)
            u = K[0, 0] * x_prime + K[0, 2]
            v = K[1, 1] * y_prime + K[1, 2]
            
            # Draw a green dot at the calculated pixel coordinate
            cv2.circle(img_cv, (int(u), int(v)), radius=3, color=(0, 255, 0), thickness=-1)
            points_drawn += 1

    # Save the final image
    cv2.imwrite(str(output_image_path), img_cv)
    print(f"\n[SUCCESS] Reprojected {points_drawn} 3D points back onto the 2D image!")
    print(f"Saved validation image to: {output_image_path}")

if __name__ == "__main__":
    validate_reprojection()
