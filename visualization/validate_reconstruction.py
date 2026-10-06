import pycolmap
import numpy as np
import cv2
from pathlib import Path

def validate_reprojection():
    print("--- Phase 6: Reprojection Validation ---")
    
    # 1. Resolve paths relative to script location
    script_dir = Path(__file__).resolve().parent
    sparse_dir = script_dir / "../outputs/sparse_scaled"
    image_dir = script_dir / "../data/scene"
    output_image_path = script_dir / "../outputs/reprojection_test.jpg"
    docs_asset_path = script_dir / "../docs/assets/reprojection.jpg"
    
    if not sparse_dir.exists():
        # Fallback: try unscaled sparse
        sparse_dir = script_dir / "../outputs/sparse"
        if not sparse_dir.exists():
            print("No reconstruction found!")
            return
        print("WARNING: Using unscaled sparse model (scaled not found).")
    
    # 2. Load Reconstruction
    reconstruction = pycolmap.Reconstruction(sparse_dir)
    print(f"Loaded reconstruction: {reconstruction.num_reg_images()} images, {reconstruction.num_points3D()} 3D points")
    
    # 3. Find the image with the MOST observed 3D points for best visualization
    best_image_id = None
    best_count = 0
    
    for image_id, img_data in reconstruction.images.items():
        count = sum(1 for p2d in img_data.points2D if p2d.has_point3D())
        if count > best_count:
            best_count = count
            best_image_id = image_id
    
    if best_image_id is None:
        print("No image with observed 3D points found!")
        return
    
    img_data = reconstruction.images[best_image_id]
    camera = reconstruction.cameras[img_data.camera_id]
    
    print(f"Selected Image: {img_data.name}")
    print(f"Visible 3D points in this view: {best_count}")
    
    # 4. Get Extrinsic and Intrinsic matrices
    R = img_data.cam_from_world().rotation.matrix()
    t = img_data.cam_from_world().translation
    K = camera.calibration_matrix()
    
    print("\n--- Extrinsic & Intrinsic Matrices ---")
    print(f"Rotation (R):\n{R}")
    print(f"Translation (t): {t}")
    print(f"Camera Matrix (K):\n{K}")
    print(f"Camera model: {camera.model_name}")
    
    # 5. Read the source 2D image
    raw_img_path = image_dir / img_data.name
    img_cv = cv2.imread(str(raw_img_path))
    if img_cv is None:
        print(f"Failed to read image at {raw_img_path}")
        return
    
    img_h, img_w = img_cv.shape[:2]
    print(f"Image resolution: {img_w} x {img_h}")
    
    # 6. Reproject all visible 3D points back onto this image
    points_drawn = 0
    errors = []
    
    for point2D in img_data.points2D:
        if point2D.has_point3D():
            point3D = reconstruction.points3D[point2D.point3D_id]
            P_world = point3D.xyz
            
            # Transform world point to camera coordinates: P_cam = R * P_world + t
            P_cam = np.dot(R, P_world) + t
            
            # Skip points behind the camera (negative depth)
            if P_cam[2] <= 0:
                continue
            
            # Perspective division: normalized image coordinates
            x_norm = P_cam[0] / P_cam[2]
            y_norm = P_cam[1] / P_cam[2]
            
            # Apply intrinsics to get pixel coordinates
            u = K[0, 0] * x_norm + K[0, 2]
            v = K[1, 1] * y_norm + K[1, 2]
            
            # Bounds check: only draw points within the image frame
            if 0 <= int(u) < img_w and 0 <= int(v) < img_h:
                # Compute per-point reprojection error against COLMAP's stored 2D observation
                obs_x, obs_y = point2D.xy
                err = np.sqrt((u - obs_x)**2 + (v - obs_y)**2)
                errors.append(err)
                
                # Draw green circle for reprojected point
                cv2.circle(img_cv, (int(u), int(v)), radius=4, color=(0, 255, 0), thickness=-1)
                # Draw thin white outline for visibility
                cv2.circle(img_cv, (int(u), int(v)), radius=4, color=(255, 255, 255), thickness=1)
                points_drawn += 1
    
    # 7. Compute and print error statistics
    if len(errors) > 0:
        errors = np.array(errors)
        print(f"\n--- Reprojection Error Statistics ---")
        print(f"Points reprojected  : {points_drawn}")
        print(f"Mean error          : {np.mean(errors):.4f} px")
        print(f"Median error        : {np.median(errors):.4f} px")
        print(f"Max error           : {np.max(errors):.4f} px")
        print(f"Std deviation       : {np.std(errors):.4f} px")
        print(f"Points < 1.0 px err : {np.sum(errors < 1.0)} / {len(errors)} ({100*np.sum(errors < 1.0)/len(errors):.1f}%)")
        print(f"Points < 2.0 px err : {np.sum(errors < 2.0)} / {len(errors)} ({100*np.sum(errors < 2.0)/len(errors):.1f}%)")
    
    # 8. Save the validation image
    cv2.imwrite(str(output_image_path), img_cv)
    print(f"\n[SUCCESS] Saved reprojection image to: {output_image_path}")
    
    # Also copy to docs assets for the web page
    docs_asset_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(docs_asset_path), img_cv)
    print(f"[SUCCESS] Copied to docs assets: {docs_asset_path}")

if __name__ == "__main__":
    validate_reprojection()
