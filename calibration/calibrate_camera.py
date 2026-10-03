import cv2
import numpy as np
import glob
import yaml
import os

# --- Configuration ---
# Your specific checkerboard has 19 interior corners horizontally and 6 vertically
CHECKERBOARD = (19, 6)
# Standard checkerboard squares are usually around 2.5cm (0.025m), but for pure calibration without scale, 1.0 is fine.
SQUARE_SIZE = 1.0 
CALIB_DIR = "../data/calibration/"
OUTPUT_YAML = "camera_intrinsics.yaml"

def calibrate():
    # 1. Prepare the mathematical "Ground Truth" 3D points
    # These are perfectly flat points like (0,0,0), (1,0,0), (2,0,0) ... (19,5,0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Arrays to store object points (3D) and image points (2D) from all images
    objpoints = [] # 3D point in real world space
    imgpoints = [] # 2D points in image plane

    # 2. Load your images
    images = glob.glob(os.path.join(CALIB_DIR, '*.jpeg'))
    
    if not images:
        print(f"Error: No images found in {CALIB_DIR}. Check your path!")
        return

    print(f"Found {len(images)} images. Searching for checkerboard corners...")

    img_shape = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1] # (width, height)

        # 3. Find the checkerboard corners
        # This is where the math might fail due to the taped seam!
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret == True:
            objpoints.append(objp)
            
            # Refine the corner locations to sub-pixel accuracy for better math
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            print(f"[SUCCESS] Found corners in {os.path.basename(fname)}")
        else:
            print(f"[FAILED] Could not find {CHECKERBOARD} grid in {os.path.basename(fname)}")

    # 4. Calculate the Calibration
    if len(objpoints) > 0:
        print(f"\nCalibrating based on {len(objpoints)} successful images...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
        
        # 5. Calculate the Reprojection Error (The quality check!)
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            # Use numpy to avoid OpenCV type mismatch errors between CV_32FC1 and CV_32FC2
            error = np.linalg.norm(imgpoints[i].reshape(-1, 2) - imgpoints2.reshape(-1, 2)) / len(imgpoints2)
            mean_error += error
            
        print(f"\n--- CALIBRATION RESULTS ---")
        print(f"Camera Matrix (Intrinsics):\n{mtx}")
        print(f"Distortion Coefficients:\n{dist}")
        print(f"Total Mean Reprojection Error: {mean_error/len(objpoints):.4f} pixels")
        
        if mean_error/len(objpoints) > 1.0:
            print("WARNING: Reprojection error is > 1.0. The taped seam caused bad calibration!")
        else:
            print("SUCCESS: Reprojection error is good (< 1.0)!")

        # 6. Save to YAML
        data = {
            'camera_matrix': mtx.tolist(),
            'dist_coeff': dist.tolist(),
            'reprojection_error': float(mean_error/len(objpoints))
        }
        with open(OUTPUT_YAML, "w") as f:
            yaml.dump(data, f)
        print(f"\nSaved results to {OUTPUT_YAML}")
            
    else:
        print("\nFATAL ERROR: OpenCV could not find the checkerboard in ANY of the images.")
        print("You must reprint the checkerboard on a single piece of paper.")

if __name__ == "__main__":
    calibrate()
