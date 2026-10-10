# Technical Report: Real-to-Sim 3D Visual Reconstruction Pipeline
**Module:** Photogrammetry, Computational Geometry & Simulation Asset Generation  
**Author:** Robotics & Computer Vision Engineering Team  
**Status:** Validated, Production-Ready, Reproducible  

---

## 1. Executive Summary & Objective

This technical report describes the architecture, mathematical foundation, and reproduction procedure for the **Real-to-Sim 3D Visual Reconstruction Pipeline**.

The primary objective of this project is to convert multi-view RGB photographs of physical objects into metric-calibrated, simulation-ready 3D digital twins. The pipeline outputs three standard digital artifacts:
1. **Dense Colored Point Cloud (`outputs/dense/fused.ply`):** High-density geometric representation containing over 1.67 million 3D vertices with spatial coordinates and true-color RGB channels.
2. **Watertight Polygon Mesh (`outputs/export/scene_mesh.ply`):** Closed manifold triangular mesh generated via Poisson Surface Reconstruction (Depth 9), providing 275,963 triangular faces for collision and contact mechanics.
3. **Robot Physics Descriptor (`outputs/export/scene.urdf`):** Standard Unified Robot Description Format XML descriptor linking visual and physical collision geometry for direct execution in PyBullet, MuJoCo, Isaac Sim, and ROS 2 Gazebo.

---

## 2. Mathematical & Algorithmic Architecture

The complete reconstruction pipeline spans seven rigorous mathematical stages:

```
[Physical Scene] ──► [Camera Calibration] ──► [Sparse SfM] ──► [Dense MVS] ──► [Poisson Meshing] ──► [Validation] ──► [Sim(3) URDF]
```

### 2.1 Camera Calibration (Phases 1 & 2)
The pinhole camera model maps a 3D Euclidean point $\mathbf{X} = [X, Y, Z]^T$ in camera coordinates to a 2D pixel coordinate $\mathbf{x} = [u, v]^T$ on the sensor plane:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix}$$

Lens distortion is compensated using the 5-parameter Brown-Conrady polynomial model:

$$x_{\text{distorted}} = x'(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + [2 p_1 x' y' + p_2(r^2 + 2x'^2)]$$

$$y_{\text{distorted}} = y'(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + [p_1(r^2 + 2y'^2) + 2 p_2 x' y']$$

where $r^2 = x'^2 + y'^2$.

### 2.2 Sparse Structure from Motion (Phase 4)
* **SIFT Feature Extraction:** Computes scale-space extrema using Difference of Gaussians (DoG) and builds 128-dimensional invariant orientation descriptors.
* **Exhaustive Feature Matching:** Computes pairwise descriptor distances across all image pairs $C(N, 2)$. Matches are geometrically verified using the Epipolar constraint:
  $$\mathbf{x}_2^T \mathbf{F} \mathbf{x}_1 = 0$$
* **Bundle Adjustment:** Jointly minimizes the non-linear reprojection error across all camera poses $(\mathbf{R}_i, \mathbf{t}_i)$ and 3D point positions $\mathbf{X}_j$ using Levenberg-Marquardt optimization:
  $$\min_{\mathbf{R}_i, \mathbf{t}_i, \mathbf{X}_j} \sum_{i} \sum_{j} \rho \left( \|\mathbf{x}_{ij} - \pi(\mathbf{K}, \mathbf{R}_i, \mathbf{t}_i, \mathbf{X}_j)\|^2 \right)$$

### 2.3 Dense Multi-View Stereo (Phase 4B)
* **Image Undistortion:** Re-samples raw input images to remove non-linear optical distortion, creating ideal planar pinhole frames.
* **PatchMatch Stereo:** Evaluates photometric consistency (Normalized Cross-Correlation) over random 3D slanted planes per pixel across multi-view epipolar lines. Iterative spatial propagation yields dense depth and normal grids.
* **Stereo Fusion:** Merges depth map estimations from multiple views by enforcing photometric and geometric depth consistency, filtering occlusions and depth discontinuities.

### 2.4 Poisson Surface Reconstruction (Phase 4C)
Transforms unorganized dense point vertices into a watertight continuous boundary surface by solving the spatial Poisson equation:

$$\nabla \cdot \nabla \chi = \nabla \cdot \mathbf{V}$$

where $\mathbf{V}$ represents an oriented normal vector field estimated via KD-Tree nearest-neighbor covariance analysis, and $\chi$ is an indicator function evaluated over an adaptive Octree hierarchy.

### 2.5 Metric Scale Calibration (Phase 7)
Because monocular SfM reconstructs scenes up to an arbitrary global scale factor, the reconstruction is mapped to real-world SI units (meters) using a 7-DOF Similarity Transformation $\text{Sim}(3)$:

$$\mathbf{X}_{\text{metric}} = s \cdot \mathbf{R} \cdot \mathbf{X}_{\text{arbitrary}} + \mathbf{t}$$

where the scalar scale $s$ is derived from the known physical measurement stored in `data/scene/measurement.txt`.

---

## 3. System Prerequisites & Environment Setup

To run this repository on any custom dataset from scratch, set up the operating environment as follows.

### 3.1 Hardware Recommendations
* **CPU:** 8+ physical cores (Intel Core i7/i9 or AMD Ryzen 7/9)
* **RAM:** Minimum 16 GB (32 GB recommended for datasets $> 80$ images)
* **GPU (Crucial for Dense MVS):** NVIDIA GPU with CUDA Compute Capability $\ge 7.5$ (RTX 20/30/40/50 series or Quadro/Tesla) with $\ge 8\text{ GB}$ VRAM.
* **Storage:** $\ge 10\text{ GB}$ free SSD storage for depth maps and dense point cloud binaries.

### 3.2 Software Installation

#### Step 1: Clone the Repository
```bash
git clone https://github.com/Dolphin-Syndrom/visual_reconstruction.git
cd visual_reconstruction
```

#### Step 2: Create a Dedicated Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Step 3: Install Required Dependencies
Install the required scientific computing, vision, and geometry packages:
```bash
pip install --upgrade pip
pip install numpy opencv-python open3d pyyaml matplotlib scipy pybullet
```

#### Step 4: Install PyCOLMAP with CUDA Acceleration
Dense Multi-View Stereo requires GPU-accelerated PyCOLMAP bindings:
```bash
# Install uv package installer
pip install uv

# Install pre-compiled CUDA 12 PyCOLMAP wheel
uv pip install pycolmap-cuda12
```

> [!NOTE]
> If an NVIDIA GPU is not available, install CPU PyCOLMAP with `pip install pycolmap`. Note that CPU PatchMatch stereo execution will take significantly longer (several hours).

---

## 4. How to Capture a Custom Dataset from Scratch

Proper photographic capture determines 90% of photogrammetric reconstruction quality. Follow these capture rules strictly.

### 4.1 Target Object Selection Criteria
* **Texture Characteristics (Crucial):** Choose objects with diffuse (matte), high-frequency visual surface textures (e.g., printed fabric, stone, textured wood, paper, chalked objects).
* **Materials to Avoid:** Avoid untextured smooth monochromatic surfaces, specular (shiny) plastics, metals, glass, mirrors, or translucent materials. Specular highlights move across frames, violating the Lambertian brightness constancy assumption and causing holes in dense clouds.

### 4.2 Multi-Tier Orbital Capture Protocol
1. **Camera Settings:**
   * Lock exposure, ISO, and white balance to manual mode.
   * Lock focus (do not use continuous autofocus between shots).
   * Maintain consistent focal length (do not zoom).
2. **Orbital Trajectory:**
   * Walk in a full 360-degree circle around the target at **three elevation tiers**:
     * **Tier 1 (Horizontal / Low Angle):** 0° to 15° elevation (capture 25–30 images).
     * **Tier 2 (Medium Angle):** 30° to 45° elevation (capture 25–30 images).
     * **Tier 3 (High Angle / Top Down):** 60° to 75° elevation (capture 20–25 images).
3. **Overlap Ratio:**
   * Maintain **75% to 85% overlap** between consecutive frames.
   * Angular spacing between shots should not exceed 10 degrees.
4. **Target Dataset Size:**
   * Standard desktop objects: 60 to 90 high-resolution images.
5. **Physical Metric Scale Reference:**
   * Measure a distinct, rigid physical dimension of your object using calipers or a ruler.
   * Record this dimension in centimeters in `data/scene/measurement.txt` (e.g., `37.5 cm`).

---

## 5. Step-by-Step Execution Workflow

Execute the following commands sequentially from the project root directory.

### Step 1: Camera Calibration (Phases 1 & 2)
1. Print a planar checkerboard target (default: 19 horizontal by 6 vertical internal corners).
2. Capture 10 to 15 calibration frames of the checkerboard from different tilt angles (pitch, yaw, roll) and distances.
3. Save the frames into `data/calibration/` (formats: `.jpeg`, `.jpg`, `.png`).
4. Execute calibration:
   ```bash
   python3 calibration/calibrate_camera.py
   ```
5. **Verification:** The script calculates camera focal lengths, principal point coordinates, and distortion vectors. The calibration must achieve an **RMS reprojection error $< 0.5\text{ px}$**. The parameters are exported to `camera_intrinsics.yaml`.

---

### Step 2: Sparse Structure from Motion (Phase 4)
1. Clear old scene frames and place your newly captured orbital images into `data/scene/`.
2. Run sparse reconstruction:
   ```bash
   python3 reconstruction/run_reconstruction.py
   ```
3. **Execution Steps:**
   * Cleans and initializes `outputs/database.db`.
   * Extracts SIFT feature points (corners, edges, texture gradients).
   * Performs exhaustive two-view matching.
   * Runs incremental mapping and bundle adjustment.
4. **Verification:**
   * Check registered image percentage (target: $\ge 95\%$).
   * Check mean reprojection error (target: $< 1.0\text{ px}$).
   * Exported files: `outputs/sparse/0/cameras.bin`, `outputs/sparse/0/images.bin`, `outputs/sparse/0/points3D.bin`.
5. **Interactive Visualization:**
   ```bash
   python3 visualization/view_cloud.py
   ```
   Renders the sparse point constellation in the interactive Open3D viewer.

---

### Step 3: Dense Multi-View Stereo (Phase 4B)
1. Run dense multi-view stereo reconstruction:
   ```bash
   python3 reconstruction/run_dense_reconstruction.py
   ```
2. **Execution Steps:**
   * **Image Undistortion:** Generates ideal pinhole frames in `outputs/dense/images/`.
   * **PatchMatch Stereo:** Spawns CUDA GPU worker threads to calculate photometric depth maps (`outputs/dense/stereo/depth_maps/*.photometric.bin`) and geometric depth maps (`*.geometric.bin`).
   * **Stereo Fusion:** Enforces multi-view geometric depth consistency and writes the fused point cloud.
3. **Output:** `outputs/dense/fused.ply`.
4. **Interactive Visualization:**
   ```bash
   python3 visualization/view_dense.py
   ```
   Inspect the dense point cloud in Open3D. Check for solid surface coverage and absence of spurious floaters.

---

### Step 4: Solid Poisson Surface Reconstruction (Phase 4C)
1. Convert the unorganized point vertices into a solid physical polygon mesh:
   ```bash
   python3 reconstruction/run_surface_reconstruction.py
   ```
2. **Execution Steps:**
   * Loads `outputs/dense/fused.ply`.
   * Estimates surface normal vectors via KD-Tree nearest neighbor query ($K = 50$).
   * Solves the spatial Poisson indicator equation at **Octree Depth 9** ($512^3$ spatial voxel hierarchy).
   * Trims the lowest 10% density vertices to eliminate background extrapolation artifacts.
3. **Output:** `outputs/export/scene_mesh.ply`.
4. **Verification:** A 3D Open3D window displays the watertight solid mesh. Verify anatomical facial contours, limb boundaries, and continuous manifold surfaces.

---

### Step 5: Full Pipeline Diagnostics (Phase 5)
Audit the disk payload, binary completeness, and numerical consistency:
```bash
python3 visualization/diagnose_and_view.py
```
* Audits sparse binary models (`cameras.bin`, `images.bin`, `points3D.bin`).
* Audits dense MVS artifacts (81 undistorted images, 162 depth maps, average depth file size: ~4.4 MB).
* Renders the complete dense reconstruction in high contrast.

---

### Step 6: Mathematical Reprojection Validation (Phase 6)
Verify the geometric accuracy of the 3D-to-2D projection:
```bash
python3 visualization/validate_reconstruction.py
```
* Projects triangulated 3D world points onto 2D image keyframes using recovered camera extrinsic matrices $[\mathbf{R} \mid \mathbf{t}]$ and intrinsic matrix $\mathbf{K}$.
* Evaluates pixel residual error statistics:
  * **Mean Error:** Target $< 1.0\text{ px}$.
  * **Inlier Ratio ($< 1.0\text{ px}$):** Target $\ge 75\%$.
  * **Inlier Ratio ($< 2.0\text{ px}$):** Target $\ge 90\%$.
* Exports overlay image: `outputs/reprojection_test.jpg` (green circles indicate reprojections).

---

### Step 7: Metric Scale Calibration & Simulation URDF Export (Phase 7)
1. Verify `data/scene/measurement.txt` contains your ground-truth length (e.g., `37.5 cm`).
2. Apply the Sim(3) metric transformation:
   ```bash
   python3 geometry/scale_scene.py
   ```
3. Export the physics simulator package:
   ```bash
   python3 visualization/export_scene.py
   ```
4. **Exported Assets:**
   * `outputs/export/scene_scaled.ply`: Metric-scaled 3D point cloud (scale factor applied).
   * `outputs/export/scene.urdf`: XML robot simulator descriptor referencing visual and collision geometry.

---

## 6. Physics Simulator Integration Guide

The exported `outputs/export/scene.urdf` descriptor allows direct import into standard robotics simulators.

### 6.1 PyBullet Integration
Create a test script `test_sim.py`:

```python
import pybullet as p
import pybullet_data
import time

# 1. Initialize PyBullet physics engine
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 2. Load ground plane and reconstructed digital twin
plane_id = p.loadURDF("plane.urdf")
scene_id = p.loadURDF(
    "outputs/export/scene.urdf",
    basePosition=[0, 0, 0],
    baseOrientation=[0, 0, 0, 1],
    useFixedBase=True
)

print(f"Successfully loaded digital twin URDF into PyBullet (ID: {scene_id})")

# 3. Step physics simulation loop
for _ in range(1000):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

p.disconnect()
```

### 6.2 MuJoCo Integration
Convert `scene_mesh.ply` to an `.obj` file or reference the PLY directly in your MJCF XML:

```xml
<mujoco model="reconstructed_scene">
  <asset>
    <mesh name="scene_mesh" file="outputs/export/scene_mesh.ply" scale="1 1 1"/>
  </asset>
  <worldbody>
    <geom name="scene_collision" type="mesh" mesh="scene_mesh" pos="0 0 0" rgba="0.8 0.6 0.4 1"/>
  </worldbody>
</mujoco>
```

---

## 7. Troubleshooting & Engineering Runbook

| Failure Symptom | Underlying Root Cause | Verified Engineering Resolution |
| :--- | :--- | :--- |
| **CUDA error during PatchMatch** | Default pip package (`pip install pycolmap`) installs CPU-only binaries without CUDA acceleration. | Install pre-compiled CUDA 12 bindings using `uv`: `uv pip install pycolmap-cuda12`. Ensure NVIDIA drivers support CUDA $\ge 12.0$. |
| **`colmap::ExistsDir` exception in stereo fusion** | `pycolmap.stereo_fusion` expects a directory path rather than a file path when validating output locations. | Pass the directory `dense_dir` to `stereo_fusion()`, capture the returned in-memory `Reconstruction` object, and call `reconstruction.export_PLY("outputs/dense/fused.ply")`. |
| **SfM registers $< 50\%$ of images** | Visual overlap between adjacent frames is too low ($< 60\%$), or camera exposure varied wildly. | Re-capture orbital images maintaining $\ge 75\%$ overlap and locked manual exposure. Ensure consecutive camera angles differ by $\le 10^\circ$. |
| **Dense cloud has large holes / voids** | Target object material is shiny, specular, or untextured plastic. | Passive photogrammetry requires diffuse, high-frequency texture. Apply temporary matte spray (chalk/developer spray) or select textured objects. |
| **Poisson mesh creates outer "bubble"** | Poisson equation extrapolates bounding volume into unobserved empty space. | The script automatically trims the lowest 10% density vertices using `run_surface_reconstruction.py`. If bubbles persist, increase trimming quantile to 15%. |
| **Scale is inverted or incorrect in simulator** | `measurement.txt` format could not be parsed by regex. | Ensure `data/scene/measurement.txt` contains a plain decimal number followed by `cm` (e.g., `37.5 cm`). |

---

## 8. Empirical Validation: Trial 1 vs. Trial 2

To evaluate the pipeline's robustness, we conducted two empirical test trials comparing rigid specular objects against textured diffuse objects:

| Metric / Parameter | Trial 1 (Rigid LEGO Block) | Trial 2 (Plush Cat Doll) | Experimental Analysis |
| :--- | :--- | :--- | :--- |
| **Input Dataset Size** | 38 images | 81 images | +113% angular density & visual overlap |
| **Object Material Class** | Specular, smooth plastic | Diffuse, textured plush fabric | High-frequency surface gradient upgrade |
| **Registered Images** | 37 / 38 (97.4%) | 81 / 81 (100.0%) | 100% multi-view trajectory convergence |
| **Sparse 3D Points** | 4,164 points | 53,524 points | 12.8× sparse landmark density |
| **Dense Fused Points** | 114,675 points | **1,676,815 points** | **14.6× Point Density Increase** |
| **Poisson Mesh Faces** | 120,442 triangles | **275,963 triangles** | Complete manifold surface without holes |
| **Mean Reprojection Error** | 0.9624 px | **0.5602 px** | **41.8% Sub-pixel Error Reduction** |
| **Median Reprojection Error** | 0.7849 px | **0.3973 px** | Highly concentrated residual error mode |
| **Points $< 1.0\text{ px}$ Error** | 62.4% | **86.3%** | High inlier confidence across 7,650 points |
| **Points $< 2.0\text{ px}$ Error** | 91.5% | **97.1%** | Minimal Gaussian outlier distribution |
| **MVS Computation Time** | ~3.2 minutes | 22.8 minutes | High-resolution volumetric depth evaluation |
| **Total Pipeline Storage** | ~340 MB | **3.2 GiB (452 items)** | Complete archive of depth and geometry maps |

### Conclusions
1. **Material Sensitivity:** Active or passive photogrammetry is constrained by surface reflectance. Diffuse, textured surfaces allow PatchMatch Stereo to resolve sub-pixel correspondences without tracking dropouts.
2. **Volumetric Resolution:** Increasing multi-tier orbital images from 38 to 81 yields a 14.6× increase in dense point cloud density (1.68M points), providing sufficient fidelity for high-precision robotic simulation and collision geometry.
3. **Sub-Pixel Certification:** Achieving a mean reprojection error of **0.5602 px** mathematically verifies that the recovered 3D geometry aligns with physical camera observations to within sub-pixel accuracy.
