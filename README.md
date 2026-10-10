# Real-to-Sim 3D Visual Reconstruction Pipeline

A photogrammetric reconstruction pipeline that converts multi-view RGB photographs into metric-scaled, simulation-ready 3D assets (Dense Point Cloud, Poisson Solid Mesh, and URDF Robot Description).

---

## 1. Pipeline Overview

This repository executes a 7-phase computer vision and computational geometry workflow:

```
[RGB Camera]
      │
      ▼
1. Camera Calibration  ────────► Intrinsic Matrix K, Brown-Conrady Distortion (RMS: 0.41 px)
      │
      ▼
2. Sparse SfM (PyCOLMAP) ──────► SIFT Extraction + Exhaustive Matching + Bundle Adjustment
      │
      ▼
3. Dense MVS (CUDA) ───────────► PatchMatch Depth Maps + Stereo Fusion (1.68M Points)
      │
      ▼
4. Poisson Meshing ────────────► KD-Tree Normals + Octree Depth 9 (275K Triangles)
      │
      ▼
5. Validation ─────────────────► 2D/3D Reprojection Test (0.56 px Mean Residual)
      │
      ▼
6. Metric Scaling & Export ────► Sim(3) Alignment + outputs/export/scene.urdf
      │
      ▼
[Physics Simulators: PyBullet / MuJoCo / Isaac Sim / Gazebo]
```

---

## 2. System Requirements

* **Operating System:** Ubuntu 22.04+ or Windows 10/11
* **Python:** Version 3.10 or higher
* **GPU (Recommended for Dense MVS):** NVIDIA GPU with CUDA 12+ (e.g., RTX 30/40/50 series)
* **Memory:** Minimum 16 GB RAM

---

## 3. Installation

Clone this repository and create a clean Python virtual environment:

```bash
git clone https://github.com/Dolphin-Syndrom/visual_reconstruction.git
cd visual_reconstruction
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

Install the required core dependencies:

```bash
# Core computer vision and geometry libraries
pip install numpy opencv-python open3d pyyaml matplotlib scipy pybullet

# GPU Acceleration for COLMAP Dense MVS (recommended)
pip install uv
uv pip install pycolmap-cuda12

# Standard CPU fallback (if CUDA GPU is unavailable):
# pip install pycolmap
```

---

## 4. Quickstart Execution (Phase-by-Phase)

Follow these steps in sequential order:

### Step 1: Camera Calibration (Phases 1 & 2)
Place checkerboard calibration images in `data/calibration/` and calculate camera intrinsics:
```bash
python3 calibration/calibrate_camera.py
```
* **Output:** `camera_intrinsics.yaml` (RMS error: 0.41 px).

### Step 2: Sparse Structure from Motion (Phase 4)
Place orbital scene images in `data/scene/` and run sparse SfM:
```bash
python3 reconstruction/run_reconstruction.py
```
* **Outputs:** `outputs/database.db`, `outputs/sparse/0/` (`cameras.bin`, `images.bin`, `points3D.bin`).

### Step 3: Dense Multi-View Stereo (Phase 4B)
Run CUDA PatchMatch stereo to generate per-pixel depth maps and fused point cloud:
```bash
python3 reconstruction/run_dense_reconstruction.py
```
* **Outputs:** `outputs/dense/fused.ply` (1,676,815 points), 162 depth binaries in `outputs/dense/stereo/depth_maps/`.

### Step 4: Poisson Surface Reconstruction (Phase 4C)
Convert unorganized points into a watertight manifold triangle mesh:
```bash
python3 reconstruction/run_surface_reconstruction.py
```
* **Output:** `outputs/export/scene_mesh.ply` (139,382 vertices, 275,963 triangles).

### Step 5: Reprojection Validation (Phase 6)
Verify sub-pixel 2D/3D correspondence collinearity:
```bash
python3 visualization/validate_reconstruction.py
```
* **Validation Target:** Mean residual $< 1.0\text{ px}$ (Observed: 0.5602 px across 7,650 points).
* **Output:** `outputs/reprojection_test.jpg`.

### Step 6: Metric Scaling & Simulation URDF Export (Phase 7)
Define ground truth measurement in `data/scene/measurement.txt` (e.g., `37.5 cm`), scale the model, and export simulator descriptors:
```bash
python3 geometry/scale_scene.py
python3 visualization/export_scene.py
```
* **Outputs:** `outputs/export/scene_scaled.ply`, `outputs/export/scene.urdf`.

---

## 5. Directory Structure

```
visual_reconstruction/
├── calibration/             # Camera calibration scripts
│   ├── calibrate_camera.py  # Intrinsic matrix & distortion solver
│   └── find_grid.py         # Sub-pixel corner detection visualizer
├── data/                    # Input datasets
│   ├── calibration/         # Checkerboard frames (14 images)
│   └── scene/               # Target orbital frames (81 images) + measurement.txt
├── docs/                    # Interactive web documentation (HTML/CSS)
│   ├── index.html           # Technical engineering log
│   ├── style.css            # Responsive layout styling
│   └── assets/              # Renderings, plots, and diagnostic screenshots
├── geometry/                # Spatial transformation and scaling utilities
│   ├── projection.py        # 3D to 2D pinhole math
│   ├── scale_scene.py       # Sim(3) ground-truth metric scaling
│   └── transforms.py        # Coordinate frame transformations
├── outputs/                 # Generated pipeline artifacts
│   ├── database.db          # SIFT feature & matching database
│   ├── sparse/0/            # COLMAP sparse model binaries
│   ├── dense/               # Undistorted frames, depth maps & fused.ply
│   └── export/              # scene_scaled.ply, scene_mesh.ply, scene.urdf
├── reconstruction/          # Core reconstruction pipelines
│   ├── run_reconstruction.py         # Sparse SfM driver
│   ├── run_dense_reconstruction.py   # CUDA MVS PatchMatch driver
│   └── run_surface_reconstruction.py # Open3D Poisson meshing driver
├── report/                  # Detailed documentation and reproduction reports
│   └── week2_reconstruction_report.md
└── visualization/           # Verification and export tools
    ├── diagnose_and_view.py          # Storage and binary verification
    ├── export_scene.py               # URDF and scaled PLY export
    ├── validate_reconstruction.py    # Reprojection error calculator
    ├── view_cloud.py                 # Open3D sparse cloud viewer
    └── view_dense.py                 # Open3D dense cloud viewer
```

---

## 6. Empirical Benchmark Results

| Parameter | Trial 1 (Rigid LEGO Block) | Trial 2 (Textured Cat Doll) | Factor Improvement |
| :--- | :--- | :--- | :--- |
| **Input Image Count** | 38 images | 81 images | +113% coverage |
| **Surface Reflectance** | Specular / Plastic | Diffuse / Textured Fabric | Material Optimization |
| **Registered Images** | 37 / 38 (97.4%) | 81 / 81 (100.0%) | Complete Registration |
| **Dense Point Cloud** | 114,675 points | **1,676,815 points** | **14.6× Point Density** |
| **Poisson Mesh Faces** | 120,442 triangles | **275,963 triangles** | 2.3× Geometric Detail |
| **Mean Reprojection Error** | 0.9624 px | **0.5602 px** | **41.8% Error Reduction** |
| **Inliers $< 1.0\text{ px}$** | 62.4% | **86.3%** | High Sub-Pixel Fidelity |
| **Pipeline Storage Size** | ~340 MB | **3.2 GiB (452 items)** | Complete Production Scale |

---

## 7. Physics Simulation Integration

Load the generated scene into PyBullet:

```python
import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load the reconstructed digital twin
scene_id = p.loadURDF("outputs/export/scene.urdf", basePosition=[0, 0, 0], useFixedBase=True)

while True:
    p.stepSimulation()
```

---

## 8. License & Authors

Developed for Real-to-Sim Robotics and Photogrammetric Computer Vision workflows. Released under the MIT License.
