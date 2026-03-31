# BEV 2D Occupancy Network
**Hackathon — Problem Statement 3: Bird's-Eye-View 2D Occupancy**

---

## Project Overview

Standard front-view cameras suffer from perspective distortion. Autonomous vehicles solve this by converting camera images into a 2D top-down **Bird's-Eye-View (BEV) Occupancy Grid** — a map where each cell represents whether that patch of ground is occupied by an obstacle or is free space.

This project implements a multi-camera BEV occupancy prediction system using the nuScenes dataset. Given images from 6 cameras mounted on the ego vehicle, the model outputs three simultaneous predictions:

1. **Current BEV Occupancy Grid** — 200×200 probability map of current obstacles
2. **Future BEV (next 3 seconds)** — predicted occupancy accumulated from next 6 LiDAR frames
3. **Uncertainty Heatmap** — per-cell confidence map; Risk = occupancy × uncertainty

**Results:**

| Metric | Value |
|--------|-------|
| Mean IoU (with TTA) | 0.1969 ± 0.0505 |
| Distance-weighted Error | 106.97 ± 6.88 |
| Training samples | 323 (nuScenes mini) |

#### Our model achieves 19.69% IoU using only 404 samples, compared to ~21% reported in literature using 700 samples. This demonstrates competitive performance despite significantly reduced training data.
---

## Model Architecture

**Input:** 200×200×3 fused BEV image (6 cameras warped via IPM and averaged)

**Pipeline:**

```
6 Camera Images
       │
       ▼
  IPM Homography Warp (per camera)
  6-camera weighted BEV fusion
       │
       ▼
  EfficientNet-b4 Encoder        ← pretrained ImageNet
       │
       ▼
  UNet Decoder (256→128→64→32→32 channels)
       │
  ┌────┴────┬─────────────┐
  ▼         ▼             ▼
head_curr  head_future  head_uncert
(Req 1)    (Req 2)      (Req 3)
200×200    200×200      200×200
```

**Key components:**

- **IPM (Inverse Perspective Mapping):** Homography matrix computed per camera warps each image to a top-down 200×200 BEV patch. All 6 patches are fused by weighted averaging, giving full 360° coverage.
- **Backbone:** Pretrained EfficientNet-b4 encoder — essential for generalisation on small datasets.
- **Multi-head decoder:** Three lightweight CNN heads share a common 32-channel feature map from the UNet decoder, each predicting a different output.
- **Loss:** Focal + Distance-weighted Focal + Dice + Tversky (β=0.7, recall-focused) for occupancy heads. Uncertainty head uses MSE against the model's own prediction error.
- **PV Supervision:** Auxiliary UNet head on the front camera image, supervised by LiDAR projected onto the image plane — provides direct gradient to backbone features.
- **Training:** Two-phase — encoder frozen (Phase 1, 10 epochs), then full fine-tune with differential learning rates (Phase 2, up to 30 epochs). Test-time augmentation (3 flips) applied at inference.

---

## Dataset Used

**nuScenes v1.0-mini**

| Property | Value |
|----------|-------|
| Scenes | 10 |
| Total samples | 404 |
| Cameras | 6 (front, front-left, front-right, back, back-left, back-right) |
| LiDAR | Velodyne HDL-32E (LIDAR_TOP) |
| BEV range | ±20m longitudinal and lateral |
| Grid resolution | 0.2m per cell → 200×200 grid |
| Train / Val split | 323 / 81 samples (80/20, seed=42) |

Download: [https://www.nuscenes.org/nuscenes#download](https://www.nuscenes.org/nuscenes#download)
Select **"Mini"** under the v1.0 section. Free registration required.

---

## Setup & Installation

### Requirements

```
Python 3.10+
CUDA GPU (T4 or better recommended)

numpy==1.26.4
torch>=2.0
torchvision>=0.15
opencv-python-headless
nuscenes-devkit==1.2.0
segmentation-models-pytorch>=0.3
albumentations>=1.3
matplotlib>=3.7
```

### Google Colab (Recommended)

1. Open `bev_occupancy.ipynb` in Google Colab
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run the install cell:

```python
!pip install numpy==1.26.4 -q
!pip install nuscenes-devkit segmentation-models-pytorch albumentations torchvision -q
```

4. **Restart the runtime** after installation (Runtime → Restart session)

### Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/bev-occupancy-hackathon.git
cd bev-occupancy-hackathon

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install numpy==1.26.4
pip install nuscenes-devkit segmentation-models-pytorch albumentations
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## How to Run

### Step 1 — Prepare the dataset

**On Colab** (if dataset is in Google Drive):

```python
from google.colab import drive
drive.mount('/content/drive')

import tarfile, os
os.makedirs('/content/nuscenes', exist_ok=True)

with tarfile.open('/content/drive/MyDrive/v1.0-mini.tgz', 'r:gz') as tar:
    tar.extractall('/content/nuscenes')
```

**Locally:** Extract the downloaded `.tgz` to a folder and update `NUSCENES_DATAROOT` in the notebook.

### Step 2 — Run the notebook

Open `bev_occupancy.ipynb` and run all cells in order:

| Cell | Action |
|------|--------|
| Cell 1 | Install dependencies |
| Cell 2 | Mount Drive + extract dataset |
| Cell 3 | Full pipeline — data loading, model, training, evaluation, save outputs |

### Step 3 — Inference only (skip training)

If `bev_final.pth` is already downloaded:

```python
# After model is defined, replace the training block with:
model.load_state_dict(torch.load('bev_final.pth'))
model.eval()
# Then run the evaluation and render_full_output section at the bottom
```

**Pretrained weights:** [Download bev_final.pth from Google Drive](https://drive.google.com/file/d/1Wd6sh73cZa_x_Z7fkYnctq-GpZpqQgW8/view?usp=drive_link)

### Training time (T4 GPU)

| Phase | Epochs | Time |
|-------|--------|------|
| Phase 1 — encoder frozen | 10 | ~8 min |
| Phase 2 — full fine-tune | up to 30 | ~30 min |
| **Total** | | **~40 min** |

---

## Example Outputs / Results

Each inference call produces a 6-panel figure saved as `submission_N.png`:

```
┌─────────────────┬──────────────────┬──────────────────┐
│ BEV Camera      │ Req 1: Current   │ Req 1: TP/FP/FN  │
│ Input           │ Occupancy Grid   │ Overlay          │
│ (6-cam fused)   │ (heatmap)        │ Green/Red/Blue   │
├─────────────────┼──────────────────┼──────────────────┤
│ Req 2: Future   │ Req 3:           │ Req 3:           │
│ BEV (next 3s)   │ Uncertainty      │ Risk Zones       │
│                 │ Heatmap          │ (occ × uncert)   │
└─────────────────┴──────────────────┴──────────────────┘
```

All maps are in ego vehicle coordinates (±20m) with compass labels, 5m range rings, and a yellow ego vehicle marker at centre.

**Final metrics:**

| Metric | Value |
|--------|-------|
| Mean IoU (TTA) | **0.1969** |
| Distance-weighted Error | **106.97** |

Sample outputs are saved in the `outputs/` folder of this repository (`submission_0.png` to `submission_80.png`).

---

## References

- Lu et al. (2024). Fast Occupancy Network. arXiv:2412.07163
- Caesar et al. (2020). nuScenes: A multimodal dataset for autonomous driving. CVPR
- Li et al. (2022). BEVFormer: Learning bird's-eye-view representation. ECCV
- Philion & Fidler (2020). Lift, Splat, Shoot. ECCV
- Lin et al. (2017). Focal loss for dense object detection. ICCV
