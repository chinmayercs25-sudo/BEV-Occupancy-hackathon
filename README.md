# BEV 2D Occupancy Network

Hackathon submission for Problem Statement 3: Bird's-Eye-View 2D Occupancy.

## Results
- Mean IoU (TTA): 0.1969
- Distance-weighted Error: 106.97
- Dataset: nuScenes mini (404 frames)

## Outputs
1. **Current BEV Occupancy Grid** — 200×200 probability heatmap
2. **Future BEV (next 3s)** — accumulated next 6 LiDAR frames
3. **Uncertainty Heatmap** — risk zones (occupancy × uncertainty)

## Architecture
- 6-camera fused IPM BEV input
- Pretrained EfficientNet-b4 encoder (ImageNet)
- UNet decoder with 3 output heads
- Losses: Focal + Dice + Tversky + Distance-weighted
- Test-time augmentation (3 flips)
- Perspective view supervision

## Reference
Inspired by Fast Occupancy Network (Lu et al., 2024)

## How to run
Open `bev_occupancy.ipynb` in Google Colab with a T4 GPU.
Dataset: nuScenes mini — place at `/content/nuscenes/`
