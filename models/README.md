# Model Checkpoints Directory

Due to GitHub's file size limit (100 MB per file), binary PyTorch checkpoint files (`*.pt`) are excluded from version control via `.gitignore`:
- `best_resnet18_binary_model.pt` (~134 MB)
- `best_resnet34_binary_model.pt` (~255 MB)
- `best_resnet18_3class_model.pt` (~134 MB)

## Checkpoint Generation

All models are trained from scratch using the authentic European Space Agency (ESA) Sentinel-1 C-band SAR dataset with synchronized AIS telemetry.

To reproduce these model checkpoints locally:
```bash
# 1. Download and extract the UEIKAP dataset:
python -m src.download_ueikap

# 2. Derive hydrodynamic ground truth:
python -m src.label_generator

# 3. Train ResNet-18 binary model:
python -m src.train --arch resnet18 --mode binary --epochs 15

# 4. Train ResNet-34 polarimetric model:
python -m src.train --arch resnet34 --mode binary --epochs 15

# 5. Train ResNet-18 3-class directional model:
python -m src.train --arch resnet18 --mode 3class --epochs 15
```

All trained checkpoints will be automatically saved to this directory.
