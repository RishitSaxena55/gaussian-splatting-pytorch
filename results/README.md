# Results

This folder contains training outputs for each scene.

## Scene Results

### Lego
- Training curves and comparison images

### Chair  
- Training curves and comparison images

### Ship
- Training curves and comparison images

### Drums
- Training curves and comparison images

## How to Add Results

Copy your output files from Google Drive to the appropriate scene folder:

```bash
# From Google Drive (3DGS_Research or 3DGS/{Scene})
cp -r /content/drive/MyDrive/3DGS/Lego/outputs/* results/lego/
cp -r /content/drive/MyDrive/3DGS/Chair/outputs/* results/chair/
cp -r /content/drive/MyDrive/3DGS/Ship/outputs/* results/ship/
cp -r /content/drive/MyDrive/3DGS/Drums/outputs/* results/drums/
```

## Expected Files

Each scene folder should contain:
- `comparison.png` - Rendered vs Ground Truth grid
- `training_curves.png` - Loss, PSNR, Gaussian count plots
- `3d_view.png` - 3D point cloud visualization
- `rendered_views.mp4` - Video of rendered training views (optional)
