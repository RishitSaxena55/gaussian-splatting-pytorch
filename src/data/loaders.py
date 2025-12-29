"""
3DGS Data Loaders - Load datasets for training.
"""
import torch
import numpy as np
import os
from tqdm import tqdm
import imageio.v2 as imageio
from ..cameras import Camera


def load_nsvf_dataset(data_path, max_images=30, downscale=4, device='cuda'):
    """
    Load NeRF Synthetic (NSVF format) dataset.
    
    Args:
        data_path: Path to scene folder (e.g., .../Lego)
        max_images: Maximum number of images to load
        downscale: Downscale factor for images
        device: Device for tensors
        
    Returns:
        images: List of (H, W, 3) tensors
        cameras: List of Camera instances
    """
    with open(f'{data_path}/intrinsics.txt', 'r') as f:
        intrinsics = [float(x) for x in f.read().split()]
    fx, fy, cx, cy = intrinsics[:4]
    
    rgb_dir = f'{data_path}/rgb'
    pose_dir = f'{data_path}/pose'
    img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])[:max_images]
    
    images, cameras = [], []
    
    for img_file in tqdm(img_files, desc='Loading NSVF'):
        img = imageio.imread(f'{rgb_dir}/{img_file}')
        if downscale > 1:
            img = img[::downscale, ::downscale]
        img = torch.from_numpy(img).float() / 255.0
        
        if img.shape[-1] == 4:
            alpha = img[..., 3:4]
            img = img[..., :3] * alpha + (1 - alpha)
        
        H, W = img.shape[:2]
        images.append(img.to(device))
        
        pose_file = img_file.replace('.png', '.txt')
        pose = np.loadtxt(f'{pose_dir}/{pose_file}').reshape(4, 4)
        c2w = torch.tensor(pose, dtype=torch.float32)
        
        cam = Camera(c2w[:3, :3].T, c2w[:3, 3], fx/downscale, fy/downscale, 
                    cx/downscale, cy/downscale, W, H, device)
        cameras.append(cam)
    
    print(f"✅ Loaded {len(images)} images ({H}x{W})")
    return images, cameras
