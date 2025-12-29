"""
3DGS - 3D Gaussian Splatting Implementation

A complete, educational PyTorch implementation of 3D Gaussian Splatting.

Usage:
    from src.cameras import Camera
    from src.gaussians import GaussianModel
    from src.renderer import render_gaussians
    from src.training import combined_loss, densify_and_prune
    from src.metrics import psnr, ssim
    from src.sh import eval_sh

Example:
    # Create model
    model = GaussianModel(50000, sh_degree=2, device='cuda')
    
    # Create camera
    cam = Camera(R, T, fx, fy, cx, cy, width, height)
    
    # Render
    image = render_gaussians(model, cam)
    
    # Compute loss
    loss = combined_loss(image, target)
"""

from .cameras import Camera
from .gaussians import GaussianModel, get_num_sh_coeffs
from .renderer import render_gaussians, render_depth
from .training import combined_loss, l1_loss, ssim_loss, psnr, densify_and_prune, save_checkpoint, load_checkpoint
from .metrics import evaluate_model
from .sh import eval_sh, rgb_to_sh, sh_to_rgb

__all__ = [
    # Camera
    'Camera',
    # Gaussian Model
    'GaussianModel', 'get_num_sh_coeffs',
    # Renderer
    'render_gaussians', 'render_depth',
    # Training
    'combined_loss', 'l1_loss', 'ssim_loss', 'psnr',
    'densify_and_prune', 'save_checkpoint', 'load_checkpoint',
    # Metrics
    'evaluate_model',
    # Spherical Harmonics
    'eval_sh', 'rgb_to_sh', 'sh_to_rgb',
]

__version__ = '1.0.0'
