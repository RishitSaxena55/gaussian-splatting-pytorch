"""
3DGS Training Utilities - Loss functions, densification, and checkpointing.
"""
import torch
import torch.nn as nn


def l1_loss(pred, target):
    """L1 (Mean Absolute Error) loss."""
    return (pred - target).abs().mean()


def ssim_loss(pred, target):
    """
    Simplified SSIM loss (1 - SSIM).
    
    Args:
        pred: Predicted image
        target: Ground truth image
        
    Returns:
        1 - SSIM scalar
    """
    C1, C2 = 0.01**2, 0.03**2
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = ((pred - mu_x)**2).mean()
    sigma_y = ((target - mu_y)**2).mean()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    
    ssim = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return 1 - ssim


def combined_loss(pred, target, lambda_ssim=0.2):
    """
    Combined L1 + SSIM loss as in 3DGS paper.
    
    Args:
        pred: Predicted image
        target: Ground truth image
        lambda_ssim: Weight for SSIM loss (default 0.2)
        
    Returns:
        Combined loss scalar
    """
    l1 = l1_loss(pred, target)
    ssim = ssim_loss(pred, target)
    return (1 - lambda_ssim) * l1 + lambda_ssim * ssim


def psnr(pred, target):
    """
    Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image
        target: Ground truth image
        
    Returns:
        PSNR in dB
    """
    mse = ((pred - target) ** 2).mean()
    return 10 * torch.log10(1.0 / mse)


def densify_and_prune(model, config, device='cuda'):
    """
    Adaptive density control: clone, split, and prune Gaussians.
    
    Args:
        model: GaussianModel instance
        config: Dict with densify_grad_thresh and prune_opacity_thresh
        device: Device string
        
    Returns:
        New number of Gaussians
    """
    grad_avg = model.xyz_grad_accum / (model.denom + 1e-6)
    
    needs_densify = grad_avg > config['densify_grad_thresh']
    is_small = model.scales.max(dim=-1).values < 0.01
    
    # Clone small Gaussians with high gradients
    to_clone = needs_densify & is_small
    if to_clone.sum() > 0:
        idx = to_clone.nonzero(as_tuple=True)[0]
        model._xyz = nn.Parameter(torch.cat([
            model._xyz.data,
            model._xyz.data[idx] + torch.randn_like(model._xyz.data[idx]) * 0.01
        ]))
        model._log_scale = nn.Parameter(torch.cat([model._log_scale.data, model._log_scale.data[idx]]))
        model._rotation = nn.Parameter(torch.cat([model._rotation.data, model._rotation.data[idx]]))
        model._opacity = nn.Parameter(torch.cat([model._opacity.data, model._opacity.data[idx]]))
        model._sh = nn.Parameter(torch.cat([model._sh.data, model._sh.data[idx]]))
    
    # Split large Gaussians with high gradients
    to_split = needs_densify & ~is_small
    if to_split.sum() > 0:
        idx = to_split.nonzero(as_tuple=True)[0]
        n = len(idx)
        scales = model.scales[idx]
        offset = scales * 0.5
        
        new_xyz1 = model._xyz.data[idx] + torch.randn(n, 3, device=device) * offset
        new_xyz2 = model._xyz.data[idx] - torch.randn(n, 3, device=device) * offset
        new_scale = model._log_scale.data[idx] - 0.5
        
        model._xyz = nn.Parameter(torch.cat([model._xyz.data, new_xyz1, new_xyz2]))
        model._log_scale = nn.Parameter(torch.cat([model._log_scale.data, new_scale, new_scale]))
        model._rotation = nn.Parameter(torch.cat([
            model._rotation.data, model._rotation.data[idx], model._rotation.data[idx]
        ]))
        model._opacity = nn.Parameter(torch.cat([
            model._opacity.data, model._opacity.data[idx], model._opacity.data[idx]
        ]))
        model._sh = nn.Parameter(torch.cat([
            model._sh.data, model._sh.data[idx], model._sh.data[idx]
        ]))
    
    # Prune low-opacity Gaussians
    keep = model.opacity.squeeze() > config['prune_opacity_thresh']
    if not keep.all():
        model._xyz = nn.Parameter(model._xyz.data[keep])
        model._log_scale = nn.Parameter(model._log_scale.data[keep])
        model._rotation = nn.Parameter(model._rotation.data[keep])
        model._opacity = nn.Parameter(model._opacity.data[keep])
        model._sh = nn.Parameter(model._sh.data[keep])
    
    # Reset accumulators
    n = len(model)
    model.xyz_grad_accum = torch.zeros(n, device=device)
    model.denom = torch.zeros(n, device=device)
    
    return n


def save_checkpoint(model, iteration, loss, path):
    """
    Save model checkpoint.
    
    Args:
        model: GaussianModel instance
        iteration: Current training iteration
        loss: Current loss value
        path: File path to save to
    """
    torch.save({
        'iteration': iteration,
        'loss': loss,
        'num_gaussians': len(model),
        'xyz': model._xyz.data,
        'log_scale': model._log_scale.data,
        'rotation': model._rotation.data,
        'opacity': model._opacity.data,
        'sh': model._sh.data,
        'xyz_grad_accum': model.xyz_grad_accum,
        'denom': model.denom,
    }, path)


def load_checkpoint(path, model_class, sh_degree=2, device='cuda'):
    """
    Load model from checkpoint.
    
    Args:
        path: Checkpoint file path
        model_class: GaussianModel class
        sh_degree: SH degree for model
        device: Device string
        
    Returns:
        (model, iteration) tuple
    """
    ckpt = torch.load(path, map_location=device)
    model = model_class(ckpt['num_gaussians'], sh_degree=sh_degree, device=device)
    
    model._xyz = nn.Parameter(ckpt['xyz'].to(device))
    model._log_scale = nn.Parameter(ckpt['log_scale'].to(device))
    model._rotation = nn.Parameter(ckpt['rotation'].to(device))
    model._opacity = nn.Parameter(ckpt['opacity'].to(device))
    model._sh = nn.Parameter(ckpt['sh'].to(device))
    
    if 'xyz_grad_accum' in ckpt:
        model.xyz_grad_accum = ckpt['xyz_grad_accum'].to(device)
        model.denom = ckpt['denom'].to(device)
    else:
        model.xyz_grad_accum = torch.zeros(len(model), device=device)
        model.denom = torch.zeros(len(model), device=device)
    
    return model, ckpt['iteration']
