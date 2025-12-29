"""
3DGS Metrics - Evaluation metrics for image quality.
"""
import torch


def psnr(pred, target):
    """
    Peak Signal-to-Noise Ratio.
    
    Higher is better. Typical values: 20-40 dB.
    
    Args:
        pred: Predicted image tensor
        target: Ground truth image tensor
        
    Returns:
        PSNR in dB (scalar)
    """
    mse = ((pred - target) ** 2).mean()
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(1.0 / mse)


def mse(pred, target):
    """Mean Squared Error."""
    return ((pred - target) ** 2).mean()


def mae(pred, target):
    """Mean Absolute Error (L1)."""
    return (pred - target).abs().mean()


def ssim(pred, target):
    """
    Simplified Structural Similarity Index.
    
    Higher is better. Range: [-1, 1], typically [0, 1].
    
    Args:
        pred: Predicted image
        target: Ground truth image
        
    Returns:
        SSIM value (scalar)
    """
    C1, C2 = 0.01**2, 0.03**2
    
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = ((pred - mu_x)**2).mean()
    sigma_y = ((target - mu_y)**2).mean()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    
    ssim_val = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_val


def evaluate_model(model, cameras, images, render_fn):
    """
    Evaluate model on all views.
    
    Args:
        model: GaussianModel instance
        cameras: List of Camera instances
        images: List of ground truth images
        render_fn: Render function (model, camera) -> image
        
    Returns:
        Dict with mean PSNR, SSIM, and per-view metrics
    """
    psnrs, ssims = [], []
    
    with torch.no_grad():
        for cam, gt in zip(cameras, images):
            rendered = render_fn(model, cam)
            psnrs.append(psnr(rendered, gt).item())
            ssims.append(ssim(rendered, gt).item())
    
    return {
        'psnr_mean': sum(psnrs) / len(psnrs),
        'psnr_std': torch.tensor(psnrs).std().item(),
        'ssim_mean': sum(ssims) / len(ssims),
        'ssim_std': torch.tensor(ssims).std().item(),
        'psnr_per_view': psnrs,
        'ssim_per_view': ssims,
    }
