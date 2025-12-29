"""
3DGS Renderer - Educational differentiable point-based renderer.
"""
import torch


def render_gaussians(model, camera):
    """
    Render Gaussians to an image using point-based splatting.
    
    This is an educational renderer that draws each Gaussian as a single pixel.
    For production quality, use gsplat or similar CUDA rasterizers.
    
    Args:
        model: GaussianModel instance
        camera: Camera instance
        
    Returns:
        (H, W, 3) Rendered RGB image
    
    Example:
        >>> image = render_gaussians(model, camera)
        >>> plt.imshow(image.cpu().numpy())
    """
    device = model.device
    width, height = camera.width, camera.height
    
    # Project Gaussians to screen
    u, v, xyz_cam = camera.project(model.xyz)
    
    # Get view-dependent colors
    view_dirs = camera.get_view_dir(model.xyz)
    colors = model.get_colors(view_dirs)
    opacity = model.opacity.squeeze(-1)
    
    # Compute 2D covariance for radius estimation
    cov2d = model.get_covariance_2d(camera, xyz_cam)
    radius = 3 * torch.sqrt(torch.max(cov2d[:, 0, 0], cov2d[:, 1, 1]))
    
    # Filter valid Gaussians
    valid = (xyz_cam[:, 2] > 0.01) & \
            (u > -radius) & (u < width + radius) & \
            (v > -radius) & (v < height + radius)
    
    u_valid = u[valid]
    v_valid = v[valid]
    colors_valid = colors[valid]
    opacity_valid = opacity[valid]
    
    # Get pixel indices
    pixel_idx = (v_valid.long().clamp(0, height-1) * width + 
                 u_valid.long().clamp(0, width-1))
    
    # Accumulate colors and weights
    weights = opacity_valid
    weighted_colors = colors_valid * weights.unsqueeze(-1)
    
    accumulated_color = torch.zeros(height * width, 3, device=device)
    accumulated_weight = torch.zeros(height * width, device=device)
    
    accumulated_color.scatter_add_(0, pixel_idx.unsqueeze(-1).expand(-1, 3), weighted_colors)
    accumulated_weight.scatter_add_(0, pixel_idx, weights)
    
    # Blend with white background
    accumulated_weight = accumulated_weight.unsqueeze(-1).clamp(min=1e-6, max=1.0)
    image = torch.ones(height * width, 3, device=device)
    image = image * (1 - accumulated_weight) + \
            accumulated_color / accumulated_weight.clamp(min=1e-6) * accumulated_weight
    
    return torch.clamp(image.view(height, width, 3), 0, 1)


def render_depth(model, camera):
    """
    Render depth map from Gaussians.
    
    Args:
        model: GaussianModel instance
        camera: Camera instance
        
    Returns:
        (H, W) Depth map
    """
    device = model.device
    width, height = camera.width, camera.height
    
    u, v, xyz_cam = camera.project(model.xyz)
    depth = xyz_cam[:, 2]
    opacity = model.opacity.squeeze(-1)
    
    valid = (depth > 0.01) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    
    pixel_idx = (v.long().clamp(0, height-1) * width + u.long().clamp(0, width-1))
    
    weights = opacity * valid.float()
    weighted_depth = depth * weights
    
    accumulated_depth = torch.zeros(height * width, device=device)
    accumulated_weight = torch.zeros(height * width, device=device)
    
    accumulated_depth.scatter_add_(0, pixel_idx, weighted_depth)
    accumulated_weight.scatter_add_(0, pixel_idx, weights)
    
    depth_map = accumulated_depth / accumulated_weight.clamp(min=1e-6)
    
    return depth_map.view(height, width)
