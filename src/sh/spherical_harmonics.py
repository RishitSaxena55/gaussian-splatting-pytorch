"""
3DGS Spherical Harmonics - View-dependent color encoding.
"""
import torch

# SH basis coefficients
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252,
    -1.0925484305920792,
    0.5462742152960396
]


def eval_sh(degree, sh_coeffs, directions):
    """
    Evaluate spherical harmonics for view-dependent colors.
    
    Args:
        degree: SH degree (0, 1, or 2)
        sh_coeffs: (N, num_coeffs, 3) SH coefficients per Gaussian
        directions: (N, 3) Normalized view directions
        
    Returns:
        (N, 3) RGB colors (before clamping)
    
    Example:
        >>> colors = eval_sh(2, model._sh, view_dirs)
        >>> colors = torch.clamp(colors, 0, 1)
    """
    # Degree 0 (constant/DC term)
    result = C0 * sh_coeffs[:, 0]
    
    if degree >= 1 and sh_coeffs.shape[1] > 1:
        x = directions[:, 0:1]
        y = directions[:, 1:2]
        z = directions[:, 2:3]
        
        # Degree 1
        result = result + C1 * (
            -y * sh_coeffs[:, 1] + 
            z * sh_coeffs[:, 2] - 
            x * sh_coeffs[:, 3]
        )
        
        if degree >= 2 and sh_coeffs.shape[1] > 4:
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            
            # Degree 2
            result = result + C2[0] * xy * sh_coeffs[:, 4]
            result = result + C2[1] * yz * sh_coeffs[:, 5]
            result = result + C2[2] * (3*zz - 1) * sh_coeffs[:, 6]
            result = result + C2[3] * xz * sh_coeffs[:, 7]
            result = result + C2[4] * (xx - yy) * sh_coeffs[:, 8]
    
    return result + 0.5  # Offset to center around gray


def rgb_to_sh(rgb):
    """
    Convert RGB colors to DC (degree 0) SH coefficient.
    
    Args:
        rgb: (N, 3) RGB colors in [0, 1]
        
    Returns:
        (N, 3) DC SH coefficient
    """
    return (rgb - 0.5) / C0


def sh_to_rgb(sh_dc):
    """
    Convert DC SH coefficient to RGB.
    
    Args:
        sh_dc: (N, 3) DC SH coefficient
        
    Returns:
        (N, 3) RGB colors
    """
    return torch.clamp(C0 * sh_dc + 0.5, 0, 1)
