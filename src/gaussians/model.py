"""
3DGS Gaussian Model - Complete Gaussian representation with covariance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..sh import eval_sh


def get_num_sh_coeffs(degree):
    """Get number of SH coefficients for given degree."""
    return (degree + 1) ** 2


class GaussianModel:
    """
    3D Gaussian Splatting Model.
    
    Each Gaussian has:
        - Position (xyz): 3D location
        - Scale: Anisotropic 3D scales (log-space)
        - Rotation: Quaternion orientation
        - Opacity: Transparency (logit-space)
        - SH coefficients: View-dependent colors
    
    Args:
        num_gaussians: Initial number of Gaussians
        sh_degree: Spherical harmonics degree (0, 1, or 2)
        device: 'cuda' or 'cpu'
    
    Example:
        >>> model = GaussianModel(50000, sh_degree=2)
        >>> colors = model.get_colors(view_dirs)
        >>> cov3d = model.get_covariance_3d()
    """
    def __init__(self, num_gaussians, sh_degree=2, device='cuda'):
        self.device = device
        self.sh_degree = sh_degree
        num_sh = get_num_sh_coeffs(sh_degree)
        
        # Learnable parameters
        self._xyz = nn.Parameter(torch.randn(num_gaussians, 3, device=device) * 2)
        self._log_scale = nn.Parameter(torch.ones(num_gaussians, 3, device=device) * -3)
        self._rotation = nn.Parameter(torch.zeros(num_gaussians, 4, device=device))
        self._rotation.data[:, 0] = 1.0  # Initialize as identity quaternion
        self._opacity = nn.Parameter(torch.zeros(num_gaussians, 1, device=device))
        self._sh = nn.Parameter(torch.zeros(num_gaussians, num_sh, 3, device=device))
        self._sh.data[:, 0, :] = 0.5  # Gray base color
        
        # Gradient accumulators for densification
        self.xyz_grad_accum = torch.zeros(num_gaussians, device=device)
        self.denom = torch.zeros(num_gaussians, device=device)
    
    @property
    def xyz(self):
        """Get 3D positions."""
        return self._xyz
    
    @property
    def scales(self):
        """Get scales (exp of log_scale)."""
        return torch.exp(self._log_scale)
    
    @property
    def rotations(self):
        """Get normalized quaternions."""
        return F.normalize(self._rotation, dim=-1)
    
    @property
    def opacity(self):
        """Get opacity (sigmoid activated)."""
        return torch.sigmoid(self._opacity)
    
    def get_colors(self, view_dirs):
        """
        Evaluate SH to get view-dependent colors.
        
        Args:
            view_dirs: (N, 3) Normalized view directions
            
        Returns:
            (N, 3) RGB colors clamped to [0, 1]
        """
        return torch.clamp(eval_sh(self.sh_degree, self._sh, view_dirs), 0, 1)
    
    def get_covariance_3d(self):
        """
        Compute 3D covariance matrices from scale and rotation.
        
        Returns:
            (N, 3, 3) Covariance matrices
        """
        S = torch.diag_embed(self.scales)
        q = self.rotations
        r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        R = torch.stack([
            torch.stack([1-2*(y*y+z*z), 2*(x*y-r*z), 2*(x*z+r*y)], dim=-1),
            torch.stack([2*(x*y+r*z), 1-2*(x*x+z*z), 2*(y*z-r*x)], dim=-1),
            torch.stack([2*(x*z-r*y), 2*(y*z+r*x), 1-2*(x*x+y*y)], dim=-1),
        ], dim=1)
        
        M = R @ S
        return M @ M.transpose(-1, -2)
    
    def get_covariance_2d(self, camera, xyz_cam):
        """
        Project 3D covariance to 2D screen space.
        
        Args:
            camera: Camera object
            xyz_cam: (N, 3) Points in camera space
            
        Returns:
            (N, 2, 2) 2D covariance matrices
        """
        cov3d = self.get_covariance_3d()
        J = camera.get_projection_jacobian(xyz_cam)
        W = camera.R.unsqueeze(0)
        T = J @ W
        cov2d = T @ cov3d @ T.transpose(-1, -2)
        
        # Add anti-aliasing
        cov2d[:, 0, 0] += 0.3
        cov2d[:, 1, 1] += 0.3
        return cov2d
    
    def __len__(self):
        return self._xyz.shape[0]
    
    def init_from_points(self, points, colors=None):
        """
        Initialize Gaussians from 3D points (e.g., from COLMAP).
        
        Args:
            points: (N, 3) 3D point positions
            colors: (N, 3) Optional RGB colors
        """
        n = len(points)
        num_sh = get_num_sh_coeffs(self.sh_degree)
        
        self._xyz = nn.Parameter(torch.tensor(points, dtype=torch.float32, device=self.device))
        self._log_scale = nn.Parameter(torch.ones(n, 3, device=self.device) * -3)
        self._rotation = nn.Parameter(torch.zeros(n, 4, device=self.device))
        self._rotation.data[:, 0] = 1.0
        self._opacity = nn.Parameter(torch.zeros(n, 1, device=self.device))
        self._sh = nn.Parameter(torch.zeros(n, num_sh, 3, device=self.device))
        
        if colors is not None:
            self._sh.data[:, 0, :] = torch.tensor(colors, dtype=torch.float32, device=self.device) - 0.5
        else:
            self._sh.data[:, 0, :] = 0.5
        
        self.xyz_grad_accum = torch.zeros(n, device=self.device)
        self.denom = torch.zeros(n, device=self.device)
    
    def get_parameters(self):
        """Get all learnable parameters as a list."""
        return [self._xyz, self._log_scale, self._rotation, self._opacity, self._sh]
