"""
3DGS Camera Module - Pinhole camera with projection and Jacobians.
"""
import torch
import torch.nn.functional as F


class Camera:
    """
    Pinhole Camera with projection, view directions, and Jacobians.
    
    Args:
        R: Rotation matrix (3x3)
        T: Translation/position vector (3,)
        fx, fy: Focal lengths
        cx, cy: Principal point
        width, height: Image dimensions
    
    Example:
        >>> cam = Camera(R, T, 500, 500, 400, 300, 800, 600)
        >>> u, v, xyz_cam = cam.project(points)  # Project 3D points to 2D
    """
    def __init__(self, R, T, fx, fy, cx, cy, width, height, device='cuda'):
        self.device = device
        self.R = R.float().to(device)
        self.T = T.float().to(device)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)
        self.near = 0.01
        self.far = 100.0
    
    def project(self, xyz):
        """
        Project 3D world points to 2D pixel coordinates.
        
        Args:
            xyz: (N, 3) 3D points in world space
            
        Returns:
            u, v: (N,) Pixel coordinates
            xyz_cam: (N, 3) Points in camera space
        """
        t = -self.R @ self.T
        xyz_cam = (self.R @ xyz.T).T + t
        z = torch.clamp(xyz_cam[:, 2], min=self.near)
        u = self.fx * xyz_cam[:, 0] / z + self.cx
        v = self.fy * xyz_cam[:, 1] / z + self.cy
        return u, v, xyz_cam
    
    def get_view_dir(self, xyz):
        """
        Get normalized view direction from camera to 3D points.
        
        Args:
            xyz: (N, 3) 3D points
            
        Returns:
            (N, 3) Normalized direction vectors
        """
        return F.normalize(xyz - self.T.unsqueeze(0), dim=-1)
    
    def get_projection_jacobian(self, xyz_cam):
        """
        Compute Jacobian of projection for EWA splatting.
        
        Args:
            xyz_cam: (N, 3) Points in camera space
            
        Returns:
            J: (N, 2, 3) Jacobian matrices
        """
        z = torch.clamp(xyz_cam[:, 2:3], min=self.near)
        z_sq = z.squeeze() ** 2
        
        J = torch.zeros(xyz_cam.shape[0], 2, 3, device=self.device)
        J[:, 0, 0] = self.fx / z.squeeze()
        J[:, 0, 2] = -self.fx * xyz_cam[:, 0] / z_sq
        J[:, 1, 1] = self.fy / z.squeeze()
        J[:, 1, 2] = -self.fy * xyz_cam[:, 1] / z_sq
        return J
    
    def get_intrinsic_matrix(self):
        """Return 3x3 intrinsic matrix K."""
        return torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
    
    def get_extrinsic_matrix(self):
        """Return 4x4 extrinsic matrix [R|t]."""
        E = torch.eye(4, device=self.device)
        E[:3, :3] = self.R
        E[:3, 3] = -self.R @ self.T
        return E
