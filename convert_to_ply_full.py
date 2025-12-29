# ============================================
# CONVERT 3DGS TO PLY FOR SUPERSPLAT (FULL SH)
# ============================================
import torch
import numpy as np
from pathlib import Path
import struct

def save_ply_full_sh(path, positions, scales, rotations, opacities, sh_dc, sh_rest):
    """Save Gaussians to PLY format with full SH coefficients for SuperSplat."""
    
    n = len(positions)
    
    # SuperSplat expects SH degree 3 = 16 coefficients per channel (48 total)
    # sh_dc: (N, 3) - DC component
    # sh_rest: (N, 15, 3) or similar - higher order terms
    
    # Build PLY header
    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
"""
    
    # Add all f_rest properties (45 for degree 3: 15 coeffs * 3 channels)
    for i in range(45):
        header += f"property float f_rest_{i}\n"
    
    header += """property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
    
    # Reshape sh_rest to (N, 45) - interleaved RGB
    if sh_rest.ndim == 3:
        # Shape is (N, num_coeffs, 3)
        num_rest_coeffs = sh_rest.shape[1]
        # Interleave: for each coeff, write R, G, B
        sh_rest_flat = np.zeros((n, 45), dtype=np.float32)
        for i in range(min(num_rest_coeffs, 15)):
            sh_rest_flat[:, i*3 + 0] = sh_rest[:, i, 0]  # R
            sh_rest_flat[:, i*3 + 1] = sh_rest[:, i, 1]  # G
            sh_rest_flat[:, i*3 + 2] = sh_rest[:, i, 2]  # B
    else:
        sh_rest_flat = np.zeros((n, 45), dtype=np.float32)
    
    print(f"📝 Writing PLY with {n:,} vertices...")
    
    with open(path, 'wb') as f:
        f.write(header.encode())
        
        for i in range(n):
            # Position (x, y, z)
            f.write(struct.pack('fff', float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])))
            # Normal (nx, ny, nz) - dummy values
            f.write(struct.pack('fff', 0.0, 0.0, 0.0))
            # SH DC (f_dc_0, f_dc_1, f_dc_2)
            f.write(struct.pack('fff', float(sh_dc[i, 0]), float(sh_dc[i, 1]), float(sh_dc[i, 2])))
            # SH rest (45 values)
            for j in range(45):
                f.write(struct.pack('f', float(sh_rest_flat[i, j])))
            # Opacity (raw, before sigmoid)
            f.write(struct.pack('f', float(opacities[i])))
            # Scale (log scale)
            f.write(struct.pack('fff', float(scales[i, 0]), float(scales[i, 1]), float(scales[i, 2])))
            # Rotation (quaternion w, x, y, z)
            f.write(struct.pack('ffff', float(rotations[i, 0]), float(rotations[i, 1]), float(rotations[i, 2]), float(rotations[i, 3])))
            
            if i % 100000 == 0 and i > 0:
                print(f"   Progress: {i:,}/{n:,}")
    
    print(f"✅ Saved PLY: {path}")
    print(f"   Vertices: {n:,}")
    print(f"   File size: {path.stat().st_size / 1024 / 1024:.1f} MB")


# ============================================
# LOAD YOUR TRAINED GAUSSIANS
# ============================================
data_dir = Path("C:/Users/rishi/Downloads/trained_gaussians/trained_gaussians/bonsai")

print(f"📂 Loading from: {data_dir}")

# Load all components
pos = torch.load(data_dir / "pos_7000.pt", map_location='cpu').detach().numpy()
scale_raw = torch.load(data_dir / "scale_raw_7000.pt", map_location='cpu').detach().numpy()
q_rot = torch.load(data_dir / "q_rot_7000.pt", map_location='cpu').detach().numpy()
opacity_raw = torch.load(data_dir / "opacity_raw_7000.pt", map_location='cpu').detach().numpy()
f_dc = torch.load(data_dir / "f_dc_7000.pt", map_location='cpu').detach().numpy()
f_rest = torch.load(data_dir / "f_rest_7000.pt", map_location='cpu').detach().numpy()

# Ensure correct shapes
opacity_raw = opacity_raw.squeeze()

print(f"✅ Loaded {len(pos):,} Gaussians")
print(f"   pos: {pos.shape}")
print(f"   scale: {scale_raw.shape}")
print(f"   rot: {q_rot.shape}")
print(f"   opacity: {opacity_raw.shape}")
print(f"   f_dc: {f_dc.shape}")
print(f"   f_rest: {f_rest.shape}")

# ============================================
# SAVE TO PLY (FULL SH FORMAT)
# ============================================
output_path = Path("C:/Users/rishi/Downloads/3dgs/bonsai_full_sh.ply")

save_ply_full_sh(
    output_path,
    positions=pos,
    scales=scale_raw,
    rotations=q_rot,
    opacities=opacity_raw,
    sh_dc=f_dc,
    sh_rest=f_rest
)

print(f"\n🎉 Done! Upload this file to SuperSplat:")
print(f"   {output_path}")
print(f"\n📌 Go to: https://playcanvas.com/supersplat")
print(f"   Click 'Open' → Select the .ply file")
print(f"   Switch to 'Splat Mode' for full rendering!")
