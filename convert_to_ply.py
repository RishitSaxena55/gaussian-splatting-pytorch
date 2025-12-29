# ============================================
# CONVERT 3DGS TO PLY FOR SUPERSPLAT
# ============================================
import torch
import numpy as np
from pathlib import Path
import struct

def save_ply_simple(path, positions, scales, rotations, opacities, sh_dc):
    """Save Gaussians to PLY format compatible with SuperSplat (simple version)."""
    
    n = len(positions)
    
    # Build PLY header (minimal for SuperSplat compatibility)
    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
"""
    
    with open(path, 'wb') as f:
        f.write(header.encode())
        
        for i in range(n):
            # Position (x, y, z)
            f.write(struct.pack('fff', float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])))
            # Scale (log scale)
            f.write(struct.pack('fff', float(scales[i, 0]), float(scales[i, 1]), float(scales[i, 2])))
            # Rotation (quaternion w, x, y, z)
            f.write(struct.pack('ffff', float(rotations[i, 0]), float(rotations[i, 1]), float(rotations[i, 2]), float(rotations[i, 3])))
            # Opacity (raw, before sigmoid)
            f.write(struct.pack('f', float(opacities[i])))
            # SH DC (f_dc_0, f_dc_1, f_dc_2)
            f.write(struct.pack('fff', float(sh_dc[i, 0]), float(sh_dc[i, 1]), float(sh_dc[i, 2])))
    
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

# Fix color convention (some implementations use different signs)
# Try negating or adjusting if colors look wrong
f_dc = -f_dc  # Negate to fix inverted colors

# Ensure correct shapes
opacity_raw = opacity_raw.squeeze()
print(f"✅ Loaded {len(pos):,} Gaussians")
print(f"   pos: {pos.shape}")
print(f"   scale: {scale_raw.shape}")
print(f"   rot: {q_rot.shape}")
print(f"   opacity: {opacity_raw.shape}")
print(f"   f_dc: {f_dc.shape}")

# ============================================
# SAVE TO PLY
# ============================================
output_path = Path("C:/Users/rishi/Downloads/3dgs/bonsai_7000.ply")

save_ply_simple(
    output_path,
    positions=pos,
    scales=scale_raw,
    rotations=q_rot,
    opacities=opacity_raw,
    sh_dc=f_dc
)

print(f"\n🎉 Done! Upload this file to SuperSplat:")
print(f"   {output_path}")
print(f"\n📌 Go to: https://playcanvas.com/supersplat")
print(f"   Click 'Open' → Select the .ply file")
