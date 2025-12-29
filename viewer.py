# ============================================
# 3D GAUSSIAN VIEWER
# Interactive visualization of trained Gaussians
# ============================================
import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Path to trained Gaussians (update if needed)
data_dir = Path("C:/Users/rishi/Downloads/trained_gaussians/trained_gaussians/bonsai")

print(f"📂 Loading from: {data_dir}")

# Load Gaussian parameters
pos = torch.load(data_dir / "pos_7000.pt", map_location='cpu').detach().numpy()
opacity_raw = torch.load(data_dir / "opacity_raw_7000.pt", map_location='cpu')
opacity = torch.sigmoid(opacity_raw).detach().numpy().squeeze()
f_dc = torch.load(data_dir / "f_dc_7000.pt", map_location='cpu').detach().numpy()
scale_raw = torch.load(data_dir / "scale_raw_7000.pt", map_location='cpu')
scale = torch.exp(scale_raw).detach().numpy()

# Convert SH DC component to RGB
C0 = 0.28209479177387814
colors = np.clip(f_dc * C0 + 0.5, 0, 1)

print(f"✅ Loaded {len(pos)} Gaussians")
print(f"   Position range: [{pos.min():.2f}, {pos.max():.2f}]")
print(f"   Opacity range: [{opacity.min():.3f}, {opacity.max():.3f}]")
print(f"   Scale range: [{scale.min():.4f}, {scale.max():.4f}]")

# Filter by opacity (keep visible ones)
opacity_threshold = 0.1
visible = opacity > opacity_threshold
pos_v = pos[visible]
colors_v = colors[visible]
opacity_v = opacity[visible]
scale_v = scale[visible]

print(f"   Visible (opacity > {opacity_threshold}): {visible.sum()} Gaussians")

# Subsample if too many points
max_points = 100000
if len(pos_v) > max_points:
    idx = np.random.choice(len(pos_v), max_points, replace=False)
    pos_v = pos_v[idx]
    colors_v = colors_v[idx]
    opacity_v = opacity_v[idx]
    print(f"   Subsampled to {max_points} points for visualization")

# Create color strings for Plotly
color_strs = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors_v]

# Create interactive 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=pos_v[:, 0],
    y=pos_v[:, 1],
    z=pos_v[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        color=color_strs,
        opacity=0.8,
    ),
    hovertemplate='<b>Position</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
)])

fig.update_layout(
    title=f'3D Gaussian Splatting - Bonsai ({len(pos_v):,} points)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data',
        camera=dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=1200,
    height=900,
    margin=dict(l=0, r=0, t=40, b=0)
)

# Save as interactive HTML
output_path = Path(__file__).parent / "bonsai_viewer.html"
fig.write_html(str(output_path))
print(f"\n🎉 Saved to: {output_path}")
print(f"   Open this file in your browser for interactive 3D view!")

# Optionally show in browser
import webbrowser
webbrowser.open(str(output_path))
