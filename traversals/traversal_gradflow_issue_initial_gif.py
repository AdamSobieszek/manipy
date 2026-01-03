# Two-panel animation (2D + 3D) with a 35° CCW rotation applied to the (x,y) coordinate frame.
# Saves to the user's original paths (if available) and also to /mnt/data for instant download here.
#
# Model:
#   f(x,y) = a(g(x,y)),   g(x,y) = x + exp(y)/3 + sin(y+0.95) + 0.2*(y+1)^2
#   a(z)   = -exp(-alpha*z),  alpha = 0.33/5  (strictly increasing)
# Dynamics: gradient ascent x' = ∇f(x,y)
# Particles start on a single level set (constant g0), y's are Gaussian spread.
#
# Rotation (display frame):
#   (x,y) simulate in original coords; for plotting use (x',y') = R(x,y), R = Rot(35° CCW).
#   Background in panels uses z = f(R^T(x',y')) so that the field is expressed in rotated coordinates.

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

# ---------------- Activation and pre-activation ----------------
alpha = 0.33 / 5.0

def activation(z):
    return -np.exp(-alpha * z)

def inv_activation(f):
    eps = 1e-12
    return -(1.0 / alpha) * np.log(np.clip(-f, eps, None))

def g_val_xy(x, y):
    return x + np.exp(y) / 3.0 + np.sin(y + 0.95) + 0.2 * (y + 1.0) ** 2

def g_baseline_of_y(y):
    return np.exp(y) / 3.0 + np.sin(y + 0.95) + 0.2 * (y + 1.0) ** 2

def f_val_xy(x, y):
    return activation(g_val_xy(x, y))

def f_val(xy):
    return f_val_xy(xy[..., 0], xy[..., 1])

def grad_f_xy(x, y):
    g = g_val_xy(x, y)
    a_prime = alpha * np.exp(-alpha * g)
    dfdx = a_prime
    dfdy = a_prime * (np.cos(y + 0.95) + 0.4 * (y + 1.0) + np.exp(y) / 3.0)
    return np.array([dfdx, dfdy])

# ---------------- Rotation utilities ----------------
theta = np.deg2rad(-15.0)  # 35° CCW
c, s = np.cos(theta), np.sin(theta)

def rot_xy_to_rotated(x, y):
    """(x,y) -> (x',y') = R (x,y)"""
    xp = c * x - s * y
    yp = s * x + c * y
    return xp, yp

def rot_rotated_to_xy(xp, yp):
    """(x',y') -> (x,y) = R^T (x',y')"""
    x =  c * xp + s * yp
    y = -s * xp + c * yp
    return x, y

# ---------------- Domains in rotated frame (for display) ----------------
xp_min, xp_max = -4.5, 6.5
yp_min, yp_max = -3.0, 5.0

# Right-panel heatmap grid in rotated coords
nx2d, ny2d = 180, 120
XP2, YP2 = np.meshgrid(np.linspace(xp_min, xp_max, nx2d),
                       np.linspace(yp_min, yp_max, ny2d))
# Evaluate f at the inverse-rotated original coordinates
X2, Y2 = rot_rotated_to_xy(XP2, YP2)
Z2 = f_val_xy(X2, Y2)

# Left-panel 3D surface grid in rotated coords
nx3d, ny3d = 50, 40
XP3, YP3 = np.meshgrid(np.linspace(xp_min, xp_max, nx3d),
                       np.linspace(yp_min, yp_max, ny3d))
X3, Y3 = rot_rotated_to_xy(XP3, YP3)
Z3 = f_val_xy(X3, Y3)

zmin, zmax = float(Z2.min()), float(Z2.max())
norm = Normalize(vmin=zmin, vmax=zmax)
cmap = cm.viridis

# ---------------- Simulation in original coords ----------------
dt = 0.95
n_steps = 50

n_particles = 70
rng = np.random.default_rng(21)
y_samples = np.sort(rng.normal(loc=-1.0, scale=0.9, size=900))
y_samples = y_samples[(y_samples >= -3.0 + 0.3) & (y_samples <= 1.0 - 0.5)] 
idx = np.linspace(0, len(y_samples) - 1, n_particles).astype(int)
ys0 = y_samples[idx]

g0 = -1.6  # initial common g-level => single initial f-level
xs0 = g0 - g_baseline_of_y(ys0)
pts0 = np.vstack([xs0, ys0]).T  # shape (N,2) in original coords

traj = [pts0.copy()]
for _ in range(n_steps):
    p = traj[-1]
    gx, gy = grad_f_xy(p[:, 0], p[:, 1])
    next_p = np.empty_like(p)
    next_p[:, 0] = p[:, 0] + dt * gx
    next_p[:, 1] = p[:, 1] + dt * gy
    traj.append(next_p)

final = traj[-1]
f_final = f_val(final)
median_f = float(np.median(f_final))
ref_idx = int(np.argmin(np.abs(f_final - median_f)))
f0 = float(f_val(pts0[0]))

# Parametric y-samples (original coords) for drawing level curves, then rotate for display
Y_line = np.linspace(-5.0, 5.0, 800)

# ---------------- Frames ----------------
frames = []
figsize = (7.0, 4.2)  # ~ 770x462 @ dpi=110
dpi = 110
x_min = xp_min *2.2
x_max = xp_max *2.2
for t_idx, pts in enumerate(traj):
    ref = pts[ref_idx]             # original coords
    f_ref = float(f_val(ref))
    g_ref = float(inv_activation(f_ref))
    g_ini = float(inv_activation(f0))

    # Level set curves in original coords: x = g - B(y)
    X_line_ref = g_ref - g_baseline_of_y(Y_line)
    X_line_ini = g_ini - g_baseline_of_y(Y_line)

    # Rotate curves for display
    XP_line_ref, YP_line_ref = rot_xy_to_rotated(X_line_ref, Y_line)
    XP_line_ini, YP_line_ini = rot_xy_to_rotated(X_line_ini, Y_line)

    # Rotate particle positions for display
    XP, YP = rot_xy_to_rotated(pts[:, 0], pts[:, 1])
    XP_ref, YP_ref = rot_xy_to_rotated(ref[0], ref[1])

    # 3D curves: z = constant
    Z_line_ref = np.full_like(Y_line, f_ref)
    Z_line_ini = np.full_like(Y_line, f0)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 2], wspace=0.0)#, hspace=0.001)

    # ----- Left: 3D surface in rotated coords -----
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    facecolors = cmap(norm(Z3))
    ax3d.plot_surface(XP3, YP3, Z3, rstride=2, cstride=2,
                      facecolors=facecolors, linewidth=0, antialiased=True, shade=False, alpha=0.7)

    mask_ref3 = (XP_line_ref >= xp_min) & (XP_line_ref <= xp_max) & (YP_line_ref >= yp_min) & (YP_line_ref <= yp_max)
    ax3d.plot(XP_line_ref[mask_ref3], YP_line_ref[mask_ref3], Z_line_ref[mask_ref3], color='black', linewidth=2.0)
    # Draw initial level-set as a persistent red guide under the moving black line
    mask_ini3 = (XP_line_ini >= xp_min) & (XP_line_ini <= xp_max) & (YP_line_ini >= yp_min) & (YP_line_ini <= yp_max)
    ax3d.plot(XP_line_ini[mask_ini3], YP_line_ini[mask_ini3], Z_line_ini[mask_ini3], color='red', linewidth=1.5, alpha=0.9)
    if t_idx == 0:
        ax3d.plot(XP_line_ini[mask_ini3], YP_line_ini[mask_ini3], Z_line_ini[mask_ini3], color='white', linestyle='--', linewidth=1.3)

    z_pts = f_val(pts)
    ax3d.scatter(XP, YP, z_pts, s=11, c='#ffd84d', depthshade=True)
    ax3d.scatter([XP_ref], [YP_ref], [f_ref], s=30, c='red', edgecolors='black', linewidths=0.6, depthshade=False)

    ax3d.set_xlim(xp_min, xp_max)
    ax3d.set_ylim(yp_min, yp_max)
    ax3d.set_zlim(zmin, zmax)
    ax3d.set_xlabel("x'", labelpad=0)
    ax3d.set_ylabel("y'", labelpad=0)
    # ax3d.set_zlabel('z=f(x,y)', labelpad=4)
    ax3d.view_init(elev=8, azim=220)
    ax3d.set_box_aspect((1, 1, 1.7))

    # ----- Right: 2D panel in rotated coords -----
    ax2d = fig.add_subplot(gs[0, 1])
    ax2d.imshow(Z2, extent=[xp_min, xp_max, yp_min, yp_max], origin='lower', cmap=cmap, aspect='auto')

    # Draw initial level-set as a persistent red guide under the moving black line
    mask_ini2 = (XP_line_ini >= xp_min) & (XP_line_ini <= xp_max) & (YP_line_ini >= yp_min) & (YP_line_ini <= yp_max)
    ax2d.plot(XP_line_ini[mask_ini2], YP_line_ini[mask_ini2], color='red', linewidth=1.5, alpha=0.9, zorder=2)
    mask_ref2 = (XP_line_ref >= xp_min) & (XP_line_ref <= xp_max) & (YP_line_ref >= yp_min) & (YP_line_ref <= yp_max)
    ax2d.plot(XP_line_ref[mask_ref2], YP_line_ref[mask_ref2], color='black', linewidth=2.0)
    if t_idx == 0:
        ax2d.plot(XP_line_ini[mask_ini2], YP_line_ini[mask_ini2], color='white', linestyle='--', linewidth=1.3)

    ax2d.scatter(XP, YP, s=10, c='#ffd84d', edgecolors='none', alpha=0.95, zorder=3)
    ax2d.scatter([XP_ref], [YP_ref], s=36, c='red', edgecolors='black', linewidth=0.6, zorder=4)
    ax2d.set_xlim(xp_min, xp_max)
    ax2d.set_ylim(yp_min, yp_max)
    ax2d.set_aspect('equal', adjustable='box')
    ax2d.set_xlabel("x'")
    ax2d.set_ylabel("y'")
    fig.suptitle("Level set drift under gradient flow", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(image)
    if t_idx == 0:
        for i in range(10):
            frames.append(image)
    plt.close(fig)

# ---------------- Save files ----------------
mp4_path = "/Users/adamsobieszek/PycharmProjects/_manipy/traversals/traversal_gradient_flow_2panel2.mp4"
webm_path = "/Users/adamsobieszek/PycharmProjects/_manipy/traversals/traversal_gradient_flow_2panel2.webm"  # fallback
gif_path = "/Users/adamsobieszek/PycharmProjects/_manipy/traversals/traversal_gradient_flow_2panel2.gif"

# Prefer widely compatible MP4 (H.264). Fallback to WEBM if MP4 fails.
video_path = mp4_path
video_ok = True
try:
    writer = imageio.get_writer(mp4_path, fps=18, codec='libx264')
    for fr in frames:
        fr_rgb = fr[..., :3] if fr.ndim == 3 and fr.shape[-1] == 4 else fr
        writer.append_data(fr_rgb)
    writer.close()
except Exception:
    video_ok = False

if not video_ok:
    try:
        video_path = webm_path
        writer = imageio.get_writer(webm_path, fps=18, codec='libvpx-vp9', quality=7)
        for fr in frames:
            fr_rgb = fr[..., :3] if fr.ndim == 3 and fr.shape[-1] == 4 else fr
            writer.append_data(fr_rgb)
        writer.close()
        video_ok = True
    except Exception:
        video_ok = False

# Also save GIF
try:
    durations = [1/12.0] * (len(frames) - 1) + [1/24.0]
    imageio.mimsave(gif_path, frames, duration=durations, loop=0)
except Exception:
    pass


(video_path, gif_path, video_ok)
