"""
Two-panel animation (2D + 3D) with a 35° CCW rotation applied to the (x,y) coordinate frame.
Saves to the user's original paths (if available) and also to /mnt/data for instant download here.

Model:
  f(x,y) = a(g(x,y)),   g(x,y) = x + exp(y)/3 + sin(y+0.95) + 0.2*(y+1)^2
  a(z)   = -exp(-alpha*z),  alpha = 0.33/5  (strictly increasing)
Dynamics: (choose one)
  - Gradient flow:   x' = ∇f(x,y)
  - Traversal (canonical): x' = ∇f / |∇f|^2  (so df/dt = 1)

Particles start on a single level set (constant g0); y's are Gaussian spread.

Rotation (display frame):
  (x,y) simulate in original coords; for plotting use (x',y') = R(x,y), R = Rot(35° CCW).
  Background in panels uses z = f(R^T(x',y')) so that the field is expressed in rotated coordinates.

MP4 ENCODING FIX:
  We force H.264 + yuv420p with ffmpeg via imageio and add '+faststart' for broad compatibility.
  If H.264 fails, we fall back to VP9 WebM; GIF is always attempted.
"""
from __future__ import annotations
import os
from typing import List, Tuple, Dict, Literal

import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

# ---------------- Activation and pre-activation ----------------
alpha = 0.33 / 5.0

def activation(z: np.ndarray) -> np.ndarray:
    return -np.exp(-alpha * z)

def inv_activation(f: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -(1.0 / alpha) * np.log(np.clip(-f, eps, None))

def g_val_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + np.exp(y) / 3.0 + np.sin(y + 0.95) + 0.2 * (y + 1.0) ** 2

def g_baseline_of_y(y: np.ndarray) -> np.ndarray:
    return np.exp(y) / 3.0 + np.sin(y + 0.95) + 0.2 * (y + 1.0) ** 2

def f_val_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return activation(g_val_xy(x, y))

def f_val(xy: np.ndarray) -> np.ndarray:
    return f_val_xy(xy[..., 0], xy[..., 1])

def grad_f_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return gradient as array shape (2, N) for vectorized x,y (N,)."""
    g = g_val_xy(x, y)
    # a'(g) = alpha * exp(-alpha*g)
    a_prime = alpha * np.exp(-alpha * g)
    dfdx = a_prime
    dfdy = a_prime * (np.cos(y + 0.95) + 0.4 * (y + 1.0) + np.exp(y) / 3.0)
    return np.array([dfdx, dfdy])

# ---------------- Rotation utilities ----------------
# 35° CCW, consistent with the description.
_theta_deg = 35.0
_theta = np.deg2rad(_theta_deg)
_c, _s = np.cos(_theta), np.sin(_theta)

def rot_xy_to_rotated(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(x,y) -> (x',y') = R(x,y) with 35° CCW."""
    xp = _c * x - _s * y
    yp = _s * x + _c * y
    return xp, yp

def rot_rotated_to_xy(xp: np.ndarray, yp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(x',y') -> (x,y) = R^T(x',y') for 35° CCW."""
    x =  _c * xp + _s * yp
    y = -_s * xp + _c * yp
    return x, y

# ---------------- Display grids & colormap ----------------

def make_display_grids(
    xp_min: float = -4.5,
    xp_max: float = 6.5,
    yp_min: float = -3.0,
    yp_max: float = 5.0,
    nx2d: int = 180,
    ny2d: int = 120,
    nx3d: int = 50,
    ny3d: int = 40,
) -> Dict[str, np.ndarray]:
    """Precompute background fields on a rotated grid for both panels."""
    XP2, YP2 = np.meshgrid(np.linspace(xp_min, xp_max, nx2d),
                           np.linspace(yp_min, yp_max, ny2d))
    X2, Y2 = rot_rotated_to_xy(XP2, YP2)
    Z2 = f_val_xy(X2, Y2)

    XP3, YP3 = np.meshgrid(np.linspace(xp_min, xp_max, nx3d),
                           np.linspace(yp_min, yp_max, ny3d))
    X3, Y3 = rot_rotated_to_xy(XP3, YP3)
    Z3 = f_val_xy(X3, Y3)

    zmin, zmax = float(Z2.min()), float(Z2.max())
    norm = Normalize(vmin=zmin, vmax=zmax)

    return dict(
        XP2=XP2, YP2=YP2, Z2=Z2,
        XP3=XP3, YP3=YP3, Z3=Z3,
        xp_min=xp_min, xp_max=xp_max, yp_min=yp_min, yp_max=yp_max,
        zmin=zmin, zmax=zmax, norm=norm, cmap=cm.viridis,
    )

# ---------------- Particle initialization ----------------

def init_particles(
    n_particles: int = 70,
    rng_seed: int = 21,
    y_mu: float = -1.0,
    y_sigma: float = 0.9,
    y_clip: Tuple[float, float] = (-3.0 + 0.3, 1.0 - 0.5),
    g0: float = -1.6,
) -> Tuple[np.ndarray, float]:
    """Return initial (N,2) pts in original coords and f0 value."""
    rng = np.random.default_rng(rng_seed)
    y_samples = np.sort(rng.normal(loc=y_mu, scale=y_sigma, size=900))
    y_samples = y_samples[(y_samples >= y_clip[0]) & (y_samples <= y_clip[1])]
    idx = np.linspace(0, len(y_samples) - 1, n_particles).astype(int)
    ys0 = y_samples[idx]

    xs0 = g0 - g_baseline_of_y(ys0)  # initial common g-level => single initial f-level
    pts0 = np.vstack([xs0, ys0]).T  # (N,2)
    f0 = float(f_val(pts0[:1])[0])
    return pts0, f0

# ---------------- Simulation ----------------
DynMode = Literal['gradient', 'traversal']

def simulate(
    pts0: np.ndarray,
    n_steps: int = 50,
    dt: float = 0.95,
    mode: DynMode = 'gradient',
    eps: float = 1e-9,
) -> List[np.ndarray]:
    """Simulate either gradient flow (x' = ∇f) or canonical traversal (x' = ∇f/|∇f|^2)."""
    traj = [pts0.copy()]
    for _ in range(n_steps):
        p = traj[-1]
        gx, gy = grad_f_xy(p[:, 0], p[:, 1])  # (N,), (N,)
        if mode == 'gradient':
            vx, vy = gx, gy
        elif mode == 'traversal':
            g2 = gx * gx + gy * gy
            inv_g2 = 1.0 / (g2 + eps)
            vx, vy = gx * inv_g2, gy * inv_g2
        else:
            raise ValueError(f"Unknown mode: {mode}")
        next_p = np.empty_like(p)
        next_p[:, 0] = p[:, 0] + dt * vx
        next_p[:, 1] = p[:, 1] + dt * vy
        traj.append(next_p)
    return traj

# ---------------- Frame rendering ----------------

def render_frames(
    traj: List[np.ndarray],
    f0: float,
    bg: Dict[str, np.ndarray],
    figsize: Tuple[float, float] = (7.0, 4.2),
    dpi: int = 110,
    title: str = "Level set drift under gradient flow",
) -> List[np.ndarray]:
    """Return a list of RGBA frames as uint8 arrays."""
    XP2, YP2, Z2 = bg['XP2'], bg['YP2'], bg['Z2']
    XP3, YP3, Z3 = bg['XP3'], bg['YP3'], bg['Z3']
    xp_min, xp_max = bg['xp_min'], bg['xp_max']
    yp_min, yp_max = bg['yp_min'], bg['yp_max']
    zmin, zmax, norm, cmap = bg['zmin'], bg['zmax'], bg['norm'], bg['cmap']

    # Parametric y-samples (original coords) for drawing level curves, then rotate for display
    Y_line = np.linspace(-5.0, 5.0, 800)

    frames: List[np.ndarray] = []
    for t_idx, pts in enumerate(traj):
        # Reference particle: the one closest to median f at the last step
        final = traj[-1]
        f_final = f_val(final)
        median_f = float(np.median(f_final))
        ref_idx = int(np.argmin(np.abs(f_final - median_f)))

        ref = pts[ref_idx]             # original coords at current time
        f_ref = float(f_val(ref.reshape(1, 2))[0])
        g_ref = float(inv_activation(np.array([f_ref]))[0])
        g_ini = float(inv_activation(np.array([f0]))[0])

        # Level set curves in original coords: x = g - B(y)
        X_line_ref = g_ref - g_baseline_of_y(Y_line)
        X_line_ini = g_ini - g_baseline_of_y(Y_line)

        # Rotate curves for display
        XP_line_ref, YP_line_ref = rot_xy_to_rotated(X_line_ref, Y_line)
        XP_line_ini, YP_line_ini = rot_xy_to_rotated(X_line_ini, Y_line)

        # Rotate particle positions for display
        XP, YP = rot_xy_to_rotated(pts[:, 0], pts[:, 1])
        XP_ref, YP_ref = rot_xy_to_rotated(ref[0:1], ref[1:2])

        # 3D curves: z = constant
        Z_line_ref = np.full_like(Y_line, f_ref)
        Z_line_ini = np.full_like(Y_line, f0)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 2], wspace=0.0)

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
        ax3d.scatter(XP_ref, YP_ref, np.array([f_ref]), s=30, c='red', edgecolors='black', linewidths=0.6, depthshade=False)

        ax3d.set_xlim(xp_min, xp_max)
        ax3d.set_ylim(yp_min, yp_max)
        ax3d.set_zlim(zmin, zmax)
        ax3d.set_xlabel("x'", labelpad=0)
        ax3d.set_ylabel("y'", labelpad=0)
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
        ax2d.scatter(XP_ref, YP_ref, s=36, c='red', edgecolors='black', linewidth=0.6, zorder=4)
        ax2d.set_xlim(xp_min, xp_max)
        ax2d.set_ylim(yp_min, yp_max)
        ax2d.set_aspect('equal', adjustable='box')
        ax2d.set_xlabel("x'")
        ax2d.set_ylabel("y'")
        fig.suptitle(title, fontsize=11, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        fig.canvas.draw()
        image = np.asarray(fig.canvas.renderer.buffer_rgba())  # (H,W,4) uint8
        frames.append(image)
        if t_idx == 0:
            # small hold on the first frame
            for _ in range(10):
                frames.append(image)
        plt.close(fig)

    return frames

# ---------------- Vector field rendering ----------------

def render_vector_field_comparison(
    bg: Dict[str, np.ndarray],
    figsize: Tuple[float, float] = (12.0, 5.0),
    dpi: int = 210,
    gradient_color: str = '#1f77b4',  # Blue
    traversal_color: str = '#d62728',  # Red
    eps: float = 1e-9,
    min_arrow_length: float = 0.05,
    max_arrow_length: float = 0.3,
) -> np.ndarray:
    """
    Generate a static PNG showing both gradient and traversal vector fields side by side.
    Arrows are scaled separately for each field to have similar lengths.
    
    Args:
        bg: Background display grids dictionary
        figsize: Figure size (width, height)
        dpi: DPI for rendering
        gradient_color: Color for gradient flow arrows
        traversal_color: Color for traversal flow arrows
        eps: Small epsilon for numerical stability
        min_arrow_length: Minimum arrow length (after scaling)
        max_arrow_length: Maximum arrow length (after scaling)
        
    Returns:
        RGBA frame as uint8 array
    """
    XP2, YP2, Z2 = bg['XP2'], bg['YP2'], bg['Z2']
    xp_min, xp_max = bg['xp_min'], bg['xp_max']
    yp_min, yp_max = bg['yp_min'], bg['yp_max']
    zmin, zmax, norm, cmap = bg['zmin'], bg['zmax'], bg['norm'], bg['cmap']
    
    # Create a coarser grid for vector field visualization
    # Use fewer points for cleaner arrow display
    nx_vec, ny_vec = 25, 20
    XP_vec, YP_vec = np.meshgrid(
        np.linspace(xp_min, xp_max, nx_vec),
        np.linspace(yp_min, yp_max, ny_vec)
    )
    
    # Convert to original coordinates for gradient computation
    X_vec, Y_vec = rot_rotated_to_xy(XP_vec, YP_vec)
    
    # Compute gradients at each grid point
    gx, gy = grad_f_xy(X_vec.flatten(), Y_vec.flatten())
    gx = gx.reshape(XP_vec.shape)
    gy = gy.reshape(YP_vec.shape)
    
    # Compute gradient flow vectors (already in original coords)
    vx_grad = gx
    vy_grad = gy
    
    # Compute traversal flow vectors: ∇f / |∇f|^2
    g2 = gx * gx + gy * gy
    inv_g2 = 1.0 / (g2 + eps)
    vx_trav = gx * inv_g2
    vy_trav = gy * inv_g2
    
    # Rotate vectors to display coordinates
    # For a vector (vx, vy) in original coords, rotated vector is R(vx, vy)
    vxp_grad, vyp_grad = rot_xy_to_rotated(vx_grad.flatten(), vy_grad.flatten())
    vxp_grad = vxp_grad.reshape(XP_vec.shape)
    vyp_grad = vyp_grad.reshape(YP_vec.shape)
    
    vxp_trav, vyp_trav = rot_xy_to_rotated(vx_trav.flatten(), vy_trav.flatten())
    vxp_trav = vxp_trav.reshape(XP_vec.shape)
    vyp_trav = vyp_trav.reshape(YP_vec.shape)
    
    # Compute arrow lengths for scaling
    lengths_grad = np.sqrt(vxp_grad**2 + vyp_grad**2)
    lengths_trav = np.sqrt(vxp_trav**2 + vyp_trav**2)
    
    # Scale each field separately to have similar arrow lengths
    # Target median length for both fields
    target_length = 0.35  # Target arrow length after scaling
    
    # Compute scale factors (use median to avoid outliers)
    valid_grad = lengths_grad > eps
    valid_trav = lengths_trav > eps
    median_grad = np.median(lengths_grad[valid_grad]) if np.any(valid_grad) else 1.0
    median_trav = np.median(lengths_trav[valid_trav]) if np.any(valid_trav) else 1.0
    
    scale_grad = target_length / median_grad if median_grad > eps else 1.0
    scale_trav = target_length / median_trav if median_trav > eps else 1.0
    
    # Apply initial scaling
    vxp_grad_scaled = vxp_grad * scale_grad
    vyp_grad_scaled = vyp_grad * scale_grad
    vxp_trav_scaled = vxp_trav * scale_trav
    vyp_trav_scaled = vyp_trav * scale_trav
    
    # Compute lengths after scaling and truncate to min/max
    lengths_grad_scaled = np.sqrt(vxp_grad_scaled**2 + vyp_grad_scaled**2)
    lengths_trav_scaled = np.sqrt(vxp_trav_scaled**2 + vyp_trav_scaled**2)
    
    # Truncate to min/max lengths by scaling each vector individually
    # For vectors that are too long, scale them down to max_arrow_length
    # For vectors that are too short, scale them up to min_arrow_length
    # For zero vectors, leave them as zero
    
    # Gradient field truncation
    with np.errstate(divide='ignore', invalid='ignore'):
        # Long vectors: scale to max_arrow_length
        mask_long = lengths_grad_scaled > max_arrow_length
        if np.any(mask_long):
            scale_factor = max_arrow_length / lengths_grad_scaled
            vxp_grad_scaled[mask_long] *= scale_factor[mask_long]
            vyp_grad_scaled[mask_long] *= scale_factor[mask_long]
        
        # Short non-zero vectors: scale to min_arrow_length
        mask_short = (lengths_grad_scaled < min_arrow_length) & (lengths_grad_scaled > eps)
        if np.any(mask_short):
            scale_factor = min_arrow_length / lengths_grad_scaled
            vxp_grad_scaled[mask_short] *= scale_factor[mask_short]
            vyp_grad_scaled[mask_short] *= scale_factor[mask_short]
    
    # Traversal field truncation
    with np.errstate(divide='ignore', invalid='ignore'):
        # Long vectors: scale to max_arrow_length
        mask_long = lengths_trav_scaled > max_arrow_length
        if np.any(mask_long):
            scale_factor = max_arrow_length / lengths_trav_scaled
            vxp_trav_scaled[mask_long] *= scale_factor[mask_long]
            vyp_trav_scaled[mask_long] *= scale_factor[mask_long]
        
        # Short non-zero vectors: scale to min_arrow_length
        mask_short = (lengths_trav_scaled < min_arrow_length) & (lengths_trav_scaled > eps)
        if np.any(mask_short):
            scale_factor = min_arrow_length / lengths_trav_scaled
            vxp_trav_scaled[mask_short] *= scale_factor[mask_short]
            vyp_trav_scaled[mask_short] *= scale_factor[mask_short]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # Left subplot: Gradient flow
    ax1.imshow(Z2, extent=[xp_min, xp_max, yp_min, yp_max], 
               origin='lower', cmap=cmap, aspect='auto', alpha=0.8)
    ax1.quiver(XP_vec, YP_vec, vxp_grad_scaled, vyp_grad_scaled, 
               color=gradient_color, 
               angles='xy', scale_units='xy', scale=1.0,
               width=0.003, headwidth=3.5, headlength=4.0, headaxislength=3.5,
               alpha=0.85, zorder=3)
    ax1.set_xlim(xp_min, xp_max)
    ax1.set_ylim(yp_min, yp_max)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel("x'")
    ax1.set_ylabel("y'")
    ax1.set_title("Gradient flow: $\\dot x = \\nabla_x f(x)$", fontsize=12, pad=10)
    
    # Right subplot: Traversal flow
    ax2.imshow(Z2, extent=[xp_min, xp_max, yp_min, yp_max], 
               origin='lower', cmap=cmap, aspect='auto', alpha=0.8)
    ax2.quiver(XP_vec, YP_vec, vxp_trav_scaled, vyp_trav_scaled, 
               color=traversal_color, 
               angles='xy', scale_units='xy', scale=1.0,
               width=0.003, headwidth=3.5, headlength=4.0, headaxislength=3.5,
               alpha=0.85, zorder=3)
    ax2.set_xlim(xp_min, xp_max)
    ax2.set_ylim(yp_min, yp_max)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel("x'")
    ax2.set_ylabel("y'")
    ax2.set_title("Traversal flow: $\\dot x = \\frac{\\nabla_x f(x)}{\\|\\nabla_x f(x)\\|^2}$", fontsize=12, pad=10)
    
    fig.suptitle("Vector field comparison", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.canvas.draw()
    image = np.asarray(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    return image

# ---------------- Saving utilities ----------------

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            # If directory is invalid (e.g., user path not present), ignore; caller can try other outputs
            pass

def _to_rgb(fr: np.ndarray) -> np.ndarray:
    """Convert RGBA or gray to RGB uint8 for video writers."""
    if fr.ndim == 3 and fr.shape[-1] == 4:
        return fr[..., :3].copy()
    if fr.ndim == 2:
        return np.repeat(fr[..., None], 3, axis=2)
    return fr


def save_video(frames: List[np.ndarray], path: str, fps: int = 18) -> bool:
    """Try writing H.264 MP4 with yuv420p; fall back handled by caller."""
    try:
        _ensure_dir(path)
        with imageio.get_writer(
            path, fps=fps, codec='libx264', format='FFMPEG',
            output_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart']
        ) as writer:
            for fr in frames:
                fr_rgb = _to_rgb(fr)
                writer.append_data(fr_rgb)
        return True
    except Exception:
        return False


def save_webm(frames: List[np.ndarray], path: str, fps: int = 18) -> bool:
    try:
        _ensure_dir(path)
        with imageio.get_writer(
            path, fps=fps, codec='libvpx-vp9', format='FFMPEG',
            output_params=['-b:v', '1M']  # modest bitrate
        ) as writer:
            for fr in frames:
                fr_rgb = _to_rgb(fr)
                writer.append_data(fr_rgb)
        return True
    except Exception:
        return False


def save_gif(frames: List[np.ndarray], path: str, fps: int = 18) -> bool:
    try:
        _ensure_dir(path)
        # durations per frame (seconds). Hold last frame slightly shorter.
        durations = [1.0 / 12.0] * (len(frames) - 1) + [1.0 / 24.0]
        imageio.mimsave(path, [_to_rgb(fr) for fr in frames], duration=durations, loop=0)
        return True
    except Exception:
        return False


def save_png(image: np.ndarray, path: str) -> bool:
    """Save a single image as PNG."""
    try:
        _ensure_dir(path)
        imageio.imwrite(path, _to_rgb(image))
        return True
    except Exception:
        return False

# ---------------- High-level run helper ----------------

def run_once(mode: DynMode, title: str) -> List[np.ndarray]:
    bg = make_display_grids()
    pts0, f0 = init_particles()
    traj = simulate(pts0, n_steps=50, dt=0.95 if mode == 'gradient' else .01, mode=mode)
    frames = render_frames(traj, f0, bg, title=title)
    return frames


def save_all_targets(frames: List[np.ndarray], base_filename: str,
                     user_dir: str, scratch_dir: str = '/mnt/data', save_webm_fallback: bool = True) -> Dict[str, List[str]]:
    """Save MP4 + GIF to both the user's preferred directory (if present) and /mnt/data.
    Returns dict with successful paths per kind.
    """
    # Build full paths
    targets = []  # (kind, path)
    # user paths
    if user_dir and os.path.isdir(user_dir):
        targets.append(('mp4', os.path.join(user_dir, f'{base_filename}.mp4')))
        targets.append(('gif', os.path.join(user_dir, f'{base_filename}.gif')))
    # scratch paths
    if scratch_dir:
        targets.append(('mp4', os.path.join(scratch_dir, f'{base_filename}.mp4')))
        targets.append(('gif', os.path.join(scratch_dir, f'{base_filename}.gif')))

    successes = {'mp4': [], 'gif': [], 'webm': []}

    # Try each target
    for kind, path in targets:
        if kind == 'mp4':
            ok = save_video(frames, path, fps=18)
            if ok:
                successes['mp4'].append(path)
            elif save_webm_fallback:
                # Replace extension with webm
                webm_path = os.path.splitext(path)[0] + '.webm'
                ok2 = save_webm(frames, webm_path, fps=18)
                if ok2:
                    successes['webm'].append(webm_path)
        elif kind == 'gif':
            ok = save_gif(frames, path, fps=18)
            if ok:
                successes['gif'].append(path)
    return successes


# ---------------- __main__ ----------------
if __name__ == '__main__':
    # User's original preferred directory if available; we detect the parent of original paths.
    user_base_dir = '/Users/adamsobieszek/PycharmProjects/_manipy/traversals'

    # 1) Gradient flow
    frames_grad = run_once('gradient', title="Level set drift under gradient flow: $\\dot x = \\nabla_x f(x)$")
    grad_success = save_all_targets(
        frames_grad,
        base_filename='traversal_gradient_flow_2panel2',
        user_dir=user_base_dir,
        scratch_dir='/mnt/data'
    )
    print('Gradient flow saved to:', grad_success)

    # 2) Canonical traversal (df/dt=1)
    frames_trav = run_once('traversal', title="Canonical traversal: $\\dot x = X_f = \\frac{\\nabla_x f(x)}{\\|\\nabla_x f(x)\\|^2}$")
    trav_success = save_all_targets(
        frames_trav,
        base_filename='traversal_canonical_2panel2',
        user_dir=user_base_dir,
        scratch_dir='/mnt/data'
    )
    print('Traversal saved to:', trav_success)

    # 3) Vector field comparison (side by side PNG)
    bg_vec = make_display_grids()
    image_vec = render_vector_field_comparison(
        bg_vec,
        gradient_color='#1f77b4',  # Blue
        traversal_color='#d62728',  # Red
    )
    vec_success = []
    # Save to user directory
    if user_base_dir and os.path.isdir(user_base_dir):
        png_path = os.path.join(user_base_dir, 'traversal_vector_field_comparison.png')
        if save_png(image_vec, png_path):
            vec_success.append(png_path)
    # Save to scratch directory
    if os.path.isdir('/mnt/data'):
        png_path = os.path.join('/mnt/data', 'traversal_vector_field_comparison.png')
        if save_png(image_vec, png_path):
            vec_success.append(png_path)
    print('Vector field comparison saved to:', vec_success)

    # Convenience: print direct download hints for this environment
    for kind, paths in (('MP4', grad_success.get('mp4', []) + trav_success.get('mp4', [])),
                        ('WEBM', grad_success.get('webm', []) + trav_success.get('webm', [])),
                        ('GIF', grad_success.get('gif', []) + trav_success.get('gif', [])),
                        ('PNG', vec_success if isinstance(vec_success, list) else [])):
        if paths:
            print(f"Saved {kind}:")
            for p in paths:
                print('  ', p)
