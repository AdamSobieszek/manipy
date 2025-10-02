import torch
import numpy as np
import matplotlib.pyplot as plt


# Training utilities
def evaluate_r2(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            y_true.append(yb.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0).ravel()
    y_pred = np.concatenate(y_pred, axis=0).ravel()
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = corr ** 2
    return r2

# ----- linear "impression" f and its geometry -----
def f_linear(x: torch.Tensor, a: float = 2.0, b: float = 1.0) -> torch.Tensor:
    """f(x) = a x1 + b x2, for x shape (N,2)."""
    return a * x[:, 0] + b * x[:, 1]

def grad_f_linear(x: torch.Tensor, a: float = 2.0, b: float = 1.0) -> torch.Tensor:
    """Constant gradient field (N,2)."""
    return torch.tensor([a, b], dtype=x.dtype, device=x.device).expand_as(x)

def ortho_frame(a: float, b: float, device, dtype):
    """
    Returns unit normal (n_hat) along ∇f and unit tangent (t_hat) perpendicular to it.
    """
    g = torch.tensor([a, b], dtype=dtype, device=device)
    gnorm = torch.norm(g)
    n_hat = g / (gnorm + 1e-12)
    t_hat = torch.tensor([-b, a], dtype=dtype, device=device) / (gnorm + 1e-12)
    return n_hat, t_hat, gnorm

# ----- variance schedule and y-marginal -----
def sigma2_of_y(y: torch.Tensor) -> torch.Tensor:
    """σ²(y) = exp(-y² / 4)."""
    return torch.exp(-0.25 * y * y)


@torch.no_grad()
def density_special_gaussian(x: torch.Tensor,
                             a: float = 2.0,
                             b: float = 1.0,
                             y_std: float = 2.0,
                             return_log: bool = False) -> torch.Tensor:
    """
    Unnormalized density p(x) corresponding to the (y, s) construction.
    Matches your NumPy expression up to a constant factor.
      p_y(y) = N(0, y_std^2)
      p(s|y) = N(0, σ²(y)), σ²(y)=exp(-y^2/4)
      y = f_lin(x) = a x1 + b x2
      s = projection of x onto tangent direction
    We return p_x ∝ p_y(y) * p(s|y) * ||∇f||  (your code’s convention).
    """
    device, dtype = x.device, x.dtype
    n_hat, t_hat, gnorm = ortho_frame(a, b, device, dtype)

    y = f_linear(x, a, b)  # (N,)
    alpha = y / (gnorm + 1e-12)   # along n_hat
    # Decompose x in (n_hat, t_hat) basis to get s
    s = (x * t_hat).sum(dim=1)    # (N,)

    # p_y
    log_p_y = -0.5 * (y * y) / (y_std * y_std) - torch.log(torch.tensor(y_std * np.sqrt(2 * np.pi), dtype=dtype, device=device))
    # sigma^2(y)
    sigma2 = sigma2_of_y(y)
    # p(s|y)
    log_p_s = -0.5 * (s * s) / (sigma2 + 1e-12) - 0.5 * torch.log(2 * torch.pi * sigma2 + 1e-12)

    # your convention multiplies by ||∇f|| (a constant factor)
    log_p = log_p_y + log_p_s + torch.log(torch.tensor(gnorm, dtype=dtype, device=device) + 1e-12)

    return log_p if return_log else torch.exp(log_p)
def make_impression_function(
    # --- main peak (a smooth hill) ---
    peak_mu=(0.2, -0.1),
    peak_cov=((0.35, 0.12), (0.12, 0.6)),
    peak_amp=1.0,
    # --- low-valued attractors (valleys/minima) ---
    attractors=(
        {"mu": (-1.0,  0.8), "cov": ((0.25, 0.00), (0.00, 0.06)), "depth": 0.9},
        {"mu": ( 0.9, -0.9), "cov": ((0.18, 0.07), (0.07, 0.12)), "depth": 0.7},
        {"mu": ( 0.3,  0.9), "cov": ((0.10, 0.00), (0.00, 0.30)), "depth": 0.6},
    ),
    # --- optional oriented lanes (anisotropic tubes, +ridge / -valley) ---
    lanes=(
        {"dir": (1.0, 0.0), "center": (0.0, 0.0), "width_perp": 0.15, "length_along": 1.6, "gain": 0.35, "sign": +1.0},
        {"dir": (0.0, 1.0), "center": (-0.6, 0.3), "width_perp": 0.10, "length_along": 1.0, "gain": 0.25, "sign": -1.0},
    ),
    # --- NEW: linear component along z = x + y ---
    lin_z_gain: float = 0.0,       # set >0 to enable (adds a plane along the 45° diagonal)
    lin_z_center: float = 0.0,     # shift: uses (z - lin_z_center)
    lin_z_tanh_alpha: float = 0.0, # if >0, saturate with tanh(alpha * (z - center))
    # --- sigmoid envelope so the landscape "ends" smoothly ---
    envelope_center=(0.0, 0.0),
    envelope_r0=2.2,
    envelope_beta=2.0,
    # --- global squashing of the final value (keeps outputs in [-scale, scale]) ---
    squash_alpha=0.9,
    squash_scale=1.0,
    # --- tiny roughness (set to 0.0 to remove) ---
    roughness_amp=0.05,
    roughness_scale=1.5,
    density = density_special_gaussian,
):
    """
    Returns: impression_function(x: (N,2)->(N,)) closure.

    Components:
      + Main anisotropic peak (μ, Σ)
      - Attractor wells (anisotropic minima)
      ± Oriented lanes (ridges/valleys)
      + Linear diagonal term: z = x1 + x2  (tilts landscape along 45°)
      × Radial sigmoid envelope (ends the landscape)
      → Optional global tanh squash
    """
    # tensors for closure
    peak_mu_t = torch.tensor(peak_mu, dtype=torch.float32)
    peak_cov_t = torch.tensor(peak_cov, dtype=torch.float32)
    env_c_t = torch.tensor(envelope_center, dtype=torch.float32)

    attract_list = []
    for it in attractors:
        mu_t = torch.tensor(it["mu"], dtype=torch.float32)
        cov_t = torch.tensor(it["cov"], dtype=torch.float32)
        depth = float(it["depth"])
        attract_list.append((mu_t, cov_t, depth))

    lane_list = []
    for it in lanes:
        d = torch.tensor(it["dir"], dtype=torch.float32)
        d = d / (torch.norm(d) + 1e-12)
        c = torch.tensor(it["center"], dtype=torch.float32)
        wp = float(it["width_perp"])
        la = float(it["length_along"])
        gain = float(it["gain"])
        sign = float(it.get("sign", +1.0))
        d_perp = torch.tensor([-d[1], d[0]], dtype=torch.float32)
        R = torch.stack([d, d_perp], dim=1)  # 2x2
        S = torch.tensor([[la**2, 0.0], [0.0, wp**2]], dtype=torch.float32)
        cov_lane = R @ S @ R.T
        lane_list.append((c, cov_lane, gain, sign))

    eye2 = torch.eye(2)

    def _precision(cov: torch.Tensor, device, dtype):
        cov = cov.to(device=device, dtype=dtype)
        return torch.inverse(cov + 1e-6 * eye2.to(device=device, dtype=dtype))

    def _gauss_aniso(x, mu, cov_prec):
        d = x - mu
        q = torch.einsum('ni,ij,nj->n', d, cov_prec, d)
        return torch.exp(-0.5 * q)

    def impression_function(x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype

        # Base: main peak
        P_peak = _precision(peak_cov_t, device, dtype)
        mu = peak_mu_t.to(device=device, dtype=dtype)
        val = _gauss_aniso(x, mu, P_peak) * float(peak_amp)

        # Subtract attractors (valleys)
        for (mu_i, cov_i, depth_i) in attract_list:
            P = _precision(cov_i, device, dtype)
            mui = mu_i.to(device=device, dtype=dtype)
            val = val - float(depth_i) * _gauss_aniso(x, mui, P)

        # Add oriented lanes
        for (c_i, cov_lane, gain_i, sign_i) in lane_list:
            P = _precision(cov_lane, device, dtype)
            ci = c_i.to(device=device, dtype=dtype)
            val = val + float(sign_i) * float(gain_i) * _gauss_aniso(x, ci, P)

        # --- NEW: linear component z = x1 + x2 ---
        if lin_z_gain != 0.0:
            z = x[:, 0] + x[:, 1]
            z_shift = z - float(lin_z_center)
            if lin_z_tanh_alpha and lin_z_tanh_alpha > 0.0:
                z_term = float(lin_z_gain) * torch.tanh(float(lin_z_tanh_alpha) * z_shift)
            else:
                z_term = float(lin_z_gain) * z_shift *z_shift*(z_shift)
            val = val + z_term

        # Optional tiny smooth roughness
        if roughness_amp != 0.0:
            s = float(roughness_scale)
            val = val + float(roughness_amp) * torch.exp(
                -0.5 * ((x[:, 0] / s) ** 2 + (x[:, 1] / s) ** 2)
            ) * (0.25 * x[:, 0] + 0.35 * x[:, 1])

        # Radial sigmoid envelope
        r = torch.norm(x - env_c_t.to(device=device, dtype=dtype), dim=1)
        gate = torch.sigmoid(float(envelope_beta) * (float(envelope_r0) - r))
        val = gate * val*(torch.sigmoid((val+.7)**3)*density(x, 2,1)+torch.sigmoid((val+.7)**3)/7)
        val = val+ float(lin_z_gain) * z_shift*3

        # Final squash
        out = float(squash_scale) * torch.tanh(float(squash_alpha) * val)
        return out

    return impression_function
import torch
import numpy as np
import time


import numpy as np
import torch
import warnings

def draw_support_boundary(ax,
                          density_fn,
                          density_kwargs=None,
                          xlim=(-3.0, 3.0),
                          ylim=(-3.0, 3.0),
                          resolution: int = 300,
                          p_min: float | None = None,
                          p_rel: float = 0.2,
                          device=None,
                          dtype=torch.float32,
                          linewidth: float = 2.0,
                          linestyle: str = "-.",
                          color: str = "white"):
    """
    Draw a support boundary as an iso-density contour:  { x : p(x) > level }.
    If p_min is None, level = p_rel * max p on the grid.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    density_fn : callable
        Function evaluating (unnormalized) density on a batch of points:
            density_fn(XY, **density_kwargs) -> (M,) tensor
        where XY is (M, 2) torch tensor.
    density_kwargs : dict or None
        Extra keyword args passed to density_fn (e.g., dict(a=2.0, b=1.0, y_std=2.0)).
    xlim, ylim : tuple
        Plot window in x1 and x2.
    resolution : int
        Grid resolution per axis.
    p_min : float or None
        Absolute density threshold. If None, uses relative threshold p_rel * max p.
    p_rel : float
        Relative threshold as a fraction of max density (used when p_min is None).
    device, dtype : torch device/dtype
        Where to compute the density.
    linewidth, linestyle, color : styling for the boundary line.

    Returns
    -------
    level : float
        The density level used for the boundary.
    """
    if device is None:
        device = torch.device("cpu")
    if density_kwargs is None:
        density_kwargs = {}

    # grid
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    XY = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=dtype, device=device)

    # evaluate density
    with torch.no_grad():
        P = density_fn(XY, **density_kwargs)
        # if density_fn returns log-density, uncomment the next line:
        # P = torch.exp(P)
    P = P.view(resolution, resolution).detach().cpu().numpy()
    Pmax = float(P.max())
    if Pmax <= 0.0:
        # nothing to draw
        return None

    level = p_min if (p_min is not None) else (p_rel * Pmax)

    # draw contour safely
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No contour levels were found")
        cs = ax.contour(X, Y, P, levels=[level],
                        linewidths=linewidth, linestyles=linestyle, colors=color)

    # proxy handle for legend (avoids touching cs.collections)
    ax.plot([], [], linestyle=linestyle, color=color, linewidth=linewidth,
            label=f"p(x) > {level:.3g}")

    return level
# --- NEW: Data distribution visualization ---
def plot_data_distribution(distribution_sampler,
                           n_samples: int = 30000,
                           show_support: bool = True,
                           support_method: str = "constraints",
                           p_min: float | None = None,
                           p_min_rel: float = 0.05,
                           bandwidth: float = 0.15,
                           title: str = "Data distribution",
                           fig = None, ax = None):
    """
    Visualize sampled data with a density-like background and (optional) support boundary.
    """
    import matplotlib.pyplot as plt

    x = distribution_sampler(n_samples).cpu()
    X = x.numpy()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.0))
    else:
        title = None
    hb = ax.hexbin(X[:, 0], X[:, 1], gridsize=65, bins='log', alpha=0.8)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log counts')

    if show_support:
        draw_support_boundary(ax,density_fn=density_special_gaussian,density_kwargs=dict(a=2.0, b=1.0, y_std=2.0),
            xlim=(-3, 3), ylim=(-3, 3), resolution=300, p_min=None, p_rel=0.25, linewidth=2.0, linestyle="-.", color="white")
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if title is not None:
        plt.show()
    else:
        return ax



def plot_trajectories_with_impression_and_support(trajectories, impression_fn,
                                                  title="",
                                                  show_support: bool = True,
                                                  support_method: str = "constraints",
                                                  p_min: float | None = None,
                                                  p_min_rel: float = 0.05,
                                                  bandwidth: float = 0.15,
                                                  xlim=(-3., 3.), ylim=(-2, 2)):
    """
    Same as your plot_trajectories_with_impression, but adds a probability-support boundary.
    - support_method="constraints" draws the analytic boundary of the sampler's base region.
    - support_method="kde" draws {x : p_hat(x) > threshold}, where threshold is p_min (or p_min_rel * max p).
    """
    x_range = np.linspace(xlim[0], xlim[1], 160)
    y_range = np.linspace(ylim[0], ylim[1], 160)
    X, Y = np.meshgrid(x_range, y_range)

    grid_points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    Z = impression_fn(grid_points).detach().numpy().reshape(X.shape)

    fig, ax = plt.subplots(1, 1, figsize=((xlim[1]-xlim[0])*2, (ylim[1]-ylim[0])*2))
    contour = ax.contourf(X, Y, Z, levels=24, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=ax, label='Impression Value')

    # Plot trajectories (expects shape (T, N, 2))
    T, N, _ = trajectories.shape
    n_plot = min(N, 100)
    for traj in trajectories.transpose(1, 0, 2)[:n_plot]:
        ax.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.3, linewidth=0.7)
        ax.scatter(traj[0, 0], traj[0, 1], c='blue', s=8, alpha=0.7)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=8, alpha=0.7)

    # NEW: support boundary
    if show_support:
        draw_support_boundary(ax,density_fn=density_special_gaussian,density_kwargs=dict(a=2.0, b=1.0, y_std=2.0),
            xlim=(-3, 3), ylim=(-3, 3), resolution=300, p_min=None, p_rel=0.25, linewidth=2.0, linestyle="-.", color="white")


    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_streamlines_with_impression_and_support(vector_field_fn, impression_fn,
                                                 title="",
                                                 show_support: bool = True,
                                                 support_method: str = "constraints",
                                                 p_min: float | None = None,
                                                 p_min_rel: float = 0.05,
                                                 bandwidth: float = 0.15,
                                                 xlim=(-3., 3.), ylim=(-2, 2),
                                                 density=1.2):
    """
    Visualize the impression function as a background and overlay streamlines of the vector field.
    vector_field_fn: function mapping (N,2) torch tensor to (N,2) torch tensor (the vector field)
    impression_fn: function mapping (N,2) torch tensor to (N,) torch tensor (the impression)
    """
    x_range = np.linspace(xlim[0], xlim[1], 160)
    y_range = np.linspace(ylim[0], ylim[1], 160)
    X, Y = np.meshgrid(x_range, y_range)
    grid_points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)

    # Impression background
    Z = impression_fn(grid_points).detach().numpy().reshape(X.shape)

    # Vector field
    V = vector_field_fn(grid_points).detach().numpy()
    Ux = V[:, 0].reshape(X.shape)
    Uy = V[:, 1].reshape(X.shape)

    fig, ax = plt.subplots(1, 1, figsize=((xlim[1]-xlim[0])*2, (ylim[1]-ylim[0])*2))
    contour = ax.contourf(X, Y, Z, levels=24, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=ax, label='Impression Value')

    # Streamlines
    strm = ax.streamplot(X, Y, Ux, Uy, color='w', density=density, linewidth=1.2, arrowsize=1.5, broken_streamlines=False)

    # Support boundary
    if show_support:
        draw_support_boundary(ax, density_fn=density_special_gaussian, density_kwargs=dict(a=2.0, b=1.0, y_std=2.0),
            xlim=(-3, 3), ylim=(-3, 3), resolution=300, p_min=None, p_rel=0.25, linewidth=2.0, linestyle="-.", color="white")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- Live dashboard (requires the StreamlineMetricsDashboard + MetricSpec definitions from the previous cell) ---

# Shape-safe wrappers (expect (N,2) -> (N,2) for vf, (N,2) -> (N,) for scalar fields)
def vf_wrapper(pts: torch.Tensor) -> torch.Tensor:
    # model returns (N,2)
    return model(pts)

def impression_wrapper(pts: torch.Tensor) -> torch.Tensor:
    out = ensemble(pts)                   # could be (N,) or (N,1)
    return out.view(-1) if out.ndim > 1 else out

def density_wrapper(pts: torch.Tensor) -> torch.Tensor:
    # match the kwargs you used for support drawing elsewhere
    return density_special_gaussian(pts, a=2.0, b=1.0, y_std=2.0)

# dash = StreamlineMetricsDashboard(
#     vector_field_fn=vf_wrapper,
#     impression_fn=impression_wrapper,
#     density_fn=density_wrapper,          # or None to hide the dashed support curve
#     device=device,
#     xlim=(-3., 3.), ylim=(-2., 2.),
#     resolution=100,                      # higher = smoother background/streamlines
#     seed_grid=(24, 18),                  # more seeds = denser streamlines
#     max_steps=700,
#     support_cutoff=0.02,
#     metrics=[
#         MetricSpec("loss", "MSE Loss"),
#         MetricSpec("lr",   "Learning Rate", dash="dot", formatter=lambda v: f"{v:.1e}"),
#     ],
#     title="PGFM Training (Streamlines + Metrics)",
# )


# Build the function once (you can tweak params live)
impression_function = make_impression_function(
    peak_mu=(2.5, 1.),
    peak_cov=((0.95, -0.40), (-0.40, 0.4)),
    peak_amp=1.8,
    attractors=(
        {"mu": (-.5,  1.), "cov": ((1.10, .20), (0.20, 0.95)), "depth": .1},
        {"mu": ( -2.5, 0.8), "cov": ((1., -0.16), (-0.16, 2.82)), "depth": 1.4},
    ),
    lanes=(
        {"dir": (-1.5, 0.), "center": (-1.2, -1.4), "width_perp": .22, "length_along": .5, "gain": 1.2, "sign": -1},
        {"dir": (-1.5, 1.), "center": (-0.2,-1.1), "width_perp": .32, "length_along": .5, "gain": .4, "sign": -1},
        {"dir": (1.5, 1.3), "center": (1., .5), "width_perp": .3, "length_along": .7, "gain": .3, "sign": 1},
    ),
    envelope_center=(0.0, 0.0), envelope_r0=2.4, envelope_beta=2.0,
    squash_alpha=0.9, squash_scale=1.0,
    roughness_amp=.0, roughness_scale=1.8, lin_z_gain=.0005

)



# results = run_all_metrics(
# model=model,
# ensemble=ensemble,
# impression_function=impression_function,
# X_start=X_low,
# X_for_ood=X[(-1.5<X[:,1])&(X[:,1] < 1.5)],
# step_size=0.05,
# n_steps=50,
# device=device,
# )

# --- NEW: Plotly epoch viewer ---
import numpy as np
import torch

import plotly.graph_objects as go
from plotly.figure_factory import create_quiver
# --- Matplotlib -> NumPy image exporter (no plt.show/printing) ---
import io, numpy as np
import matplotlib.pyplot as plt

def render_streamlines_image(
    vector_field_fn, impression_fn,
    show_support=True,
    xlim=(-3., 3.), ylim=(-2., 2.),
    density_fn=None, density_kwargs=None, cutoff_p=0.02,
    resolution=100, dpi=200, plot=False, device=None
):
    fig, ax = plt.subplots(1, 1, figsize=((xlim[1]-xlim[0])*2, (ylim[1]-ylim[0])*2), dpi=dpi)
    # Build grid
    x_range = np.linspace(xlim[0], xlim[1], resolution)
    y_range = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x_range, y_range)
    grid_points = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32).to(device)

    # Evaluate fields
    with torch.no_grad():
        Z = impression_fn(grid_points).detach().cpu().numpy().reshape(X.shape)
        V = vector_field_fn(grid_points).detach().cpu().numpy()
    Ux = V[:, 0].reshape(X.shape)
    Uy = V[:, 1].reshape(X.shape)

    # --- draw with Matplotlib (OFF-SCREEN) ---
    cf = ax.contourf(X, Y, Z, levels=24, cmap='viridis', alpha=0.75)
    ax.streamplot(X, Y, Ux, Uy, color='w', density=1., linewidth=1.1, arrowsize=1.6, broken_streamlines=False)

    if show_support and density_fn is not None:
        with torch.no_grad():
            P = density_fn(grid_points, **(density_kwargs or {})).detach().cpu().numpy().reshape(X.shape)
        thr = cutoff_p * np.max(P)
        ax.contour(X, Y, P, levels=[thr], colors='k', linewidths=2.0, linestyles='-.')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.set_title('PGFM Trajectories (Matplotlib)')
    ax.grid(True, alpha=0.3)

    
    # Load as numpy array for Plotly
    if not plot:
        # Save to RGBA array without opening a new output
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        plt.close(fig)  # <- CRUCIAL: avoid extra notebook prints/outputs
        buf.seek(0)
        import PIL.Image as Image
        im = np.array(Image.open(buf).convert('RGBA'))
        buf.close()
        return im
    else:
        plt.show()


import torch
import numpy as np
import ot  # pip install POT

@torch.no_grad()
def _safe_grad_f(x, f_func):
    """Compute ∇f(x) in a grad-safe block (no graph kept)."""
    x_req = x.detach().clone().requires_grad_(True)
    f = f_func(x_req)  # (N,)
    g = torch.autograd.grad(f.sum(), x_req, create_graph=False)[0]
    return g.detach()

@torch.no_grad()
def ot_pair_points_with_asymmetric_cost(
    x_all: torch.Tensor,
    f_func,
    # ------ splitting ------
    split_quantile: float = 0.4,     # e.g., 0.4 means bottom 40% vs top 60%
    eps: float = 1e-8,
    large_cost_val: float = 1e10,
    # ------ OT solver ------
    ot_numItermax: int = 200000,
    device=None,
    dtype=None,
):
    """
    Inputs
    ------
    x_all : (N,D) torch tensor
    f_func: callable (x)->(N,) differentiable

    Returns
    -------
    x0_pairs : (K,D)
    x1_pairs : (K,D)
    u_hat    : (K,D)   where u_hat = (x1 - x0) / (f1 - f0)
    """
    if device is None: device = x_all.device
    if dtype  is None: dtype  = x_all.dtype

    # 1) Evaluate f and split by quantile
    f_all = f_func(x_all).reshape(-1)  # (N,)
    thr = torch.quantile(f_all.float(), split_quantile).to(device=device, dtype=dtype)

    low_mask  = f_all <= thr
    high_mask = f_all  > thr

    X0 = x_all[low_mask]
    F0 = f_all[low_mask]
    X1 = x_all[high_mask]
    F1 = f_all[high_mask]

    if X0.numel() == 0 or X1.numel() == 0:
        # no pairs possible
        return (torch.empty(0, x_all.shape[1], device=device, dtype=dtype),
                torch.empty(0, x_all.shape[1], device=device, dtype=dtype),
                torch.empty(0, x_all.shape[1], device=device, dtype=dtype))

    m, n = X0.shape[0], X1.shape[0]
             # (m,D)

    # 3) Pairwise deltas
    dX = X1.unsqueeze(0) - X0.unsqueeze(1)         # (m,n,D)
    dF = F1.unsqueeze(0) - F0.unsqueeze(1)         # (m,n)

    # 4) Build custom asymmetric cost
    #    4a) geometric term
    geom = (dX ** 2).sum(dim=2)                    # (m,n)

    #    4b) regression-target term: û = dX / (Δf + eps)
    u_hat = dX / (dF.unsqueeze(2).abs() + eps)     # (m,n,D)
    # broadcast target (m,1,D)

    #    4c) inverse-gap term (optional)
    inv_gap = (geom / ((dF ** 2) + eps))#**0.5            # (m,n)

    #    4e) final cost
    C =  inv_gap

    #    4f) hard mask for Δf <= 0 : set to very large
    C = torch.where(dF > 0.0, C, torch.full_like(C, large_cost_val))

    # 5) Solve OT (POT)
    C_np = C.detach().cpu().numpy()
    a = np.ones(m, dtype=np.float64) / float(m)
    b = np.ones(n, dtype=np.float64) / float(n)
    C_np[np.isinf(C_np)] = large_cost_val

    try:
        pi = ot.emd(a, b, C_np, numItermax=ot_numItermax)  # (m,n)
    except Exception as e:
        print(f"[OT] POT emd failed: {e}")
        return (torch.empty(0, x_all.shape[1], device=device, dtype=dtype),
                torch.empty(0, x_all.shape[1], device=device, dtype=dtype),
                torch.empty(0, x_all.shape[1], device=device, dtype=dtype))

    # 6) Extract hard pairs where transport mass > 0
    idx = np.argwhere(pi > 0)
    if idx.size == 0:
        return (torch.empty(0, x_all.shape[1], device=device, dtype=dtype),
                torch.empty(0, x_all.shape[1], device=device, dtype=dtype),
                torch.empty(0, x_all.shape[1], device=device, dtype=dtype))

    i_sel = torch.tensor(idx[:, 0], device=device, dtype=torch.long)
    j_sel = torch.tensor(idx[:, 1], device=device, dtype=torch.long)

    x0_pairs = X0[i_sel]
    x1_pairs = X1[j_sel]
    dF_sel = (F1[j_sel] - F0[i_sel]).unsqueeze(1)
    u_hat_pairs = (x1_pairs - x0_pairs) / (dF_sel.abs() + eps)

    return x0_pairs, x1_pairs, u_hat_pairs

@torch.no_grad()
def create_ot_pairs_from_sampler(batch_size,
                                 distribution_sampler,
                                 f_func,
                                 # you can pass through any of the cost params:
                                 split_quantile=0.65,
                                  eps=1e-8,
                                 oversample_factor=1):
    """
    Samples points, runs OT pairing with the custom asymmetric metric,
    and returns up to `batch_size` pairs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype  = torch.float32

    # oversample to ensure enough valid pairs
    x_all = distribution_sampler(batch_size * oversample_factor).to(device=device, dtype=dtype)

    x0, x1, u_hat = ot_pair_points_with_asymmetric_cost(
        x_all, f_func,
        split_quantile=split_quantile,
  eps=eps
    )
    # trim to batch_size
    if x0.shape[0] > batch_size:
        print(x0.shape[0], batch_size)
        sel = torch.randperm(x0.shape[0], device=x0.device)[:batch_size]
        x0, x1, u_hat = x0[sel], x1[sel], u_hat[sel]
    return x0, x1




@torch.no_grad()
def split_by_impression(x, quantile=0.5, impression_function=None):
    """Split points into low and high impression based on function values"""
    impressions = impression_function(x).reshape(-1)
    threshold = torch.quantile(impressions, quantile)

    low_mask = impressions <= threshold
    high_mask = impressions > threshold

    x_low = x[low_mask]
    x_high = x[high_mask]

    return x_low, x_high, impressions

def create_impression_pairs(batch_size, distribution_sampler, impression_function):
    """
    Create pairs of points for flow matching based on impression values
    Returns matched pairs where x0 has lower impression than x1
    """
    # Sample from distribution
    n_samples = batch_size * 4  # Oversample to ensure enough pairs
    x_all = distribution_sampler(n_samples)

    # Split by impression
    x_low, x_high, _ = split_by_impression(x_all, quantile=0.5, impression_function=impression_function)

    # For pairing, we can use random matching or optimal transport
    # Here using random matching for simplicity
    n_pairs = min(len(x_low), len(x_high), batch_size)

    idx_low = torch.randperm(len(x_low))[:n_pairs]
    idx_high = torch.randperm(len(x_high))[:n_pairs]

    x0 = x_low[idx_low]
    x1 = x_high[idx_high]

    return x0, x1




sample_points_from_convex_distribution = lambda n, X, device: torch.cat([sample_polytope(n=n//(10+n//200) +1, x_points=X, k=3, device=device) for _ in range(10+n//200)])[torch.randperm((n//(10+n//200)+1)*(10+n//200))[:n]]
