

def model_vector_field_on_grid(model,
                               t_eval: float,
                               xlim=(-3.0, 3.0),
                               ylim=(-3.0, 3.0),
                               N: int = 30,
                               device=None,
                               dtype=torch.float32):
    """
    Evaluate v_theta(x, t_eval) on a grid; returns (X, Y, v_all) where
      X,Y are (N,N) numpy grids, v_all is (N,N,2) numpy.
    """
    if device is None:
        device = torch.device("cpu")

    xs = np.linspace(*xlim, N)
    ys = np.linspace(*ylim, N)
    X, Y = np.meshgrid(xs, ys)
    XY = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=dtype, device=device)

    t = torch.full((XY.shape[0], 1), float(t_eval), dtype=dtype, device=device)
    with torch.no_grad():
        v = model(torch.cat([XY, t], dim=-1)).detach()
    v = v.view(N, N, 2).cpu().numpy()
    return X, Y, v

def decompose_flow(v_xy: torch.Tensor,
                   grad_xy: torch.Tensor,
                   eps: float = 1e-12):
    """
    v_xy: (M,2) model velocities at points
    grad_xy: (M,2) gradients of f at points
    Returns v_grad (M,2), v_tan (M,2), scalar projections v·n_hat (M,)
    """
    gnorm = torch.norm(grad_xy, dim=1, keepdim=True).clamp_min(eps)
    v_norm = torch.norm(v_xy, dim=1, keepdim=True).clamp_min(eps)
    v_hat = v_xy / v_norm
    n_hat = grad_xy / gnorm
    v_dot_n = (v_hat * n_hat).sum(dim=1, keepdim=True)    # component increasing f
    v_grad = v_dot_n * n_hat
    v_tan = v_xy - v_grad
    return v_grad, v_tan, v_dot_n.squeeze(1)

def grid_density_for_plot(density_fn, X, Y, a=2.0, b=1.0, y_std=2.0, device=None, dtype=torch.float32):
    """Evaluate (unnormalized) density on the grid for contour background."""
    if device is None:
        device = torch.device("cpu")

    XY = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=dtype, device=device)
    with torch.no_grad():
        p = density_fn(XY, a=a, b=b, y_std=y_std)
    return p.view(*X.shape).cpu().numpy()

def draw_pmin_boundary(ax, X, Y, P, p_min=None, p_rel=0.2, **contour_kwargs):
    """
    Draws a contour of p(x) > threshold. If p_min is None, uses p_rel * max(P).
    """
    level = (P.max() * p_rel) if (p_min is None) else p_min
    cs = ax.contour(X, Y, P, levels=[level], **contour_kwargs)
    # proxy legend handle
    ax.plot([], [], linestyle=contour_kwargs.get("linestyles", "-."), color=contour_kwargs.get("colors", "white"),
            linewidth=contour_kwargs.get("linewidths", 2.0), label=f"p(x) > {level:.2g}")
    return cs

def plot_flow_decomposition_panels(model,
                                   grad_f_fn=None,    # function (x)->grad, defaults to grad_f_linear
                                   a: float = 2.0,
                                   b: float = 1.0,
                                   y_std: float = 2.0,
                                   t_eval: float = 0.5,
                                   xlim=(-3.0, 3.0),
                                   ylim=(-3.0, 3.0),
                                   N: int = 30,
                                   quiver_step: int = 2,
                                   device=None,
                                   dtype=torch.float32,
                                   add_pmin_boundary: bool = True,
                                   p_rel: float = 0.2):
    """
    Multi-panel decomposition:
      (1) total model v with density
      (2) gradient-aligned component v_grad
      (3) tangential residual v_tan
      (4) scalar map of v·n_hat (rate of increase of f)
    """
    import matplotlib.pyplot as plt

    if device is None:
        device = torch.device("cpu")
    if grad_f_fn is None:
        grad_f_fn = lambda x: grad_f_linear(x, a=a, b=b)

    # Grid + model field
    X, Y, v_np = model_vector_field_on_grid(model, t_eval, xlim, ylim, N, device, dtype)
    XY = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=dtype, device=device)
    v = torch.tensor(v_np.reshape(-1, 2), dtype=dtype, device=device)

    # Gradient field on grid
    with torch.no_grad():
        g = grad_f_fn(XY)  # (N*N, 2)

    # Decompose
    v_grad, v_tan, vdotn = decompose_flow(v, g)
    v_tot = v

    # For plotting
    P = grid_density_for_plot(density_special_gaussian, X, Y, a=a, b=b, y_std=y_std, device=device, dtype=dtype)

    def _quiver(ax, U, V, title, scale=20, color=None):
        ax.contourf(X, Y, P, levels=20, cmap='viridis', alpha=0.65)
        if add_pmin_boundary:
            draw_pmin_boundary(ax, X, Y, P, p_min=None, p_rel=p_rel, linewidths=2.0, linestyles="-.", colors="white")
        ax.quiver(X[::quiver_step, ::quiver_step],
                  Y[::quiver_step, ::quiver_step],
                  U[::quiver_step, ::quiver_step],
                  V[::quiver_step, ::quiver_step],
                  scale=scale, width=0.003, color=color)
        ax.set_title(title)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    # reshape to grid
    VG = v_grad.view(N, N, 2).cpu().numpy()
    VT = v_tan.view(N, N, 2).cpu().numpy()
    V  = v_tot.view(N, N, 2).cpu().numpy()
    Vdot = vdotn.view(N, N).cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (1) total
    _quiver(axes[0, 0], V[:, :, 0], V[:, :, 1], f"Total model field v(x, t={t_eval:.2f})", scale=25, color='red')

    # (2) gradient-aligned
    _quiver(axes[0, 1], VG[:, :, 0], VG[:, :, 1], "Gradient component (increases f)", scale=25, color='blue')

    # (3) tangential
    _quiver(axes[1, 0], VT[:, :, 0], VT[:, :, 1], "Tangential residual (shape control)", scale=8, color='green')

    # (4) scalar map: v · n̂
    im = axes[1, 1].contourf(X, Y, Vdot, levels=20, cmap='coolwarm', alpha=0.95)
    axes[1, 1].contour(X, Y, P, levels=10, linewidths=0.6, colors='k', alpha=0.3)
    if add_pmin_boundary:
        draw_pmin_boundary(axes[1, 1], X, Y, P, p_min=None, p_rel=p_rel, linewidths=2.0, linestyles="-.", colors="white")
    cbar = fig.colorbar(im, ax=axes[1, 1], label=r"$v \cdot \hat{n}$ (rate of increase of $f$)")
    axes[1, 1].set_title("Scalar projection on gradient")
    axes[1, 1].set_xlabel("x₁"); axes[1, 1].set_ylabel("x₂")
    axes[1, 1].set_xlim(*xlim); axes[1, 1].set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

# --- NEW: Gradient flow simulation (ascent or descent) starting from high-impression seeds ---
def simulate_gradient_flow(x0: torch.Tensor,
                           steps: int = 15,
                           step_size: float = 0.2,
                           ascent: bool = True) -> torch.Tensor:
    """
    Euler integrate dx/dt = ±∇f(x) from initial seeds x0.
    Returns trajectories as (T, N, 2) tensor.
    """
    sign = 1.0 if ascent else -1.0
    x = x0.clone()
    traj = [x.clone()]
    for _ in range(steps - 1):
        # compute gradient (requires a gradient-enabled copy)
        x_req = x.clone().detach().requires_grad_(True)
        g = compute_impression_gradient(x_req).detach()
        x = x + step_size * sign * g
        traj.append(x.clone())
    return torch.stack(traj, dim=0)  # (T, N, 2)

def plot_gradient_flow_from_high(distribution_sampler,
                                 impression_fn,
                                 n_seed_pool: int = 5000,
                                 n_seeds: int = 120,
                                 quantile: float = 0.5,
                                 steps: int = 150,
                                 step_size: float = 0.02,
                                 ascent: bool = True,
                                 xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
                                 show_support: bool = True,
                                 support_method: str = "constraints",
                                 p_min: float | None = None,
                                 p_min_rel: float = 0.05,
                                 bandwidth: float = 0.15,
                                 title: str | None = None):
    """
    Draw gradient flow trajectories starting at the high-impression half of the distribution.
    """
    import matplotlib.pyplot as plt

    # Seed selection from high half
    x_all = distribution_sampler(n_seed_pool)
    _, x_high, _ = split_by_impression(x_all, quantile=quantile)
    if len(x_high) == 0:
        raise RuntimeError("No high-impression seeds found; try lowering quantile.")
    idx = torch.randperm(len(x_high))[:min(n_seeds, len(x_high))]
    seeds = x_high[idx]

    # Simulate flow (gradient ascent by default)
    traj = simulate_gradient_flow(seeds, steps=steps, step_size=step_size, ascent=ascent).cpu().numpy()

    # Background: impression heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    xr = np.linspace(xlim[0], xlim[1], 160)
    yr = np.linspace(ylim[0], ylim[1], 160)
    Xg, Yg = np.meshgrid(xr, yr)
    G = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=torch.float32)
    Z = impression_fn(G).detach().numpy().reshape(Xg.shape)
    cont = ax.contourf(Xg, Yg, Z, levels=24, cmap='viridis', alpha=0.6)
    plt.colorbar(cont, ax=ax, label='Impression value')

    # Plot gradient-flow trajectories
    for p in traj.transpose(1, 0, 2):
        ax.plot(p[:, 0], p[:, 1], 'k-', alpha=0.35, linewidth=0.8)
        ax.scatter(p[0, 0], p[0, 1], c='white', s=8, alpha=0.8, edgecolor='k')  # start

    # Optional support boundary
    if show_support:
      draw_support_boundary(ax,density_fn=density_special_gaussian,density_kwargs=dict(a=2.0, b=1.0, y_std=2.0),
            xlim=(-3, 3), ylim=(-3, 3), resolution=300, p_min=None, p_rel=0.25, linewidth=2.0, linestyle="-.", color="white")


    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    if title is None:
        title = f"Gradient {'ascent' if ascent else 'descent'} from high-impression half"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()



# ----- (optional) compatibility alias if other code expects the old name -----
# --- Lightweight Torch KDE for 2D (chunked; no external deps) ---
@torch.no_grad()
def kde_on_grid(samples: torch.Tensor,
                grid_points: torch.Tensor,
                bandwidth: float = 0.15,
                sample_chunk: int = 4096,
                grid_chunk: int = 8192) -> torch.Tensor:
    """
    Gaussian KDE estimate p(x) on grid_points.
    samples: (N,2), grid_points: (G,2)
    Returns: density (G,)
    """
    N, d = samples.shape
    assert d == 2, "This KDE is implemented for 2D."
    const = 1.0 / (N * ((2.0 * np.pi) ** (d / 2)) * (bandwidth ** d))
    dens = torch.zeros(grid_points.shape[0], dtype=torch.float32, device=samples.device)
    h2 = bandwidth * bandwidth

    for gi in range(0, grid_points.shape[0], grid_chunk):
        Pg = grid_points[gi:gi + grid_chunk]               # (Gg,2)
        acc = torch.zeros(Pg.shape[0], device=samples.device)
        for si in range(0, samples.shape[0], sample_chunk):
            S = samples[si:si + sample_chunk]              # (Ss,2)
            diff = Pg[:, None, :] - S[None, :, :]          # (Gg,Ss,2)
            dist2 = (diff * diff).sum(-1)                  # (Gg,Ss)
            acc += torch.exp(-0.5 * dist2 / h2).sum(dim=1) # (Gg,)
        dens[gi:gi + grid_chunk] = const * acc
    return dens

def _make_grid(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), n=300, device="cpu"):
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    X, Y = np.meshgrid(xs, ys)
    G = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
    return X, Y, G

@torch.enable_grad()
def compute_impression_gradient(x):
    """Compute the gradient of the impression function at points x"""
    x = x.clone().detach().requires_grad_(True)
    f = impression_function(x)
    grad = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
    return grad

grad_impression =compute_impression_gradient

# Modified training loop for Psychometric Gradient Flow Matching
def plot_trajectories_with_impression(trajectories, impression_fn, title=""):
    """Plot trajectories overlaid on impression function heatmap"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Create grid for impression function visualization
    x_range = np.linspace(-1.5, 1.5, 100)
    y_range = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Compute impression values on grid
    grid_points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    Z = impression_fn(grid_points).detach().numpy().reshape(X.shape)

    # Plot impression function as heatmap
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=ax, label='Impression Value')

    # Plot trajectories
    for traj in trajectories.transpose(1, 0, 2)[:100]:  # Plot subset of trajectories
        ax.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.3, linewidth=0.5)
        ax.scatter(traj[0, 0], traj[0, 1], c='blue', s=5, alpha=0.5)  # Start
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=5, alpha=0.5)  # End

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
