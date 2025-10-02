
def sample_gaussian_around_x(n: int,
                             x_points: torch.Tensor,
                             kernel_std: float = 0.1,
                             device=None,
                             dtype=torch.float32) -> torch.Tensor:
    """
    Sample n points from a Gaussian kernel centered at a randomly selected x from x_points.

    Args:
        n: Number of samples to generate.
        x_points: Tensor of shape (M, D) containing candidate x points.
        kernel_std: Standard deviation of the Gaussian kernel.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        samples: Tensor of shape (n, D) of samples.
    """
    if device is None:
        device = x_points.device
    x_points = x_points.to(device=device, dtype=dtype)
    M, D = x_points.shape
    idx = torch.randint(0, M, (1,))
    x_center = x_points[idx].squeeze(0)  # shape (D,)
    samples = x_center + torch.randn(n, D, device=device, dtype=dtype) * kernel_std
    return samples

import numpy as np
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def prob_in_random_triangle(
    query_points: torch.Tensor,
    samples: torch.Tensor,
    n_triangles: int = 50000,
    tri_batch: int = 256,
    point_chunk: int = 65536,
    eps: float = 1e-9,
) -> torch.Tensor:
    device = samples.device
    dtype = torch.float32

    N = samples.shape[0]
    if N < 3:
        return torch.zeros(query_points.shape[0], device=device, dtype=dtype)

    P_all = query_points.to(device=device, dtype=dtype)
    S = samples.to(device=device, dtype=dtype)

    counts = torch.zeros(P_all.shape[0], device=device, dtype=dtype)
    valid_total = 0.0

    ones_w = torch.ones(N, device=device)

    def cross2(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    for t0 in range(0, n_triangles, tri_batch):
        tb = min(tri_batch, n_triangles - t0)
        idx = torch.stack([torch.multinomial(ones_w, 3, replacement=False) for _ in range(tb)], dim=0)

        A = S[idx[:, 0]]
        B = S[idx[:, 1]]
        C = S[idx[:, 2]]

        AB = B - A
        AC = C - A
        area2 = torch.abs(cross2(AB, AC))
        valid = area2 > eps
        if valid.sum().item() == 0:
            continue

        AB = AB[valid]
        BC = (C - B)[valid]
        CA = (A - C)[valid]
        A_v = A[valid]
        B_v = B[valid]
        C_v = C[valid]

        for p0 in range(0, P_all.shape[0], point_chunk):
            P = P_all[p0:p0 + point_chunk]
            PA = P[None, :, :] - A_v[:, None, :]
            PB = P[None, :, :] - B_v[:, None, :]
            PC = P[None, :, :] - C_v[:, None, :]

            c1 = cross2(AB[:, None, :], PA)
            c2 = cross2(BC[:, None, :], PB)
            c3 = cross2(CA[:, None, :], PC)

            inside = ((c1 >= -eps) & (c2 >= -eps) & (c3 >= -eps)) | ((c1 <= eps) & (c2 <= eps) & (c3 <= eps))
            counts[p0:p0 + P.shape[0]] += inside.sum(dim=0).to(dtype)

        valid_total += float(valid.sum().item())

    if valid_total == 0.0:
        return torch.zeros_like(counts)

    probs = counts / valid_total
    return probs.clamp_(0.0, 1.0)

def _convex_hull_monotone_chain(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float32)

@torch.no_grad()
def visualize_triangle_prob_sets(
    sample_sets,
    titles=None,
    n_triangles: int = 30000,
    grid_res: int = 200,
    margin_frac: float = 0.08,
    tri_batch: int = 512,
    point_chunk: int = 65536,
    scatter_max: int = 400,
    cmap: str = "viridis",
    device: str = None,
):
    """
    Visualize P[x in random triangle from samples] for multiple 2D sample sets.
    - sample_sets: list of array-like/Tensor, each of shape (N,2)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Bounds shared across subplots for comparability
    all_xy = np.concatenate([np.asarray(s, dtype=np.float32) for s in sample_sets if len(s) > 0], axis=0)
    xmin, ymin = all_xy.min(0)
    xmax, ymax = all_xy.max(0)
    dx, dy = xmax - xmin, ymax - ymin
    pad_x, pad_y = margin_frac * (dx if dx > 0 else 1.0), margin_frac * (dy if dy > 0 else 1.0)
    xmin, xmax = float(xmin - pad_x), float(xmax + pad_x)
    ymin, ymax = float(ymin - pad_y), float(ymax + pad_y)

    # Grid
    xs = torch.linspace(xmin, xmax, grid_res, device=device)
    ys = torch.linspace(ymin, ymax, grid_res, device=device)
    Yg, Xg = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)  # (G,2)

    # Compute fields
    fields = []
    for s in sample_sets:
        s_t = torch.as_tensor(np.asarray(s, dtype=np.float32), device=device)
        if s_t.shape[0] < 3:
            fields.append(torch.zeros(grid_res, grid_res, device=device))
            continue
        probs = prob_in_random_triangle(
            query_points=grid,
            samples=s_t,
            n_triangles=n_triangles,
            tri_batch=tri_batch,
            point_chunk=point_chunk,
        ).view(grid_res, grid_res)
        fields.append(probs)

    vmin = min(float(f.min().item()) for f in fields) if fields else 0.0
    vmax = max(float(f.max().item()) for f in fields) if fields else 1.0

    # Plot
    n = len(sample_sets)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.5*nrows), squeeze=False)
    for i, (s, field) in enumerate(zip(sample_sets, fields)):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        im = ax.imshow(
            field.detach().cpu().numpy(),
            extent=(xmin, xmax, ymin, ymax),
            origin="lower",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation="bilinear",
            aspect="equal",
        )
        # Contours
        try:
            cs = ax.contour(
                np.linspace(xmin, xmax, grid_res),
                np.linspace(ymin, ymax, grid_res),
                field.detach().cpu().numpy(),
                levels=[0.1, 0.25, 0.5, 0.75],
                colors="k",
                linewidths=0.7,
                alpha=0.7,
            )
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
        except Exception:
            pass

        # Samples
        s_np = np.asarray(s, dtype=np.float32)
        if s_np.shape[0] > 0:
            idx = np.random.choice(s_np.shape[0], size=min(scatter_max, s_np.shape[0]), replace=False)
            ax.scatter(s_np[idx, 0], s_np[idx, 1], s=8, c="white", edgecolors="k", linewidths=0.3, alpha=0.8, zorder=3)

            hull = _convex_hull_monotone_chain(s_np)
            if len(hull) >= 3:
                ax.plot(np.r_[hull[:,0], hull[0,0]], np.r_[hull[:,1], hull[0,1]], c="w", lw=1.0, alpha=0.9)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        if titles and i < len(titles):
            ax.set_title(titles[i])

    # Hide unused axes
    for j in range(i+1, nrows*ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar.set_label("P(point ∈ random triangle from samples)")
    plt.tight_layout()
    plt.show()
