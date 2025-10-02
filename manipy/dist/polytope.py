import torch
import numpy as np
import torch.nn.functional as F


@torch.no_grad()
def kde_hull_blur(
    query_points: torch.Tensor,
    samples: torch.Tensor,
    bandwidth: float = 0.25,
    pixel_size: float = None,
    pad_sigmas: float = 4.0,
    pix_chunk: int = 262144,
) -> torch.Tensor:
    """
    Density = Gaussian-blurred convex hull indicator sampled at arbitrary points.
    - samples: (N,2)
    - query_points: (G,2) arbitrary coordinates (need NOT be on a grid)
    - bandwidth: Gaussian sigma in coordinate units.
    Returns: (G,) tensor on samples.device
    """
    device = samples.device
    dtype = torch.float32

    # Convex hull via monotone chain (CPU/NumPy)
    smp = samples.detach().to('cpu', torch.float32)
    pts = np.unique(smp.numpy(), axis=0)
    if pts.shape[0] < 3:
        return torch.zeros(query_points.shape[0], dtype=dtype, device=device)

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

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
    hull_np = np.array(lower[:-1] + upper[:-1], dtype=np.float32)  # (M,2)
    hull = torch.from_numpy(hull_np).to(device=device, dtype=dtype)  # (M,2)

    # Rasterization setup (no requirement on query_points to be gridded)
    xmin, ymin = hull.min(dim=0).values
    xmax, ymax = hull.max(dim=0).values
    margin = float(pad_sigmas * bandwidth)
    xmin = float(xmin.item() - margin)
    xmax = float(xmax.item() + margin)
    ymin = float(ymin.item() - margin)
    ymax = float(ymax.item() + margin)

    if pixel_size is None:
        pixel_size = max(bandwidth / 3.0, 1e-3)

    W = int(np.ceil((xmax - xmin) / pixel_size)) + 1
    H = int(np.ceil((ymax - ymin) / pixel_size)) + 1
    W = max(W, 3)
    H = max(H, 3)

    # Pixel-center coordinates: x = xmin + (i+0.5)*ps, y = ymin + (j+0.5)*ps
    xs = xmin + (torch.arange(W, device=device, dtype=dtype) + 0.5) * pixel_size
    ys = ymin + (torch.arange(H, device=device, dtype=dtype) + 0.5) * pixel_size
    Xg, Yg = torch.meshgrid(ys, xs, indexing='ij')  # (H,W) note: Y first (rows), then X

    # Point-in-convex-polygon mask on the raster
    v = hull  # (M,2)
    e = v[(torch.arange(v.shape[0], device=device) + 1) % v.shape[0]] - v  # (M,2)
    all_pts = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)  # (H*W,2)

    inside = torch.zeros(all_pts.shape[0], dtype=torch.bool, device=device)
    eps = 1e-9
    for start in range(0, all_pts.shape[0], pix_chunk):
        P = all_pts[start:start + pix_chunk]  # (Pc,2)
        diff = P[None, :, :] - v[:, None, :]  # (M,Pc,2)
        crosses = e[:, None, 0] * diff[:, :, 1] - e[:, None, 1] * diff[:, :, 0]  # (M,Pc)
        mask_pos = (crosses >= -eps).all(dim=0)
        mask_neg = (crosses <= eps).all(dim=0)
        inside[start:start + pix_chunk] = mask_pos | mask_neg

    mask2d = inside.view(H, W).to(dtype)  # 1 inside hull, 0 outside

    # Gaussian blur (separable, isotropic in pixel units)
    sigma_px = max(bandwidth / pixel_size, 1e-6)

    def gaussian_kernel1d(sigma: float):
        r = int(np.ceil(3.0 * sigma))
        x = torch.arange(-r, r + 1, device=device, dtype=dtype)
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = g / g.sum()
        return g

    k = gaussian_kernel1d(sigma_px)
    img = mask2d.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # X blur
    pad_x = (k.numel() // 2, k.numel() // 2, 0, 0)
    img = F.pad(img, pad=pad_x, mode='constant', value=0.0)
    img = F.conv2d(img, k.view(1, 1, 1, -1))
    # Y blur
    pad_y = (0, 0, k.numel() // 2, k.numel() // 2)
    img = F.pad(img, pad=pad_y, mode='constant', value=0.0)
    img = F.conv2d(img, k.view(1, 1, -1, 1))

    blurred = img  # (1,1,H,W)

    # Sample blurred raster at arbitrary query_points via bilinear sampling
    qp = query_points.to(device=device, dtype=dtype)

    # Map to normalized coords for grid_sample with align_corners=True
    # -1 corresponds to pixel center at index 0; +1 corresponds to pixel center at index W-1/H-1
    x_n = 2.0 * (qp[:, 0] - (xmin + 0.5 * pixel_size)) / (max(W - 1, 1) * pixel_size) - 1.0
    y_n = 2.0 * (qp[:, 1] - (ymin + 0.5 * pixel_size)) / (max(H - 1, 1) * pixel_size) - 1.0
    grid = torch.stack([x_n, y_n], dim=-1).view(1, -1, 1, 2)  # (1,G,1,2)

    samp = F.grid_sample(
        blurred, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )  # (1,1,G,1)

    dens = samp.view(-1).to(dtype)
    return dens


def sample_polytope(n: int,
                       x_points: torch.Tensor,
                       k: int,
                       device=None,
                       dtype=torch.float32) -> torch.Tensor:
    """
    Sample n points from the convex hull of k randomly selected points from x_points.
    Uses Dirichlet weights (uniform over the simplex) to form convex combinations.

    Args:
        n: Number of samples to generate.
        x_points: Tensor of shape (M, D) containing candidate x points.
        k: Number of vertices to define the convex hull (randomly chosen from x_points).
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        samples: Tensor of shape (n, D) of samples from the convex hull.
    """
    if device is None:
        device = x_points.device
    x_points = x_points.to(device=device, dtype=dtype)

    M, D = x_points.shape
    if M == 0:
        raise ValueError("x_points is empty.")
    if k < 1:
        raise ValueError("k must be >= 1.")

    # Choose k points (without replacement if possible)
    if k <= M:
        idx = torch.randperm(M, device=device)[:k]
    else:
        idx = torch.randint(0, M, (k,), device=device)

    vertices = x_points[idx]  # (k, D)

    # Dirichlet: sample k exponential(1) variables and normalize
    exp_samples = torch.empty((n, k), device=device, dtype=dtype).exponential_(1.0)
    weights = exp_samples / exp_samples.sum(dim=1, keepdim=True)

    samples = weights @ vertices  # (n, D)
    return samples

