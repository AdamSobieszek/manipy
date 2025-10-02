"""
levelset_viz.py — Visualize ∂/∂f and ∂/∂d vector fields

A small, self-contained PyTorch module to visualize, on a 2D slice of R^k,
  • the transverse field  v2 = ∇f / ||∇f||^2   (your “∂/∂f” direction), and
  • the eikonal field    n  = ∇f / ||∇f||      (your “∂/∂d” direction),
for a given differentiable nn.Module f: R^k → R.

Core ideas
----------
We choose a 2D affine plane Π ⊂ R^k represented as x(u) = x0 + B u with u ∈ R^2
and B ∈ R^{k×2} having orthonormal columns (we QR-orthonormalize if needed).
On a grid in (u1,u2)-coordinates we evaluate f, compute per-point gradients
via autograd, then project the ambient vector fields to Π via the 2D coordinate
map. Visualizations use matplotlib (quiver + contours).

Quick start
-----------
from levelset_viz import Projection2D, sample_and_fields, visualize_fields
import torch

# 1) Define your model f: R^k → R
class Toy(torch.nn.Module):
    def forward(self, x):
        # x: (..., k)
        return (x[...,0]**2 + 0.5*x[...,1]**2 - 1.0)  # R^2→R example

f = Toy()

# 2) Choose a 2D slice of R^k (here k=2, so the whole space)
proj = Projection2D.base_dims(k=2, dims=(0,1))

# 3) Sample, compute fields, and visualize
uv_bounds = ((-1.8, 1.8), (-1.8, 1.8))
U, V, out = sample_and_fields(f, proj, uv_bounds, resolution=121, eps=1e-6, device='cpu')
visualize_fields(U, V, out, level_y=0.0, show_v2=True, show_n=True,
                 quiver_stride=4, title='∂/∂f (v2) and ∂/∂d (n) for a toy f')

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import numpy as np
import matplotlib.pyplot as plt

Tensor = torch.Tensor

# -----------------------------------------------------------------------------
# Geometry of the 2D slice Π ⊂ R^k
# -----------------------------------------------------------------------------

@dataclass
class Projection2D:
    """A 2D affine slice Π of R^k:  x(u) = x0 + B u,  u ∈ R^2.

    Parameters
    ----------
    x0 : (k,) tensor — base point of the plane
    B  : (k,2) tensor — columns are spanning vectors (not necessarily orthonormal)

    Internals
    ---------
    We store an orthonormal basis Q with shape (k,2) satisfying Q^T Q = I_2 and
    Q spans the same plane as B. This ensures projection and coordinates are
    numerically stable and metric-consistent.
    """
    x0: Tensor  # (k,)
    B: Tensor   # (k,2)

    def __post_init__(self):
        assert self.x0.ndim == 1, "x0 must be 1D (k,)"
        assert self.B.ndim == 2 and self.B.shape[1] == 2, "B must be (k,2)"
        # Orthonormalize columns of B via QR
        # Ensure on the same device/dtype as B
        Q, R = torch.linalg.qr(self.B, mode='reduced')  # (k,2), (2,2)
        # Guard against degenerate B
        if torch.linalg.matrix_rank(R) < 2:
            raise ValueError("Projection2D: provided basis B is rank-deficient.")
        self.Q = Q  # (k,2), orthonormal columns
        self.k = self.x0.numel()

    # Embedding from 2D coords to ambient
    def embed(self, U: Tensor, V: Tensor) -> Tensor:
        """Map meshgrid (U,V) to ambient points X on Π.
        U,V: (H,W)
        Returns X: (H,W,k)
        """
        H, W = U.shape
        uv = torch.stack([U, V], dim=-1)          # (H,W,2)
        X = self.x0.view(1,1,-1) + uv @ self.Q.mT # (H,W,k)
        return X

    # Project an ambient vector field to Π and express in (u,v) coordinates
    def project_vectors(self, Vfield: Tensor) -> Tuple[Tensor, Tensor]:
        """Project ambient vectors to Π and express as 2D components.
        Vfield: (H,W,k) ambient vectors at grid points on Π
        Returns (Vu, Vv): both (H,W) giving 2D components in the Q-basis.
        Since Q has orthonormal columns, coordinates are simply Q^T v.
        """
        # coords (H,W,2) = (H,W,k) · (k,2)
        uv = Vfield @ self.Q  # (H,W,2)
        return uv[...,0], uv[...,1]

    # Build from canonical coordinate axes (useful when k≥2)
    @staticmethod
    def base_dims(k: int, dims: Tuple[int,int] = (0,1), x0: Optional[Tensor] = None) -> 'Projection2D':
        x0 = torch.zeros(k) if x0 is None else x0
        e = torch.eye(k)
        B = torch.stack([e[:,dims[0]], e[:,dims[1]]], dim=1)  # (k,2)
        return Projection2D(x0=x0, B=B)

# -----------------------------------------------------------------------------
# Sampling and autodiff machinery
# -----------------------------------------------------------------------------

def _ensure_callable_model(f):
    if not hasattr(f, 'forward'):
        raise TypeError("Expected an nn.Module-like object with a forward(x) method.")

@torch.no_grad()
def _make_meshgrid(bounds: Tuple[Tuple[float,float], Tuple[float,float]], resolution: int, device: str, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    (u0,u1), (v0,v1) = bounds
    u = torch.linspace(u0, u1, resolution, device=device, dtype=dtype)
    v = torch.linspace(v0, v1, resolution, device=device, dtype=dtype)
    U, V = torch.meshgrid(u, v, indexing='ij')
    return U, V


def _eval_and_grad(f, X: Tensor, create_graph: bool = False) -> Tuple[Tensor, Tensor]:
    """Evaluate f and compute per-point gradients ∇f(x) for a batch of X.

    X: (N,k) with requires_grad=True
    Returns: (fvals, grad) with shapes (N,1) and (N,k)
    """
    # Evaluate
    y = f(X)  # expect (N,) or (N,1)
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    if y.shape[-1] != 1:
        raise ValueError("Model must output a scalar per input point.")
    # Compute gradient per sample using autograd with a matching grad_outputs
    grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(outputs=y, inputs=X, grad_outputs=grad_outputs,
                               create_graph=create_graph, retain_graph=create_graph,
                               only_inputs=True)[0]
    return y, grad


def sample_and_fields(
    f,
    proj: Projection2D,
    uv_bounds: Tuple[Tuple[float,float], Tuple[float,float]],
    resolution: int = 121,
    eps: float = 1e-6,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    compute_graph: bool = False,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """Sample a grid on Π, evaluate f and compute fields.

    Returns
    -------
    U, V : (H,W) coordinate grids
    out  : dict of tensors, each (H,W,*) shaped
        'X'    : (H,W,k) ambient points
        'f'    : (H,W) scalar field values f(x)
        'grad' : (H,W,k) gradient ∇f
        'norm' : (H,W,1) ||∇f||
        'n'    : (H,W,k) normalized gradient (∂/∂d)
        'v2'   : (H,W,k) transverse field (∂/∂f) = ∇f / ||∇f||^2
        'mask' : (H,W,1) boolean where ||∇f|| > eps
    """
    _ensure_callable_model(f)
    device = str(device)
    U, V = _make_meshgrid(uv_bounds, resolution, device=device, dtype=dtype)

    # Embed to ambient and flatten for batch eval
    X = proj.embed(U, V)                              # (H,W,k)
    H, W, k = X.shape
    Xf = X.reshape(-1, k).to(device=device, dtype=dtype)
    Xf.requires_grad_(True)

    # Evaluate and grad
    y, g = _eval_and_grad(f, Xf, create_graph=compute_graph)
    y = y.reshape(H, W, 1)
    g = g.reshape(H, W, k)

    # Norms and fields
    norm = g.pow(2).sum(dim=-1, keepdim=True).sqrt()            # (H,W,1)
    mask = (norm > eps)
    safe = torch.where(mask, norm, torch.full_like(norm, eps))  # avoid div by 0

    n = g / safe                                                # ∂/∂d
    v2 = g / (safe*safe)                                       # ∂/∂f

    out = {
        'X': X, 'f': y.squeeze(-1), 'grad': g, 'norm': norm,
        'n': n, 'v2': v2, 'mask': mask
    }
    return U, V, out

# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def visualize_fields(
    U: Tensor,
    V: Tensor,
    out: Dict[str, Tensor],
    *,
    level_y: Optional[float] = None,
    show_v2: bool = True,
    show_n: bool = True,
    quiver_stride: int = 5,
    scale: Optional[float] = None,
    arrow_width: float = 0.002,
    figsize: Tuple[float,float] = (7.0, 6.0),
    title: Optional[str] = None,
    cmap: str = 'viridis',
    show: bool = True,
):
    """Plot contours of f and quiver plots for v2 (∂/∂f) and n (∂/∂d).

    Parameters
    ----------
    level_y : value for the highlighted level set {f = level_y}. If None, we
               draw default contours of f.
    quiver_stride : subsampling step for the quiver grid
    scale : matplotlib quiver scale; if None, auto-scales based on field magnitudes
    """
    f = out['f'].detach().cpu().numpy()       # (H,W)
    mask = out['mask'].squeeze(-1).cpu().numpy().astype(bool)

    # Project ambient vector fields to 2D components (u,v)
    # We need the Projection2D to do this; infer Q from the sampling output X
    # Trick: recover Q by finite differences in U,V → X mapping if not passed.
    # Easier: we stored fields in ambient; here we approximate 2D components by
    # taking dot products with the basis that generated U,V. To keep API simple,
    # we assume U,V came from Projection2D.embed with an orthonormal Q and we
    # reconstruct Q numerically from the first row variations.
    #
    # Robust reconstruction (exact if grids are regular):
    Xu0 = out['X'][1,0] - out['X'][0,0]   # ≈ Q[:,0] * Δu
    Xv0 = out['X'][0,1] - out['X'][0,0]   # ≈ Q[:,1] * Δv
    du = float(U[1,0] - U[0,0].item())
    dv = float(V[0,1] - V[0,0].item())
    q0 = (Xu0 / (du + 1e-12))
    q1 = (Xv0 / (dv + 1e-12))
    # Orthonormalize in case of numerical noise
    Q = torch.stack([q0, q1], dim=1)
    Q, _ = torch.linalg.qr(Q, mode='reduced')   # (k,2)

    def to_uv(vecHWk: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        uv = (vecHWk @ Q).detach().cpu().numpy()   # (H,W,2)
        return uv[...,0], uv[...,1]

    U_np = U.detach().cpu().numpy(); V_np = V.detach().cpu().numpy()
    H, W = U_np.shape

    # Subsample for quiver
    sl = (slice(None,None,quiver_stride), slice(None,None,quiver_stride))
    Uq, Vq = U_np[sl], V_np[sl]

    plt.figure(figsize=figsize)

    # Contours of f
    if level_y is not None:
        # Highlight the requested level and add context contours
        CS = plt.contour(U_np, V_np, f, levels=[level_y], colors='k', linewidths=2.0)
        plt.contour(U_np, V_np, f, levels=15, cmap=cmap, alpha=0.6)
        plt.clabel(CS, inline=True, fontsize=9, fmt={level_y: f"f={level_y}"})
    else:
        plt.contour(U_np, V_np, f, levels=20, cmap=cmap)

    # v2 quiver (∂/∂f)
    if show_v2:
        Vu, Vv = to_uv(out['v2'])
        Vuq, Vvq = Vu[sl], Vv[sl]
        Mk = mask[sl]
        plt.quiver(Uq[Mk], Vq[Mk], Vuq[Mk], Vvq[Mk],
                   angles='xy', scale_units='xy', scale=scale, width=arrow_width,
                   color='tab:orange', alpha=0.9, label='∂/∂f (v2)')

    # n quiver (∂/∂d)
    if show_n:
        Nu, Nv = to_uv(out['n'])
        Nuq, Nvq = Nu[sl], Nv[sl]
        Mk = mask[sl]
        plt.quiver(Uq[Mk], Vq[Mk], Nuq[Mk], Nvq[Mk],
                   angles='xy', scale_units='xy', scale=scale, width=arrow_width,
                   color='tab:blue', alpha=0.7, label='∂/∂d (n)')

    plt.xlabel('u'); plt.ylabel('v')
    if title: plt.title(title)
    plt.legend(loc='upper right')
    plt.axis('equal'); plt.tight_layout()
    if show:
        plt.show()

# -----------------------------------------------------------------------------
# Convenience: streamlines using matplotlib's streamplot (for ∂/∂f or ∂/∂d)
# -----------------------------------------------------------------------------

def streamplot_field(
    U: Tensor, V: Tensor, out: Dict[str, Tensor], which: str = 'v2',
    density: float = 1.0, minlength: float = 0.1, maxlength: float = 4.0,
    color: str = 'k', linewidth: float = 1.0, title: Optional[str] = None,
    show: bool = True,
):
    """Streamplot of either 'v2' (∂/∂f) or 'n' (∂/∂d)."""
    assert which in ('v2','n')
    f = out['f'].detach().cpu().numpy()
    mask = out['mask'].squeeze(-1).cpu().numpy().astype(bool)

    Xu0 = out['X'][1,0] - out['X'][0,0]
    Xv0 = out['X'][0,1] - out['X'][0,0]
    du = float(U[1,0] - U[0,0].item()); dv = float(V[0,1] - V[0,0].item())
    q0 = (Xu0 / (du + 1e-12)); q1 = (Xv0 / (dv + 1e-12))
    Q = torch.stack([q0, q1], dim=1)
    Q, _ = torch.linalg.qr(Q, mode='reduced')

    def to_uv(vecHWk: Tensor):
        uv = (vecHWk @ Q).detach().cpu().numpy()
        return uv[...,0], uv[...,1]

    U_np = U.detach().cpu().numpy(); V_np = V.detach().cpu().numpy()
    Vu, Vv = to_uv(out[which])
    Vu = np.where(mask, Vu, 0.0); Vv = np.where(mask, Vv, 0.0)

    plt.figure(figsize=(7,6))
    plt.contour(U_np, V_np, f, levels=20, cmap='Greys', alpha=0.6)
    plt.streamplot(U_np, V_np, Vu, Vv, density=density, minlength=minlength,
                   maxlength=maxlength, color=color, linewidth=linewidth)
    if title: plt.title(title)
    plt.xlabel('u'); plt.ylabel('v'); plt.axis('equal'); plt.tight_layout()
    if show:
        plt.show()

# -----------------------------------------------------------------------------
# If run as a script, demo on a simple analytic f
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    class Toy(torch.nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return (x[...,0]**2 + 0.5*x[...,1]**2 - 1.0)

    f = Toy()
    proj = Projection2D.base_dims(k=2, dims=(0,1))
    U, V, out = sample_and_fields(f, proj, ((-1.8,1.8),(-1.8,1.8)), resolution=141)

    visualize_fields(U, V, out, level_y=0.0, quiver_stride=4,
                     title='Toy f: ∂/∂f (orange) and ∂/∂d (blue)')
    streamplot_field(U, V, out, which='v2', density=1.0,
                     title='Streamlines of ∂/∂f (v2)')
