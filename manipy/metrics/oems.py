import torch
import numpy as np

def plot_mean_increase_and_mse_along_flow(
    model,
    ensemble,
    impression_function,
    X: torch.Tensor,
    *,
    step_size: float = 0.05,
    n_steps: int = 150,
    normalize_step: bool = True,
    show_predicted_curve: bool = True,
    figsize=(11, 4.2),
    device: torch.device | None = None,
):
    """
    Integrate the learned flow field from starting points X and summarize
    how the impression evolves with traveled distance.

    Plots:
      Left:  Mean increase of TRUE impression (impression_function) vs. distance.
             Optionally overlays mean increase of the ENSEMBLE (dashed).
      Right: MSE of ensemble vs. true impression along the traversed distance.

    Args:
        model:             vector field R^2 -> R^2 (torch.nn.Module or callable)
        ensemble:          scalar field R^2 -> R (your predictor)
        impression_function: ground-truth scalar field R^2 -> R
        X:                 (N, 2) starting points
        step_size:         arc-length step per integration step (if normalize_step=True)
        n_steps:           number of integration steps
        normalize_step:    if True, take steps of fixed arc length (recommended)
        show_predicted_curve: overlay ensemble's mean ∆ as dashed curve
        figsize:           matplotlib figure size
        device:            torch device (defaults to model's device or X.device)

    Returns:
        Dict with numpy arrays for further analysis:
            {
              "distances": (T+1,),
              "mean_delta_true": (T+1,),
              "std_delta_true": (T+1,),
              "mean_delta_pred": (T+1,),     # if show_predicted_curve
              "mse": (T+1,)
            }
    """
    import matplotlib.pyplot as plt

    model_device = next(model.parameters()).device if hasattr(model, "parameters") else None
    if device is None:
        device = getattr(X, "device", None) or model_device or torch.device("cpu")

    X = X.detach().to(device).float()
    y = impression_function(X).reshape(-1)
    X = X[y.flatten() < y.median()]
    N = X.shape[0]
    assert X.ndim == 2 and X.shape[1] == 2, "X must be (N,2)"

    T = n_steps
    with torch.no_grad():
        # Storage
        pos = torch.empty((T + 1, N, 2), device=device)
        true_vals = torch.empty((T + 1, N), device=device)
        pred_vals = torch.empty((T + 1, N), device=device)

        # t = 0
        x = X.clone()
        pos[0] = x
        tv0 = impression_function(x).reshape(-1)
        pv0 = ensemble(x).reshape(-1)
        true_vals[0] = tv0
        pred_vals[0] = pv0

        eps = 1e-8
        for t in range(1, T + 1):
            v = model(x)                                # (N,2)
            if normalize_step:
                v_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
                dx = (step_size * v / v_norm)           # fixed arc-length step
            else:
                dx = step_size * v                      # Euler step in field units

            x = x + dx
            pos[t] = x
            true_vals[t] = impression_function(x).reshape(-1)
            pred_vals[t] = ensemble(x).reshape(-1)

        # Distances traveled (arc length if normalize_step=True)
        distances = torch.arange(T + 1, device=device, dtype=torch.float32) * step_size

        # Mean increase relative to start (true and predicted)
        delta_true = true_vals - true_vals[0:1]         # (T+1,N)
        mean_delta_true = delta_true.mean(dim=1)        # (T+1,)
        std_delta_true = delta_true.std(dim=1)          # (T+1,)

        delta_pred = pred_vals - pred_vals[0:1]
        mean_delta_pred = delta_pred.mean(dim=1)

        # MSE of ensemble vs true along the traversed positions
        mse = ((pred_vals - true_vals) ** 2).mean(dim=1)  # (T+1,)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    d = distances.detach().cpu().numpy()
    m_true = mean_delta_true.detach().cpu().numpy()
    s_true = std_delta_true.detach().cpu().numpy()
    m_pred = mean_delta_pred.detach().cpu().numpy()
    mse_np = mse.detach().cpu().numpy()

    # Left subplot: mean ∆ impression vs distance
    ax1.plot(d, m_true, label="True mean Δ impression", linewidth=2)
    ax1.fill_between(d, m_true - s_true, m_true + s_true, alpha=0.2, label="±1σ (true)")
    if show_predicted_curve:
        ax1.plot(d, m_pred, linestyle="--", linewidth=2, label="Ensemble mean Δ impression")
    ax1.set_xlabel("Distance traveled")
    ax1.set_ylabel("Mean Δ impression")
    ax1.set_title("Mean increase along model flow")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right subplot: MSE vs distance
    ax2.plot(d, mse_np, linewidth=2)
    ax2.set_xlabel("Distance traveled")
    ax2.set_ylabel("MSE: (ensemble − true)$^2$")
    ax2.set_title("Ensemble MSE along model flow")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    out = {
        "distances": d,
        "mean_delta_true": m_true,
        "std_delta_true": s_true,
        "mse": mse_np,
    }
    if show_predicted_curve:
        out["mean_delta_pred"] = m_pred
    return out

# Assume: model (R^2->R^2), ensemble (R^2->R), impression_function (R^2->R) are defined,
# and X is a batch of starting points (N,2) — e.g., a row along the x-axis at y=const.

# Example: start from a row on the x-axis
# xs = torch.linspace(-1.5, 1.5, 128)
# X_row = torch.stack([xs, torch.zeros_like(xs)], dim=-1)
# =========================
# Manipulation Path Toolkit
# =========================
import torch
import numpy as np
from typing import Callable, Dict, Tuple, Optional

# -----------------------------
# 0) Path integration utilities
# -----------------------------
@torch.no_grad()
def integrate_flow_arc_length(
    model: Callable[[torch.Tensor], torch.Tensor],
    X0: torch.Tensor,
    *,
    step_size: float = 0.05,
    n_steps: int = 150,
    normalize_step: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Follow the learned manipulation vector field to generate paths γ(ℓ).
    If `normalize_step=True`, each Euler step has length `step_size`,
    so the horizontal axis is exactly traveled distance ℓ.

    Returns:
        {
          "pos": (T+1, N, d) positions along path,
          "dx": (T,   N, d) increments per step,
          "distances": (T+1,) ℓ grid,
        }
    """
    eps = 1e-8
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = X0.device
    X0 = X0.detach().to(device).float()
    T = int(n_steps)
    N, d = X0.shape

    pos = torch.empty((T + 1, N, d), device=device)
    dxs = torch.empty((T, N, d), device=device)

    x = X0.clone()
    pos[0] = x
    for t in range(1, T + 1):
        v = model(x)                                  # (N,d)
        if normalize_step:
            v = v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))
        dx = step_size * v
        x = x + dx
        dxs[t - 1] = dx
        pos[t] = x

    distances = torch.arange(T + 1, device=device, dtype=torch.float32) * step_size \
                if normalize_step else \
                torch.cat([torch.zeros(1, device=device),
                           dxs.norm(dim=-1).sum(dim=0).mean().new_tensor(
                               [dxs.norm(dim=-1).mean(dim=1)[:t].sum().item()
                                for t in range(1, T+1)]
                           )])  # fallback if not normalized

    return {"pos": pos, "dx": dxs, "distances": distances}


# -------------------------------------------------------
# 1) Path risk  ρ(ℓ) := MSE(f̂_y(x) , y(x)) along γ(ℓ)
# -------------------------------------------------------
@torch.no_grad()
def compute_path_risk(
    ensemble: Callable[[torch.Tensor], torch.Tensor],
    impression_function: Callable[[torch.Tensor], torch.Tensor],
    pos: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Computes the expected risk at each arc-length step:
        ρ(ℓ_t) = E_paths[ (ensemble(x_t) - y_true(x_t))^2 ].
    pos: (T+1, N, d) positions along paths.
    """
    T1, N, _ = pos.shape
    y_true = impression_function(pos.view(-1, pos.shape[-1])).view(T1, N)
    y_pred = ensemble(pos.view(-1, pos.shape[-1])).view(T1, N)
    se = (y_pred - y_true) ** 2
    mse = se.mean(dim=1)            # (T+1,)
    se_std = se.std(dim=1)          # optional dispersion
    return {"mse": mse, "se_std": se_std, "y_true": y_true, "y_pred": y_pred}


def plot_path_risk(distances: torch.Tensor, mse: torch.Tensor, se_std: Optional[torch.Tensor] = None,
                   title: str = "Risk along manipulation path (MSE)") -> None:
    import matplotlib.pyplot as plt
    d = distances.detach().cpu().numpy()
    m = mse.detach().cpu().numpy()
    plt.figure(figsize=(6.5, 4.0))
    plt.plot(d, m, linewidth=2, label="MSE (ensemble vs. true)")
    if se_std is not None:
        s = se_std.detach().cpu().numpy()
        plt.fill_between(d, np.maximum(0.0, m - s), m + s, alpha=0.2, label="±1σ across paths")
    plt.xlabel("Distance ℓ")
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 2) Manipulation efficiency η(ℓ): increase per unit distance
#     - cumulative increase:   Δy(ℓ) = E[ y(ℓ) - y(0) ]
#     - cumulative efficiency: η(ℓ) = Δy(ℓ) / max(ℓ, ε)
#     - optional local slope:  d/dℓ E[y(ℓ)] (finite-difference)
# -------------------------------------------------------------------
@torch.no_grad()
def compute_manipulation_efficiency(
    y_true_along: torch.Tensor,        # (T+1, N)
    distances: torch.Tensor,           # (T+1,)
) -> Dict[str, torch.Tensor]:
    """
    Implements the paper's manipulation-efficiency notion:
        Δy(ℓ)  := E_paths[ y(ℓ) - y(0) ]
        η(ℓ)   := Δy(ℓ) / ℓ   (with η(0) undefined -> NaN)

    Also returns a smoothed finite-difference estimate of local efficiency slope.
    """
    eps = 1e-12
    mu_y = y_true_along.mean(dim=1)                         # (T+1,)
    delta = mu_y - mu_y[0]                                  # Δy(ℓ)
    eff = delta / torch.clamp(distances, min=eps)           # η(ℓ); η(0)=Δ/0 -> inf -> set to NaN
    eff[0] = torch.tensor(float("nan"), device=eff.device)

    # Local slope dy/dℓ via central difference (for analysis/optional plotting)
    d = distances
    dy = torch.empty_like(mu_y)
    dy[0] = (mu_y[1] - mu_y[0]) / (d[1] - d[0] + eps)
    dy[-1] = (mu_y[-1] - mu_y[-2]) / (d[-1] - d[-2] + eps)
    dy[1:-1] = (mu_y[2:] - mu_y[:-2]) / (d[2:] - d[:-2] + eps)

    return {"delta_mean": delta, "efficiency": eff, "mean_y": mu_y, "local_slope": dy}


def plot_manipulation_efficiency(
    distances: torch.Tensor,
    delta_mean: torch.Tensor,
    efficiency: torch.Tensor,
    local_slope: Optional[torch.Tensor] = None,
    title_left: str = "Manipulation increase Δy(ℓ)",
    title_right: str = "Efficiency η(ℓ) = Δy(ℓ)/ℓ",
) -> None:
    import matplotlib.pyplot as plt
    d = distances.detach().cpu().numpy()
    dm = delta_mean.detach().cpu().numpy()
    eff = efficiency.detach().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(d, dm, linewidth=2)
    axs[0].set_xlabel("Distance ℓ")
    axs[0].set_ylabel("Mean increase Δy")
    axs[0].set_title(title_left)
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(d, eff, linewidth=2)
    axs[1].set_xlabel("Distance ℓ")
    axs[1].set_ylabel("η(ℓ)")
    axs[1].set_title(title_right)
    axs[1].grid(True, alpha=0.3)

    if local_slope is not None:
        ls = local_slope.detach().cpu().numpy()
        axs[0].plot(d, ls, linestyle="--", alpha=0.6, label="local slope dy/dℓ")
        axs[0].legend()

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# 3) KDE on training set X  →  OOD likelihood along the path
#    - Gaussian product kernel with diagonal bandwidth
#    - Scott's rule (default) or fixed bandwidth
# -------------------------------------------------------------
@torch.no_grad()
def fit_gaussian_kde(
    X_train: torch.Tensor,
    *,
    bandwidth: Optional[torch.Tensor] = None,  # (d,) or scalar
    rule: str = "scott",                       # "scott" | "silverman" | "fixed"
    max_ref_points: int = 5000,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Simple Gaussian KDE with diagonal bandwidth for arbitrary d.
    Uses Scott/Silverman rules unless `bandwidth` is provided.
    Subsamples X_train to at most `max_ref_points` for tractability.
    """
    if device is None:
        device = X_train.device
    X_train = X_train.detach().to(device).float()
    n, d = X_train.shape

    # Subsample reference set for speed if needed
    if n > max_ref_points:
        idx = torch.randperm(n, device=device)[:max_ref_points]
        X_ref = X_train[idx]
    else:
        X_ref = X_train

    # Diagonal bandwidth
    if bandwidth is None:
        # Per-dim std
        std = X_train.std(dim=0, unbiased=True).clamp_min(1e-8)
        if rule.lower() == "scott":
            h_scalar = (X_train.shape[0]) ** (-1.0 / (d + 4))
            bw = h_scalar * std
        elif rule.lower() == "silverman":
            h_scalar = (X_train.shape[0] * (d + 2) / 4.0) ** (-1.0 / (d + 4))
            bw = h_scalar * std
        else:
            raise ValueError("When rule='fixed', provide `bandwidth`.")
    else:
        bw = bandwidth.to(device).float()
        if bw.ndim == 0:
            bw = bw * torch.ones(X_train.shape[1], device=device)

    log_norm = -0.5 * d * np.log(2.0 * np.pi) - torch.log(bw).sum()
    return {"X_ref": X_ref, "bw": bw, "log_norm": log_norm}


@torch.no_grad()
def kde_log_prob(X_eval: torch.Tensor, kde: Dict[str, torch.Tensor], chunk: int = 8192, return_probs: bool = False) -> torch.Tensor:
    """
    Evaluate log p̂(x) under a diagonal Gaussian KDE using log-sum-exp.
    Returns (N_eval,) log-densities.
    """
    X_ref = kde["X_ref"]
    bw = kde["bw"]
    log_norm = kde["log_norm"]

    device = X_ref.device
    X_eval = X_eval.detach().to(device).float()
    N = X_eval.shape[0]
    M = X_ref.shape[0]

    out = []
    # work in chunks over eval points (rows)
    for i in range(0, N, chunk):
        Xe = X_eval[i:i+chunk]                               # (n_c, d)
        # compute scaled squared distances to all ref points
        # (n_c, M, d) -> sum over d
        z = (Xe[:, None, :] - X_ref[None, :, :]) / bw[None, None, :]
        sq = -0.5 * (z ** 2).sum(dim=-1)                     # (n_c, M)
        # log mean of exponentials
        m = torch.logsumexp(sq, dim=1) - np.log(M)
        out.append(log_norm + m)
    return torch.cat(out, dim=0) if not return_probs else torch.exp(torch.cat(out, dim=0))                           # (N,)


@torch.no_grad()
def compute_ood_along_path(
    pos: torch.Tensor,                   # (T+1, N, d)
    log_prob_fn: Callable,
    X_train_for_cdf: Optional[torch.Tensor] = None,  # optional: calibrate to empirical CDF of train densities
) -> Dict[str, torch.Tensor]:
    """
    For each step ℓ_t, compute:
      - mean log-density:    E_paths[ log p̂( x_t ) ]
      - OOD likelihood:      lower-tail probability under empirical CDF
                              of train log p̂ (optional but recommended).
    """
    T1, N, d = pos.shape
    # Evaluate log p for all path points
    logp = log_prob_fn(pos.view(-1, d)).view(T1, N)    # (T+1, N)

    # Mean/stats across paths at each ℓ
    mean_logp = logp.mean(dim=1)                             # (T+1,)
    std_logp  = logp.std(dim=1)

    # Optional calibration: map logp to lower-
    # tail OOD probability in [0,1]
    ood_prob = None
    if X_train_for_cdf is not None:
        lp_train = log_prob_fn(X_train_for_cdf.view(-1, d))  # (n_train,)
        lp_train_sorted = torch.sort(lp_train)[0]
        def ecdf(vals: torch.Tensor) -> torch.Tensor:
            # fraction of train with logp <= each val (lower-tail)
            idx = torch.searchsorted(lp_train_sorted, vals, right=True)
            return idx.to(vals.dtype) / lp_train_sorted.numel()
        ood_prob = ecdf(mean_logp)  # use mean logp per step; or ecdf per path then mean

    out = {"mean_logp": mean_logp, "std_logp": std_logp}
    if ood_prob is not None:
        out["ood_prob"] = 1-ood_prob
    return out


def plot_ood_vs_distance(
    distances: torch.Tensor,
    mean_logp: torch.Tensor,
    std_logp: Optional[torch.Tensor] = None,
    ood_prob: Optional[torch.Tensor] = None,
    title_left: str = "Mean log density log p̂(x) vs. distance",
    title_right: str = "OOD likelihood (empirical lower-tail)",
) -> None:
    import matplotlib.pyplot as plt
    d = distances.detach().cpu().numpy()
    mlp = mean_logp.detach().cpu().numpy()

    if ood_prob is None:
        plt.figure(figsize=(6.5, 4.0))
        plt.plot(d, mlp, linewidth=2, label="E[ log p̂(x_ℓ) ]")
        if std_logp is not None:
            s = std_logp.detach().cpu().numpy()
            plt.fill_between(d, mlp - s, mlp + s, alpha=0.2, label="±1σ")
        plt.xlabel("Distance ℓ")
        plt.ylabel("Mean log density")
        plt.title(title_left)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        axs[0].plot(d, mlp, linewidth=2, label="E[ log p̂(x_ℓ) ]")
        if std_logp is not None:
            s = std_logp.detach().cpu().numpy()
            axs[0].fill_between(d, mlp - s, mlp + s, alpha=0.2, label="±1σ")
        axs[0].set_xlabel("Distance ℓ"); axs[0].set_ylabel("Mean log density")
        axs[0].set_title(title_left); axs[0].grid(True, alpha=0.3); axs[0].legend()

        op = ood_prob.detach().cpu().numpy()
        axs[1].plot(d, op, linewidth=2)
        axs[1].set_xlabel("Distance ℓ"); axs[1].set_ylabel("OOD likelihood (↓ better)")
        axs[1].set_title(title_right); axs[1].grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()


# -----------------------
# Example usage (modular)
# -----------------------
# Assumes you already restricted starts to low points, e.g.:
# X = X.detach().to(device).float()
# y = impression_function(X).reshape(-1)
# X_low = X[y.flatten() < y.median()]

@torch.no_grad()
def run_all_metrics(
    model,
    ensemble,
    impression_function,
    X_start: torch.Tensor,
    X_for_ood: torch.Tensor,
    *,
    step_size: float = 0.05,
    n_steps: int = 150,
    device: Optional[torch.device] = None,
    kde_rule: str = "scott",
    kde_max_ref_points: int = 5000,
):
    """
    Convenience driver that:
      (1) Integrates paths,
      (2) Computes/plots Path Risk,
      (3) Computes/plots Manipulation Efficiency,
      (4) Fits KDE on X (your dataset of valid stimuli) and computes/plots OOD likelihood.
    """
    if device is None:
        device = X_start.device
    # 1) Integrate
    traj = integrate_flow_arc_length(model, X_start, step_size=step_size, n_steps=n_steps,
                                     normalize_step=True, device=device)
    pos, distances = traj["pos"], traj["distances"]

    # 2) Path risk (Def. Path risk)
    risk = compute_path_risk(ensemble, impression_function, pos)
    plot_path_risk(distances, risk["mse"], risk["se_std"])

    # 3) Manipulation efficiency (Δy / ℓ)
    y_true_along = risk["y_true"]                  # re-use
    eff = compute_manipulation_efficiency(y_true_along, distances)
    plot_manipulation_efficiency(distances, eff["delta_mean"], eff["efficiency"], eff["local_slope"])

    # Calibrate OOD to empirical CDF of training log-densities (optional but helpful)
    ood = compute_ood_along_path(pos, lambda x: kde_hull_blur(x, samples=X_for_ood), X_train_for_cdf=X_for_ood)
    plot_ood_vs_distance(distances, ood["mean_logp"], ood["std_logp"], ood_prob=ood.get("ood_prob"))

    return {"traj": traj, "risk": risk, "eff": eff, "ood": ood}


# --- Plotly Streamline + Metrics Dashboard (asynchronous, training-friendly) ---
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch
from concurrent.futures import ThreadPoolExecutor, Future

try:
    # in notebooks
    from IPython.display import display
except Exception:
    display = print


# ---------- Metric configuration ----------

@dataclass(frozen=True)
class MetricSpec:
    """
    Describes a single time-series diagnostic to plot (one subplot per metric).
    key:       Identifier for this metric. Use this in request_update(metrics={key: value, ...}).
    label:     Legend/axis label for the metric subplot.
    dash:      Optional Plotly dash style: "dash", "dot", "dashdot", etc.
    width:     Line width.
    formatter: Optional callable to format values in the left panel overlay box.
    """
    key: str
    label: str
    dash: Optional[str] = None
    width: int = 2
    formatter: Optional[Callable[[float], str]] = None


# ---------- Streamline utilities ----------

def _bilinear_sample(field: np.ndarray, x: float, y: float,
                     x_range: np.ndarray, y_range: np.ndarray) -> float:
    """
    Bilinear sample of a scalar field defined on meshgrid(X=x_range, Y=y_range).
    Returns np.nan if (x,y) is out of bounds.
    """
    nx, ny = len(x_range), len(y_range)
    if not (x_range[0] <= x <= x_range[-1] and y_range[0] <= y <= y_range[-1]):
        return np.nan

    # normalized indices
    tx = (x - x_range[0]) / (x_range[-1] - x_range[0]) * (nx - 1)
    ty = (y - y_range[0]) / (y_range[-1] - y_range[0]) * (ny - 1)
    i0 = int(np.floor(tx)); i1 = min(i0 + 1, nx - 1)
    j0 = int(np.floor(ty)); j1 = min(j0 + 1, ny - 1)
    a = tx - i0; b = ty - j0

    f00 = field[j0, i0]
    f10 = field[j0, i1]
    f01 = field[j1, i0]
    f11 = field[j1, i1]
    return (1-a)*(1-b)*f00 + a*(1-b)*f10 + (1-a)*b*f01 + a*b*f11


def _bilinear_vector_sample(Ux: np.ndarray, Uy: np.ndarray, x: float, y: float,
                            x_range: np.ndarray, y_range: np.ndarray) -> Tuple[float, float]:
    vx = _bilinear_sample(Ux, x, y, x_range, y_range)
    vy = _bilinear_sample(Uy, x, y, x_range, y_range)
    return vx, vy


def _rk4_step(pos: np.ndarray, ds: float,
              sampler: Callable[[float, float], Tuple[float, float]]) -> np.ndarray:
    """RK4 in arc-length parameter s (we normalize by speed inside)."""
    def f(p):
        vx, vy = sampler(p[0], p[1])
        v = np.array([vx, vy], dtype=np.float64)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < 1e-12:
            return np.array([np.nan, np.nan])
        return v / n  # unit direction → spatial step ~ ds

    k1 = f(pos)
    if not np.all(np.isfinite(k1)): return np.array([np.nan, np.nan])
    k2 = f(pos + 0.5 * ds * k1)
    if not np.all(np.isfinite(k2)): return np.array([np.nan, np.nan])
    k3 = f(pos + 0.5 * ds * k2)
    if not np.all(np.isfinite(k3)): return np.array([np.nan, np.nan])
    k4 = f(pos + ds * k3)
    if not np.all(np.isfinite(k4)): return np.array([np.nan, np.nan])

    return pos + (ds / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def _integrate_line(x0: float, y0: float, ds: float, max_steps: int,
                    sampler: Callable[[float, float], Tuple[float, float]],
                    bounds: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate streamline forward & backward and join."""
    x_min, x_max, y_min, y_max = bounds

    def inside(p):
        return (x_min <= p[0] <= x_max) and (y_min <= p[1] <= y_max)

    def trace(start, direction):
        pts = [start.copy()]
        p = start.copy()
        for _ in range(max_steps):
            step = _rk4_step(p, ds * direction, sampler)
            if not np.all(np.isfinite(step)):
                break
            if not inside(step):
                break
            p = step
            pts.append(p.copy())
        return np.array(pts)

    start = np.array([x0, y0], dtype=np.float64)
    if not inside(start):
        return np.empty((0,)), np.empty((0,))

    fwd = trace(start, +1.0)
    bwd = trace(start, -1.0)

    # join (exclude duplicate start)
    path = np.vstack((bwd[::-1], fwd[1:])) if len(bwd) and len(fwd) else (
        fwd if len(fwd) else bwd
    )
    if path is None or len(path) < 2:
        return np.empty((0,)), np.empty((0,))
    return path[:, 0], path[:, 1]


def _build_streamlines(Ux: np.ndarray, Uy: np.ndarray,
                       x_range: np.ndarray, y_range: np.ndarray,
                       xlim: Tuple[float, float], ylim: Tuple[float, float],
                       seed_grid: Tuple[int, int] = (22, 16),
                       ds: Optional[float] = None,
                       max_steps: int = 600,
                       speed_thresh: float = 1e-6,
                       max_streams: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (xs, ys) with None separators for a single Scattergl trace.
    """
    nx, ny = len(x_range), len(y_range)
    dx = (x_range[-1] - x_range[0]) / (nx - 1)
    dy = (y_range[-1] - y_range[0]) / (ny - 1)
    if ds is None:
        ds = 0.75 * min(dx, dy)

    sampler = lambda x, y: _bilinear_vector_sample(Ux, Uy, x, y, x_range, y_range)
    sx, sy = seed_grid
    xs_all, ys_all = [], []
    x_seeds = np.linspace(xlim[0], xlim[1], sx)
    y_seeds = np.linspace(ylim[0], ylim[1], sy)

    count = 0
    for yy in y_seeds:
        for xx in x_seeds:
            vx, vy = sampler(xx, yy)
            if not np.isfinite(vx) or not np.isfinite(vy):
                continue
            if (vx*vx + vy*vy) < speed_thresh:
                continue
            x_path, y_path = _integrate_line(xx, yy, ds, max_steps, sampler,
                                             (xlim[0], xlim[1], ylim[0], ylim[1]))
            if len(x_path) >= 2:
                xs_all.extend(list(x_path) + [None])
                ys_all.extend(list(y_path) + [None])
                count += 1
                if max_streams is not None and count >= max_streams:
                    break
        if max_streams is not None and count >= max_streams:
            break

    return np.asarray(xs_all, dtype=object), np.asarray(ys_all, dtype=object)


# ---------- The async dashboard ----------

class StreamlineMetricsDashboard:
    """
    Live training dashboard:
      - Left:   Streamline plot over an "impression" background (Z(x,y) contour).
      - Right:  One small subplot per metric (time series).
    Everything updates asynchronously to avoid blocking your training loop.

    Parameters
    ----------
    vector_field_fn : Callable[[torch.Tensor], torch.Tensor]
        Accepts (N,2) XY points on `device`, returns (N,2) velocities.
    impression_fn : Callable[[torch.Tensor], torch.Tensor]
        Accepts (N,2) XY points on `device`, returns (N,) scalar field (for background contour).
    density_fn : Optional[Callable[[torch.Tensor], torch.Tensor]]
        If provided, its iso-level at `support_cutoff * max` is drawn as a dashed contour on top.
    device : torch.device
        Where to evaluate the functions.
    xlim, ylim : Tuple[float,float]
        Plot domain.
    resolution : int
        Number of grid points per axis for evaluating fields.
    seed_grid : Tuple[int,int]
        Seed grid (cols, rows) for streamlines.
    ds : Optional[float]
        Step length for RK4 (defaults to ~0.75 * grid spacing).
    metrics : Sequence[MetricSpec]
        One subplot per metric.
    title : str
        Figure title.
    """

    def __init__(
        self,
        *,
        vector_field_fn: Callable[[torch.Tensor], torch.Tensor],
        impression_fn: Callable[[torch.Tensor], torch.Tensor],
        density_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        xlim: Tuple[float, float] = (-3., 3.),
        ylim: Tuple[float, float] = (-2., 2.),
        resolution: int = 80,
        seed_grid: Tuple[int, int] = (22, 16),
        ds: Optional[float] = None,
        max_steps: int = 600,
        support_cutoff: float = 0.02,
        metrics: Sequence[MetricSpec] = (),
        title: str = "Training Dashboard",
    ) -> None:
        self.vector_field_fn = vector_field_fn
        self.impression_fn = impression_fn
        self.density_fn = density_fn
        self.device = device
        self.xlim, self.ylim = xlim, ylim
        self.resolution = int(resolution)
        self.seed_grid = seed_grid
        self.ds = ds
        self.max_steps = int(max_steps)
        self.support_cutoff = float(support_cutoff)

        # Concurrency
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._future: Optional[Future] = None
        self._latest_payload: Optional[Any] = None

        # Metrics config & history
        self.metric_specs: List[MetricSpec] = list(metrics)
        self._metric_by_key: Dict[str, MetricSpec] = {m.key: m for m in self.metric_specs}
        self.step_hist: List[int] = []
        self.metric_hists: Dict[str, List[float]] = {m.key: [] for m in self.metric_specs}

        # Precompute grid
        self.x_range = np.linspace(xlim[0], xlim[1], self.resolution)
        self.y_range = np.linspace(ylim[0], ylim[1], self.resolution)
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)
        self.grid_points = torch.tensor(
            np.stack([self.X.ravel(), self.Y.ravel()], 1),
            dtype=torch.float32
        )
        if self.device is not None:
            self.grid_points = self.grid_points.to(self.device)

        # Figure with dynamic rows = #metrics (minimum 1 row)
        n_rows = max(len(self.metric_specs), 1)
        specs = [[{"rowspan": n_rows}, {"type": "xy"}]] + [
            [None, {"type": "xy"}] for _ in range(n_rows - 1)
        ]

        self.fig = go.FigureWidget(make_subplots(
            rows=n_rows, cols=2, specs=specs,
            column_widths=[0.67, 0.33],
            vertical_spacing=0.05,
            subplot_titles=(("Streamlines",) + tuple(m.label for m in self.metric_specs)) if self.metric_specs else ("Streamlines", "Metric"),
        ))

        self._init_static_figure(title)
        display(self.fig)

    # ---------- public API ----------

    def request_update(
        self,
        *,
        step: int,
        metrics: Dict[str, Optional[float]] = None,
        annotation_extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Queue an async recomputation of the left panel (and push metric values).
        If a previous update is still running, this call is ignored.

        step:             Training step (x-axis for right plots).
        metrics:          Mapping MetricSpec.key → float|None
        annotation_extras:extra lines for the overlay on the left.
        """
        if self._future and not self._future.done():
            return

        metrics = metrics or {}

        # Build overlay text
        lines = [f"step: {step}"]
        for spec in self.metric_specs:
            v = metrics.get(spec.key, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                disp = "NA"
            else:
                disp = spec.formatter(v) if spec.formatter else f"{float(v):.4f}"
            lines.append(f"{spec.label}: {disp}")
        if annotation_extras:
            for k, v in annotation_extras.items():
                if isinstance(v, float):
                    lines.append(f"{k}: {v:.4f}")
                else:
                    lines.append(f"{k}: {v}")
        metrics_text = "<br>".join(lines)

        self._future = self._executor.submit(
            self._prepare_payload, step, metrics, metrics_text
        )

    def update_plot_if_ready(self) -> bool:
        """
        If an async payload is ready, apply it to the figure and append histories.
        Returns True if the plot was updated (or a completed payload was consumed).
        """
        payload_to_plot = None
        with self._lock:
            if self._latest_payload is not None:
                payload_to_plot = self._latest_payload
                self._latest_payload = None

        # propagate any executor exception
        if self._future and self._future.done():
            self._future.result()

        if payload_to_plot is None:
            return self._future is not None and self._future.done()

        step, metrics, Z, Ux, Uy, support_isoline, SL_xs, SL_ys, metrics_text = payload_to_plot

        # Update histories
        self.step_hist.append(step)
        for spec in self.metric_specs:
            v = metrics.get(spec.key, None)
            self.metric_hists[spec.key].append(np.nan if v is None else float(v))

        # Update plot
        with self.fig.batch_update():
            # Background contour
            self.fig.data[self._idx_contour].x = self.x_range
            self.fig.data[self._idx_contour].y = self.y_range
            self.fig.data[self._idx_contour].z = Z

            # Streamlines
            self.fig.data[self._idx_stream].x = SL_xs
            self.fig.data[self._idx_stream].y = SL_ys

            # Optional support isoline
            if support_isoline is not None:
                self.fig.data[self._idx_support].x = self.x_range
                self.fig.data[self._idx_support].y = self.y_range
                self.fig.data[self._idx_support].z = support_isoline
                self.fig.data[self._idx_support].visible = True
            else:
                self.fig.data[self._idx_support].visible = False

            # Metric subplots
            for i, spec in enumerate(self.metric_specs):
                tr_idx = self._metric_trace_start + i
                self.fig.data[tr_idx].x = self.step_hist
                self.fig.data[tr_idx].y = self.metric_hists[spec.key]

            # Overlay text
            self.fig.layout.annotations[self._ann_idx].text = metrics_text

        return True

    def shutdown(self) -> None:
        """Shut down the background executor."""
        self._executor.shutdown()

    # ---------- internals ----------

    def _init_static_figure(self, title: str) -> None:
        n_rows = max(len(self.metric_specs), 1)

        # Left panel layers: (0) background contour, (1) streamlines, (2) optional support isoline
        self.fig.add_trace(
            go.Contour(
                x=self.x_range, y=self.y_range, z=np.zeros_like(self.X),
                colorscale="Viridis", showscale=False, contours=dict(coloring="heatmap"))
            , row=1, col=1
        )
        self._idx_contour = len(self.fig.data) - 1

        self.fig.add_trace(
            go.Scattergl(
                x=[], y=[], mode="lines",
                line=dict(width=1.2, color="white"),
                name="Streamlines", showlegend=False),
            row=1, col=1
        )
        self._idx_stream = len(self.fig.data) - 1

        self.fig.add_trace(
            go.Contour(
                x=self.x_range, y=self.y_range, z=np.zeros_like(self.X),
                contours=dict(start=0, end=0, size=1, coloring="lines"),
                line=dict(width=2, dash="dashdot", color="black"),
                showscale=False, visible=False, name="Support"
            ),
            row=1, col=1
        )
        self._idx_support = len(self.fig.data) - 1

        # Metric traces
        self._metric_trace_start = len(self.fig.data)
        for i, spec in enumerate(self.metric_specs):
            line = dict(width=spec.width)
            if spec.dash:
                line["dash"] = spec.dash
            self.fig.add_trace(
                go.Scatter(mode="lines+markers", name=spec.label, line=line),
                row=i + 1, col=2
            )
            # y-axis title per subplot
            self.fig.update_yaxes(title_text=spec.label, row=i + 1, col=2)

        # Overlay metrics box (top-left corner inside left panel)
        self.fig.add_annotation(
            xref="x1", yref="y1", x=self.xlim[0] + 0.02*(self.xlim[1]-self.xlim[0]),
            y=self.ylim[1] - 0.02*(self.ylim[1]-self.ylim[0]),
            text="", showarrow=False, align="left",
            bgcolor="rgba(255,255,255,0.78)", bordercolor="#444", borderwidth=1,
            font=dict(size=12, family="Courier New"),
            xanchor="left", yanchor="top"
        )
        self._ann_idx = len(self.fig.layout.annotations) - 1

        # Layout & axes
        self.fig.update_layout(
            width=1400, height=900, title_text=title, title_x=0.5,
            margin=dict(l=40, r=20, t=60, b=40),
            legend=dict(orientation="h", x=0.52, y=1.03, xanchor="left", yanchor="bottom"),
        )
        # Equal aspect on left
        self.fig.update_xaxes(range=list(self.xlim), title_text="x₁", row=1, col=1, scaleanchor="y1")
        self.fig.update_yaxes(range=list(self.ylim), title_text="x₂", row=1, col=1)

        # Shared x for metric plots = "Step"
        for i in range(n_rows):
            self.fig.update_xaxes(title_text="Step", row=i + 1, col=2)

    @torch.no_grad()
    def _prepare_payload(self, step: int, metrics: Dict[str, Optional[float]], metrics_text: str):
        # Evaluate fields on grid
        pts = self.grid_points
        Z = self.impression_fn(pts).detach().cpu().numpy().reshape(self.Y.shape)

        V = self.vector_field_fn(pts).detach().cpu().numpy()
        Ux = V[:, 0].reshape(self.Y.shape)
        Uy = V[:, 1].reshape(self.Y.shape)

        # Optional support isoline: put the *raw* density grid here so the isoline can be defined at a single value
        support_grid = None
        if self.density_fn is not None:
            P = self.density_fn(pts).detach().cpu().numpy().reshape(self.Y.shape)
            thr = self.support_cutoff * float(np.max(P) if np.isfinite(P).any() else 1.0)
            # For isoline at "thr", pass (P - thr) so the 0-level set is the boundary
            support_grid = (P - thr)

        # Streamlines (single trace with None-separated segments)
        SL_xs, SL_ys = _build_streamlines(
            Ux, Uy, self.x_range, self.y_range,
            self.xlim, self.ylim,
            seed_grid=self.seed_grid,
            ds=self.ds,
            max_steps=self.max_steps,
        )

        with self._lock:
            self._latest_payload = (step, metrics, Z, Ux, Uy, support_grid, SL_xs, SL_ys, metrics_text)