
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from typing import List, Dict, Sequence, Tuple, Iterable, Union, Optional

from .gradients import batch_true_gradients
from .metrics import traversal_mse_for_epsilons, prediction_bias_variance

# ---------- Computation helpers (module-based, no training) ----------

@torch.no_grad()
def compute_prediction_bias_variance_modules(
    f_true: torch.nn.Module,
    est_modules: Sequence[torch.nn.Module],
    X: torch.Tensor
) -> Dict[str, float]:
    """
    Given a true module f_true and a list of estimated modules f_hat^{(s)},
    compute prediction bias, variance, and MSE on a fixed test design X.
    If only a single estimated module is provided, variance=0 and MSE=bias.
    Returns dict with {'bias','variance','mse'} as Python floats.
    """
    y_true = f_true(X).view(-1)
    preds = []
    for m in est_modules:
        y_hat = m(X).view(-1)
        preds.append(y_hat.detach())
    preds_mat = torch.stack(preds, dim=0)  # [S, n]
    b, v, mse = prediction_bias_variance(y_true, preds_mat)
    return {"bias": b, "variance": v, "mse": mse}

@torch.no_grad()
def compute_traversal_mse_modules(
    f_true: torch.nn.Module,
    est_modules: Sequence[torch.nn.Module],
    X: torch.Tensor,
    epsilons: List[float],
) -> pd.DataFrame:
    """
    For a true module and a list of estimated modules, compute traversal MSE
    over a grid of epsilons, averaging across modules.
    Returns a tidy DataFrame with columns ['epsilon','mse_mean','mse_std']
    """
    grad_true = batch_true_gradients(f_true, X)  # [n,d]
    mse_collect = {eps: [] for eps in epsilons}
    for m in est_modules:
        grad_hat = batch_true_gradients(m, X)  # [n,d] general module
        res = traversal_mse_for_epsilons(grad_true, grad_hat, epsilons)
        for eps, val in res.items():
            mse_collect[eps].append(val)
    rows = []
    for eps in epsilons:
        vals = torch.tensor(mse_collect[eps], dtype=torch.float64)
        rows.append({"epsilon": float(eps),
                     "mse_mean": float(vals.mean().item()),
                     "mse_std":  float(vals.std(unbiased=False).item())})
    return pd.DataFrame(rows)

# ---------- Plotting helpers (Plotly figures) ----------

def plot_traversal_heatmap(
    traversal_summary: Union[pd.DataFrame, Iterable[Dict]],
    value_col: str = "mse_mean",
    lambda_col: str = "lambda",
    epsilon_col: str = "epsilon",
    title: str = "Traversal MSE = Error of unit increase step",
    log_x: bool = False,
    log_y: bool = False,
    log_error: bool = False,
) -> go.Figure:
    """
    Build a 2D heatmap (lambda x epsilon) of traversal MSE using Plotly.
    traversal_summary: DataFrame or records with at least [lambda, epsilon, value_col].
    """
    df = pd.DataFrame(traversal_summary).copy()
    # Ensure numeric types
    df[lambda_col] = pd.to_numeric(df[lambda_col])
    df[epsilon_col] = pd.to_numeric(df[epsilon_col])
    df[value_col]   = pd.to_numeric(df[value_col])
    # Pivot to matrix
    mat = df.pivot(index=lambda_col, columns=epsilon_col, values=value_col).sort_index().sort_index(axis=1)
    print(mat.values)
    # Set default Plotly width/height (in px)
    default_width, default_height = 700, 450
    # mat.shape: (n_rows, n_cols)
    n_rows, n_cols = mat.shape
    max_n = max(n_rows, n_cols)
    cell_width, cell_height = 700/max_n, 700/max_n
    mat_width = n_cols * cell_width
    mat_height = n_rows * cell_height
    fig_width = int((default_width + mat_width) / 2)
    fig_height = int((default_height + mat_height) / 2)
    if log_error:
        mat_values = np.log10(-1/2*mat.values**0.5)
    else:
        mat_values = mat.values
    fig = px.imshow(mat_values,
                    x=[f"{c:g}" for c in mat.columns],
                    y=[f"{r:g}" for r in mat.index],
                    aspect="auto",
                    labels=dict(x="epsilon (traversal ridge)", y="lambda (model ridge)", color=value_col),
                    title=title,
                    width=fig_width,
                    height=fig_height)
                
    # Optional log ticks (display only; the values plotted are linear MSEs)
    if log_x:
        fig.update_xaxes(type="category", title="epsilon (log grid)")
    if log_y:
        fig.update_yaxes(type="category", title="lambda (log grid)")
    return fig

def plot_prediction_curve(
    pred_summary: Union[pd.DataFrame, Iterable[Dict]],
    metric: str = "mse",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Line plot of prediction metric vs lambda. metric in {'mse','bias','variance'}.
    """
    df = pd.DataFrame(pred_summary).copy()
    if title is None:
        title = f"Prediction {metric.upper()} vs λ"
    df = df.sort_values("lambda")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["lambda"], y=df[metric], mode="lines+markers", name=metric.upper()))
    fig.update_layout(xaxis_title="lambda (model ridge)", yaxis_title=metric.upper(), title=title)
    return fig

def plot_traversal_curve(
    traversal_summary: Union[pd.DataFrame, Iterable[Dict]],
    lambda_value: float,
    value_col: str = "mse_mean",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Line plot of traversal metric vs epsilon for a fixed lambda.
    """
    df = pd.DataFrame(traversal_summary).copy()
    df = df[df["lambda"] == lambda_value].copy()
    df = df.sort_values("epsilon")
    if title is None:
        title = f"Traversal {value_col} vs ε at λ={lambda_value:g}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["epsilon"], y=df[value_col], mode="lines+markers", name=value_col))
    fig.update_layout(xaxis_title="epsilon (traversal ridge)", yaxis_title=value_col, title=title)
    return fig
