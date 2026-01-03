"""Plotly visualisations for latent-space linearity diagnostics."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import torch

from .metrics import batch_evaluate_attributes, get_preferred_device


def summarise_results_to_frame(
    results: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Flatten the nested metrics dictionary into a tabular structure."""

    records = []
    for attribute, payload in results.items():
        row = {
            "attribute": attribute,
            "n_samples": payload.get("n_samples"),
            "latent_dim": payload.get("latent_dim"),
        }

        slope = payload.get("local_slope", {}) or {}
        row.update(
            {
                "mean_abs_slope": slope.get("mean_abs_slope"),
                "median_abs_slope": slope.get("median_abs_slope"),
                "slope_std": slope.get("slope_std"),
                "mean_local_r2": slope.get("mean_local_r2"),
                "median_local_r2": slope.get("median_local_r2"),
            }
        )

        midpoint = payload.get("midpoint", {}) or {}
        row.update(
            {
                "midpoint_mean_abs_error": midpoint.get("mean_abs_error"),
                "midpoint_median_abs_error": midpoint.get("median_abs_error"),
                "midpoint_max_abs_error": midpoint.get("max_abs_error"),
                "midpoint_mean_midpoint_distance": midpoint.get(
                    "mean_midpoint_distance"
                ),
            }
        )

        convexity = payload.get("convexity", {}) or {}
        row.update(
            {
                "convexity_mean_abs_violation": convexity.get("mean_abs_violation"),
                "convexity_median_abs_violation": convexity.get("median_abs_violation"),
                "convexity_max_abs_violation": convexity.get("max_abs_violation"),
                "convexity_samples_used": convexity.get("samples_used"),
            }
        )

        records.append(row)

    frame = pd.DataFrame.from_records(records)
    return frame.sort_values("attribute").reset_index(drop=True)


def plot_metric_bar(
    frame: pd.DataFrame,
    metric: str,
    *,
    sort_descending: bool = False,
    title: Optional[str] = None,
    hover_precision: int = 4,
) -> go.Figure:
    """Create a bar chart for a single metric across attributes."""

    data = frame.sort_values(metric, ascending=not sort_descending)
    values = data[metric]
    hover_template = (
        f"<b>%{{x}}</b><br>{metric}: %{{y:.{hover_precision}f}}<extra></extra>"
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=data["attribute"],
                y=values,
                text=values.round(hover_precision),
                textposition="auto",
                hovertemplate=hover_template,
            )
        ]
    )
    fig.update_layout(
        title=title or metric.replace("_", " ").title(),
        xaxis_title="Attribute",
        yaxis_title=metric.replace("_", " ").title(),
        template="plotly_white",
    )
    return fig


def plot_metric_scatter(
    frame: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    *,
    color_metric: Optional[str] = None,
    title: Optional[str] = None,
    hover_precision: int = 4,
) -> go.Figure:
    """Scatter plot contrasting two metrics per attribute."""

    hover_template = (
        f"<b>%{{text}}</b><br>{x_metric}: %{{x:.{hover_precision}f}}<br>"
        f"{y_metric}: %{{y:.{hover_precision}f}}"
    )
    if color_metric is not None:
        hover_template += (
            f"<br>{color_metric}: %{{marker.color:.{hover_precision}f}}"
        )
    hover_template += "<extra></extra>"

    marker_kwargs = {}
    if color_metric is not None and color_metric in frame:
        marker_kwargs["color"] = frame[color_metric]
        marker_kwargs["colorbar"] = {"title": color_metric.replace("_", " ").title()}
        marker_kwargs["colorscale"] = "Viridis"

    fig = go.Figure(
        data=[
            go.Scatter(
                x=frame[x_metric],
                y=frame[y_metric],
                mode="markers+text",
                text=frame["attribute"],
                textposition="top center",
                hovertemplate=hover_template,
                marker={"size": 12, **marker_kwargs},
            )
        ]
    )
    fig.update_layout(
        title=title
        or f"{y_metric.replace('_', ' ').title()} vs {x_metric.replace('_', ' ').title()}",
        xaxis_title=x_metric.replace("_", " ").title(),
        yaxis_title=y_metric.replace("_", " ").title(),
        template="plotly_white",
    )
    return fig


def generate_attribute_figures(
    photo_to_coords,
    dim_to_photo_to_rating,
    *,
    attributes: Optional[Iterable[str]] = None,
    n_neighbors: int = 32,
    midpoint_pairs: int = 5000,
    convexity_samples: int = 2000,
    random_state: Optional[int] = None,
    device: Optional[str] = None,
    neighbor_block_size: int = 1024,
    nearest_block_size: int = 1024,
) -> Tuple[pd.DataFrame, Mapping[str, Mapping[str, object]], Mapping[str, go.Figure]]:
    """Compute metrics and assemble a small dashboard of Plotly figures."""

    if isinstance(device, str):
        if device.lower() == "auto":
            resolved_device = get_preferred_device()
        else:
            resolved_device = torch.device(device)
    else:
        resolved_device = device

    if isinstance(resolved_device, torch.device):
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if resolved_device.type == "mps" and (
            not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS requested but not available.")

    results = batch_evaluate_attributes(
        photo_to_coords,
        dim_to_photo_to_rating,
        attributes=attributes,
        n_neighbors=n_neighbors,
        midpoint_pairs=midpoint_pairs,
        convexity_samples=convexity_samples,
        random_state=random_state,
        device=resolved_device,
        neighbor_block_size=neighbor_block_size,
        nearest_block_size=nearest_block_size,
    )
    frame = summarise_results_to_frame(results)

    figures = {
        "midpoint_error": plot_metric_bar(
            frame,
            "midpoint_mean_abs_error",
            title="Mean Midpoint Absolute Error",
        ),
        "convexity_violation": plot_metric_bar(
            frame,
            "convexity_mean_abs_violation",
            title="Mean Convexity Violation",
        ),
        "slope_vs_r2": plot_metric_scatter(
            frame,
            x_metric="mean_abs_slope",
            y_metric="mean_local_r2",
            color_metric="midpoint_mean_abs_error",
            title="Local Slope Consistency vs R²",
        ),
    }

    return frame, results, figures


__all__ = [
    "batch_evaluate_attributes",
    "generate_attribute_figures",
    "plot_metric_bar",
    "plot_metric_scatter",
    "summarise_results_to_frame",
]
