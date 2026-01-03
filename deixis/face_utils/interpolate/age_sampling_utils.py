"""Utilities for analysing latent coverage of age ratings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from deixis.face_utils.interpolate.metrics import get_preferred_device

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from deixis.face_utils.interpolate.load_interp_data import load_interp_data  # type: ignore
else:
    from .load_interp_data import load_interp_data


def _ensure_tensor(value: Sequence[float] | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def align_latents_and_age(
    photo_to_coords: Mapping[str, Sequence[float] | torch.Tensor],
    age_map: Mapping[str, float],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, list[str]]:
    shared = sorted(set(photo_to_coords).intersection(age_map))
    if not shared:
        raise ValueError("No overlapping entries between coordinates and age ratings.")

    latents = torch.stack([_ensure_tensor(photo_to_coords[key], device=device).view(-1) for key in shared])
    ages = torch.tensor([float(age_map[key]) for key in shared], dtype=torch.float32, device=device)
    return latents, ages, shared


def try_load_w_avg(device: torch.device) -> Optional[torch.Tensor]:
    try:
        from manipy.stylegan.utils import get_w_avg  # type: ignore

        w_avg_tensor = get_w_avg(device=device)
        if isinstance(w_avg_tensor, dict) and "w_avg" in w_avg_tensor:
            w_avg_tensor = w_avg_tensor["w_avg"]
        return _ensure_tensor(w_avg_tensor, device=device)
    except Exception:
        return None


@dataclass
class SphericalMetrics:
    dataframe: pd.DataFrame
    cosine_indices: torch.Tensor
    cosine_weights: torch.Tensor
    density_distance: torch.Tensor
    local_variance: torch.Tensor
    local_spherical_age: torch.Tensor


def compute_spherical_metrics(
    latents: torch.Tensor,
    ages: torch.Tensor,
    *,
    w_avg: Optional[torch.Tensor] = None,
    neighbor_k: int = 32,
) -> SphericalMetrics:
    device = latents.device
    if w_avg is None:
        w_avg = latents.mean(dim=0)

    latents_centered = latents - w_avg
    norms = latents_centered.norm(dim=1, keepdim=True).clamp_min(1e-6)
    unit_latents = latents_centered / norms

    cosine_sim = unit_latents @ unit_latents.t()
    cosine_sim.fill_diagonal_(-1.0)

    neighbor_k = min(neighbor_k, latents.shape[0] - 1)
    cosine_vals, cosine_idx = torch.topk(cosine_sim, k=neighbor_k, dim=1)

    weights = torch.relu(cosine_vals)
    weight_sums = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    normalized_weights = weights / weight_sums

    neighbor_ages = ages[cosine_idx]
    local_spherical_age = (normalized_weights * neighbor_ages).sum(dim=1)

    latent_dist = torch.cdist(latents, latents, p=2)
    neighbor_distances = latent_dist[torch.arange(latents.shape[0]).unsqueeze(1), cosine_idx]
    density_distance = (normalized_weights * neighbor_distances).sum(dim=1)
    local_variance = (normalized_weights * (neighbor_ages - local_spherical_age.unsqueeze(1)) ** 2).sum(dim=1)

    alpha = density_distance.mean().item() + 1e-6
    beta = local_variance.mean().item() + 1e-6
    spherical_confidence = torch.exp(-density_distance / alpha) * torch.exp(-local_variance / beta)

    frame = pd.DataFrame(
        {
            "density_distance": density_distance.cpu().numpy(),
            "local_variance": local_variance.cpu().numpy(),
            "local_spherical_age": local_spherical_age.cpu().numpy(),
            "spherical_confidence": spherical_confidence.cpu().numpy(),
        }
    )

    return SphericalMetrics(
        dataframe=frame,
        cosine_indices=cosine_idx,
        cosine_weights=normalized_weights,
        density_distance=density_distance,
        local_variance=local_variance,
        local_spherical_age=local_spherical_age,
    )


def orthogonal_design_scores(latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latents_centered = latents - latents.mean(dim=0, keepdim=True)
    cov = latents_centered.t().matmul(latents_centered) / (latents_centered.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)
    projections = latents_centered @ eigvecs
    scores = (projections.pow(2) / (eigvals + 1e-6)).sum(dim=1)
    return scores, eigvals, eigvecs


def select_candidates_per_age_bin(
    latents: torch.Tensor,
    ages: torch.Tensor,
    shared_keys: Iterable[str],
    metrics: SphericalMetrics,
    *,
    per_bin: int = 5,
) -> pd.DataFrame:
    scores, _, _ = orthogonal_design_scores(latents)
    age_np = ages.cpu().numpy()
    keys_list = list(shared_keys)

    age_min = int(np.floor(age_np.min()))
    age_max = int(np.ceil(age_np.max()))
    bins = list(range(age_min, age_max))

    spherical_df = metrics.dataframe.copy()
    spherical_df["photo"] = keys_list
    spherical_df["age"] = age_np
    spherical_df["design_score"] = scores.cpu().numpy()

    results = []
    for start in bins:
        end = start + 1
        mask = (spherical_df["local_spherical_age"] >= start) & (spherical_df["local_spherical_age"] < end)
        if not mask.any():
            continue
        subset = spherical_df.loc[mask].copy()
        subset["density_rank"] = subset["density_distance"].rank()
        subset["variance_rank"] = subset["local_variance"].rank()
        subset["combined_rank"] = subset["density_rank"] + subset["variance_rank"]
        best = subset.nsmallest(per_bin, "combined_rank")
        best = best.assign(age_bin=f"[{start}, {end})")
        results.append(best)

    if not results:
        return pd.DataFrame(columns=["age_bin", "photo", "local_spherical_age", "density_distance", "local_variance", "design_score"])

    candidates = pd.concat(results, ignore_index=True)
    return candidates.sort_values(["age_bin", "design_score"], ascending=[True, False])


def run_analysis(per_bin: int = 5, neighbor_k: int = 32, device: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    device_t = torch.device(device) if device else get_preferred_device()
    photo_to_coords, dim_to_photo_to_rating = load_interp_data()

    age_key = next((key for key in dim_to_photo_to_rating if key.lower().startswith("age")), None)
    if age_key is None:
        raise KeyError("Could not locate an age rating column.")

    latents, ages, keys = align_latents_and_age(photo_to_coords, dim_to_photo_to_rating[age_key], device=device_t)
    w_avg = try_load_w_avg(device_t)

    metrics = compute_spherical_metrics(latents, ages, w_avg=w_avg, neighbor_k=neighbor_k)
    candidates = select_candidates_per_age_bin(latents, ages, keys, metrics, per_bin=per_bin)
    metrics_with_keys = metrics.dataframe.copy()
    metrics_with_keys.insert(0, "photo", keys)
    metrics_with_keys.insert(1, "age", ages.cpu().numpy())

    return metrics_with_keys, candidates, keys


def main() -> None:
    metrics_df, candidates_df, _ = run_analysis()
    print("Top spherical confidence rows:")
    print(metrics_df.sort_values("spherical_confidence", ascending=False).head(10))
    print("\nTop candidate suggestions:")
    print(candidates_df.head(20))


if __name__ == "__main__":  # pragma: no cover
    main()
