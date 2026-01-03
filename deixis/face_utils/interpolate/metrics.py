"""Tensor-based utilities for analysing latent-space attribute linearity."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


ArrayLike = Sequence[float]


def get_preferred_device() -> torch.device:
    """Return CUDA or MPS device if available, else CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class NeighborGraph:
    """Container holding neighbour information for each latent sample."""

    indices: torch.Tensor  # shape [N, k], dtype long
    distances: torch.Tensor  # shape [N, k], dtype float
    device: torch.device


def align_latents_and_ratings(
    photo_to_coords: Mapping[str, ArrayLike],
    photo_to_rating: Mapping[str, float],
    *,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Return tensors of latents and ratings on the shared photo keys."""

    device = device or get_preferred_device()
    shared_keys = sorted(set(photo_to_coords).intersection(photo_to_rating))
    if not shared_keys:
        raise ValueError("No overlapping photo keys between coords and ratings.")

    latents = []
    ratings = []
    for key in shared_keys:
        raw = photo_to_coords[key]
        if isinstance(raw, torch.Tensor):
            coord = raw.detach().to(dtype=torch.float32, device=device)
        else:
            coord = torch.as_tensor(raw, dtype=torch.float32, device=device)
        if coord.ndim != 1:
            coord = coord.reshape(-1)
        latents.append(coord)
        ratings.append(float(photo_to_rating[key]))

    latents_tensor = torch.stack(latents, dim=0)
    ratings_tensor = torch.tensor(ratings, dtype=torch.float32, device=device)
    return latents_tensor, ratings_tensor, shared_keys


def _batched_cdist(
    xs: torch.Tensor,
    ys: torch.Tensor,
    *,
    block_size: int,
) -> Iterable[Tuple[int, int, torch.Tensor]]:
    """Yield pairwise distances between `xs` chunks and all of `ys`."""

    n = xs.shape[0]
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        chunk = xs[start:end]
        dist = torch.cdist(chunk, ys)
        yield start, end, dist


@torch.no_grad()
def build_neighbor_graph(
    latents: torch.Tensor,
    n_neighbors: int,
    *,
    block_size: int = 1024,
) -> NeighborGraph:
    """Compute the neighbour graph around each latent point."""

    num_points = latents.shape[0]
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive.")
    n_neighbors = min(n_neighbors, num_points)

    indices_list = []
    distances_list = []
    for start, end, dist in _batched_cdist(latents, latents, block_size=block_size):
        values, idx = torch.topk(dist, k=n_neighbors, largest=False)
        indices_list.append(idx)
        distances_list.append(values)

    indices = torch.cat(indices_list, dim=0)
    distances = torch.cat(distances_list, dim=0)
    return NeighborGraph(indices=indices, distances=distances, device=latents.device)


@contextmanager
def _temp_manual_seed(seed: Optional[int]) -> Iterable[None]:
    """Context manager to restore RNG state after using `torch.manual_seed`."""

    if seed is None:
        yield
        return

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_states = None
    if hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps_states = torch.mps.get_rng_state()
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_states is not None:
            torch.mps.set_rng_state(mps_states)


@torch.no_grad()
def _nearest_points(
    latents: torch.Tensor,
    queries: torch.Tensor,
    *,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (distances, indices) of nearest latent point to each query."""

    device = latents.device
    best_dist = torch.full(
        (queries.shape[0],), float("inf"), device=device, dtype=torch.float32
    )
    best_idx = torch.zeros((queries.shape[0],), device=device, dtype=torch.long)

    for start, end, dist in _batched_cdist(queries, latents, block_size=block_size):
        min_dist, min_idx = dist.min(dim=1)
        best_chunk = best_dist[start:end]
        update = min_dist < best_chunk
        if torch.any(update):
            updated_dist = torch.where(update, min_dist, best_chunk)
            current_idx = best_idx[start:end]
            updated_idx = torch.where(update, min_idx, current_idx)
            best_dist[start:end] = updated_dist
            best_idx[start:end] = updated_idx

    return best_dist, best_idx


def _gather_neighbors(
    neighbor_graph: NeighborGraph,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return neighbour distances and indices excluding the self-neighbour."""

    distances = neighbor_graph.distances[:, 1:]
    indices = neighbor_graph.indices[:, 1:]
    return distances, indices


@torch.no_grad()
def compute_local_slope_statistics(
    latents: torch.Tensor,
    ratings: torch.Tensor,
    neighbor_graph: NeighborGraph,
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Summarise how consistent the local slopes are around each sample."""

    distances, indices = _gather_neighbors(neighbor_graph)
    device = latents.device

    valid_mask = distances > eps
    if not torch.any(valid_mask):
        raise ValueError("Neighbour graph does not contain positive distances.")

    slopes = torch.zeros_like(distances, device=device)
    rating_deltas = ratings[indices] - ratings.unsqueeze(1)
    slopes[valid_mask] = rating_deltas[valid_mask] / distances[valid_mask]

    abs_slopes = torch.abs(slopes[valid_mask])

    distances_clamped = torch.where(distances > eps, distances, distances.new_full((), eps))
    weighted_slope = torch.sum(slopes * distances_clamped, dim=1) / (
        torch.sum(distances_clamped * distances_clamped, dim=1) + eps
    )
    fitted = weighted_slope.unsqueeze(1) * distances_clamped
    residuals = rating_deltas - fitted
    ss_res = torch.sum(residuals * residuals, dim=1)
    demeaned = rating_deltas - rating_deltas.mean(dim=1, keepdim=True)
    ss_tot = torch.sum(demeaned * demeaned, dim=1) + eps
    r2 = 1.0 - ss_res / ss_tot

    stats = {
        "mean_abs_slope": float(abs_slopes.mean().item()),
        "median_abs_slope": float(abs_slopes.median().item()),
        "slope_variance": float(torch.var(slopes[valid_mask], unbiased=False).item()),
        "slope_std": float(torch.std(slopes[valid_mask], unbiased=False).item()),
        "mean_local_r2": float(r2.mean().item()),
        "median_local_r2": float(r2.median().item()),
    }

    return stats


@torch.no_grad()
def compute_midpoint_deviation(
    latents: torch.Tensor,
    ratings: torch.Tensor,
    neighbor_graph: NeighborGraph,
    *,
    max_pairs: int = 5000,
    block_size: int = 1024,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Measure how close midpoint ratings stay to linear interpolation."""

    device = latents.device
    distances, indices = _gather_neighbors(neighbor_graph)
    num_points, neighbor_count = indices.shape

    sources = torch.arange(num_points, device=device).unsqueeze(1).expand(-1, neighbor_count)
    mask = sources < indices
    if not torch.any(mask):
        raise ValueError("Need at least one neighbour pair to compute midpoints.")

    pair_i = sources[mask]
    pair_j = indices[mask]
    pairs = torch.stack([pair_i, pair_j], dim=1)

    with _temp_manual_seed(random_state):
        if pairs.shape[0] > max_pairs:
            perm = torch.randperm(pairs.shape[0], device=device)[:max_pairs]
            pairs = pairs[perm]

    latents_i = latents[pairs[:, 0]]
    latents_j = latents[pairs[:, 1]]
    midpoints = 0.5 * (latents_i + latents_j)
    rating_pred = 0.5 * (ratings[pairs[:, 0]] + ratings[pairs[:, 1]])

    midpoint_distances, nearest_idx = _nearest_points(
        latents, midpoints, block_size=block_size
    )
    rating_true = ratings[nearest_idx]
    errors = torch.abs(rating_true - rating_pred)

    return {
        "mean_abs_error": float(errors.mean().item()),
        "median_abs_error": float(errors.median().item()),
        "max_abs_error": float(errors.max().item()),
        "mean_midpoint_distance": float(midpoint_distances.mean().item()),
    }


@torch.no_grad()
def compute_convexity_violation(
    latents: torch.Tensor,
    ratings: torch.Tensor,
    neighbor_graph: NeighborGraph,
    *,
    n_samples: int = 2000,
    block_size: int = 1024,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Assess convexity by sampling barycentric combinations of neighbours."""

    device = latents.device
    distances, indices = _gather_neighbors(neighbor_graph)
    neighbors_per_point = indices.shape[1]
    if neighbors_per_point < 2:
        raise ValueError("Convexity metric requires at least three neighbours.")

    num_points = latents.shape[0]
    n_samples = min(n_samples, num_points * neighbors_per_point)

    with _temp_manual_seed(
        (random_state + 1) if random_state is not None else None
    ):
        anchors = torch.randint(0, num_points, (n_samples,), device=device)
        neighbor_pool = indices[anchors]
        random_scores = torch.rand((n_samples, neighbors_per_point), device=device)
        _, top2 = torch.topk(random_scores, k=2, dim=1)
        chosen_neighbors = neighbor_pool.gather(1, top2)

    triplet_indices = torch.cat(
        [anchors.unsqueeze(1), chosen_neighbors],
        dim=1,
    )

    with _temp_manual_seed(
        (random_state + 2) if random_state is not None else None
    ):
        weights = torch.rand((n_samples, 3), device=device)
    weights = weights / weights.sum(dim=1, keepdim=True)

    latent_triplets = latents[triplet_indices]
    barycenters = torch.sum(weights.unsqueeze(-1) * latent_triplets, dim=1)
    rating_pred = torch.sum(weights * ratings[triplet_indices], dim=1)

    distances_to_set, nearest_idx = _nearest_points(
        latents, barycenters, block_size=block_size
    )
    rating_true = ratings[nearest_idx]
    violations = torch.abs(rating_true - rating_pred)

    return {
        "mean_abs_violation": float(violations.mean().item()),
        "median_abs_violation": float(violations.median().item()),
        "max_abs_violation": float(violations.max().item()),
        "samples_used": int(violations.numel()),
        "mean_barycenter_distance": float(distances_to_set.mean().item()),
    }


@torch.no_grad()
def evaluate_attribute_linearity(
    photo_to_coords: Mapping[str, ArrayLike],
    photo_to_rating: Mapping[str, float],
    *,
    n_neighbors: int = 32,
    neighbor_block_size: int = 1024,
    midpoint_pairs: int = 5000,
    convexity_samples: int = 2000,
    nearest_block_size: int = 1024,
    random_state: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Convenience wrapper computing all metrics for a single attribute."""

    if n_neighbors < 3:
        raise ValueError("n_neighbors must be at least 3 for the metrics.")

    device = device or get_preferred_device()
    latents, ratings, keys = align_latents_and_ratings(
        photo_to_coords,
        photo_to_rating,
        device=device,
    )

    neighbor_graph = build_neighbor_graph(
        latents,
        n_neighbors=min(n_neighbors, latents.shape[0]),
        block_size=neighbor_block_size,
    )

    slope_stats = compute_local_slope_statistics(latents, ratings, neighbor_graph)
    midpoint_stats = compute_midpoint_deviation(
        latents,
        ratings,
        neighbor_graph,
        max_pairs=min(midpoint_pairs, latents.shape[0] * (n_neighbors - 1) // 2),
        block_size=nearest_block_size,
        random_state=random_state,
    )
    convexity_stats = compute_convexity_violation(
        latents,
        ratings,
        neighbor_graph,
        n_samples=min(convexity_samples, 10 * latents.shape[0]),
        block_size=nearest_block_size,
        random_state=random_state,
    )

    return {
        "n_samples": int(latents.shape[0]),
        "latent_dim": int(latents.shape[1]),
        "keys": keys,
        "device": str(device),
        "local_slope": slope_stats,
        "midpoint": midpoint_stats,
        "convexity": convexity_stats,
    }


@torch.no_grad()
def batch_evaluate_attributes(
    photo_to_coords: Mapping[str, ArrayLike],
    dim_to_photo_to_rating: Mapping[str, Mapping[str, float]],
    *,
    attributes: Optional[Iterable[str]] = None,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Dict[str, Dict[str, object]]:
    """Evaluate linearity diagnostics for the selected rating dimensions."""

    selected = attributes if attributes is not None else dim_to_photo_to_rating.keys()
    device = device or get_preferred_device()

    results: Dict[str, Dict[str, object]] = {}
    for attribute in selected:
        rating_map = dim_to_photo_to_rating.get(attribute)
        if rating_map is None:
            raise KeyError(f"Attribute '{attribute}' not present in ratings map.")
        results[attribute] = evaluate_attribute_linearity(
            photo_to_coords,
            rating_map,
            device=device,
            **kwargs,
        )

    return results


__all__ = [
    "batch_evaluate_attributes",
    "build_neighbor_graph",
    "compute_convexity_violation",
    "compute_local_slope_statistics",
    "compute_midpoint_deviation",
    "evaluate_attribute_linearity",
    "get_preferred_device",
    "align_latents_and_ratings",
]
