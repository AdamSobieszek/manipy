"""Cluster utilities for interpolation-safe regions and training context prep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .metrics import align_latents_and_ratings, build_neighbor_graph, get_preferred_device


@dataclass(frozen=True)
class ClusterConfig:
    """Parameters that control how supervised clusters are formed."""

    x_radius: float  # maximum latent-space distance between neighbours
    y_radius: float  # maximum rating distance between neighbours
    n_neighbors: int = 48  # neighbours considered per point
    min_cluster_size: int = 8  # discard connected components smaller than this
    neighbor_block_size: int = 1024  # block size when querying neighbours
    rating_metric: str = "l2"  # either 'l1' or 'l2'


@dataclass
class ClusterSummary:
    """Stores aggregate cluster information for downstream use."""

    cluster_id: int
    members: List[int]
    centroid_latent: torch.Tensor
    centroid_rating: torch.Tensor
    latent_cov: torch.Tensor
    rating_cov: torch.Tensor

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class TrainingContext:
    """Container that packages clustering outputs for model consumption."""

    attributes: List[str]
    keys: List[str]
    latents: torch.Tensor
    ratings: torch.Tensor
    clusters: List[ClusterSummary]
    soft_membership: torch.Tensor
    hard_labels: torch.Tensor


def assemble_dataset(
    photo_to_coords: Mapping[str, Sequence[float]],
    dim_to_photo_to_rating: Mapping[str, Mapping[str, float]],
    *,
    attributes: Optional[Iterable[str]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Return latent matrix, rating matrix, shared keys, and attribute names."""

    device = device or get_preferred_device()
    attribute_list = list(attributes) if attributes is not None else sorted(dim_to_photo_to_rating)
    if not attribute_list:
        raise ValueError("No attributes provided for dataset assembly.")

    shared_keys = set(photo_to_coords)
    for attr in attribute_list:
        rating_map = dim_to_photo_to_rating.get(attr)
        if rating_map is None:
            raise KeyError(f"Attribute '{attr}' not present in rating dictionary.")
        shared_keys &= set(rating_map)

    if not shared_keys:
        raise ValueError("No overlapping keys across selected attributes and latents.")

    sorted_keys = sorted(shared_keys)
    latents = []
    for key in sorted_keys:
        raw = photo_to_coords[key]
        if isinstance(raw, torch.Tensor):
            tensor = raw.detach().to(device=device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(raw, dtype=torch.float32, device=device)
        if tensor.ndim != 1:
            tensor = tensor.reshape(-1)
        latents.append(tensor)

    latent_tensor = torch.stack(latents, dim=0)

    rating_columns = []
    for attr in attribute_list:
        column = []
        mapping = dim_to_photo_to_rating[attr]
        for key in sorted_keys:
            column.append(float(mapping[key]))
        rating_columns.append(column)

    rating_tensor = torch.tensor(rating_columns, dtype=torch.float32, device=device).transpose(0, 1)
    return latent_tensor, rating_tensor, sorted_keys, attribute_list


def _compute_rating_distance(
    rating_deltas: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """Compute rating distance per neighbour edge."""

    if metric == "l1":
        return rating_deltas.abs().sum(dim=2)
    if metric == "l2":
        return torch.linalg.norm(rating_deltas, dim=2)
    raise ValueError(f"Unsupported rating metric '{metric}'. Use 'l1' or 'l2'.")


def find_interpolation_safe_clusters(
    latents: torch.Tensor,
    ratings: torch.Tensor,
    config: ClusterConfig,
) -> List[ClusterSummary]:
    """Identify clusters where latent and rating distances stay within thresholds."""

    if latents.shape[0] != ratings.shape[0]:
        raise ValueError("Latent and rating tensors must share the same number of rows.")

    if latents.shape[0] == 0:
        return []

    neighbor_graph = build_neighbor_graph(
        latents,
        n_neighbors=config.n_neighbors,
        block_size=config.neighbor_block_size,
    )

    distances = neighbor_graph.distances[:, 1:]
    indices = neighbor_graph.indices[:, 1:]
    rating_deltas = ratings[indices] - ratings.unsqueeze(1)
    if ratings.ndim == 1:
        rating_deltas = rating_deltas.unsqueeze(-1)

    rating_distance = _compute_rating_distance(rating_deltas, config.rating_metric)

    mask = (distances <= config.x_radius) & (rating_distance <= config.y_radius)

    if not torch.any(mask):
        return []

    edge_rows = torch.nonzero(mask, as_tuple=False)
    parents = list(range(latents.shape[0]))

    def find_parent(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(a: int, b: int) -> None:
        root_a = find_parent(a)
        root_b = find_parent(b)
        if root_a == root_b:
            return
        if root_a < root_b:
            parents[root_b] = root_a
        else:
            parents[root_a] = root_b

    neighbour_indices_cpu = indices.cpu()
    for row in edge_rows.cpu():
        anchor = int(row[0].item())
        neighbour_offset = int(row[1].item())
        neighbour = int(neighbour_indices_cpu[anchor, neighbour_offset])
        union(anchor, neighbour)

    clusters_map: Dict[int, List[int]] = {}
    for node in range(len(parents)):
        root = find_parent(node)
        clusters_map.setdefault(root, []).append(node)

    summaries: List[ClusterSummary] = []
    for cluster_id, members in clusters_map.items():
        if len(members) < config.min_cluster_size:
            continue
        member_idx = torch.tensor(members, device=latents.device)
        latent_subset = latents.index_select(0, member_idx)
        rating_subset = ratings.index_select(0, member_idx)
        centroid_latent = latent_subset.mean(dim=0)
        centroid_rating = rating_subset.mean(dim=0)
        latent_centered = latent_subset - centroid_latent
        rating_centered = rating_subset - centroid_rating
        latent_cov = latent_centered.t().matmul(latent_centered) / max(len(members) - 1, 1)
        rating_cov = rating_centered.t().matmul(rating_centered) / max(len(members) - 1, 1)
        summaries.append(
            ClusterSummary(
                cluster_id=cluster_id,
                members=members,
                centroid_latent=centroid_latent,
                centroid_rating=centroid_rating,
                latent_cov=latent_cov,
                rating_cov=rating_cov,
            )
        )

    return summaries


def compute_soft_membership(
    latents: torch.Tensor,
    ratings: torch.Tensor,
    clusters: Sequence[ClusterSummary],
    *,
    latent_weight: float = 1.0,
    rating_weight: float = 1.0,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Return soft membership matrix using cluster centroids."""

    if not clusters:
        raise ValueError("Cannot compute memberships without clusters.")

    device = latents.device
    cluster_latents = torch.stack([c.centroid_latent.to(device=device) for c in clusters], dim=0)
    cluster_ratings = torch.stack([c.centroid_rating.to(device=device) for c in clusters], dim=0)

    latent_dist = torch.cdist(latents, cluster_latents)
    rating_dist = torch.cdist(ratings, cluster_ratings)

    combined = latent_weight * latent_dist.pow(2) + rating_weight * rating_dist.pow(2)
    temp = max(float(temperature), 1e-6)
    logits = -combined / temp
    return torch.softmax(logits, dim=1)


def build_training_context(
    photo_to_coords: Mapping[str, Sequence[float]],
    dim_to_photo_to_rating: Mapping[str, Mapping[str, float]],
    *,
    attributes: Optional[Iterable[str]] = None,
    cluster_config: Optional[ClusterConfig] = None,
    latent_weight: float = 1.0,
    rating_weight: float = 1.0,
    temperature: float = 0.5,
    device: Optional[torch.device] = None,
) -> TrainingContext:
    """Assemble latent/rating tensors, clusters, and soft labels for training."""

    device = device or get_preferred_device()
    latents, ratings, keys, attribute_list = assemble_dataset(
        photo_to_coords,
        dim_to_photo_to_rating,
        attributes=attributes,
        device=device,
    )

    config = cluster_config or ClusterConfig(x_radius=0.8, y_radius=2.0)
    clusters = find_interpolation_safe_clusters(latents, ratings, config)
    if not clusters:
        raise ValueError("No clusters satisfied the provided configuration.")

    membership = compute_soft_membership(
        latents,
        ratings,
        clusters,
        latent_weight=latent_weight,
        rating_weight=rating_weight,
        temperature=temperature,
    )
    hard_labels = torch.argmax(membership, dim=1)

    return TrainingContext(
        attributes=attribute_list,
        keys=keys,
        latents=latents,
        ratings=ratings,
        clusters=clusters,
        soft_membership=membership,
        hard_labels=hard_labels,
    )


__all__ = [
    "ClusterConfig",
    "ClusterSummary",
    "TrainingContext",
    "assemble_dataset",
    "build_training_context",
    "compute_soft_membership",
    "find_interpolation_safe_clusters",
]
