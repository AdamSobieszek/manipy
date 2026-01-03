"""Transformer-based feature space training for latent-to-rating regression.

This version drops the pre-clustering stage and instead enforces a
neighbourhood energy objective that penalises large rating jumps relative
to latent distance directly in the loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    print(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from deixis.face_utils.interpolate.cluster_context import assemble_dataset  # type: ignore
    from deixis.face_utils.interpolate.metrics import build_neighbor_graph, get_preferred_device  # type: ignore
else:
    from .cluster_context import assemble_dataset
    from .metrics import build_neighbor_graph, get_preferred_device


@dataclass
class TransformerFeatureConfig:
    latent_dim: int = 512
    token_dim: int = 64
    num_tokens: int = 32
    embed_dim: int = 768
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_dropout: float = 0.1
    representation_dim: int = 768
    output_dropout: float = 0.1


@dataclass
class TrainingConfig:
    attributes: Optional[Iterable[str]] = None
    feature: TransformerFeatureConfig = field(default_factory=TransformerFeatureConfig)
    batch_size: int = 64
    epochs: int = 25
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    neighbour_k: int = 32
    energy_lambda: float = 1.0
    representation_lambda: float = 0.1
    validation_split: float = 0.15
    max_train_points: Optional[int] = 20000
    random_seed: int = 42
    device: Optional[str] = None
    verbose: bool = True


class LatentNeighbourDataset(Dataset):
    """Return latents, ratings, masks, and neighbourhood information."""

    def __init__(
        self,
        latents: torch.Tensor,
        ratings: torch.Tensor,
        mask: torch.Tensor,
        neighbours: torch.Tensor,
        neighbour_distances: torch.Tensor,
    ):
        self.latents = latents
        self.ratings = ratings
        self.mask = mask
        self.neighbours = neighbours
        self.neighbour_distances = neighbour_distances

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        neighbour_idx = self.neighbours[idx]
        return {
            "latent": self.latents[idx],
            "rating": self.ratings[idx],
            "mask": self.mask[idx],
            "neighbour_latent": self.latents[neighbour_idx],
            "neighbour_rating": self.ratings[neighbour_idx],
            "neighbour_mask": self.mask[neighbour_idx],
            "neighbour_dist": self.neighbour_distances[idx],
        }


class LatentTokeniser(nn.Module):
    def __init__(self, config: TransformerFeatureConfig):
        super().__init__()
        self.config = config
        total_token_dim = config.num_tokens * config.token_dim
        self.to_tokens = nn.Linear(config.latent_dim, total_token_dim)
        self.token_projection = nn.Linear(config.token_dim, config.embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.positional = nn.Parameter(torch.zeros(1, config.num_tokens + 1, config.embed_dim))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.positional, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        tokens = self.to_tokens(x).view(batch, self.config.num_tokens, self.config.token_dim)
        tokens = self.token_projection(tokens)
        cls = self.cls.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return tokens + self.positional


class TransformerFeatureEncoder(nn.Module):
    def __init__(self, config: TransformerFeatureConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.transformer_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.tokeniser = LatentTokeniser(config)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        self.projection = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.representation_dim),
            nn.GELU(),
            nn.Dropout(config.output_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokeniser(x)
        encoded = self.encoder(tokens)
        cls_repr = encoded[:, 0]
        return self.projection(cls_repr)


class TransformerRegressionModel(nn.Module):
    def __init__(self, config: TransformerFeatureConfig, target_dim: int):
        super().__init__()
        self.encoder = TransformerFeatureEncoder(config)
        self.head = nn.Linear(config.representation_dim, target_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        representation = self.encoder(x)
        prediction = self.head(representation)
        return prediction, representation


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    diff = torch.where(mask, diff, torch.zeros_like(diff))
    denom = mask.sum().clamp_min(1.0)
    return diff.pow(2).sum() / denom


def _energy_loss(
    pred: torch.Tensor,
    neighbour_pred: torch.Tensor,
    target: torch.Tensor,
    neighbour_target: torch.Tensor,
    mask: torch.Tensor,
    neighbour_mask: torch.Tensor,
    distances: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    diff_pred = neighbour_pred - pred.unsqueeze(1)
    diff_target = neighbour_target - target.unsqueeze(1)
    joint_mask = mask.unsqueeze(1) & neighbour_mask

    denom = distances.unsqueeze(-1).clamp_min(eps)
    energy_pred = diff_pred.abs() / denom
    energy_true = diff_target.abs() / denom

    residual = energy_pred - energy_true
    residual = torch.where(joint_mask, residual, torch.zeros_like(residual))
    weights = torch.exp(-distances.unsqueeze(-1))

    denom_weights = (joint_mask * weights).sum().clamp_min(1.0)
    return (residual.pow(2) * weights).sum() / denom_weights


def _representation_loss(rep: torch.Tensor, neighbour_rep: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    diff = neighbour_rep - rep.unsqueeze(1)
    penalty = diff.pow(2).sum(dim=-1)
    weights = torch.exp(-distances)
    return (penalty * weights).mean()


def _compute_neighbours(latents: torch.Tensor, n_neighbors: int, block_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    graph = build_neighbor_graph(latents, n_neighbors=n_neighbors + 1, block_size=block_size)
    return graph.indices[:, 1:], graph.distances[:, 1:]


def _prepare_dataset(
    photo_to_coords: Mapping[str, Sequence[float]],
    dim_to_photo_to_rating: Mapping[str, Mapping[str, float]],
    config: TrainingConfig,
    device: torch.device,
) -> Tuple[LatentNeighbourDataset, List[str]]:
    latents, ratings, _, attributes = assemble_dataset(
        photo_to_coords,
        dim_to_photo_to_rating,
        attributes=config.attributes,
        device=device,
    )

    mask = torch.isfinite(ratings)
    latents = latents.to(device)
    ratings = torch.where(mask, ratings, torch.zeros_like(ratings))
    mask = mask.to(device)

    total = latents.shape[0]
    if config.max_train_points is not None and total > config.max_train_points:
        generator = torch.Generator(device="cpu").manual_seed(config.random_seed)
        subset_idx = torch.randperm(total, generator=generator)[: config.max_train_points].to(device)
        latents = latents.index_select(0, subset_idx)
        ratings = ratings.index_select(0, subset_idx)
        mask = mask.index_select(0, subset_idx)

    neighbours, distances = _compute_neighbours(latents, config.neighbour_k)

    dataset = LatentNeighbourDataset(
        latents=latents.cpu(),
        ratings=ratings.cpu(),
        mask=mask.cpu(),
        neighbours=neighbours.cpu(),
        neighbour_distances=distances.cpu(),
    )
    return dataset, attributes


def _split_dataset(dataset: LatentNeighbourDataset, validation_split: float, seed: int) -> Tuple[Dataset, Dataset]:
    if validation_split <= 0 or validation_split >= 1:
        return dataset, dataset
    total = len(dataset)
    val = max(1, int(total * validation_split))
    train = total - val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train, val], generator=generator)


def _collate(batch: List[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in batch[0]}


class TransformerFeatureTrainer:
    def __init__(self, model: TransformerRegressionModel, config: TrainingConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def _step(self, batch: dict[str, torch.Tensor], train: bool = True) -> dict[str, float]:
        latent = batch["latent"].to(self.device)
        rating = batch["rating"].to(self.device)
        mask = batch["mask"].to(self.device)

        pred, rep = self.model(latent)
        supervised = _masked_mse(pred, rating, mask)

        neighbour_latent = batch["neighbour_latent"].to(self.device)
        neighbour_rating = batch["neighbour_rating"].to(self.device)
        neighbour_mask = batch["neighbour_mask"].to(self.device)
        neighbour_dist = batch["neighbour_dist"].to(self.device)

        flat_neighbour_latent = neighbour_latent.view(-1, neighbour_latent.shape[-1])
        neighbour_pred, neighbour_rep = self.model(flat_neighbour_latent)
        neighbour_pred = neighbour_pred.view(latent.shape[0], -1, pred.shape[-1])
        neighbour_rep = neighbour_rep.view(latent.shape[0], -1, rep.shape[-1])

        energy = _energy_loss(
            pred,
            neighbour_pred,
            rating,
            neighbour_rating.to(self.device),
            mask,
            neighbour_mask,
            neighbour_dist,
        )
        representation_penalty = _representation_loss(rep, neighbour_rep, neighbour_dist)

        total_loss = supervised + self.config.energy_lambda * energy + self.config.representation_lambda * representation_penalty

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "supervised": supervised.item(),
            "energy": energy.item(),
            "representation": representation_penalty.item(),
        }

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "supervised": 0.0, "energy": 0.0, "representation": 0.0}
        for batch in loader:
            metrics = self._step(batch, train=True)
            for key in totals:
                totals[key] += metrics[key]
        for key in totals:
            totals[key] /= len(loader)
        return totals

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        totals = {"loss": 0.0, "supervised": 0.0, "energy": 0.0, "representation": 0.0}
        with torch.no_grad():
            for batch in loader:
                metrics = self._step(batch, train=False)
                for key in totals:
                    totals[key] += metrics[key]
        for key in totals:
            totals[key] /= len(loader)
        return totals


def train_transformer_feature_space(
    photo_to_coords: Mapping[str, Sequence[float]],
    dim_to_photo_to_rating: Mapping[str, Mapping[str, float]],
    config: Optional[TrainingConfig] = None,
) -> Tuple[TransformerRegressionModel, dict[str, List[float]]]:
    config = config or TrainingConfig()
    device = torch.device(config.device) if config.device else get_preferred_device()

    dataset, attributes = _prepare_dataset(photo_to_coords, dim_to_photo_to_rating, config, device)
    train_set, val_set = _split_dataset(dataset, config.validation_split, config.random_seed)

    def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            collate_fn=_collate,
        )

    train_loader = _loader(train_set, True)
    val_loader = _loader(val_set, False)

    model = TransformerRegressionModel(config.feature, target_dim=len(attributes))
    trainer = TransformerFeatureTrainer(model, config, device)

    history = {"train_loss": [], "val_loss": [], "val_supervised": [], "val_energy": [], "attributes": attributes}
    for epoch in range(config.epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_supervised"].append(val_metrics["supervised"])
        history["val_energy"].append(val_metrics["energy"])
        if config.verbose:
            print(
                f"[Epoch {epoch + 1:02d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_supervised={val_metrics['supervised']:.4f} "
                f"val_energy={val_metrics['energy']:.4f}"
            )

    return model, history


__all__ = [
    "TransformerFeatureConfig",
    "TrainingConfig",
    "TransformerRegressionModel",
    "train_transformer_feature_space",
]


def _default_data_loader() -> Tuple[Mapping[str, Sequence[float]], Mapping[str, Mapping[str, float]]]:
    try:
        if __package__ in (None, ""):
            from deixis.face_utils.interpolate.load_interp_data import load_interp_data  # type: ignore
        else:
            from .load_interp_data import load_interp_data  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("`load_interp_data` must be available to run as a script.") from exc

    return load_interp_data()


def _parse_cli_config() -> TrainingConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Train transformer feature space.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--energy-lambda", type=float, default=None)
    parser.add_argument("--representation-lambda", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attributes", type=str, nargs="*", default=None)
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.energy_lambda is not None:
        cfg.energy_lambda = args.energy_lambda
    if args.representation_lambda is not None:
        cfg.representation_lambda = args.representation_lambda
    if args.device is not None:
        cfg.device = args.device
    if args.attributes:
        cfg.attributes = args.attributes
    if args.max_points is not None:
        cfg.max_train_points = args.max_points

    if cfg.device is None:
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            cfg.device = "mps"
        elif torch.cuda.is_available():
            cfg.device = "cuda"
        else:
            cfg.device = "cpu"

    return cfg


def _run_cli() -> None:
    photo_to_coords, dim_to_photo_to_rating = _default_data_loader()
    config = _parse_cli_config()
    if config.attributes is None:
        config.attributes = sorted(dim_to_photo_to_rating.keys())
    model, history = train_transformer_feature_space(photo_to_coords, dim_to_photo_to_rating, config=config)
    print("Training complete. Final validation loss:", history["val_loss"][-1])


if __name__ == "__main__":  # pragma: no cover
    _run_cli()
