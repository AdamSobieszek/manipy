"""
Natural-gradient refinement of StyleGAN2 W-space inversions.

This module implements a geometry-aware refinement loop that combines
per-image posterior covariances (returned by :func:`load_interp_data`)
with local covariance estimates drawn from random W-space samples.
The resulting metric is used to precondition the feature-space gradient,
yielding a natural-gradient descent update that keeps the optimisation
within the high-density manifold of the generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from manipy.stylegan.utils import show_faces

try:
    from face_utils.sg_opt.sg_output_analysis import build_feature_extractor
except ImportError as exc:  # pragma: no cover - imported at runtime within project
    raise ImportError(
        "build_feature_extractor is required. Ensure 'deixis.face_utils' "
        "is importable from your PYTHONPATH."
    ) from exc


# -------------------------------------------------------------------------
# Configuration dataclasses
# -------------------------------------------------------------------------


@dataclass
class NaturalProjectionConfig:
    """Hyperparameters controlling the refinement loop."""

    steps: int = 50
    learning_rate: float = 0.08
    lr_decay: float = 0.75
    lr_decay_every: int = 20
    prior_weight: float = 0.08
    prior_face_weight: float = 0.35
    covariance_blend: float = 0.6  # weight on face-specific covariance
    damping: float = 1e-3
    candidate_neighbor_k: int = 256
    metric: str = "diag"  # {"diag", "full"}
    local_stats_refresh: int = 10
    truncation_psi: float = 1.0
    noise_mode: str = "random"
    max_norm_std: Optional[float] = 2.5
    early_stop_tol: Optional[float] = 1e-4
    early_stop_patience: int = 5
    device: Optional[str] = None
    target_feature: str = "vgg16_512"
    display_every: Optional[int] = None
    identity_lambda: float = 0.0
    grad_accum: int = 6
    norm_target: float = 20.0
    norm_weight: float = 0.1


@dataclass
class CandidateDistribution:
    """
    Random W-space samples gathered from the generator.

    Attributes:
        samples: (M, D) tensor of W vectors.
        w_avg: (D,) StyleGAN moving-average center.
        global_mean: (D,) empirical mean of samples.
        global_cov_diag: (D,) diagonal covariance estimate.
        norm_mean: mean radius ||w - w_avg|| across samples.
        norm_std: std-dev of radius ||w - w_avg|| across samples.
    """

    samples: torch.Tensor
    w_avg: torch.Tensor
    global_mean: torch.Tensor
    global_cov_diag: torch.Tensor
    norm_mean: float
    norm_std: float

    @classmethod
    @torch.no_grad()
    def from_generator(
        cls,
        G: torch.nn.Module,
        *,
        num_samples: int = 10_000,
        batch_size: int = 512,
        truncation_psi: float = 1.0,
        device: torch.device | str,
    ) -> "CandidateDistribution":
        """
        Draw random latents from `G` to define a prior over W.

        Args:
            G: StyleGAN generator (expects `.mapping` and `.synthesis`).
            num_samples: total W samples to draw.
            batch_size: minibatch size for sampling loop.
            truncation_psi: truncation for mapping network.
            device: torch device used for sampling.
        """

        device = torch.device(device)
        mapping = G.mapping
        z_dim = mapping.z_dim
        w_avg = mapping.w_avg.detach().to(device)

        samples: list[torch.Tensor] = []
        remaining = num_samples
        while remaining > 0:
            cur = min(batch_size, remaining)
            z = torch.randn(cur, z_dim, device=device)
            w = mapping(z, None, truncation_psi=truncation_psi)[:, 0]
            samples.append(w.detach())
            remaining -= cur

        w_samples = torch.cat(samples, dim=0)
        global_mean = w_samples.mean(dim=0)
        centered = w_samples - global_mean
        global_cov_diag = centered.pow(2).mean(dim=0)

        centered_to_avg = w_samples - w_avg
        norms = centered_to_avg.norm(dim=1)
        norm_mean = norms.mean().item()
        norm_std = norms.std(unbiased=False).item()

        return cls(
            samples=w_samples,
            w_avg=w_avg,
            global_mean=global_mean,
            global_cov_diag=global_cov_diag,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

    def to(self, device: torch.device | str) -> "CandidateDistribution":
        """Return a distribution with tensors moved to `device`."""
        device = torch.device(device)
        return CandidateDistribution(
            samples=self.samples.to(device),
            w_avg=self.w_avg.to(device),
            global_mean=self.global_mean.to(device),
            global_cov_diag=self.global_cov_diag.to(device),
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
        )

    @torch.no_grad()
    def local_statistics(
        self,
        latent: torch.Tensor,
        *,
        k: int,
        metric: str = "diag",
        eps: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute local mean & covariance around `latent` using the closest `k` samples.
        """

        if k <= 0 or k > self.samples.shape[0]:
            raise ValueError(f"Invalid neighbour count k={k}")

        latent = latent.detach()
        diff = self.samples - latent.unsqueeze(0)
        dist2 = diff.pow(2).sum(dim=1)
        knn = torch.topk(dist2, k=k, largest=False).indices
        subset = self.samples.index_select(0, knn)
        local_mean = subset.mean(dim=0)

        centered = subset - local_mean
        if metric == "full":
            cov = centered.t().matmul(centered) / float(k)
            cov = cov + torch.eye(cov.shape[0], device=cov.device) * eps
        else:
            cov = centered.pow(2).mean(dim=0) + eps

        return local_mean, cov


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------


def _load_image_tensor(path: Path | str, device: torch.device) -> torch.Tensor:
    """
    Load an image file as a StyleGAN-compatible tensor in [-1, 1].
    """

    image = Image.open(path).convert("RGB")
    array = torch.from_numpy(np.array(image, dtype=np.float32))
    array = array.permute(2, 0, 1) / 255.0  # (3, H, W) in [0,1]
    tensor = array.unsqueeze(0) * 2.0 - 1.0
    return tensor.to(device)


def _preprocess_for_vgg(
    images: torch.Tensor,
    out_res: int = 224,
    *,
    crop_coords: Optional[Union[torch.Tensor, Tuple[int, int]]] = None,
    return_coords: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Resize and normalise images for the VGG feature extractor.

    StyleGAN images are typically in [-1, 1]; we map them to [0, 1],
    downsample to 512, take a random 224 crop (unless `crop_coords` provided),
    and apply ImageNet stats.
    """

    if images.min() < 0.0 or images.max() > 1.0:
        images = (images + 1.0) * 0.5
    images = images.clamp(0.0, 1.0)

    pooled = F.avg_pool2d(images, kernel_size=2, stride=2)
    batch, _, h, w = pooled.shape
    if h < out_res or w < out_res:
        raise ValueError(f"Cannot crop {out_res} from spatial size {(h, w)}")

    device = pooled.device
    max_top = h - out_res
    max_left = w - out_res

    if crop_coords is None:
        if max_top == 0 and max_left == 0:
            tops = torch.zeros(batch, dtype=torch.long, device=device)
            lefts = torch.zeros(batch, dtype=torch.long, device=device)
        else:
            tops = torch.randint(0, max_top + 1, (batch,), device=device)
            lefts = torch.randint(0, max_left + 1, (batch,), device=device)
    else:
        crop_tensor = torch.as_tensor(crop_coords, device=device, dtype=torch.long)
        if crop_tensor.ndim == 1:
            crop_tensor = crop_tensor.unsqueeze(0)
        if crop_tensor.shape[0] != batch:
            if crop_tensor.shape[0] == 1:
                crop_tensor = crop_tensor.expand(batch, -1)
            else:
                raise ValueError("crop_coords batch dimension mismatch")
        tops, lefts = crop_tensor[:, 0], crop_tensor[:, 1]
        tops = tops.clamp(0, max_top)
        lefts = lefts.clamp(0, max_left)

    crops = []
    for idx in range(batch):
        top = int(tops[idx].item())
        left = int(lefts[idx].item())
        crops.append(pooled[idx : idx + 1, :, top : top + out_res, left : left + out_res])
    cropped = torch.cat(crops, dim=0)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    normalised = (cropped - mean) / std

    if return_coords:
        coords = torch.stack([tops, lefts], dim=1)
        return normalised, coords
    return normalised


def _extract_features(
    images: torch.Tensor,
    extractor: torch.nn.Module,
    *,
    crop_coords: Optional[Union[torch.Tensor, Tuple[int, int]]] = None,
    return_crop: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run VGG feature extractor (expects module already on device)."""

    preprocessed, coords = _preprocess_for_vgg(
        images,
        crop_coords=crop_coords,
        return_coords=True,
    )
    feats = extractor(preprocessed)
    if return_crop:
        return feats, coords
    return feats


def _quad_form(
    diff: torch.Tensor,
    precision: torch.Tensor,
    *,
    metric: str,
) -> torch.Tensor:
    """Compute 0.5 * diff^T * precision * diff for diag/full metrics."""

    if metric == "diag":
        return 0.5 * (diff.pow(2) * precision).sum()
    return 0.5 * diff.unsqueeze(0).matmul(precision).matmul(diff.unsqueeze(1)).squeeze()


def _apply_metric(grad: torch.Tensor, covariance: torch.Tensor, *, metric: str) -> torch.Tensor:
    """Apply covariance metric to gradient (natural gradient step)."""

    if grad.ndim > 1:
        grad = grad.view(-1)

    if metric == "diag":
        if covariance.ndim == 2:
            cov_vec = covariance.diagonal()
        else:
            cov_vec = covariance
        return grad * cov_vec

    if covariance.ndim == 1:
        cov_matrix = torch.diag(covariance)
    else:
        cov_matrix = covariance
    step = cov_matrix.matmul(grad.unsqueeze(-1))
    return step.view(-1)


def _blend_covariances(
    face_cov: torch.Tensor,
    local_cov: torch.Tensor,
    *,
    blend: float,
    damping: float,
    metric: str,
    identity_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine posterior (face-specific) covariance with local manifold covariance.
    """

    blend = float(blend)
    if not (0.0 <= blend <= 1.0):
        raise ValueError(f"blend must be within [0,1], got {blend}")

    if not (0.0 <= identity_lambda <= 1.0):
        raise ValueError(f"identity_lambda must be within [0,1], got {identity_lambda}")

    if metric == "diag":
        face_diag = face_cov
        local_diag = local_cov
        combined = blend * face_diag + (1.0 - blend) * local_diag
        if identity_lambda:
            identity = torch.ones_like(combined)
            combined = (1.0 - identity_lambda) * combined + identity_lambda * identity
        combined = combined + damping
        precision = 1.0 / combined
    else:
        combined = blend * face_cov + (1.0 - blend) * local_cov
        eye = torch.eye(combined.shape[0], device=combined.device, dtype=combined.dtype)
        if identity_lambda:
            combined = (1.0 - identity_lambda) * combined + identity_lambda * eye
        combined = combined + eye * damping
        precision = torch.linalg.inv(combined)
    return combined, precision


# -------------------------------------------------------------------------
# Natural-gradient projector
# -------------------------------------------------------------------------


@dataclass
class ProjectionOutcome:
    latent: torch.Tensor
    loss_history: torch.Tensor
    prior_mean: torch.Tensor
    combined_covariance: torch.Tensor
    target_feats: Optional[torch.Tensor] = None


class NaturalGradientProjector:
    """
    Geometry-aware latent refinement using natural-gradient descent.
    """

    def __init__(
        self,
        G: torch.nn.Module,
        candidate_distribution: CandidateDistribution,
        *,
        config: Optional[NaturalProjectionConfig] = None,
        extractor: Optional[torch.nn.Module] = None,
    ) -> None:
        self.config = config or NaturalProjectionConfig()
        self.device = torch.device(self.config.device or candidate_distribution.samples.device)
        self.G = G.to(self.device).eval()
        self.candidates = candidate_distribution.to(self.device)
        self.metric = self.config.metric

        if extractor is None:
            extractor = build_feature_extractor(self.config.target_feature, device=self.device)
        self.extractor = extractor.to(self.device).eval()
        for p in self.extractor.parameters():
            p.requires_grad_(False)

        self.num_ws = getattr(self.G.synthesis, "num_ws", 18)

    def _project_to_norm_ball(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Constrain latent radius w.r.t. generator average according to candidate stats.
        """

        if self.config.max_norm_std is None:
            return latent

        centered = latent - self.candidates.w_avg
        radius = centered.norm()
        limit = 20 #self.candidates.norm_mean + self.config.max_norm_std * self.candidates.norm_std
        if radius >= limit:
            return latent
        return self.candidates.w_avg + centered * (limit / (radius + 1e-8))

    def _maybe_refresh_geometry(
        self,
        latent: torch.Tensor,
        face_cov: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-estimate manifold statistics around the current latent.
        """

        local_mu, local_cov = self.candidates.local_statistics(
            latent,
            k=self.config.candidate_neighbor_k,
            metric=self.metric,
        )
        combined_cov, combined_precision = _blend_covariances(
            face_cov,
            local_cov,
            blend=self.config.covariance_blend,
            damping=self.config.damping,
            metric=self.metric,
            identity_lambda=self.config.identity_lambda,
        )

        prior_mu = self.config.prior_face_weight * latent + (1.0 - self.config.prior_face_weight) * local_mu
        return prior_mu.detach(), combined_cov.detach(), combined_precision.detach()

    def refine_single(
        self,
        target_image: torch.Tensor,
        *,
        initial_latent: torch.Tensor,
        face_covariance: torch.Tensor,
        target_path: Optional[Path | str] = None,
    ) -> ProjectionOutcome:
        """
        Refine a single latent code against `target_image`.

        Args:
            target_image: tensor in [-1,1], shape (1,3,H,W).
            initial_latent: tensor (D,) initial W estimate.
            face_covariance: tensor (D,) or (D,D) giving posterior covariance.
            target_path: optional path to the reference image for visualisation.
        """

        device = self.device
        target_image = target_image.to(device)
        latent = initial_latent.to(device).float()
        face_covariance = face_covariance.to(device).float()

        prior_mu, combined_cov, combined_precision = self._maybe_refresh_geometry(latent, face_covariance)
        history: list[float] = []
        best_latent = latent.clone()
        best_loss = math.inf
        patience_counter = 0
        lr = self.config.learning_rate
        grad_accum = max(1, int(self.config.grad_accum))
        last_target_feats: Optional[torch.Tensor] = None

        target_base = target_image

        for step in range(self.config.steps):
            should_display = False
            if self.config.display_every:
                should_display = (step % self.config.display_every == 0) or (step == self.config.steps - 1)

            lat_param = latent.detach().clone().requires_grad_(True)
            w_batch = lat_param.unsqueeze(0).repeat(grad_accum, 1)
            w_batch = w_batch.unsqueeze(1).repeat(1, self.num_ws, 1).contiguous()

            synth = self.G.synthesis(w_batch, noise_mode=self.config.noise_mode)
            feat, crop = _extract_features(synth, self.extractor, return_crop=True)

            with torch.no_grad():
                target_batch = target_base.repeat(grad_accum, 1, 1, 1)
                target_feat = _extract_features(
                    target_batch,
                    self.extractor,
                    crop_coords=crop,
                )

            feat_residual = feat - target_feat
            feat_loss = 0.5 * feat_residual.pow(2).sum(dim=1)
            data_loss = feat_loss.mean()
            prior_loss = _quad_form(lat_param - prior_mu, combined_precision, metric=self.metric)
            if self.config.norm_weight > 0.0:
                center_diff = lat_param - self.candidates.w_avg
                radius = center_diff.norm()
                norm_loss = 0.5 * (radius - self.config.norm_target) ** 2
            else:
                norm_loss = lat_param.new_tensor(0.0)
            loss = (
                data_loss
                + self.config.prior_weight * prior_loss
                + self.config.norm_weight * norm_loss
            )

            loss.backward()
            grad = lat_param.grad.detach()

            natural_step = _apply_metric(grad, combined_cov, metric=self.metric)
            latent = (lat_param - lr * natural_step).detach()
            latent = self._project_to_norm_ball(latent)
            loss_value = loss.item()
            history.append(loss_value)
            last_target_feats = target_feat.mean(dim=0).detach()

            if should_display and target_path is not None:
                try:
                    show_faces(
                        [latent.detach().cpu(), str(target_path)],
                        grid=True,
                        rows=1,
                        device=str(self.device),
                        is_w=True,
                        _G=self.G,
                    )
                except Exception as exc:  # pragma: no cover - display best effort
                    print(f"[display] Failed to show faces at step {step}: {exc}")
                        
            if loss_value + 1e-6 < best_loss:
                best_loss = loss_value
                best_latent = latent.clone()
                patience_counter = 0
            else:
                patience_counter += 1

            if self.config.early_stop_tol is not None:
                if patience_counter >= self.config.early_stop_patience:
                    recent = history[-self.config.early_stop_patience :]
                    if max(recent) - min(recent) < self.config.early_stop_tol:
                        break

            if (step + 1) % self.config.local_stats_refresh == 0:
                prior_mu, combined_cov, combined_precision = self._maybe_refresh_geometry(latent, face_covariance)

            if (step + 1) % self.config.lr_decay_every == 0:
                lr *= self.config.lr_decay

        return ProjectionOutcome(
            latent=best_latent.detach(),
            loss_history=torch.tensor(history, device=device),
            prior_mean=prior_mu.detach(),
            combined_covariance=combined_cov.detach(),
            target_feats=last_target_feats.detach() if last_target_feats is not None else None,
        )

    def refine_folder(
        self,
        folder: Path | str,
        photo_to_latent: Dict[str, torch.Tensor],
        photo_to_covariance: Dict[str, torch.Tensor],
        *,
        output_dir: Optional[Path | str] = None,
        overwrite: bool = False,
    ) -> Dict[str, ProjectionOutcome]:
        """
        Refine all images inside `folder` whose filenames match the supplied priors.

        Args:
            folder: directory containing target images.
            photo_to_latent: mapping filename -> initial latent (D,).
            photo_to_covariance: mapping filename -> covariance (D,) or (D,D).
            output_dir: optional directory to persist `.pt` results.
            overwrite: if False, existing outputs are skipped.
        """

        folder = Path(folder)
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, ProjectionOutcome] = {}
        device = self.device

        for name, latent in photo_to_latent.items():
            img_path = folder / name
            if not img_path.exists():
                continue
            if output_dir is not None:
                out_file = output_dir / f"{Path(name).stem}.pt"
                if out_file.exists() and not overwrite:
                    continue

            target = _load_image_tensor(img_path, device=device)
            cov = photo_to_covariance[name]

            outcome = self.refine_single(
                target,
                initial_latent=latent,
                face_covariance=cov,
                target_path=img_path,
            )
            results[name] = outcome

            if output_dir is not None:
                torch.save(
                    {
                        "w": outcome.latent.cpu(),
                        "loss_history": outcome.loss_history.cpu(),
                        "prior_mean": outcome.prior_mean.cpu(),
                        "combined_covariance": outcome.combined_covariance.cpu(),
                    },
                    out_file,
                )

        return results


__all__ = [
    "CandidateDistribution",
    "NaturalGradientProjector",
    "NaturalProjectionConfig",
    "ProjectionOutcome",
]
