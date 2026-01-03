"""
Utilities for loading age predictors on StyleGAN outputs.

We support two complementary predictors:

1. An image-based predictor backed by FaceFusion's FairFace ONNX model.
   It consumes rendered StyleGAN images (RGB) and returns an age estimate
   derived from the predicted age bucket.

2. A latent-space predictor operating directly in W-space, using the
   ensemble provided in ``psychGAN/content/final_models/ensemble_age.pt``.

Both predictors expose a common ``predict`` interface returning a 1D tensor
of age estimates (in years) for each sample in the batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import sys
sys.path.append('/Users/adamsobieszek/PycharmProjects/_manipy')

import numpy as np
import torch
from torch import Tensor, nn

from manipy.models.layers import EnsembleRegressor, MeanRegressor
from manipy.stylegan.core import setup_stylegan

from deixis.face_utils .facefusion_wrapper import ensure_facefusion_models, init_facefusion_state

try:
    # FaceFusion modules are available either as top-level ``facefusion`` or via the
    # vendored package inside ``deixis.face_utils.facefusion``. The wrapper already
    # aliases them, so importing from ``facefusion`` works in both cases.
    from facefusion import face_analyser
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "FaceFusion modules are required for the image-based age predictor. "
        "Ensure `deixis.face_utils.facefusion` is importable."
    ) from exc


_FACEFUSION_READY = False


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_facefusion_ready() -> None:
    global _FACEFUSION_READY
    if _FACEFUSION_READY:
        return
    init_facefusion_state()
    ensure_facefusion_models(use_landmarker_68=False)
    _FACEFUSION_READY = True


def _tensor_to_vision_frame(img: Tensor) -> np.ndarray:
    """
    Convert a StyleGAN tensor (C,H,W) to the BGR uint8 frame expected by FaceFusion.
    Supports inputs in [-1,1], [0,1], or [0,255].
    """
    if img.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C,H,W); got {tuple(img.shape)}")

    img_cpu = img.detach().to(torch.float32).cpu()
    data_min = float(img_cpu.min().item())
    data_max = float(img_cpu.max().item())

    if data_min >= -1.01 and data_max <= 1.01:
        # StyleGAN range [-1,1]
        img_cpu = (img_cpu.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0
    elif data_min >= 0.0 and data_max <= 1.01:
        img_cpu = img_cpu.clamp(0.0, 1.0) * 255.0
    else:
        # Assume already scaled to 0-255
        img_cpu = img_cpu.clamp(0.0, 255.0)

    np_img = img_cpu.round().to(torch.uint8).permute(1, 2, 0).numpy()
    # Convert RGB -> BGR for OpenCV / FaceFusion
    return np_img[:, :, ::-1]


def _age_range_to_years(age_range: range | int | float) -> float:
    """
    FaceFusion returns age buckets as ``range`` instances.
    Convert them to a representative age in years (midpoint of the inclusive range).
    """
    if isinstance(age_range, range):
        low = age_range.start
        high = max(age_range.stop - 1, low)
        return float(low + (high - low) * 0.5)
    return float(age_range)


@dataclass
class FairFaceAgePredictor:
    """
    Image-based age predictor using FaceFusion's FairFace classifier.

    Call ``predict`` with a batch of StyleGAN images (N,C,H,W) and receive
    a 1D tensor of age estimates (in years). Frames where no face is detected
    return ``nan``.
    """

    auto_init: bool = True

    def __post_init__(self) -> None:
        if self.auto_init:
            _ensure_facefusion_ready()

    def predict(self, images: Tensor) -> Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"Expected image batch with shape (N,C,H,W); got {tuple(images.shape)}")

        predictions: List[float] = []
        for img in images:
            try:
                frame = _tensor_to_vision_frame(img)
                faces = face_analyser.get_many_faces([frame])  # returns list[Face]
                face = face_analyser.get_one_face(faces)
                if face is None or face.age is None:
                    predictions.append(float("nan"))
                    continue
                predictions.append(_age_range_to_years(face.age))
            except Exception:
                predictions.append(float("nan"))

        return torch.tensor(predictions, dtype=torch.float32)


class WSpaceAgePredictor(nn.Module):
    """
    Ensemble regressor operating on StyleGAN W-space vectors.

    The checkpoint is expected at ``psychGAN/content/final_models/ensemble_age.pt``
    relative to the repository root. Set ``normalize_output`` to True to report
    normalized scores (0-1); otherwise the raw model output is returned which
    typically corresponds to age in years.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        device: Optional[torch.device] = None,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.device = device or _default_device()
        self.normalize_output = normalize_output

        if checkpoint_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            checkpoint_path = repo_root / "psychGAN" / "content" / "final_models" / "ensemble_age.pt"

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Age ensemble checkpoint not found at {checkpoint_path}. "
                "Verify that the psychGAN assets are available."
            )

        self.model = EnsembleRegressor(
            [MeanRegressor(512, 1) for _ in range(8)],
            model_kwargs={},
        )
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    def forward(self, w: Tensor) -> Tensor:
        """
        Forward pass accepting either W (N,512) or W+ (N,18,512).
        """
        if w.ndim == 3:
            w = w[:, 0, :]
        if w.ndim != 2 or w.shape[-1] != 512:
            raise ValueError(f"Expected W vectors with shape (N,512); got {tuple(w.shape)}")

        with torch.no_grad():
            preds = self.model(w.to(self.device)).squeeze(-1)

        if self.normalize_output:
            preds = preds / 100.0
        return preds


@dataclass
class AgeComparisonResult:
    image_ages: Tensor
    w_ages: Tensor
    ws: Tensor
    images: Tensor


def compare_age_predictors(
    *,
    generator,
    z: Tensor,
    truncation_psi: Optional[float] = None,
    normalize_w: bool = False,
    mapping_kwargs: Optional[dict] = None,
    synthesis_kwargs: Optional[dict] = None,
) -> AgeComparisonResult:
    """
    Run both predictors on the provided latent batch.

    Args:
        generator: StyleGAN generator (expects ``mapping`` and ``synthesis`` methods).
        z: (N,512) latent tensor sampled in Z space.
        truncation_psi: Optional truncation value for the mapping network.
        normalize_w: If True, report W predictions normalized to [0,1].
        mapping_kwargs: Extra keyword arguments forwarded to ``generator.mapping``.
        synthesis_kwargs: Extra keyword arguments forwarded to ``generator.synthesis``.

    Returns:
        AgeComparisonResult containing per-sample predictions and intermediate tensors.
    """
    device = next(generator.parameters()).device
    mapping_kwargs = dict(mapping_kwargs or {})
    if truncation_psi is not None:
        mapping_kwargs.setdefault("truncation_psi", truncation_psi)

    ws = generator.mapping(z.to(device), None, **mapping_kwargs)

    synthesis_kwargs = dict(synthesis_kwargs or {})
    images = generator.synthesis(ws, **synthesis_kwargs)

    image_predictor = FairFaceAgePredictor()
    img_preds = image_predictor.predict(images.detach().cpu())

    w_predictor = WSpaceAgePredictor(device=device, normalize_output=normalize_w)
    w_preds = w_predictor(ws[:, 0, :]).detach().cpu()

    return AgeComparisonResult(
        image_ages=img_preds,
        w_ages=w_preds,
        ws=ws.detach().cpu(),
        images=images.detach().cpu(),
    )


def sample_and_compare_age(
    num_samples: int = 32,
    truncation_psi: float = 0.7,
    seed: Optional[int] = None,
    normalize_w: bool = False,
) -> AgeComparisonResult:
    """
    Convenience wrapper that samples random latents, renders images, and runs both predictors.
    """
    device = _default_device()
    generator, _, gan_device = setup_stylegan(device=str(device))
    device = gan_device
    generator = generator.to(device).eval()

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    z = torch.randn(num_samples, generator.mapping.z_dim, device=device)
    return compare_age_predictors(
        generator=generator,
        z=z,
        truncation_psi=truncation_psi,
        normalize_w=normalize_w,
    )


def comparison_to_dataframe(result: AgeComparisonResult):
    """
    Convert an ``AgeComparisonResult`` into a pandas DataFrame with summary statistics.
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "sample": np.arange(result.image_ages.shape[0]),
            "age_image": result.image_ages.numpy(),
            "age_w": result.w_ages.numpy(),
        }
    )
    df["difference"] = df["age_w"] - df["age_image"]
    return df


if __name__ == "__main__":  # pragma: no cover - manual utility
    comparison = sample_and_compare_age(num_samples=8, truncation_psi=0.7, seed=42)
    df = comparison_to_dataframe(comparison)
    print(df.to_string(index=False))
    print("\nDifference summary:")
    print(df["difference"].describe())
