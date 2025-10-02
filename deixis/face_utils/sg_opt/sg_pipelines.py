# sgpipelines/analysis.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal, Callable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================
# Feature extractors
# ============================

class _IdentityPool(nn.Module):
    def forward(self, x): return x

class FeatureExtractor(nn.Module):
    """
    Wraps a backbone and produces a single flat feature vector per image.
    Expect input: float tensor in [0,1], shape (N,3,H,W), RGB.
    """
    def __init__(self, trunk: nn.Module, pool: nn.Module | None = None, proj: nn.Module | None = None):
        super().__init__()
        self.trunk = trunk
        self.pool = pool if pool is not None else _IdentityPool()
        self.proj = proj if proj is not None else _IdentityPool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.trunk(x)
        if isinstance(y, (list, tuple)):
            y = y[-1]
        # if feature map, global-average pool
        if y.ndim == 4:
            y = torch.flatten(F.adaptive_avg_pool2d(y, 1), 1)
        y = self.pool(y)
        y = self.proj(y)
        return y


def build_feature_extractor(name: str, device: torch.device | str = "cpu") -> FeatureExtractor:
    """
    name ∈ {"vgg16","resnet50","lpips","clip-vitb32", ...}
    (LPIPS and CLIP are optional; we fall back gracefully if not installed.)
    """
    n = name.lower()

    if n == "vgg16":
        from torchvision.models import vgg16, VGG16_Weights
        m = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23]  # conv3_3
        return FeatureExtractor(nn.Sequential(m)).to(device).eval()

    if n == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        trunk = nn.Sequential(*(list(m.children())[:-1]))  # → (N,2048,1,1)
        proj = nn.Flatten(1)
        return FeatureExtractor(trunk, proj=proj).to(device).eval()

    if n == "lpips":
        try:
            import lpips  # pip install lpips
        except Exception as e:
            raise RuntimeError("LPIPS not available; pip install lpips") from e
        net = lpips.LPIPS(net='vgg')  # returns (N,1,1,1) if called on image pairs
        # Wrap to behave like a feature extractor on single images:
        class LPIPSFeat(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            def forward(self, x):
                # Compare to black image to get a (pseudo) embedding; not a true embedding but useful for sensitivity
                z = torch.zeros_like(x)
                d = self.net(x*2-1, z*2-1)  # LPIPS expects [-1,1]
                return d.view(x.shape[0], -1)
        return FeatureExtractor(LPIPSFeat(net)).to(device).eval()

    if n in {"clip", "clip-vitb32"}:
        try:
            import clip  # pip install git+https://github.com/openai/CLIP.git
        except Exception as e:
            raise RuntimeError("CLIP not available; install openai-clip") from e
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        class CLIPImageFeat(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                # CLIP expects normalized [-1,1] w/ mean/std; we accept [0,1] and convert
                mean = torch.tensor([0.48145466,0.4578275,0.40821073], device=x.device)[None,:,None,None]
                std  = torch.tensor([0.26862954,0.26130258,0.27577711], device=x.device)[None,:,None,None]
                x_n = (x - mean) / std
                return self.m.encode_image(x_n).float()
        return FeatureExtractor(CLIPImageFeat(model)).to(device).eval()

    raise ValueError(f"Unknown feature extractor: {name}")


# ============================
# Differentiable coords → patch → features
# ============================

@dataclass
class CoordsToFeaturesConfig:
    extractor_name: str = "resnet50"
    out_res: int = 224            # feature extractor input resolution (square)
    patch_px: int = 128           # physical patch width/height in pixels (on the source image)
    coords_mode: Literal["pixels","normalized"] = "pixels"  # input coords convention
    clamp: bool = True            # clamp sampling grid to [-1,1]


def _to_norm_xy(xy: torch.Tensor, H: int, W: int, mode: str) -> torch.Tensor:
    """
    Convert (x,y) in pixels or normalized to normalized coords in [-1,1] (align_corners=True).
    xy: (...,2)
    """
    if mode == "normalized":
        return xy
    # pixels → normalized
    x, y = xy[..., 0], xy[..., 1]
    xn = 2.0 * x / max(W-1, 1) - 1.0
    yn = 2.0 * y / max(H-1, 1) - 1.0
    return torch.stack([xn, yn], dim=-1)


def _make_patch_grid(
    center_xy_norm: torch.Tensor,  # (2,), requires_grad=True
    H: int, W: int,
    out_res: int,
    patch_px: int,
    clamp: bool = True,
) -> torch.Tensor:
    """
    Create a sampling grid (1, out_res, out_res, 2) in normalized coords so that
    grid_sample(image, grid) returns a patch centered at center_xy_norm with physical
    size ~ patch_px × patch_px (in source pixel units).
    """
    device = center_xy_norm.device
    # Base grid in [-1,1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, out_res, device=device),
        torch.linspace(-1, 1, out_res, device=device),
        indexing="ij"
    )
    base = torch.stack([xx, yy], dim=-1)  # (H,W,2)

    # Half-size of patch in normalized coords: (patch_px/2) * (2/W or 2/H) = patch_px/W (or /H)
    sx = (patch_px / float(W))
    sy = (patch_px / float(H))
    scaled = torch.stack([sx * base[..., 0], sy * base[..., 1]], dim=-1)

    # Shift to center
    grid = scaled + center_xy_norm[None, None, :]
    if clamp:
        grid = torch.clamp(grid, -1.0, 1.0)
    return grid.unsqueeze(0)  # (1, out_res, out_res, 2)


def coords_to_features(
    image_bchw: torch.Tensor,        # (1,3,H,W), RGB, float in [0,1]
    center_xy: torch.Tensor,         # (2,) x,y in pixels or normalized
    cfg: CoordsToFeaturesConfig,
    extractor: Optional[FeatureExtractor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (features, patch_bchw). Differentiable w.r.t. center_xy (and image if you want).
    """
    assert image_bchw.ndim == 4 and image_bchw.shape[0] == 1 and image_bchw.shape[1] == 3
    _, _, H, W = image_bchw.shape
    device = image_bchw.device

    center_xy = center_xy.to(device).float()
    center_xy_norm = _to_norm_xy(center_xy, H, W, cfg.coords_mode)
    center_xy_norm.requires_grad_(True)

    grid = _make_patch_grid(center_xy_norm, H, W, cfg.out_res, cfg.patch_px, cfg.clamp)
    patch = F.grid_sample(image_bchw, grid, mode="bilinear", padding_mode="border", align_corners=True)

    if extractor is None:
        extractor = build_feature_extractor(cfg.extractor_name, device=device)
    with torch.set_grad_enabled(True):
        feats = extractor(patch)  # (1,D)
    return feats, patch


# ============================
# Sensitivity (grad/Hessian) at a coordinate
# ============================

@dataclass
class SensitivityResult:
    feats: torch.Tensor             # (D,)
    energy: torch.Tensor            # scalar g = 0.5||f||^2 or other reduction
    grad_xy: torch.Tensor           # (2,) ∂g/∂(x,y)   in chosen coords_mode
    unit_dir: torch.Tensor          # (2,) normalized gradient direction (steepest ascent)
    hess_xy: torch.Tensor           # (2,2) Hessian of g wrt (x,y)
    eigvals: torch.Tensor           # (2,) principal curvatures
    eigvecs: torch.Tensor           # (2,2) columns = principal directions
    patch: torch.Tensor             # (1,3,out_res,out_res)

def feature_sensitivity_at(
    image_bchw: torch.Tensor,            # (1,3,H,W), RGB, [0,1]
    center_xy: Tuple[float,float] | torch.Tensor,  # (x,y) in pixels by default
    extractor_name: str = "resnet50",
    coords_mode: Literal["pixels","normalized"] = "pixels",
    patch_px: int = 128,
    out_res: int = 224,
    reduction: Literal["l2","l2_half","l1"] = "l2_half",
) -> SensitivityResult:
    """
    Compute ∂g/∂(x,y) and ∂²g/∂(x,y)² where g is a scalar energy of features f(patch(x,y)).
    - Default g = 0.5*||f||^2 (smooth and convenient): 'l2_half'
    - 'l2' uses ||f||^2, 'l1' uses ||f||_1 (subgradient-friendly).
    Returns gradient direction (steepest ascent), Hessian, and eigendecomposition.
    """
    if isinstance(center_xy, tuple):
        center_xy = torch.tensor(center_xy, dtype=torch.float32, device=image_bchw.device)

    cfg = CoordsToFeaturesConfig(
        extractor_name=extractor_name,
        out_res=out_res,
        patch_px=patch_px,
        coords_mode=coords_mode,
        clamp=True,
    )

    # Build extractor once
    extractor = build_feature_extractor(extractor_name, device=image_bchw.device)

    # Wrap coords as a leaf tensor we can differentiate w.r.t.
    xy = center_xy.detach().clone().requires_grad_(True)

    # Inner function that maps coords → scalar energy
    def energy_from_xy(xy_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats, patch = coords_to_features(image_bchw, xy_in, cfg, extractor)
        feats = feats.view(-1)  # (D,)

        if reduction == "l2_half":
            g = 0.5 * torch.dot(feats, feats)          # 0.5 * ||f||^2
        elif reduction == "l2":
            g = torch.dot(feats, feats)                # ||f||^2
        elif reduction == "l1":
            g = feats.abs().sum()
        else:
            raise ValueError("Unknown reduction")

        return g, feats, patch

    # First-order gradient
    print(xy.shape)
    g, fvec, patch = energy_from_xy(xy)
    grad = torch.autograd.grad(g, xy, create_graph=True, retain_graph=True)[0]  # (2,)

    # Second-order: Hessian (2x2)
    # Small helper for autograd.functional.hessian expects a function returning scalar
    def _scalar_fn(xy_in: torch.Tensor) -> torch.Tensor:
        g2, _, _ = energy_from_xy(xy_in)
        return g2

    H = torch.autograd.functional.hessian(_scalar_fn, xy, create_graph=False)  # (2,2)

    # Principal directions/curvatures
    # symmetric 2x2 → eigh is stable
    evals, evecs = torch.linalg.eigh(H)

    # Steepest-ascent direction in the image plane
    grad_dir = grad / (grad.norm() + 1e-8)

    return SensitivityResult(
        feats=fvec.detach(),
        energy=g.detach(),
        grad_xy=grad.detach(),
        unit_dir=grad_dir.detach(),
        hess_xy=H.detach(),
        eigvals=evals.detach(),
        eigvecs=evecs.detach(),   # columns are eigenvectors
        patch=patch.detach()
    )