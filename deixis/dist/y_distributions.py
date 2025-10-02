# y_distributions.py
# ------------------------------------------------------------
# 1D target marginals + losses for constant-marginal training.
# ------------------------------------------------------------
from __future__ import annotations

from typing import Optional, Sequence, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution

def to_tensor(x: np.ndarray | list | torch.Tensor):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        x_tensor = torch.tensor(x.flatten(), dtype=torch.float32)
    elif isinstance(x, list):
        x_tensor = torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, torch.Tensor):
        x_tensor = x.to(torch.float32)
    else:
        try:
            x_tensor = torch.tensor(list(x), dtype=torch.float32)
        except:
            raise ValueError(f"Unsupported type: {type(x)}")
    return x_tensor

# =========================
# Base API
# =========================
class Distribution1D(Distribution):
    """
    Abstract univariate distribution aligned with torch.distributions.Distribution.

    Subclasses must implement:
      - cdf(x):     R^N -> [0,1]^N
      - icdf(u):    [0,1]^N -> R^N
      - log_prob(x): log pdf(x)

    Notes:
      - event_shape is always ().
      - rsample defaults to inverse-CDF sampling of uniform noise (non-reparameterized).
    """

    has_rsample = False

    def __init__(self, *, batch_shape: torch.Size = torch.Size(), validate_args: Optional[bool] = None) -> None:
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size(), validate_args=validate_args)

    # --- torch Distribution required API ---
    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return {}

    @property
    def support(self) -> Optional[constraints.Constraint]:
        return constraints.real

    @property
    def mean(self) -> Tensor:
        raise NotImplementedError

    @property
    def variance(self) -> Tensor:
        raise NotImplementedError

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        device, dtype = self._param_device_dtype()
        u = torch.rand(self._extended_shape(sample_shape), device=device, dtype=dtype)
        return self.icdf(u)

    def expand(self, batch_shape: torch.Size, _instance=None):
        new = self._get_checked_instance(Distribution1D, _instance)
        for k, v in self.__dict__.items():
            setattr(new, k, v)
        new._batch_shape = torch.Size(batch_shape)
        return new

    # --- convenience utilities ---
    def _param_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        for v in self.__dict__.values():
            if isinstance(v, torch.Tensor):
                return v.device, v.dtype
        return torch.device("cpu"), torch.float32

    def hist(
        self,
        n: int = 10000,
        bins: int = 50,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        ax=None,
        show: bool = True,
        **kwargs,
    ):
        """
        Visualize the distribution by sampling and plotting a histogram.

        Args:
            n: Number of samples to draw.
            bins: Number of histogram bins.
            device: Device override for plotting tensor.
            dtype: Dtype override for plotting tensor.
            ax: Optional matplotlib axis to plot on.
            show: Whether to call plt.show().
            **kwargs: Additional arguments to plt.hist.
        """
        import matplotlib.pyplot as plt

        samples = self.sample((n,))
        if device is not None or dtype is not None:
            samples = samples.to(device or samples.device, dtype or samples.dtype)
        samples_np = samples.detach().cpu().numpy()
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(samples_np, bins=bins, density=True, alpha=0.7, **kwargs)
        ax.set_title(f"Histogram of {self.__class__.__name__} samples")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        if show:
            plt.show()
        return ax


# =========================
# Parametric families
# =========================
class Uniform1D(Distribution1D):
    def __init__(self, low: float | Tensor, high: float | Tensor, *, validate_args: Optional[bool] = None):
        low_t = torch.as_tensor(low, dtype=torch.float32)
        high_t = torch.as_tensor(high, dtype=torch.float32)
        if torch.any(high_t <= low_t):
            raise AssertionError("Uniform: high must be greater than low.")
        self.low = low_t
        self.high = high_t
        super().__init__(batch_shape=torch.broadcast_shapes(self.low.shape, self.high.shape), validate_args=validate_args)

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return {"low": constraints.real, "high": constraints.dependent}

    @property
    def support(self) -> constraints.Constraint:
        return constraints.interval(self.low, self.high)

    @property
    def mean(self) -> Tensor:
        return (self.low + self.high) / 2

    @property
    def variance(self) -> Tensor:
        return ((self.high - self.low) ** 2) / 12

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low.to(x.device, x.dtype)
        high = self.high.to(x.device, x.dtype)
        return torch.clamp((x - low) / (high - low), 0.0, 1.0)

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        low = self.low.to(u.device, u.dtype)
        high = self.high.to(u.device, u.dtype)
        return low + torch.clamp(u, 0.0, 1.0) * (high - low)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low.to(x.device, x.dtype)
        high = self.high.to(x.device, x.dtype)
        inside = (x >= low) & (x <= high)
        out = torch.full_like(x, -float("inf"))
        out[inside] = -torch.log(high - low)
        return out


class Normal1D(Distribution1D):
    def __init__(self, mean: float | Tensor, std: float | Tensor, *, validate_args: Optional[bool] = None):
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        std_t = torch.as_tensor(std, dtype=torch.float32)
        if torch.any(std_t <= 0):
            raise AssertionError("Normal: std must be positive.")
        self.loc = mean_t
        self.scale = std_t
        self._std_norm = torch.distributions.Normal(0.0, 1.0)
        super().__init__(batch_shape=torch.broadcast_shapes(self.loc.shape, self.scale.shape), validate_args=validate_args)

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return {"loc": constraints.real, "scale": constraints.positive}

    @property
    def support(self) -> constraints.Constraint:
        return constraints.real

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return self.scale**2

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        loc = self.loc.to(x.device, x.dtype)
        scale = self.scale.to(x.device, x.dtype)
        z = (x - loc) / scale
        return self._std_norm.cdf(z)

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        loc = self.loc.to(u.device, u.dtype)
        scale = self.scale.to(u.device, u.dtype)
        u = torch.clamp(u, 1e-6, 1 - 1e-6)
        return loc + scale * self._std_norm.icdf(u)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        loc = self.loc.to(x.device, x.dtype)
        scale = self.scale.to(x.device, x.dtype)
        z = (x - loc) / scale
        return -0.5 * (z**2 + 2 * torch.log(scale) + torch.log(torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype)))


class MixtureOfNormals1D(Distribution1D):
    """
    Finite Gaussian mixture with weights normalized to sum 1.
    Uses per-element bisection for icdf (no closed form).
    """
    def __init__(self, weights: Sequence[float] | Tensor, means: Sequence[float] | Tensor, stds: Sequence[float] | Tensor, *, validate_args: Optional[bool] = None):
        w = torch.as_tensor(weights, dtype=torch.float32)
        w = w / w.sum()
        self.K = int(w.numel())
        m = torch.as_tensor(means, dtype=torch.float32)
        s = torch.as_tensor(stds, dtype=torch.float32)
        assert m.numel() == self.K and s.numel() == self.K, "Mismatched mixture sizes."
        if torch.any(s <= 0):
            raise AssertionError("All stds must be positive.")
        self.weights = w.reshape(-1)
        self.means = m.reshape(-1)
        self.stds = s.reshape(-1)
        self._std_norm = torch.distributions.Normal(0.0, 1.0)
        super().__init__(batch_shape=torch.Size(), validate_args=validate_args)

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return {"weights": constraints.simplex, "means": constraints.real, "stds": constraints.positive}

    @property
    def support(self) -> constraints.Constraint:
        return constraints.real

    @property
    def mean(self) -> Tensor:
        w = self.weights
        mu = self.means
        return (w * mu).sum()

    @property
    def variance(self) -> Tensor:
        w = self.weights
        mu = self.means
        var_components = self.stds**2 + mu**2
        mean_sq = (w * mu).sum() ** 2
        return (w * var_components).sum() - mean_sq

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        means = self.means.to(x.device, x.dtype)
        stds = self.stds.to(x.device, x.dtype)
        weights = self.weights.to(x.device, x.dtype)
        xk = (x[..., None] - means[None, ...]) / stds[None, ...]
        Fk = self._std_norm.cdf(xk)
        return (Fk * weights).sum(dim=-1)

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(u, 1e-6, 1 - 1e-6)
        lo = (self.means.min() - 8.0 * self.stds.max()).item()
        hi = (self.means.max() + 8.0 * self.stds.max()).item()
        lo_t = torch.full_like(u, lo)
        hi_t = torch.full_like(u, hi)
        for _ in range(64):
            mid = 0.5 * (lo_t + hi_t)
            c = self.cdf(mid)
            left = c < u
            lo_t = torch.where(left, mid, lo_t)
            hi_t = torch.where(~left, mid, hi_t)
        return 0.5 * (lo_t + hi_t)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        means = self.means.to(x.device, x.dtype)
        stds = self.stds.to(x.device, x.dtype)
        weights = self.weights.to(x.device, x.dtype)
        xk = (x[..., None] - means[None, ...]) / stds[None, ...]
        logpk = -0.5 * (xk**2 + 2 * torch.log(stds)[None, ...] + torch.log(torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype)))
        return torch.logsumexp(torch.log(weights)[None, ...] + logpk, dim=-1)


# =========================
# Empirical distributions
# =========================
class EmpiricalECDF1D(Distribution1D):
    """
    Empirical (weighted) CDF with linear interpolation; exact inverse-CDF sampling.
    """
    def __init__(self, samples: torch.Tensor, weights: Optional[torch.Tensor] = None, *, validate_args: Optional[bool] = None):
        samples, weights = to_tensor(samples), to_tensor(weights)
        assert samples.ndim == 1, "samples must be 1D"
        s = samples.detach().cpu().to(torch.float32)
        sort_vals, sort_idx = torch.sort(s)
        if weights is None:
            w = torch.ones_like(sort_vals) / sort_vals.numel()
        else:
            assert weights.shape == samples.shape, "weights shape mismatch"
            w = weights.detach().cpu().to(torch.float32).clamp(min=0) + 1e-12
            w = w / w.sum()
            w = w[sort_idx]
        self.s = sort_vals                            # [M]
        self.w = w                                    # [M]
        self.cumw = torch.cumsum(self.w, dim=0)       # [M]
        super().__init__(batch_shape=torch.Size(), validate_args=validate_args)

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return {"s": constraints.real, "w": constraints.simplex}

    @property
    def support(self) -> constraints.Constraint:
        return constraints.interval(self.s[0], self.s[-1])

    @property
    def mean(self) -> Tensor:
        return torch.sum(self.s * self.w)

    @property
    def variance(self) -> Tensor:
        m = self.mean
        return torch.sum(self.w * (self.s - m) ** 2)

    @staticmethod
    def from_numpy(y: np.ndarray, weights: Optional[np.ndarray] = None) -> "EmpiricalECDF1D":
        s = to_tensor(y)
        w = to_tensor(weights) if weights is not None else None
        return EmpiricalECDF1D(s, w)

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        x_cpu = x.detach().cpu()
        # right insertion index; clamp to [1, M-1] for interpolation
        idx = torch.searchsorted(self.s, x_cpu, right=True).clamp(min=1, max=self.s.numel() - 1)
        x0, x1 = self.s[idx - 1], self.s[idx]
        y0, y1 = self.cumw[idx - 1], self.cumw[idx]
        t = torch.where((x1 - x0) > 0, (x_cpu - x0) / (x1 - x0), torch.zeros_like(x_cpu))
        F = y0 + t * (y1 - y0)
        # clip outside support
        F = torch.where(x_cpu < self.s[0], torch.tensor(0.0), F)
        F = torch.where(x_cpu >= self.s[-1], torch.tensor(1.0), F)
        return F.to(x.device, x.dtype)

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(u.detach().cpu(), 1e-6, 1 - 1e-6)
        idx = torch.searchsorted(self.cumw, u, right=True).clamp(min=1, max=self.cumw.numel() - 1)
        y0, y1 = self.cumw[idx - 1], self.cumw[idx]
        x0, x1 = self.s[idx - 1], self.s[idx]
        t = torch.where((y1 - y0) > 0, (u - y0) / (y1 - y0), torch.zeros_like(u))
        x = x0 + t * (x1 - x0)
        return x.to(u.device, torch.float32)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # piecewise-constant pdf from linear ECDF: density on [s[i-1], s[i]) is w[i]/(s[i]-s[i-1])
        x_cpu = x.detach().cpu()
        s = self.s
        w = self.w
        M = s.numel()
        # handle outside support
        out = torch.full_like(x_cpu, -float("inf"))
        inside = (x_cpu >= s[0]) & (x_cpu <= s[-1])
        if M >= 2:
            idx = torch.searchsorted(s, x_cpu.clamp(max=s[-1] - 1e-12), right=True).clamp(min=1, max=M - 1)
            widths = (s[1:] - s[:-1]).clamp(min=1e-12)
            dens = (w[1:] / widths)[idx - 1]
            out[inside] = torch.log(dens)
        return out.to(x.device, x.dtype)


class EmpiricalKDE1D(Distribution1D):
    """
    Gaussian KDE target; CDF is mean of Normal CDFs, icdf via bisection.
    Bandwidth defaults to Silverman's rule of thumb.
    """
    def __init__(self, samples: torch.Tensor, bandwidth: Optional[float] = None, *, validate_args: Optional[bool] = None):
        samples = to_tensor(samples)
        assert samples.ndim == 1, "samples must be 1D"
        s = samples.detach().cpu().to(torch.float32)
        self.s = s
        if bandwidth is None:
            std = torch.std(s)
            n = s.numel()
            h = 1.06 * std * (n ** (-1 / 5)) + 1e-8
        else:
            h = float(bandwidth)
        self.h = torch.tensor(h, dtype=torch.float32)
        self._std_norm = torch.distributions.Normal(0.0, 1.0)
        super().__init__(batch_shape=torch.Size(), validate_args=validate_args)

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return {"s": constraints.real, "h": constraints.positive}

    @property
    def support(self) -> constraints.Constraint:
        return constraints.real

    @property
    def mean(self) -> Tensor:
        return self.s.mean()

    @property
    def variance(self) -> Tensor:
        # mixture of Gaussians: Var = Var(data) + h^2
        return self.s.var(unbiased=False) + self.h**2

    @staticmethod
    def from_numpy(y: np.ndarray, bandwidth: Optional[float] = None) -> "EmpiricalKDE1D":
        s = torch.tensor(y, dtype=torch.float32)
        return EmpiricalKDE1D(s, bandwidth)

    def _z(self, x: torch.Tensor) -> torch.Tensor:
        # broadcast x against all centers on CPU for stability
        return (x.detach().cpu()[..., None] - self.s[None, ...]) / (self.h + 1e-8)

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        z = self._z(x)
        Fk = self._std_norm.cdf(z)           # [..., M]
        out = Fk.mean(dim=-1)                # [...]
        return out.to(x.device, x.dtype)

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(u, 1e-6, 1 - 1e-6).detach().cpu()
        lo = (self.s.min() - 8.0 * self.h).item()
        hi = (self.s.max() + 8.0 * self.h).item()
        lo_t = torch.full_like(u, lo)
        hi_t = torch.full_like(u, hi)
        for _ in range(48):
            mid = 0.5 * (lo_t + hi_t)
            Fm = self.cdf(mid)
            left = Fm < u
            lo_t = torch.where(left, mid, lo_t)
            hi_t = torch.where(~left, mid, hi_t)
        return 0.5 * (lo_t + hi_t).to(u.device, torch.float32)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z = self._z(x)
        two_pi = torch.tensor(2 * torch.pi, device=z.device, dtype=z.dtype)
        logp = -0.5 * (z**2 + torch.log(two_pi) + 2 * torch.log(self.h))
        # log mean exp over centers
        val = torch.logsumexp(logp, dim=-1) - torch.log(torch.tensor(float(self.s.numel()), device=z.device, dtype=z.dtype))
        return val.to(x.device, x.dtype)


# =========================
# Generic losses to any 1D target
# =========================
def w2_quantile_loss_to_target(x: torch.Tensor, target: Distribution1D) -> torch.Tensor:
    """
    1D Wasserstein-2 (squared) via quantile matching.
    Match sorted x to target quantiles q_i = F^{-1}((i-0.5)/N).
    """
    x_sorted, _ = torch.sort(x.view(-1))
    n = x_sorted.numel()
    u = torch.linspace(0.5 / n, 1 - 0.5 / n, n, device=x.device, dtype=x.dtype)
    q = target.icdf(u).to(x.device, x.dtype)
    return F.mse_loss(x_sorted, q)


def cvm_loss_to_target(x: torch.Tensor, target: Distribution1D) -> torch.Tensor:
    """
    Cramér–von Mises distance to target CDF:
        ω^2 = 1/(12n) + mean_i [ F(x_(i)) - (2i-1)/(2n) ]^2
    """
    x_sorted, _ = torch.sort(x.view(-1))
    n = x_sorted.numel()
    u_emp = torch.linspace(0.5 / n, 1 - 0.5 / n, n, device=x.device, dtype=x.dtype)
    F_t = target.cdf(x_sorted).to(x.device, x.dtype)
    return (1.0 / (12.0 * n)) + torch.mean((F_t - u_emp) ** 2)


# =========================
# Worked example: build a target from data (natural marginal)
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")

    # Pretend these are "natural" attribute values (e.g., f(w) sampled from a dataset)
    y_natural = torch.cat([
        torch.randn(800) * 10 + 50,   # majority mode ~ N(50, 10^2)
        torch.randn(200) * 5  + 80,   # secondary mode ~ N(80,  5^2)
    ])

    # 1) Empirical ECDF target (non-parametric)
    dist_ecdf = EmpiricalECDF1D(y_natural)

    # 2) Smooth KDE target (Gaussian kernels)
    dist_kde  = EmpiricalKDE1D(y_natural)  # bandwidth via Silverman

    # 3) Parametric alternatives
    dist_uniform = Uniform1D(0.0, 100.0)
    dist_gauss   = Normal1D(mean=float(y_natural.mean()), std=float(y_natural.std()))

    # Suppose these are model outputs we aim to reshape to the target marginal
    x_model = torch.randn(512, device=device) * 20 + 60

    # Compute losses against each target
    losses = {
        "W2->ECDF":               float(w2_quantile_loss_to_target(x_model, dist_ecdf)),
        "CVM->ECDF":              float(cvm_loss_to_target(x_model, dist_ecdf)),
        "W2->KDE":                float(w2_quantile_loss_to_target(x_model, dist_kde)),
        "CVM->KDE":               float(cvm_loss_to_target(x_model, dist_kde)),
        "W2->Uniform[0,100]":     float(w2_quantile_loss_to_target(x_model, dist_uniform)),
        "CVM->Uniform[0,100]":    float(cvm_loss_to_target(x_model, dist_uniform)),
        "W2->Gaussian(mean,std)": float(w2_quantile_loss_to_target(x_model, dist_gauss)),
        "CVM->Gaussian(mean,std)":float(cvm_loss_to_target(x_model, dist_gauss)),
    }

    print("Losses (lower is better):")
    for k, v in losses.items():
        print(f"  {k:>24s}: {v:.6f}")

    # Show a few target quantiles (ECDF vs KDE)
    u_test = torch.tensor([0.1, 0.5, 0.9])
    q_ecdf = dist_ecdf.icdf(u_test)
    q_kde  = dist_kde.icdf(u_test)
    print("\nTarget quantiles (u=0.1, 0.5, 0.9):")
    print("  ECDF:", [round(float(q), 3) for q in q_ecdf])
    print("  KDE :", [round(float(q), 3) for q in q_kde])

    # Example samples from targets
    s_ecdf = dist_ecdf.sample((5,))
    s_kde  = dist_kde.sample((5,))
    print("\nExample samples from targets:")
    print("  ECDF:", [round(float(v), 2) for v in s_ecdf])
    print("  KDE :", [round(float(v), 2) for v in s_kde])