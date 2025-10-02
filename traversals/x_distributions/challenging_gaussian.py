
from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal, constraints

def _random_orthogonal(d: int, device=None, dtype=None, generator: Optional[torch.Generator] = None) -> Tensor:
    A = torch.randn((d, d), device=device, dtype=dtype, generator=generator)
    Q, R = torch.linalg.qr(A)
    signs = torch.sign(torch.diagonal(R))
    Q = Q * signs
    return Q

def _eigs_from_colinearity(d: int, colinearity: float, total_variance: float,
                           min_cond: float = 1.0, max_cond: float = 1e4) -> Tensor:
    col = float(torch.clamp(torch.tensor(colinearity), 0.0, 1.0))
    cond = min_cond * (max_cond / min_cond) ** col
    if d == 1:
        return torch.tensor([total_variance], dtype=torch.get_default_dtype())
    r = cond ** (-1.0 / (d - 1))
    lambdas = torch.tensor([r ** i for i in range(d)], dtype=torch.get_default_dtype())
    lambdas = lambdas / lambdas.sum() * (d * total_variance)
    return lambdas

class ChallengingGaussian(Distribution):
    arg_constraints = {"coskewness": constraints.unit_interval, "colinearity": constraints.unit_interval}
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, dim: int, mean: Optional[Tensor] = None, colinearity: float = 0.0,
                 coskewness: float = 0.0, total_variance: float = 1.0,
                 seed: Optional[int] = None, device=None, dtype=None) -> None:
        super().__init__(validate_args=False)
        self.dim = int(dim)
        self.device = device
        self.dtype = dtype or torch.get_default_dtype()

        self._gen = torch.Generator(device=device)
        if seed is not None:
            self._gen.manual_seed(int(seed))

        if mean is None:
            self._mean = torch.zeros(self.dim, device=device, dtype=self.dtype)
        else:
            m = torch.as_tensor(mean, device=device, dtype=self.dtype)
            if m.numel() != self.dim:
                raise ValueError(f"mean must have shape ({self.dim},)")
            self._mean = m

        self.colinearity = float(max(0.0, min(1.0, colinearity)))
        lambdas = _eigs_from_colinearity(self.dim, self.colinearity, total_variance)
        Q = _random_orthogonal(self.dim, device=device, dtype=self.dtype, generator=self._gen)
        Sigma = Q @ torch.diag(lambdas.to(self.dtype).to(device)) @ Q.T

        self._cov = Sigma
        try:
            self._chol = torch.linalg.cholesky(Sigma)
        except RuntimeError:
            jitter = (Sigma.diagonal().mean() * 1e-8).clamp_min(1e-12)
            self._cov = Sigma + torch.eye(self.dim, device=device, dtype=self.dtype) * jitter
            self._chol = torch.linalg.cholesky(self._cov)

        self._mvn = MultivariateNormal(self._mean, covariance_matrix=self._cov)

        self.coskewness = float(max(0.0, min(1.0, coskewness)))
        w = torch.randn(self.dim, device=device, dtype=self.dtype, generator=self._gen)
        self._w = w / w.norm(p=2).clamp_min(1e-12)
        self._gamma_max = 0.75
        self._gamma = self.coskewness * self._gamma_max

        self._batch_shape = torch.Size([])
        self._event_shape = torch.Size([self.dim])

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def covariance_matrix(self) -> Tensor:
        return self._cov

    @property
    def precision_matrix(self) -> Tensor:
        return torch.cholesky_inverse(self._chol)

    @property
    def event_shape(self) -> torch.Size:
        return self._event_shape

    @property
    def batch_shape(self) -> torch.Size:
        return self._batch_shape

    def _rsample_base(self, sample_shape: torch.Size) -> Tensor:
        shape = sample_shape + self.event_shape
        z = torch.randn(shape, device=self._chol.device, dtype=self._chol.dtype, generator=self._gen)
        if self._gamma > 0.0:
            u = (z * self._w).sum(dim=-1, keepdim=True)
            delta = self._gamma * (u**3 - 3.0 * u)
            z = z + delta * self._w
        x = self._mean + torch.matmul(z, self._chol.T)
        return x

    def rsample(self, sample_shape: torch.Size | int = torch.Size()) -> Tensor:
        if isinstance(sample_shape, int):
            sample_shape = torch.Size([sample_shape])
        return self._rsample_base(sample_shape)

    def sample(self, sample_shape: torch.Size | int = torch.Size()) -> Tensor:
        if isinstance(sample_shape, int):
            sample_shape = torch.Size([sample_shape])
        return self._rsample_base(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        if self._gamma == 0.0:
            return self._mvn.log_prob(value)
        raise NotImplementedError("log_prob only available when coskewness == 0.")

    def sample_dataset(self, n: int) -> Tensor:
        return self.sample((n,))

    def to(self, device=None, dtype=None) -> "ChallengingGaussian":
        device = device if device is not None else self._cov.device
        dtype = dtype if dtype is not None else self._cov.dtype
        new = ChallengingGaussian(
            dim=self.dim,
            mean=self._mean.to(device=device, dtype=dtype),
            colinearity=self.colinearity,
            coskewness=self.coskewness,
            total_variance=(self._cov.trace() / self.dim).item(),
            seed=None,
            device=device,
            dtype=dtype,
        )
        new._cov = self._cov.to(device=device, dtype=dtype)
        new._chol = self._chol.to(device=device, dtype=dtype)
        new._mvn = MultivariateNormal(new._mean, covariance_matrix=new._cov)
        new._w = self._w.to(device=device, dtype=dtype)
        new._gamma = self._gamma
        return new
