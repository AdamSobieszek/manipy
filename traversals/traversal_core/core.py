
from __future__ import annotations
from typing import Callable, Tuple, Dict, Any
import torch
from torch import Tensor

def traversal_vector(grad: Tensor, lam: float | Tensor, eps: float = 0.0) -> Tensor:
    lam_t = torch.as_tensor(lam, device=grad.device, dtype=grad.dtype)
    denom = (grad * grad).sum(dim=-1, keepdim=True) + lam_t
    if eps:
        denom = denom.clamp_min(eps)
    return grad / denom

def euler_traverse(
    x0: Tensor,
    grad_hat_fn: Callable[[Tensor], Tensor],
    lam: float | Tensor,
    dt: float,
    n_steps: int,
    *, record_path: bool = True,
) -> Tuple[Tensor, Dict[str, Any]]:
    x = x0.clone()
    waypoints = [x] if record_path else None
    speeds = []

    for _ in range(n_steps):
        g = grad_hat_fn(x)
        X = traversal_vector(g, lam)
        spd = (g * X).sum(dim=-1)
        speeds.append(spd)
        x = x + dt * X
        if record_path:
            waypoints.append(x)

    path = torch.stack(waypoints, dim=0) if record_path else x
    info: Dict[str, Any] = {"speeds": torch.stack(speeds, dim=0)}
    return path, info

def speed_error_against_true(x: Tensor, grad_true_fn: Callable[[Tensor], Tensor], X_vec: Tensor) -> Tensor:
    g_true = grad_true_fn(x)
    return (g_true * X_vec).sum(dim=-1) - 1.0
