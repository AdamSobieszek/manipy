
import torch
from typing import Tuple, Dict, List

def prediction_bias_variance(y_true: torch.Tensor, preds: torch.Tensor) -> Tuple[float, float, float]:
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.view(-1)
    assert preds.ndim == 2 and preds.shape[1] == y_true.shape[0], "preds must be [S, n]"
    mean_pred = preds.mean(dim=0)
    bias_per_point = (mean_pred - y_true)**2
    var_per_point = preds.var(dim=0, unbiased=False)
    bias = bias_per_point.mean().item()
    var = var_per_point.mean().item()
    mse = ((preds - y_true.unsqueeze(0))**2).mean().item()
    return bias, var, mse

def traversal_mse_for_epsilons(grad_true: torch.Tensor,
                               grad_hat: torch.Tensor,
                               epsilons: List[float]) -> Dict[float, float]:
    if grad_hat.ndim == 1:
        v = grad_hat
    elif grad_hat.ndim == 2 and grad_hat.shape[0] == 1:
        v = grad_hat.view(-1)
    else:
        results = {}
        for eps in epsilons:
            denom = (grad_hat.norm(dim=1)**2 + eps)
            X_hat = grad_hat / denom.unsqueeze(1)
            dot = (X_hat * grad_true).sum(dim=1)
            mse = ((dot - 1.0)**2).mean().item()
            results[eps] = mse
        return results

    v_norm2 = (v @ v).item()
    results = {}
    for eps in epsilons:
        X_hat = v / (v_norm2 + eps)
        dot = (grad_true @ X_hat)
        mse = ((dot - 1.0)**2).mean().item()
        results[eps] = mse
    return results
