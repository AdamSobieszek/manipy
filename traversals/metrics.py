
import torch
from typing import Tuple, Dict, List

def prediction_bias_variance(y_true: torch.Tensor, preds: torch.Tensor) -> Tuple[float, float, float]:
    """
    Bias-variance decomposition (no test noise).
    Args:
        y_true: [n] true targets (noise-free)
        preds:  [S, n] matrix of predictions from S simulation runs
    Returns:
        bias, variance, mse  (as Python floats)
    """
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
    """
    Compute traversal MSE for a set of epsilons:
    X_hat_eps = grad_hat / (||grad_hat||^2 + eps)
    loss = E[(<X_hat_eps, grad_true> - 1)^2]
    For linear ridge, grad_hat is constant across x (can pass shape [d] or [1,d]).
    Args:
        grad_true: [n, d] true gradients at test points
        grad_hat:  [d] or [n, d] estimated gradients (if [d], will broadcast)
        epsilons: list of epsilon values (traversal ridge penalty)
    Returns:
        dict mapping epsilon -> mean squared error (float)
    """
    if grad_hat.ndim == 1:
        v = grad_hat
    elif grad_hat.ndim == 2 and grad_hat.shape[0] == 1:
        v = grad_hat.view(-1)
    else:
        # If varying by x, average per-sample with its own vector v_i
        results = {}
        for eps in epsilons:
            denom = (grad_hat.norm(dim=1)**2 + eps)  # [n]
            X_hat = grad_hat / denom.unsqueeze(1)    # [n,d]
            dot = (X_hat * grad_true).sum(dim=1)     # [n]
            mse = ((dot - 1.0)**2).mean().item()
            results[eps] = mse
        return results

    # Constant gradient case
    v_norm2 = (v @ v).item()
    results = {}
    for eps in epsilons:
        X_hat = v / (v_norm2 + eps)
        dot = (grad_true @ X_hat)                   # [n]
        mse = ((dot - 1.0)**2).mean().item()
        results[eps] = mse
    return results
