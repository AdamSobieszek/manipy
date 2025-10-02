
from typing import List, Dict, Any, Optional
import torch
from .ridge import fit_ridge, predict_linear, linear_grad
from .gradients import batch_true_gradients
from .metrics import prediction_bias_variance, traversal_mse_for_epsilons

def _set_seed(seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)

def run_simulation(
    X_dist,                      # object with .sample(n) -> [n,d] torch tensor
    f_true,                      # torch.nn.Module: R^d -> R
    lambdas: List[float],        # ridge penalties for prediction model
    epsilons: List[float],       # traversal penalties
    n_train: int,
    n_test: int,
    n_sims: int = 100,
    noise_std: float = 0.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
    fit_intercept: bool = True,
) -> Dict[str, Any]:
    """
    Monte Carlo simulation that, for each lambda (model ridge penalty),
    - fits a ridge model across n_sims resamples,
    - computes prediction bias/variance/MSE on a fixed test set,
    - computes traversal MSEs for each epsilon using the model from each sim,
      then averages those traversal MSEs across sims.

    Returns a dict with:
      'pred_summary': list of dicts with keys {'lambda','bias','variance','mse'}
      'traversal_summary': list of dicts with keys {'lambda','epsilon','mse_mean','mse_std'}
    """
    _set_seed(seed)

    # Fixed test set (noise-free targets)
    X_test = X_dist.sample(n_test).to(device=device, dtype=dtype)
    with torch.no_grad():
        y_test_true = f_true(X_test)
        if y_test_true.ndim == 2 and y_test_true.shape[1] == 1:
            y_test_true = y_test_true.view(-1)

    # True gradients on test set
    grad_true = batch_true_gradients(f_true, X_test)  # [n_test, d]

    pred_summary = []
    trav_rows = []

    for lam in lambdas:
        # Collect predictions across sims for this lambda
        preds_mat = torch.empty((n_sims, n_test), dtype=dtype, device=device)
        # Collect traversal MSEs per sim for this lambda/each epsilon
        trav_mse_per_eps = {eps: torch.empty((n_sims,), dtype=dtype, device=device) for eps in epsilons}

        for s in range(n_sims):
            # Training sample
            X_tr = X_dist.sample(n_train).to(device=device, dtype=dtype)
            with torch.no_grad():
                y_tr_true = f_true(X_tr).view(-1)
                if noise_std > 0:
                    y_tr = y_tr_true + noise_std * torch.randn_like(y_tr_true)
                else:
                    y_tr = y_tr_true

            # Fit ridge model (outer loop model)
            model = fit_ridge(X_tr, y_tr, lam=lam, fit_intercept=fit_intercept)

            # Predictions on fixed test set
            with torch.no_grad():
                y_pred = predict_linear(model, X_test).view(-1)
            preds_mat[s] = y_pred

            # Traversal metrics for various epsilons using THIS model
            # Estimated gradient of model is constant = w
            g_hat = model["w"].detach()
            for eps in epsilons:
                mse = traversal_mse_for_epsilons(grad_true=grad_true,
                                                 grad_hat=g_hat,
                                                 epsilons=[eps])[eps]
                trav_mse_per_eps[eps][s] = torch.tensor(mse, dtype=dtype, device=device)

        # Prediction bias/variance summary for this lambda
        bias, var, mse = prediction_bias_variance(y_true=y_test_true, preds=preds_mat)
        pred_summary.append({"lambda": float(lam), "bias": bias, "variance": var, "mse": mse})

        # Traversal summary rows for this lambda
        for eps in epsilons:
            vals = trav_mse_per_eps[eps]
            trav_rows.append({
                "lambda": float(lam),
                "epsilon": float(eps),
                "mse_mean": float(vals.mean().item()),
                "mse_std":  float(vals.std(unbiased=False).item())
            })

    return {"pred_summary": pred_summary, "traversal_summary": trav_rows}