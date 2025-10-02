
import torch

def fit_ridge(X: torch.Tensor, y: torch.Tensor, lam: float, fit_intercept: bool = True):
    dtype = X.dtype
    device = X.device
    n, d = X.shape
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.view(-1)
    assert y.ndim == 1 and y.shape[0] == n, "y must be shape [n]"
    if fit_intercept:
        X_mean = X.mean(dim=0, keepdim=True)
        y_mean = y.mean()
        Xc = X - X_mean
        yc = y - y_mean
    else:
        X_mean = torch.zeros((1, d), dtype=dtype, device=device)
        y_mean = torch.tensor(0., dtype=dtype, device=device)
        Xc, yc = X, y

    XtX = Xc.T @ Xc
    A = XtX + lam * torch.eye(d, dtype=dtype, device=device)
    Xty = Xc.T @ yc
    w = torch.linalg.solve(A, Xty)
    b = y_mean - (X_mean @ w.view(-1, 1)).view(())
    return {"w": w, "b": b, "lam": lam}

def predict_linear(model: dict, X: torch.Tensor) -> torch.Tensor:
    return X @ model["w"] + model["b"]

def linear_grad(model: dict, X: torch.Tensor) -> torch.Tensor:
    n, d = X.shape
    w = model["w"].view(1, -1).expand(n, -1)
    return w
