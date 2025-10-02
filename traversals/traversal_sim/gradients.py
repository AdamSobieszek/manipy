
import torch

@torch.no_grad()
def _ensure_1d(y: torch.Tensor) -> torch.Tensor:
    return y.view(-1) if y.ndim > 1 else y

def batch_true_gradients(f_module, X: torch.Tensor) -> torch.Tensor:
    X = X.detach().requires_grad_(True)
    y = f_module(X)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.view(-1)
    assert y.ndim == 1 and y.shape[0] == X.shape[0], "f(x) must return [n] or [n,1]"
    ones = torch.ones_like(y)
    grads = torch.autograd.grad((y * ones).sum(), X, create_graph=False, retain_graph=False)[0]
    return grads
