
"""
Example analysis script that imports traversal_sim package and runs a simulation.

Assumptions:
- You already have an X_dist object with .sample(n) -> torch.Tensor[n,d].
- You already have a true function f_true: torch.nn.Module mapping R^d -> R.
If not, this script includes simple toy implementations you can use.
"""

import torch
import pandas as pd
from traversal_sim import run_simulation

# ---------- Toy defaults (you can replace these with your own) ----------
class GaussianX:
    def __init__(self, d, mean=None, cov=None, device="cpu", dtype=torch.float64):
        self.d = d
        self.mean = torch.zeros(d, dtype=dtype, device=device) if mean is None else mean
        self.cov = torch.eye(d, dtype=dtype, device=device) if cov is None else cov
        self.device = device
        self.dtype = dtype
        self._L = torch.linalg.cholesky(self.cov)

    def sample(self, n):
        z = torch.randn(n, self.d, dtype=self.dtype, device=self.device)
        return z @ self._L.T + self.mean

class TrueLinear(torch.nn.Module):
    def __init__(self, w, b=0.0):
        super().__init__()
        self.w = torch.nn.Parameter(w.clone().detach(), requires_grad=False)
        self.b = torch.nn.Parameter(torch.tensor(b, dtype=w.dtype, device=w.device), requires_grad=False)
    def forward(self, X):
        return X @ self.w + self.b

class TrueQuadratic(torch.nn.Module):
    """f(x) = 0.5 x^T Q x + a^T x + c, with symmetric Q."""
    def __init__(self, Q, a=None, c=0.0):
        super().__init__()
        Q = 0.5 * (Q + Q.T)  # symmetrize
        self.Q = torch.nn.Parameter(Q.clone().detach(), requires_grad=False)
        d = Q.shape[0]
        a = torch.zeros(d, dtype=Q.dtype, device=Q.device) if a is None else a
        self.a = torch.nn.Parameter(a.clone().detach(), requires_grad=False)
        self.c = torch.nn.Parameter(torch.tensor(c, dtype=Q.dtype, device=Q.device), requires_grad=False)
    def forward(self, X):
        XQ = X @ self.Q
        quad = 0.5 * (XQ * X).sum(dim=1)
        lin = X @ self.a
        return quad + lin + self.c

# ------------------------ Configure experiment -------------------------
device = "cpu"
dtype = torch.float64
d = 10

# X distribution and true function
X_dist = GaussianX(d=d, device=device, dtype=dtype)
w_true = torch.linspace(1.0, 2.0, d, dtype=dtype, device=device)
f_true = TrueLinear(w=w_true, b=0.0)

# Grid of lambdas (model ridge penalties) and epsilons (traversal penalties)
lambdas = [0.0, 0.1, 0.3, 1.0, 3.0]
epsilons = [0.0, 1e-3, 1e-2, 1e-1, 1.0]

# Simulation parameters
n_train = 200
n_test = 2000
n_sims = 50
noise_std = 0.5
seed = 123

# ----------------------------- Run ------------------------------------
out = run_simulation(X_dist=X_dist,
                     f_true=f_true,
                     lambdas=lambdas,
                     epsilons=epsilons,
                     n_train=n_train,
                     n_test=n_test,
                     n_sims=n_sims,
                     noise_std=noise_std,
                     device=device,
                     dtype=dtype,
                     seed=seed,
                     fit_intercept=True)

pred_df = pd.DataFrame(out["pred_summary"])
trav_df = pd.DataFrame(out["traversal_summary"])

print("\n=== Prediction bias/variance summary (by lambda) ===")
print(pred_df.to_string(index=False))

print("\n=== Traversal MSE summary (by lambda, epsilon) ===")
print(trav_df.pivot(index="lambda", columns="epsilon", values="mse_mean").round(6).to_string())

# Optionally, save CSVs
pred_df.to_csv("prediction_summary.csv", index=False)
trav_df.to_csv("traversal_summary.csv", index=False)
print("\nWrote prediction_summary.csv and traversal_summary.csv")
