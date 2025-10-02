
"""
Example: run a simulation and visualize results with plotly.
Generates a heatmap over (lambda, epsilon) and 1D curves.
"""
import torch
import pandas as pd
from traversal_sim import run_simulation, plot_traversal_heatmap, plot_prediction_curve, plot_traversal_curve

# ---- Toy data generators (same as previous example) ----
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

# ---- Config ----
device = "cpu"
dtype = torch.float64
d = 8
X_dist = GaussianX(d=d, device=device, dtype=dtype)
w_true = torch.linspace(1.0, 1.5, d, dtype=dtype, device=device)
f_true = TrueLinear(w=w_true, b=0.0)

lambdas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
epsilons = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
n_train, n_test, n_sims = 150, 1500, 100
noise_std = 0.3

# ---- Run simulation ----
out = run_simulation(X_dist, f_true, lambdas, epsilons, n_train, n_test, n_sims, noise_std, device, dtype, seed=42)

pred_df = pd.DataFrame(out["pred_summary"])
trav_df = pd.DataFrame(out["traversal_summary"])

# ---- Build figures ----
fig_heat = plot_traversal_heatmap(trav_df, value_col="mse_mean", title="Traversal MSE Heatmap")
fig_pred = plot_prediction_curve(pred_df, metric="mse", title="Prediction MSE vs λ")
# pick a lambda to slice
lambda_slice = lambdas[3]
fig_trav_curve = plot_traversal_curve(trav_df, lambda_value=lambda_slice, value_col="mse_mean",
                                      title=f"Traversal MSE vs ε at λ={lambda_slice}")

# ---- Save as HTML ----
fig_heat.write_html("heatmap_traversal.html")
fig_pred.write_html("prediction_curve.html")
fig_trav_curve.write_html("traversal_curve.html")
print("Wrote: heatmap_traversal.html, prediction_curve.html, traversal_curve.html")
