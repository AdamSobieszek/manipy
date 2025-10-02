# Gradient Alignment Visualiser — FigureWidget Edition (PyTorch)
# ==============================================================
# Replaces sklearn's LinearRegression and Lasso with their explicit
# matrix-form solutions for OLS and Ridge regression using PyTorch.
# --------------------------------------------------------------
# ▸ Drop‑in replacement for previous version.
# --------------------------------------------------------------

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from ipywidgets import FloatSlider, HBox, IntSlider, Text, VBox, Layout
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
DATA_ROOT = Path("/Users/adamsobieszek/PycharmProjects/psychGAN/content/")  # adjust to your paths
CONFIG = {
    "means": DATA_ROOT / "omi/attribute_means.csv",
    "ratings": DATA_ROOT / "omi/attribute_ratings.csv",
    "dlatents": DATA_ROOT / "coords_wlosses.csv",
}
DEFAULT_ATTRIBUTE = "trustworthy"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def orthogonalize(v: np.ndarray, b: np.ndarray) -> np.ndarray:
    b_unit = b / (np.linalg.norm(b) + 1e-9)
    return v - np.dot(v, b_unit) * b_unit


# @lru_cache(maxsize=16)
def load_attribute_data(attr: str):
    means_df = pd.read_csv(CONFIG["means"])
    ratings_df = pd.read_csv(CONFIG["ratings"])
    dlatents_df = pd.read_csv(CONFIG["dlatents"])

    if isinstance(dlatents_df["dlatents"].iloc[0], str):
        dlatents_df["dlatents"] = dlatents_df["dlatents"].apply(eval)

    if attr not in ratings_df.attribute.unique():
        raise ValueError(f"Attribute '{attr}' not found.")

    ratings_df = ratings_df.query("stimulus <= 1004 and attribute == @attr")
    selected = means_df.loc[means_df[attr] > 0, "stimulus"]
    ratings_df = ratings_df[ratings_df.stimulus.isin(selected)]
    dlatents_df = dlatents_df[dlatents_df.stimulus.isin(selected)]

    mean_r = ratings_df.groupby("stimulus")["rating"].mean()
    # print(mean_r)
    X = np.stack(dlatents_df.set_index("stimulus").loc[mean_r.index, "dlatents"].values).astype(np.float64)
    X = X-X.mean(axis=0)
    X = X/np.linalg.norm(X)
    y = (-mean_r.values).astype(np.float64)

    cov = np.cov(X, rowvar=False)
    pca = PCA(min(100, X.shape[1])).fit(X)
    pc1 = pca.components_[0]
    true_grad = LinearRegression().fit(X, y).coef_
    return X, y, cov, pc1, true_grad, pca

# ------------------------------------------------------------------
# Widget class
# ------------------------------------------------------------------
class GradientAlignmentWidget(VBox):
    def __init__(
        self,
        attribute: str = DEFAULT_ATTRIBUTE,
        n_samples: int | None = None,
        lasso_alpha: float = 0.1,
        layout: Optional[Layout] = None,
    ):
        # Call parent without layout when None to avoid TraitError
        if layout is None:
            super().__init__()
        else:
            super().__init__(layout=layout)

        # Widgets ----------------------------------------------------
        self.attr_text = Text(value=attribute, description="Attribute:")
        self.n_slider = IntSlider(value=100, min=50, max=1000, step=10, description="N")
        self.alpha_slider = FloatSlider(value=lasso_alpha, min=-10, max=1, step=.1, description="Ridge α", readout_format=".3f")

        # FigureWidget
        self.fig = go.FigureWidget(
            make_subplots(
                rows=1,
                cols=3,
                column_widths=[0.40, 0.30, 0.30],
                subplot_titles=(
                    "Gradient Alignment vs. Covariance",
                    "Whitening & Regularisation",
                    "Cosine Similarity to Top‑10 PCs",
                ),
            )
        )
        self.fig.update_layout(height=620, width=1850, showlegend=False, margin=dict(l=30, r=30, t=50, b=40))

        # Compose layout
        self.children = [HBox([self.attr_text, self.n_slider, self.alpha_slider]), self.fig]

        # Cache
        self._data: Dict[str, Tuple] = {}

        # Observe
        for w in (self.attr_text, self.n_slider, self.alpha_slider):
            w.observe(self._on_change, names="value")

        self._draw()

    # --------------------------------------------------------
    def _on_change(self, _):
        self._draw()

    # --------------------------------------------------------
    def _compute(self):
        attr = self.attr_text.value.strip()
        if attr not in self._data:
            self._data[attr] = load_attribute_data(attr)
        X, y, _cov, pc1, true_grad, pca = self._data[attr]

        self.n_slider.max = len(X)
        n = min(self.n_slider.value, len(X))
        alpha = self.alpha_slider.value
        alpha = np.exp(alpha)

        Xn, yn = X[:n], y[:n]
        Xn_scaled = Xn

        # --- PyTorch Implementation ---
        # Convert to PyTorch tensors
        X_t = torch.from_numpy(Xn_scaled).double()
        y_t = torch.from_numpy(yn).double().unsqueeze(1) # Ensure y is a column vector
        n_features = X_t.shape[1]
        
        # Precompute X.T @ X
        XtX = X_t.T @ X_t
        
        # 1. OLS (Linear Regression) solution: w = (X.T @ X)^-1 @ X.T @ y
        try:
            w_lr_t = torch.inverse(XtX) @ X_t.T @ y_t
            lr_grad = w_lr_t.squeeze().numpy()
            lr_grad = lr_grad / (np.linalg.norm(lr_grad) + 1e-19) * 10
        except torch.linalg.LinAlgError: # Handle singular matrix case
            lr_grad = np.zeros(n_features)

        # 2. Ridge Regression solution: w = (X.T @ X + aI)^-1 @ X.T @ y
        identity = torch.eye(n_features, dtype=torch.double)
        w_ridge_t = torch.inverse(XtX + alpha * identity) @ X_t.T @ y_t
        ridge_grad = w_ridge_t.squeeze().numpy()
        ridge_grad = ridge_grad / (np.linalg.norm(ridge_grad) + 1e-19) * 10
        # --- End PyTorch Implementation ---

        ci = np.cov(Xn_scaled, rowvar=False) @ (lr_grad / (np.linalg.norm(lr_grad) + 1e-19))
        ortho = orthogonalize(lr_grad, pc1)
        return Xn, yn, pc1, ortho, lr_grad, ci, ridge_grad, true_grad, pca

    # --------------------------------------------------------
    def _project(self, data, b1, b2):
        b1u = b1 / (np.linalg.norm(b1) + 1e-9)
        b2u = b2 / (np.linalg.norm(b2) + 1e-9)
        # rotate b1u and b2u by pi/6 degrees in the b1u-b2u plane
        rotation = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)], [np.sin(np.pi/6), np.cos(np.pi/6)]])
        b1u, b2u = rotation @ np.vstack([b1u, b2u]), rotation @ np.vstack([b1u, b2u])
        trans = np.vstack([b1u, b2u])
        d = data.reshape(1, -1) if data.ndim == 1 else data
        proj = d @ trans.T
        return proj[0] if data.ndim == 1 else proj

    # --------------------------------------------------------
    def _arrow(self, vec2d, name, color):
        if np.allclose(vec2d, 0):
            return go.Scatter(x=[], y=[])
        v = vec2d / (np.linalg.norm(vec2d) + 1e-9) * 0.8
        return go.Scatter(x=[0, v[0]], y=[0, v[1]], mode="lines+markers", marker=dict(size=[0, 5], color=color), line=dict(color=color, width=3), name=name, showlegend=True)

    # --------------------------------------------------------
    def _draw(self):
        X, y, pc1, ortho, lr_grad, ci, ridge_grad, true_grad, pca = self._compute()
        with self.fig.batch_update():
            self.fig.data = []
            self.fig.layout.shapes = ()

            # Panel 1
            X2d = self._project(X, pc1, ortho)
            self.fig.add_trace(go.Scatter(x=X2d[:,0]*100, y=X2d[:,1]*100, mode="markers", marker=dict(size=5, color=y, colorscale="Viridis", showscale=False), name="Samples"), row=1, col=1)
            for name, vec, col in [("PC1", pc1, "blue"), ("LR", lr_grad, "black"), ("CI", ci, "purple"), ("Ridge", ridge_grad, "orange")]:
                self.fig.add_trace(self._arrow(self._project(vec*2, pc1, ortho), name, col), row=1, col=1)
            self.fig.update_xaxes(title_text="Proj. on PC1", range=[-1.5,1.5], row=1, col=1)
            self.fig.update_yaxes(title_text="Proj. on Orthogonal", range=[-1.5,1.5],  row=1, col=1)

            # Panel 2
            for name, vec, col in [("LR", lr_grad, "black"), ("True", true_grad, "red"), ("CI", ci, "purple"), ("Ridge", ridge_grad, "orange")]:
                self.fig.add_trace(self._arrow(self._project(vec, pc1, ortho), name, col), row=1, col=2)
            self.fig.update_xaxes(range=[-1.1,1.1], row=1, col=2)
            self.fig.update_yaxes(range=[-1.1,1.1], row=1, col=2)

            # Panel 3
            pcs = pca.components_[:10]
            labels = [f"PC{i+1}" for i in range(10)]
            vecs = [pc1, lr_grad, ci, ridge_grad]
            names = ["PC1","LR","CI","Ridge"]
            sim = np.zeros((len(vecs), 10))
            for i,v in enumerate(vecs):
                vn = v / (np.linalg.norm(v)+1e-9)
                for j,pc in enumerate(pcs):
                    sim[i,j] = float(np.dot(vn, pc/ (np.linalg.norm(pc)+1e-9)))
            self.fig.add_trace(go.Heatmap(z=sim, x=labels, y=names, zmin=-1, zmax=1, colorscale="RdBu"), row=1, col=3)
            self.fig.update_xaxes(title_text="Principal Component", row=1, col=3)
            self.fig.update_yaxes(title_text="Vector", row=1, col=3)

# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------

def launch_gradient_alignment_widget(attr: str = DEFAULT_ATTRIBUTE):
    return GradientAlignmentWidget(attribute=attr)

launch_gradient_alignment_widget("trustworthy")
# ------------------------------------------------------------------
#  Ridge-path figure (Plotly, same projection as the widget)
#  – arrowheads via annotations, labels next to vectors
# ------------------------------------------------------------------
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ---------- utilities ---------------------------------------------
def _project(data: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Rotate the (b1,b2) plane by π/10 then project – matches the widget."""
    b1u = b1 / (np.linalg.norm(b1) + 1e-9)
    b2u = b2 / (np.linalg.norm(b2) + 1e-9)
    rot  = np.array([[np.cos(np.pi/10), -np.sin(np.pi/10)],
                     [np.sin(np.pi/10),  np.cos(np.pi/10)]])
    trans = rot @ np.vstack([b1u, b2u])
    d = data.reshape(1, -1) if data.ndim == 1 else data
    out = d @ trans.T
    return out[0] if data.ndim == 1 else out


def _add_arrow(fig: go.Figure,
               vec2d: np.ndarray,
               color: str,
               label: str | None = None,
               opacity: float = 1.0):
    """
    Draw a vector from (0,0) with an arrowhead.  If `label` is given,
    place LaTeX text 5 % beyond the arrow tip.
    """
    if np.allclose(vec2d, 0):
        return
    vec2d = vec2d / (np.linalg.norm(vec2d) + 1e-9) * 0.7

    # 1) arrow itself
    fig.add_annotation(
        x=vec2d[0], y=vec2d[1],           # arrowhead position
        ax=0, ay=0,                       # tail at origin
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=3, arrowsize=1, arrowwidth=2,
        arrowcolor=color,
        opacity=opacity,
        text=""                           # no text here
    )

    # 2) optional label, nudged 5 % further out
    if label is not None:
        tip = vec2d * 1.05
        fig.add_annotation(
            x=tip[0]*1.05, y=tip[1]*1.05,
            xref="x", yref="y",
            showarrow=False,
            text=label,
            font=dict(size=36, color=color),
            xanchor="center", yanchor="bottom"
        )

import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# This function assumes 'load_attribute_data' and 'orthogonalize' 
# helpers are defined as in your original script.

def create_ridge_alignment_plot(
    attribute: str = "trustworthy",
    n_samples: int | None = 1004,
    alphas: np.ndarray | None = None,
    n_alphas_to_show: int = 20,
    width: int = 800,
    height: int = 550,
):
    """
    • scatter of stimuli
    • β̂_OLS and v_CI vectors with arrowheads & on-plot LaTeX labels
    • dashed Ridge path with arrowheads for selected α
    """
    # 1 – load data & key vectors
    X, y, Σ_v, pc1, _, _ = load_attribute_data(attribute)
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]

    β_ols = LinearRegression(fit_intercept=False).fit(X, y).coef_
    β_ols /= np.linalg.norm(β_ols) + 1e-12
    v_ci  = Σ_v @ β_ols
    v_ci /= np.linalg.norm(v_ci) + 1e-12

    if alphas is None:
        alphas = np.logspace(-8, 3.5, 40)
    XtX, I = X.T @ X, np.eye(X.shape[1])
    β_ridge = np.stack([np.linalg.solve(XtX + α*I, X.T @ y) for α in alphas])
    β_ridge /= np.linalg.norm(β_ridge, axis=1, keepdims=True)

    idx_show = np.linspace(len(alphas)*2//5,
                           len(alphas) - len(alphas)*2//5,
                           n_alphas_to_show//2, dtype=int)

    # 2 – projection
    ortho = orthogonalize(β_ols, pc1)
    X2d         = _project(X,        pc1, ortho) * 100
    β_ols_2d    = _project(β_ols,    pc1, ortho)
    v_ci_2d     = _project(v_ci,     pc1, ortho)
    ridge2d     = _project(β_ridge,  pc1, ortho)
    ridge2d     = ridge2d / (np.linalg.norm(ridge2d, axis=1, keepdims=True)+1e-9) * 0.7
    ridge2d_sub = ridge2d[idx_show]

    # 3 – figure
    fig = go.Figure()

    # stimuli scatter
    fig.add_trace(
        go.Scatter(
            x=X2d[:, 0], y=X2d[:, 1],
            mode="markers",
            marker=dict(size=5, color=100+y, colorscale="Viridis", showscale=True),
            name=r"$\Large \text{Stimuli}$",
            
        )
    )

    # β̂_OLS and v_CI arrows with labels
    _add_arrow(fig, β_ols_2d, "black",  r"$\Large \hat{\beta}_{\mathrm{OLS}}$")
    _add_arrow(fig, v_ci_2d,  "purple", r"$\Large v_{\mathrm{CI}}$")

    # dashed Ridge path (legend entry kept)
    fig.add_trace(
        go.Scatter(
            x=ridge2d[:, 0], y=ridge2d[:, 1],
            mode="lines",
            line=dict(color="orange", dash="dash", width=2),
            name=r"$\Large \text{Ridge path}$",

        )
    )

    # arrowheads on selected α (no legend)
    α_colors = np.linspace(0.0, 1.0, len(idx_show))
    for vec2d, frac in zip(ridge2d_sub, α_colors):
        _add_arrow(
            fig, vec2d, "orange",
            opacity=(min(frac, 1.25-frac)*1.5)
        )

    # --- MODIFICATIONS START HERE ---

    # axes & layout

    fig.update_xaxes(
        # MODIFIED LINE: Added \Large to the LaTeX string
        title_text=r"$\Large \text{Projection on }\mathrm{PC}_1$",
        range=[-2, 2],
        zeroline=False,
        showline=True, linecolor='black', linewidth=1,
        title_font=dict(size=24), # Keep this to ensure proper spacing
        tickfont_size=20,
    )
    fig.update_yaxes(
        # MODIFIED LINE: Added \Large to the LaTeX string
        title_text=r"$\Large \text{Projection on }\hat{\beta}_{\mathrm{OLS}}$",
        range=[-1.2, 1.3],
        zeroline=False,
        showline=True, linecolor='black', linewidth=1,
        title_font=dict(size=24), # Keep this to ensure proper spacing
        tickfont_size=20,
    )
    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor='white',  # White plot background
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(size=20, color="black", family="Serif")  # Set global font to Serif
    )
    
    # --- MODIFICATIONS END HERE ---
    
    return fig

# You can run the script as before
if __name__ == "__main__":
    # Make sure the helper functions (load_attribute_data, _project, etc.) 
    # are defined in the same scope or imported correctly.
    
    # fig = create_ridge_alignment_plot()
    # fig.show(config={"mathjax": "cdn"})
    # fig.write_image("ridge_alignment_appendix.png", scale=3)
    pass # Placeholder to prevent execution errors if run standalone without helpers


# ------------------------------------------------------------------
# run example
# ------------------------------------------------------------------
if __name__ == "__main__":

    fig = create_ridge_alignment_plot()
    # Outside Jupyter this loads MathJax so LaTeX renders:
    # fig.show(config={"mathjax": "cdn"})
    fig.write_image("ridge_alignment_appendix.png", scale=3)  # needs kaleido