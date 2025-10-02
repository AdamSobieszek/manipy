from levelset_viz import Projection2D, sample_and_fields
from levelset_viz_plotly import plotly_viewer, save_html
import torch, torch.nn as nn
import plotly.io as pio
pio.renderers.default = "browser"   # or "vscode" in VS Code

class Toy(nn.Module):
    def forward(self, x):  # R^2→R
        return x[...,0]**2 + 0.5*x[...,1]**2 - 1.0

f = Toy()
proj = Projection2D.base_dims(k=2, dims=(0,1))
U, V, out = sample_and_fields(f, proj, ((-1.8,1.8), (-1.8,1.8)), resolution=100)

fig = plotly_viewer(
    U, V, out,
    num_levels=31,
    show_n=True,
    show_v2=False,
    show_grad_s2=True,          # level-dependent
    show_log_gauss_grad=True,   # level-dependent
    log_gauss_sigma_uv=2.0,
    log_gauss_band_sigma=0.5,
    quiver_stride=6,
    normalize_arrows=True,
    normalize_to=0.8
)

out_html = "/Users/adamsobieszek/PycharmProjects/_manipy/traversals/levelset_viewer.html"
save_html(fig, out_html, auto_open=True, include_plotlyjs='cdn')
