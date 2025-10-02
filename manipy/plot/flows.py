# paste this (uses the same _to_data_uri helper as before)
import io, base64, json, uuid
from PIL import Image
from IPython.display import HTML, display
import torch
import numpy as np
import matplotlib.pyplot as plt
from manipy.core import with_manipy
from sklearn.linear_model import LinearRegression

def _to_data_uri(arr, fmt="png", optimize=True):
    im = Image.fromarray(arr)
    buf = io.BytesIO(); im.save(buf, format=fmt, optimize=optimize)
    return "data:image/{};base64,".format(fmt.lower()) + base64.b64encode(buf.getvalue()).decode("ascii")

def diplay_epoch_images(images, title="Epoch viewer", width=None, height=None, fps=20, loop=False):
    datauris = [_to_data_uri(arr) for arr in images]
    uid = "fs_" + uuid.uuid4().hex
    style_wh = (f"width:{int(width)}px;" if width else "") + (f"height:{int(height)}px;" if height else "")
    loop_js = "true" if loop else "false"

    html = f"""
    <div id="{uid}" style="font-family:system-ui,Segoe UI,Arial; border:1px solid #ddd; padding:10px; border-radius:10px;">
      <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
        <span style="font-weight:600">{title}</span>
        <button class="fs-play">Play</button>
        <button class="fs-pause" disabled>Pause</button>
        <label style="margin-left:8px;">FPS <input class="fs-fps" type="number" min="1" max="60" value="{int(fps)}" style="width:4em;"></label>
        <span style="margin-left:auto;">Frame: <span class="fs-idx">1</span> / {len(datauris)}</span>
      </div>
      <img class="fs-img" style="max-width:100%; {style_wh} display:block; margin:auto;" />
      <input class="fs-slider" type="range" min="1" max="{len(datauris)}" value="1" step="1" style="width:100%; margin-top:10px;">
    </div>
    <script>
    (function() {{
        const root = document.getElementById("{uid}");
        const imgs = {json.dumps(datauris)};
        const imgEl = root.querySelector('.fs-img');
        const slider = root.querySelector('.fs-slider');
        const idxLabel = root.querySelector('.fs-idx');
        const btnPlay = root.querySelector('.fs-play');
        const btnPause = root.querySelector('.fs-pause');
        const fpsInput = root.querySelector('.fs-fps');
        const LOOP = {loop_js};
        const LAST = imgs.length;
        let timer = null;

        // Preload all images
        imgs.forEach(src => {{ const i = new Image(); i.src = src; }});

        function setIndex(i) {{
            if (!imgs.length) return;
            i = Math.min(Math.max(1, i), LAST);
            if (imgEl.src !== imgs[i-1]) imgEl.src = imgs[i-1];
            if (slider.value != i) slider.value = i;
            idxLabel.textContent = i;
        }}

        slider.addEventListener('input', () => setIndex(parseInt(slider.value)));

        // keyboard navigation
        root.addEventListener('keydown', (ev) => {{
            if (ev.key === 'ArrowRight') setIndex(parseInt(slider.value)+1);
            if (ev.key === 'ArrowLeft')  setIndex(parseInt(slider.value)-1);
        }});
        root.tabIndex = 0;

        btnPlay.addEventListener('click', () => {{
            if (timer) return;
            // If at last frame, start from beginning
            if (parseInt(slider.value) >= LAST) {{
                setIndex(1);
            }}
            // If already running, don't start again
            if (timer) return;
            // If not looping and at last frame, don't start (handled above)
            if (!LOOP && parseInt(slider.value) >= LAST) return;

            const step = () => {{
                let i = parseInt(slider.value) + 1;
                if (i > LAST) {{
                    if (LOOP) {{
                        i = 1;
                    }} else {{
                        clearInterval(timer); timer = null;
                        btnPlay.disabled = false; btnPause.disabled = true;
                        return;
                    }}
                }}
                setIndex(i);
            }};
            const interval = 1000 / Math.max(1, parseInt(fpsInput.value || 20));
            timer = setInterval(step, interval);
            btnPlay.disabled = true; btnPause.disabled = false;
        }});

        btnPause.addEventListener('click', () => {{
            if (timer) {{ clearInterval(timer); timer = null; }}
            btnPlay.disabled = false; btnPause.disabled = true;
        }});

        setIndex(1);
    }})();
    </script>
    """
    display(HTML(html))







# Helpers to compute gradient field on a grid
@torch.no_grad()
def get_data_bounds(X):
    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)
    pad_x = 0.1 * (xmax - xmin)
    pad_y = 0.1 * (ymax - ymin)
    return (xmin - pad_x, xmax), (ymin - pad_y, ymax)


@with_manipy(enable_grad=True)
def compute_grad_field(model, grid_points, device=None):

    if isinstance(model, LinearRegression):
        grad = torch.zeros_like(grid_points)
        coefs = model.coef_.reshape(-1)
        grad[:, 0] = float(coefs[0])
        grad[:, 1] = float(coefs[1])
        return grad.detach()
    else:
        model.eval()
        # need gradient, so no torch.no_grad here
        pts = grid_points.clone().detach().requires_grad_(True)
        out = model(pts).squeeze(-1)
        grad = torch.autograd.grad(outputs=out.sum(), inputs=pts, create_graph=False)[0]
        return grad.detach()


@with_manipy(no_grad=True)
def build_distance_mask(X, G, resolution, scale=1.0, device=None):
    """
    Build distance-based mask: hide regions far from any dataset example.

    Args:
        X (np.ndarray): Data points, shape (N, d)
        G (torch.Tensor): Grid points, shape (resolution*resolution, d)
        resolution (int): Number of grid points per axis

    Returns:
        mask (np.ndarray): Boolean mask of shape (resolution, resolution)
    """
    Xt = torch.from_numpy(X).to(device)
    with torch.no_grad():
        N = Xt.shape[0]
        if N > 4000:
            idx_sub = torch.randperm(N)[:4000]
            Xsub = Xt[idx_sub]
        else:
            Xsub = Xt
        D_xx = torch.cdist(Xsub, Xsub)
        D_xx = D_xx + torch.eye(Xsub.shape[0], device=device) * 1e6  # ignore self-distance
        nn_dist = D_xx.min(dim=1).values
        radius = scale *3.0* nn_dist.median().item()  # data-driven support radius
        d_grid = torch.cdist(G, Xt).min(dim=1).values.reshape(resolution, resolution).cpu().numpy()
        mask = (d_grid.T > radius)  # transpose to match U.T/V.T orientation below
    return mask

@with_manipy(no_grad=True)
def plot_model_gradient_flow(models, X, y, resolution=100, mask=True, inv_mask=None, scale=2.5, title="Model", device=None):
    """
    Plot gradient fields for a list of models (or a single model).
    If len(models) == 1, plot a single panel; otherwise, plot a grid.
    """
    # Build grid
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    (xmin, xmax), (ymin, ymax) = get_data_bounds(X)
    xs = torch.linspace(xmin, xmax, resolution)
    ys = torch.linspace(ymin, ymax, resolution)
    GX, GY = torch.meshgrid(xs, ys, indexing="ij")
    G = torch.stack([GX.reshape(-1), GY.reshape(-1)], dim=1).to(device)

    # Compute gradient fields
    if isinstance(models, (list, tuple)):
        assert len(models) > 0, "models must be a non-empty list or tuple"
        grads_per_model = [compute_grad_field(m, G) for m in models]
        n_models = len(models)
    else:
        grads_per_model = [compute_grad_field(models, G)]
        n_models = 1
    if mask is True:
        mask = build_distance_mask(X, G, resolution, scale=scale)
        # value = torch.stack([m(G) for m in models]).mean(dim=0)
        # mask = mask | (value < value.mean()).cpu().numpy().reshape(resolution, resolution)
    elif mask is False:
        mask = None

    # Background: scatter colored by y
    sc_x = X[:, 0]
    sc_y = X[:, 1]
    sc_c = y[:, 0]

    # Prepare axes
    if n_models == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6.5), constrained_layout=True)
        axes = [ax]
    else:
        ncols = 2
        nrows = (n_models + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), constrained_layout=True)
        axes = axes.ravel()

    for i in range(n_models):
        ax = axes[i]
        U = grads_per_model[i][:, 0].reshape(resolution, resolution).cpu().numpy()
        V = grads_per_model[i][:, 1].reshape(resolution, resolution).cpu().numpy()        # Streamlines (use grid vectors; transpose because meshgrid uses indexing='ij')
        if inv_mask is not None:
            U_plot = np.ma.array(U.T, mask=~inv_mask) 
            V_plot = np.ma.array(V.T, mask=~inv_mask) 
            ax.streamplot(xs.numpy(), ys.numpy(), U_plot, V_plot, color="lightgray", density=1., linewidth=0.6, arrowsize=1.4, broken_streamlines=False)
        U_plot = np.ma.array(U.T, mask=mask) if mask is not None else U.T
        V_plot = np.ma.array(V.T, mask=mask) if mask is not None else V.T
        ax.streamplot(xs.numpy(), ys.numpy(), U_plot, V_plot, color="k", density=1., linewidth=0.6, arrowsize=1.4, broken_streamlines=False)
        ax.scatter(sc_x, sc_y, c=sc_c, cmap="coolwarm", s=25, alpha=0.85)
        ax.set_title(f"{title} {i+1} gradient flow" if n_models > 1 else f"{title} gradient flow")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal", adjustable="box")
    # Hide unused axes if any
    if n_models > 1 and len(axes) > n_models:
        for j in range(n_models, len(axes)):
            axes[j].axis("off")
    plt.show()