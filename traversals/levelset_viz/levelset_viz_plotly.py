# levelset_viz_plotly.py (Model mode enabled)
# Full Dash app: left-side Mantine control panel + live Plotly figure on the right.
# - Preserves legend visibility across level changes (tracked in dcc.Store via restyleData)
# - Add/remove/reorder vector fields
# - Per-field hyperparameters + styling
# - Grid resolution + level slider
# - Sticky toolbar, responsive layout
# - NEW: Torch model gradient mode — load a model with custom loader code, pick 2 input dims,
#        visualize f(x) heatmap / levelset and its gradient ∇f over the chosen 2D slice.

from __future__ import annotations
from typing import Optional, Tuple, Dict, List, Any
import math
import numpy as np
import torch
import torch.nn.functional as F

import plotly.graph_objects as go
import plotly.figure_factory as ff

from dash import Dash, dcc, html, Input, Output, State, MATCH, ALL, ctx
import dash
import dash_mantine_components as dmc

Tensor = torch.Tensor

# ===============================
# === Core math (from notes) ===
# ===============================


def compute_level_scalar_fields(out: Dict[str, Tensor], level_y: float, eps: float = 1e-9) -> Dict[str, np.ndarray]:
    f = out['f']                # (H,W)
    norm = out['norm']          # (H,W,1)
    mask = out['mask']          # (H,W,1) boolean

    f_np = f.detach().cpu().numpy()
    norm_np = norm.detach().cpu().numpy()[..., 0]
    mask_np = mask.detach().cpu().numpy()[..., 0].astype(bool)

    safe_norm = np.where(mask_np, norm_np, eps)
    d_lin = (f_np - float(level_y)) / safe_norm
    s2 = 0.5 * d_lin**2

    return {'f': f_np, 'norm': norm_np, 'mask': mask_np, 'd_lin': d_lin, 's2': s2}


def compute_level_vector_fields(out: Dict[str, Tensor], level_y: float, eps: float = 1e-9) -> Dict[str, Tensor]:
    g = out['grad']      # (H,W,k)
    norm = out['norm']   # (H,W,1)
    mask = out['mask']   # (H,W,1)
    f = out['f']         # (H,W)

    safe = torch.where(mask, norm, torch.full_like(norm, eps))
    n = g / safe
    v2 = g / (safe * safe)

    d_lin = (f - float(level_y)) / safe.squeeze(-1)  # (H,W)
    grad_s2 = d_lin.unsqueeze(-1) * n                # (H,W,k)

    return {'grad': g, 'n': n, 'v2': v2, 'grad_d': n, 'grad_s2': grad_s2}


def _gaussian_kernel_1d(sigma_px: float, *, truncate: float = 3.0, device=None, dtype=None) -> Tuple[Tensor, int]:
    sigma_px = float(max(1e-6, sigma_px))
    radius = max(1, int(math.ceil(truncate * sigma_px)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma_px) ** 2)
    k = k / (k.sum() + 1e-12)
    return k, radius


def _separable_gaussian_conv2d(img: Tensor, sigma_u_px: float, sigma_v_px: float, *, truncate: float = 3.0) -> Tensor:
    assert img.dim() == 2, "img must be (H,W)"
    device, dtype = img.device, img.dtype
    H, W = int(img.shape[-2]), int(img.shape[-1])

    k_v, r_v = _gaussian_kernel_1d(float(sigma_v_px), truncate=truncate, device=device, dtype=dtype)
    k_u, r_u = _gaussian_kernel_1d(float(sigma_u_px), truncate=truncate, device=device, dtype=dtype)

    max_r_v = max(0, W - 1)
    max_r_u = max(0, H - 1)

    if r_v > max_r_v:
        r_v = max_r_v
        x = torch.arange(-r_v, r_v + 1, device=device, dtype=dtype)
        k_v = torch.exp(-0.5 * (x / float(max(1e-12, sigma_v_px))) ** 2)
        k_v = k_v / (k_v.sum() + 1e-12)

    if r_u > max_r_u:
        r_u = max_r_u
        x = torch.arange(-r_u, r_u + 1, device=device, dtype=dtype)
        k_u = torch.exp(-0.5 * (x / float(max(1e-12, sigma_u_px))) ** 2)
        k_u = k_u / (k_u.sum() + 1e-12)

    x = img.unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    if r_v > 0:
        x = F.pad(x, (r_v, r_v, 0, 0), mode='reflect')
    x = F.conv2d(x, k_v.view(1, 1, 1, -1))
    if r_u > 0:
        x = F.pad(x, (0, 0, r_u, r_u), mode='reflect')
    x = F.conv2d(x, k_u.view(1, 1, -1, 1))
    return x.squeeze(0).squeeze(0)


def compute_loggauss_potential_gradient(out: Dict[str, Tensor], U: Tensor, V: Tensor,
                                        level_y: float, *, sigma_uv: float,
                                        band_sigma: float, eps: float = 1e-9,
                                        truncate: float = 3.0) -> Tensor:
    f = out['f']              # (H,W)
    X = out['X']              # (H,W,k)
    mask = out['mask']        # (H,W,1)
    H, W = f.shape
    device, dtype = f.device, f.dtype

    du = (U[1, 0] - U[0, 0]).abs().item() if H > 1 else 1.0
    dv = (V[0, 1] - V[0, 0]).abs().item() if W > 1 else 1.0
    du = du if du > 0 else 1.0
    dv = dv if dv > 0 else 1.0
    sigma_u_px = float(sigma_uv / du)
    sigma_v_px = float(sigma_uv / dv)

    w_band = torch.exp(-0.5 * ((f - float(level_y)) / float(max(band_sigma, 1e-12))) ** 2)
    m = mask[..., 0].to(dtype)
    w0 = w_band * m

    P = _separable_gaussian_conv2d(w0, sigma_u_px, sigma_v_px, truncate=truncate)
    P = torch.clamp(P, min=eps)

    kdim = X.shape[-1]
    M_list = []
    for kd in range(kdim):
        Wx = w0 * X[..., kd]
        Mk = _separable_gaussian_conv2d(Wx, sigma_u_px, sigma_v_px, truncate=truncate)
        M_list.append(Mk)
    M = torch.stack(M_list, dim=-1)  # (H,W,k)
    mu = M / P.unsqueeze(-1)
    grad = 2.0 * (X - mu)
    grad = grad * mask
    return grad


def reconstruct_Q(U: Tensor, V: Tensor, X: Tensor) -> Tensor:
    Xu0 = X[1, 0] - X[0, 0]
    Xv0 = X[0, 1] - X[0, 0]
    du = float((U[1, 0] - U[0, 0]).item())
    dv = float((V[0, 1] - V[0, 0]).item())
    q0 = (Xu0 / (du + 1e-12)); q1 = (Xv0 / (dv + 1e-12))
    Q = torch.stack([q0, q1], dim=1)
    Q, _ = torch.linalg.qr(Q, mode='reduced')
    return Q  # (k,2)


def project_ambient_field_to_uv(vecHWk: Tensor, Q: Tensor) -> Tuple[np.ndarray, np.ndarray]:
    uv = (vecHWk @ Q).detach().cpu().numpy()  # (H,W,2)
    return uv[..., 0], uv[..., 1]


def _apply_arrow_scaling(Uq: np.ndarray, Vq: np.ndarray, mask: np.ndarray, *,
                         arrow_cap: Optional[float], arrow_cap_quantile: Optional[float],
                         normalize: bool, normalize_to: float,
                         internal_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    Uc, Vc = Uq.copy(), Vq.copy()
    mag = np.hypot(Uc, Vc)
    cap = None
    if arrow_cap_quantile is not None:
        valid = mag[mask]
        if valid.size > 0:
            qcap = float(np.quantile(valid, arrow_cap_quantile))
            cap = qcap if cap is None else min(cap, qcap)
    if arrow_cap is not None:
        cap = arrow_cap if cap is None else min(cap, arrow_cap)
    if cap is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            sf = np.minimum(1.0, cap / (mag + 1e-12))
        Uc *= sf; Vc *= sf
        mag = np.hypot(Uc, Vc)
    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            sf = np.where(mag > 0, (normalize_to / (mag + 1e-12)), 0.0)
        Uc *= sf; Vc *= sf
    Uc *= float(internal_scale); Vc *= float(internal_scale)
    return Uc, Vc


def add_quiver(fig: go.Figure,
               U: Tensor, V: Tensor,
               fieldHWk: Tensor, Q: Tensor, maskHW: np.ndarray,
               *,
               stride: int = 5,
               name: str = 'quiver',
               legendgroup: str = 'quiver',
               color: str = 'black',
               scale: float = 0.9,
               arrow_cap: Optional[float] = None,
               arrow_cap_quantile: Optional[float] = 0.98,
               normalize_arrows: bool = False,
               normalize_to: float = 1.0,
               internal_scale: float = 1.0,
               visible: Optional[bool] = True,
               opacity: float = 1.0) -> List[int]:
    U_np = U.detach().cpu().numpy(); V_np = V.detach().cpu().numpy()
    Vu, Vv = project_ambient_field_to_uv(fieldHWk, Q)
    sl = (slice(None, None, stride), slice(None, None, stride))
    Xq = V_np[sl]; Yq = U_np[sl]
    Uq_raw = np.where(maskHW[sl], Vu[sl], 0.0)
    Vq_raw = np.where(maskHW[sl], Vv[sl], 0.0)

    Uq, Vq = _apply_arrow_scaling(Uq_raw, Vq_raw, maskHW[sl],
                                  arrow_cap=arrow_cap,
                                  arrow_cap_quantile=arrow_cap_quantile,
                                  normalize=normalize_arrows, normalize_to=normalize_to,
                                  internal_scale=internal_scale)
    qfig = ff.create_quiver(Xq, Yq, Uq, Vq, name=name, line_color=color, scale=scale)
    qfig.update_traces(opacity=opacity)
    idxs: List[int] = []
    first = True
    for tr in qfig.data:
        tr.legendgroup = legendgroup
        tr.showlegend = first  # one legend entry per group
        first = False
        if visible is False:
            tr.visible = 'legendonly'
        fig.add_trace(tr)
        idxs.append(len(fig.data) - 1)
    return idxs


def make_heatmap_with_level(U: Tensor, V: Tensor, f_np: np.ndarray, level_y: float,
                            *, colorscale='Viridis') -> go.Figure:
    U_np = U.detach().cpu().numpy(); V_np = V.detach().cpu().numpy()
    x_vec = V_np[0, :]; y_vec = U_np[:, 0]

    fig = go.Figure()
    fig.add_trace(go.Contour(x=x_vec, y=y_vec, z=f_np,
                             contours=dict(coloring='heatmap'),
                             colorscale=colorscale,
                             colorbar=dict(title='f'),
                             name='f'))
    fig.add_trace(go.Contour(x=x_vec, y=y_vec, z=f_np,
                             contours=dict(coloring='lines', start=level_y, end=level_y, size=1e-9),
                             line=dict(color='black', width=2),
                             showscale=False,
                             name=f'level f={level_y:.3g}'))
    # Axis labels
    fig.update_layout(xaxis_title='v (2nd axis)', yaxis_title='u (1st axis)')
    # Explicit limits to the ends of the heatmap
    x_min, x_max = float(x_vec.min()), float(x_vec.max())
    y_min, y_max = float(y_vec.min()), float(y_vec.max())
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max], scaleanchor='x', scaleratio=1)
    # Default canvas aspect ratio 4:3

    fig.update_layout(uirevision='levelset_viz_persist')
    return fig

# =======================================
# === Demo sampler (replace if needed) ===
# =======================================


def demo_sampler(H: int, W: int) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """Simple bowl + tilt demo. Replace this with your own sampler if desired."""
    # Make the computational domain wider than tall (4:3 aspect)
    u = torch.linspace(-3.0, 3.0, H)   # height extent = 6
    v = torch.linspace(-4.0, 4.0, W)   # width  extent = 8
    U, V = torch.meshgrid(u, v, indexing='ij')  # (H,W)
    X = torch.stack([U, V], dim=-1)  # (H,W,2)
    f = 0.5 * (U**2 + V**2) + 0.2 * V
    grad = torch.stack([U, V + 0.2 * torch.ones_like(V)], dim=-1)  # (H,W,2)
    norm = torch.linalg.norm(grad, dim=-1, keepdim=True)           # (H,W,1)
    mask = torch.ones(H, W, 1, dtype=torch.bool)
    out = {'f': f, 'grad': grad, 'norm': norm, 'mask': mask, 'X': X}
    return U, V, out


# =======================================
# === Model sampler (NEW)            ====
# =======================================

# Holds the in-memory model object. We keep it out of dcc.Store since that's JSON-only.
LOADED_MODEL: Optional[torch.nn.Module] = None

DEFAULT_LOADER_CODE = (
    "def load(path):\n"
    "    # add here: create and load your torch.nn.Module\n"
    "    # Example:\n"
    "    # model = MyNet(...)\n"
    "    # state = torch.load(path, map_location='cpu')\n"
    "    # model.load_state_dict(state)\n"
    "    # model.eval()\n"
    "    return model\n"
)


def model_sampler(H: int, W: int, *, model: torch.nn.Module, dim_in: int, dims: Tuple[int, int]) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """
    Build a 2D slice over (dims[0], dims[1]) with other coordinates fixed to 0,
    run f(x) and compute its gradient w.r.t. those 2 dims.
    """
    assert dim_in >= 2, "Model input dimensionality must be >= 2"
    i, j = int(dims[0]), int(dims[1])
    i = max(0, min(i, dim_in - 1)); j = max(0, min(j, dim_in - 1))
    if i == j:
        j = min((j + 1), dim_in - 1) if j < dim_in - 1 else max(0, j - 1)

    # Domain: keep the same extents as demo for now
    u = torch.linspace(-3.0, 3.0, H)
    v = torch.linspace(-4.0, 4.0, W)
    U, V = torch.meshgrid(u, v, indexing='ij')  # (H,W)

    # Full input tensor (H,W,dim_in), zeros elsewhere
    Xfull = torch.zeros(H, W, dim_in, dtype=torch.float32)
    Xfull[..., i] = U
    Xfull[..., j] = V

    flat = Xfull.reshape(-1, dim_in).requires_grad_(True)

    model.eval()
    with torch.no_grad():
        _ = None  # just to make intent explicit (we'll enable grad only for inputs)
    # Forward in no-grad would block input grads; so use grad guard only for params.
    for p in model.parameters():
        p.requires_grad_(False)

    y = model(flat)  # shape (N,) or (N,1) or (N,C)
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    if y.dim() > 1:
        y = y[..., 0]
    y = y.reshape(-1)
    f = y.reshape(H, W)

    # Sum-reduce and take gradient w.r.t. inputs
    grad_flat = torch.autograd.grad(f.sum(), flat, create_graph=False, retain_graph=False)[0]
    g0 = grad_flat[:, i].reshape(H, W)
    g1 = grad_flat[:, j].reshape(H, W)
    grad = torch.stack([g0, g1], dim=-1)  # (H,W,2)

    X = torch.stack([U, V], dim=-1)
    norm = torch.linalg.norm(grad, dim=-1, keepdim=True)
    mask = torch.ones(H, W, 1, dtype=torch.bool)
    out = {'f': f, 'grad': grad, 'norm': norm, 'mask': mask, 'X': X}
    return U, V, out


# ======================================================
# === Figure builder driven by app state (Dash side) ===
# ======================================================


def build_figure(state: Dict[str, Any]) -> go.Figure:
    """Build a Plotly figure from the current app state dict."""
    H = int(state['grid']['H']); W = int(state['grid']['W'])
    level_y = float(state['level_y'])

    # Choose data source: model (if loaded) or demo
    use_model = bool(state.get('model', {}).get('loaded')) and (LOADED_MODEL is not None)
    if use_model:
        dims = tuple(state['model'].get('dims', [0, 1]))
        dim_in = int(state['model'].get('dim_in', 2))
        try:
            U, V, out = model_sampler(H, W, model=LOADED_MODEL, dim_in=dim_in, dims=(int(dims[0]), int(dims[1])))
        except Exception as e:
            # Fallback to demo if anything goes wrong, but keep an obvious title note
            U, V, out = demo_sampler(H, W)
            state['title'] = f"Levelset Viz (model error: {type(e).__name__})"
    else:
        U, V, out = demo_sampler(H, W)

    scalars = compute_level_scalar_fields(out, level_y)
    f_np, mask_np = scalars['f'], scalars['mask']
    fig = make_heatmap_with_level(U, V, f_np, level_y)
    Q = reconstruct_Q(U, V, out['X'])

    # Add vector fields in user-specified order
    legend_state = state.get('legend', {})  # {'n': True/False, ...}
    arrow_common = state['arrows']  # global arrow knobs
    for fld in state['active_fields']:
        key = fld['key']            # e.g., 'n', 'v2', 'grad_s2', 'logG', 'grad'
        label = fld['label']        # legend label
        color = fld['color']
        stride = int(fld.get('stride', arrow_common['stride']))
        internal_scale = float(fld.get('internal_scale', arrow_common['internal_scale']))
        opacity = float(fld.get('opacity', 1.0))
        visible = legend_state.get(key, True)

        # compute vectors
        if key in ('n', 'v2', 'grad_s2', 'grad'):
            vecs = compute_level_vector_fields(out, level_y)
            if key == 'grad_s2':
                field_tensor = vecs['grad_s2']
            elif key == 'grad':
                field_tensor = vecs['grad']
            else:
                field_tensor = vecs[key]
        elif key == 'logG':
            grad_log = compute_loggauss_potential_gradient(
                out, U, V, level_y,
                sigma_uv=float(fld['sigma_uv']),
                band_sigma=float(fld['band_sigma'])
            )
            field_tensor = grad_log
        else:
            continue  # unknown field

        add_quiver(fig, U, V, field_tensor, Q, mask_np,
                   stride=stride,
                   name=label,
                   legendgroup=key,
                   color=color,
                   scale=float(arrow_common['plotly_scale']),
                   arrow_cap=(arrow_common['arrow_cap'] if arrow_common['arrow_cap'] is not None else None),
                   arrow_cap_quantile=(arrow_common['arrow_cap_quantile'] if arrow_common['arrow_cap_quantile'] is not None else None),
                   normalize_arrows=bool(arrow_common['normalize_arrows']),
                   normalize_to=float(arrow_common['normalize_to']),
                   internal_scale=internal_scale,
                   visible=visible,
                   opacity=opacity)

    # Keep layout tidy
    fig.update_layout(
        title=state.get('title', 'Levelset Viz'),
        legend=dict(orientation='h', groupclick='toggleitem'),
        margin=dict(l=10, r=10, t=40, b=10),
        uirevision='levelset_viz_persist',
    )
    x_vec = V[0, :]; y_vec = U[:, 0]
    y_min, y_max = float(y_vec.min()), float(y_vec.max())
    x_min, x_max = float(x_vec.min()), float(x_vec.max())
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max], scaleanchor='x', scaleratio=1)

    return fig


# ===================================
# === Dash app + Mantine UI/UX    ===
# ===================================


def make_field_defaults() -> List[Dict[str, Any]]:
    # initial active fields (order matters)
    return [
        dict(key='n',    label='∂/∂d (n)',      color='#1f77b4', stride=5, internal_scale=0.7),
        dict(key='v2',   label='∂/∂f (v2)',     color='#ff7f0e', stride=5, internal_scale=0.7),
        # You can add the others via the "Available fields" section
    ]


AVAILABLE_FIELDS = {
    'n'      : dict(label='∂/∂d (n)',        needs={'sigma': False}),
    'v2'     : dict(label='∂/∂f (v2)',       needs={'sigma': False}),
    'grad_s2': dict(label='∇(½ d²)',         needs={'sigma': False}),
    'grad'   : dict(label='∇f (model)',      needs={'sigma': False}),  # NEW
    'logG'   : dict(label='∇V_logG',         needs={'sigma': True}),   # needs sigma_uv & band_sigma
}


def initial_state() -> Dict[str, Any]:
    return {
        'title': 'Levelset Viz',
        'grid': {'H': 100, 'W': 100},
        'level_y': 0.0,
        'active_fields': make_field_defaults(),
        'legend': {},  # field-key -> bool, filled dynamically as user toggles legend
        'arrows': {
            'stride': 5,
            'plotly_scale': 0.9,
            'arrow_cap': None,              # or numeric
            'arrow_cap_quantile': 0.98,
            'normalize_arrows': False,
            'normalize_to': 1.0,
            'internal_scale': 0.7,
            'opacity': 1.0,
        },
        # NEW: model mode meta (JSON-serializable)
        'model': {
            'loaded': False,
            'code': DEFAULT_LOADER_CODE,
            'ckpt_path': '',
            'dim_in': 2,
            'dims': [0, 1],  # (u,v) slice indices into R^D
        },
    }


app = Dash(__name__)
server = app.server  # for gunicorn/WSGI

# Stores
app.layout = dmc.MantineProvider(
    withCssVariables=True,
    children=html.Div(
        style={'display': 'flex', 'height': '100vh', 'overflow': 'hidden'},
        children=[
            # ================= Left Sidebar =================
            html.Div(
                id='left-sidebar',
                style={'width': '360px', 'minWidth': '320px', 'maxWidth': '420px',
                       'borderRight': '1px solid #eaeaea', 'overflowY': 'auto'},
                children=[
                    dmc.Affix(
                        position={'top': 0, 'left': 0},
                        zIndex=10,
                        children=dmc.Paper(
                            shadow='xs', radius='0', p='sm',
                            withBorder=True,
                            style={'background': 'white', 'position': 'sticky', 'top': 0},
                            children=dmc.Group([
                                dmc.Text("Levelset Viz Controls", fw=600),
                                dmc.Button("Reset legend vis", id='reset-legend', size='xs', variant='light'),
                            ])
                        ),
                    ),
                    dmc.Space(h=8),
                    # State stores (hidden)
                    dcc.Store(id='state', data=initial_state()),
                    dcc.Store(id='legend-state', data={}),

                    # ========== Theme + Data/Grid ==========
                    dmc.Container([
                        dmc.SegmentedControl(
                            id='theme-toggle',
                            data=[{"label": "Light", "value": "light"}, {"label": "Dark", "value": "dark"}],
                            value='light', fullWidth=True, size='xs'
                        ),
                        dmc.Divider(my='sm', label="Data / Grid"),
                        dmc.Grid(grow=True, children=[
                            dmc.GridCol(dmc.NumberInput(id='grid-H', label="Grid H", value=100, min=10, max=600, step=10), span=6),
                            dmc.GridCol(dmc.NumberInput(id='grid-W', label="Grid W", value=100, min=10, max=600, step=10), span=6),
                        ]),
                        dmc.Grid(grow=True, children=[
                            dmc.GridCol(dmc.NumberInput(id='dim-u', label="Dim for u (row)", value=0, min=0, step=1), span=6),
                            dmc.GridCol(dmc.NumberInput(id='dim-v', label="Dim for v (col)", value=1, min=0, step=1), span=6),
                        ]),
                        dmc.Divider(my='sm', label="Level"),
                        dmc.Slider(id='level-slider', min=-5, max=5, step=0.01, value=0.0, marks=[{"value": 0.0, "label": "0"}]),
                        dmc.Space(h=8),

                        # ========== Model Gradient Mode (collapsible) ==========
                        dmc.Divider(my='sm', label="Model gradient mode"),
                        dmc.Accordion(
                            id='model-acc',
                            multiple=False,
                            value=[],
                            children=[
                                dmc.AccordionItem([
                                    dmc.AccordionControl("Torch model loader (customizable)"),
                                    dmc.AccordionPanel(
                                        children=[
                                            dmc.Textarea(
                                                id='model-code',
                                                label="Loader function (eval'd / exec'd)",
                                                description="Provide a callable or a def load(path): ... -> model",
                                                value=DEFAULT_LOADER_CODE,
                                                autosize=True, minRows=6
                                            ),
                                            dmc.TextInput(id='ckpt-path', label="Checkpoint path", placeholder="/path/to/model.pt"),
                                            dmc.NumberInput(id='model-dim', label="Input dimensionality D", value=2, min=2, step=1),
                                            dmc.Group([
                                                dmc.Button("Load model", id='load-model', variant='filled'),
                                                dmc.Badge("not loaded", id='loaded-badge', variant='light', color='gray')
                                            ], mt='xs'),
                                            html.Div(id='model-status', style={'marginTop': '6px'}),
                                        ]
                                    )
                                ], value='loader')
                            ]
                        ),

                        # ========== Available / Active fields ==========
                        dmc.Divider(my='sm', label="Available fields"),
                        dmc.Group(wrap=True, gap="xs", children=[
                            dmc.Button("Add ∂/∂d (n)",    id={'type': 'add-field', 'field': 'n'}, size='xs', variant='outline'),
                            dmc.Button("Add ∂/∂f (v2)",   id={'type': 'add-field', 'field': 'v2'}, size='xs', variant='outline'),
                            dmc.Button("Add ∇(½ d²)",     id={'type': 'add-field', 'field': 'grad_s2'}, size='xs', variant='outline'),
                            dmc.Button("Add ∇f (model)",  id={'type': 'add-field', 'field': 'grad'}, size='xs', variant='outline'),
                            dmc.Button("Add ∇V_logG",     id={'type': 'add-field', 'field': 'logG'}, size='xs', variant='outline'),
                        ]),
                        dmc.Divider(my='sm', label="Active vector fields"),
                        html.Div(id='active-fields-panel'),

                        dmc.Divider(my='sm', label="Global arrow controls"),
                        dmc.Grid(grow=True, children=[
                            dmc.GridCol(dmc.NumberInput(id='glob-stride', label="Stride", value=5, min=1, max=30, step=1), span=4),
                            dmc.GridCol(dmc.NumberInput(id='glob-scale', label="Plotly scale", value=0.9, step=0.1, min=0.1, max=3.0), span=8),
                        ]),
                        dmc.Grid(grow=True, children=[
                            dmc.GridCol(dmc.Switch(id='glob-norm', label="Normalize arrows", checked=False), span=6),
                            dmc.GridCol(dmc.NumberInput(id='glob-normto', label="Normalize to", value=1.0, step=0.1, min=0.1, max=5.0), span=6),
                        ]),
                        dmc.Grid(grow=True, children=[
                            dmc.GridCol(dmc.NumberInput(id='glob-cap', label="Arrow cap (abs)", value=0.33, min=0.1, step=0.1), span=6),
                            dmc.GridCol(dmc.NumberInput(id='glob-qcap', label="Cap quantile", value=0.98, min=0.5, max=1.0, step=0.01), span=6),
                        ]),
                        dmc.Space(h=80),  # breathing room
                    ], fluid=True, px='sm')
                ],
            ),
            # ================= Right Main (Toolbar + Graph) =================
            html.Div(
                id='right-main',
                style={'flex': 1, 'display': 'flex', 'flexDirection': 'column', 'minWidth': 0},
                children=[
                    dmc.Paper(
                        withBorder=True, p='xs',
                        style={'position': 'sticky', 'top': 0, 'zIndex': 9, 'background': 'white'},
                        children=dmc.Group([
                            dmc.Text("Viewer", fw=600),
                            dmc.Button("Reset view", id='reset-view', size='xs', variant='light'),
                        ], justify='space-between')
                    ),
                    html.Div(style={'flex': 1, 'minHeight': 0}, children=[
                        dcc.Graph(id='ls-graph', style={'height': '100%', 'width': '100%'})
                    ]),
                ]
            ),
        ]
    )
)

# =======================
# === UI helpers =========
# =======================


def field_card(fld: Dict[str, Any], index: int) -> dmc.Card:
    key = fld['key']; label = fld['label']
    color = fld.get('color', '#000000')
    stride = fld.get('stride', 5)
    internal_scale = fld.get('internal_scale', 0.7)
    opacity = fld.get('opacity', 1.0)
    children = [
        dmc.Group(justify="space-between", children=[
            dmc.Text(label, fw=600),
            dmc.Group(gap='xs', children=[
                dmc.ActionIcon("▲", id={'type': 'reorder', 'field': key, 'action': 'up'},   variant='light', size='sm'),
                dmc.ActionIcon("▼", id={'type': 'reorder', 'field': key, 'action': 'down'}, variant='light', size='sm'),
                dmc.ActionIcon("✖", id={'type': 'reorder', 'field': key, 'action': 'remove'}, variant='outline', color='red', size='sm'),
            ])
        ]),
        dmc.Space(h=6),
        dmc.Grid(grow=True, children=[
            dmc.GridCol(dmc.ColorInput(id={'type': 'param', 'field': key, 'name': 'color'}, label="Color", value=color, disallowInput=False), span=6),
            dmc.GridCol(dmc.NumberInput(id={'type': 'param', 'field': key, 'name': 'stride'}, label="Stride (override)", value=stride, min=1, max=30, step=1), span=6),
        ]), 
        dmc.Grid(grow=True, children=[
            dmc.GridCol(dmc.NumberInput(id={'type': 'param', 'field': key, 'name': 'internal_scale'}, label="Internal scale", value=internal_scale, step=0.1, min=0.05, max=5.0), span=6),
            dmc.GridCol(dmc.NumberInput(id={'type': 'param', 'field': key, 'name': 'opacity'}, label="Opacity", value=opacity, step=0.1, min=0.05, max=5.0), span=6),
        ]),
    ]
    if key == 'logG':
        sigma_uv = fld.get('sigma_uv', 1.5)
        band_sigma = fld.get('band_sigma', 0.5)
        children += [
            dmc.NumberInput(id={'type': 'param', 'field': key, 'name': 'sigma_uv'}, label="σ (UV blur)", value=sigma_uv, step=0.1, min=0.05, max=20.0),
            dmc.NumberInput(id={'type': 'param', 'field': key, 'name': 'band_sigma'}, label="τ (band in f)", value=band_sigma, step=0.05, min=0.01, max=10.0),
        ]
    return dmc.Card(children=children, withBorder=True, shadow='xs', p='sm', mt='xs', radius='md')

# ============================
# === Callbacks: UI wiring ===
# ============================

# 1) Render active field cards from state
@app.callback(Output('active-fields-panel', 'children'),
              Input('state', 'data'))
def _render_active_fields(state):
    cards = []
    for i, fld in enumerate(state['active_fields']):
        cards.append(field_card(fld, i))
    if not cards:
        return dmc.Alert("No active fields. Use the buttons above to add some.", color='gray')
    return cards


# 2) Add field buttons (robust: read clicked field from triggered_id)
@app.callback(Output('state', 'data', allow_duplicate=True),
              [Input({'type': 'add-field', 'field': ALL}, 'n_clicks')],
              State('state', 'data'),
              prevent_initial_call=True)
def _add_fields(_n_clicks_list, state):
    trig = ctx.triggered_id
    if not trig:
        raise dash.exceptions.PreventUpdate
    field_key = trig.get('field') if isinstance(trig, dict) else None
    if field_key is None or field_key not in AVAILABLE_FIELDS:
        raise dash.exceptions.PreventUpdate
    # Avoid duplicates
    if not any(f['key'] == field_key for f in state['active_fields']):
        cfg = dict(
            key=field_key,
            label=AVAILABLE_FIELDS[field_key]['label'],
            color={'n': '#1f77b4', 'v2': '#ff7f0e', 'grad_s2': '#2ca02c', 'logG': '#9467bd', 'grad': '#d62728'}[field_key],
            stride=state['arrows']['stride'],
            internal_scale=state['arrows']['internal_scale'],
            opacity=state['arrows']['opacity'],
        )
        if field_key == 'logG':
            cfg.update({'sigma_uv': 1.5, 'band_sigma': 0.5})
        state['active_fields'].append(cfg)
    return state


# 3) Reorder/remove field cards
@app.callback(Output('state', 'data', allow_duplicate=True),
              Input({'type': 'reorder', 'field': ALL, 'action': ALL}, 'n_clicks'),
              State('state', 'data'),
              prevent_initial_call=True)
def _reorder_remove(_clicks, state):
    trig = ctx.triggered_id
    if not trig:
        raise dash.exceptions.PreventUpdate
    key = trig['field']; action = trig['action']
    idx = next((i for i, f in enumerate(state['active_fields']) if f['key'] == key), None)
    if idx is None:
        return state
    if action == 'up' and idx > 0:
        state['active_fields'][idx-1], state['active_fields'][idx] = state['active_fields'][idx], state['active_fields'][idx-1]
    elif action == 'down' and idx < len(state['active_fields'])-1:
        state['active_fields'][idx+1], state['active_fields'][idx] = state['active_fields'][idx], state['active_fields'][idx+1]
    elif action == 'remove':
        state['active_fields'].pop(idx)
        # also drop legend remembered visibility to avoid stale keys
        state['legend'].pop(key, None)
    return state


# 4) Per-field parameter changes (pattern-matching)
@app.callback(Output('state', 'data', allow_duplicate=True),
              Input({'type': 'param', 'field': ALL, 'name': ALL}, 'value'),
              State('state', 'data'),
              prevent_initial_call=True)
def _update_field_params(values, state):
    # We receive all current values; update matched fields in state
    ids = ctx.inputs_list[0]
    for comp in ids:
        cid = comp['id']; val = comp['value']
        field_key = cid['field']; pname = cid['name']
        for f in state['active_fields']:
            if f['key'] == field_key:
                f[pname] = val
                break
    return state


# 5) Global arrow controls
@app.callback(Output('state', 'data', allow_duplicate=True),
              Input('glob-stride', 'value'),
              Input('glob-scale', 'value'),
              Input('glob-norm', 'checked'),
              Input('glob-normto', 'value'),
              Input('glob-cap', 'value'),
              Input('glob-qcap', 'value'),
              State('state', 'data'),
              prevent_initial_call=True)
def _update_global_arrows(stride, scale, norm, normto, cap, qcap, state):
    arr = state['arrows']
    arr['stride'] = stride
    arr['plotly_scale'] = scale
    arr['normalize_arrows'] = norm
    arr['normalize_to'] = normto
    arr['arrow_cap'] = cap
    arr['arrow_cap_quantile'] = qcap
    state['arrows'] = arr
    return state


# 6) Grid + level + dims updates into state
@app.callback(Output('state', 'data', allow_duplicate=True),
              Input('grid-H', 'value'),
              Input('grid-W', 'value'),
              Input('level-slider', 'value'),
              Input('dim-u', 'value'),
              Input('dim-v', 'value'),
              State('state', 'data'),
              prevent_initial_call=True)
def _update_grid_level(H, W, level, dim_u, dim_v, state):
    state['grid']['H'] = int(H)
    state['grid']['W'] = int(W)
    state['level_y'] = float(level)
    # dims are meaningful even if model not loaded yet; we'll clamp on use
    state.setdefault('model', {})
    state['model'].setdefault('dims', [0, 1])
    state['model']['dims'] = [int(dim_u or 0), int(dim_v or 1)]
    return state


# 7) Legend visibility tracker (preserve on recompute)
@app.callback(Output('legend-state', 'data', allow_duplicate=True),
              Input('ls-graph', 'restyleData'),
              State('ls-graph', 'figure'),
              State('legend-state', 'data'),
              prevent_initial_call=True)
def _capture_legend(restyleData, fig, legend_state):
    # restyleData looks like: [{'visible': ['legendonly']}, [indices...]] or {'visible':[True]}
    if restyleData is None:
        raise dash.exceptions.PreventUpdate
    vis_change = restyleData[0].get('visible')
    indices = restyleData[1]
    if vis_change is None or not indices:
        raise dash.exceptions.PreventUpdate
    newvis = vis_change[0]  # True / 'legendonly'
    for idx in indices:
        tr = fig['data'][idx]
        group = tr.get('legendgroup') or tr.get('name')
        if not group:
            continue
        legend_state[group] = (newvis is True)
    return legend_state


# 8) Reset legend visibility button
@app.callback(Output('legend-state', 'data', allow_duplicate=True),
              Input('reset-legend', 'n_clicks'),
              prevent_initial_call=True)
def _reset_legend(_n):
    return {}


# 9) Build the figure from state + legend-state (live updates)
@app.callback(Output('ls-graph', 'figure'),
              Input('state', 'data'),
              Input('legend-state', 'data'))
def _update_figure(state, legend_state):
    # merge legend_state into state and build
    state = dict(state)  # shallow copy
    state['legend'] = legend_state or {}
    fig = build_figure(state)
    return fig


# 10) Reset view (auto-range) by bumping uirevision (minimal hack: change title temporarily)
@app.callback(Output('state', 'data', allow_duplicate=True),
              Input('reset-view', 'n_clicks'),
              State('state', 'data'),
              prevent_initial_call=True)
def _reset_view(_n, state):
    # No direct "reset camera" API; bump title to force relayout autorange via uirevision stable state is preserved anyway.
    state['title'] = (state.get('title') or 'Levelset Viz')
    return state


# 11) Load model callback (eval/exec user code)
@app.callback(
    Output('state', 'data', allow_duplicate=True),
    Output('model-status', 'children'),
    Output('loaded-badge', 'children'),
    Output('loaded-badge', 'color'),
    Input('load-model', 'n_clicks'),
    State('model-code', 'value'),
    State('ckpt-path', 'value'),
    State('model-dim', 'value'),
    State('state', 'data'),
    prevent_initial_call=True,
)
def _load_model(_n, code_text, ckpt_path, dim_in, state):
    global LOADED_MODEL
    if not _n:
        raise dash.exceptions.PreventUpdate

    # Try to obtain a loader callable either via eval(text) or exec(text) with def load(...)
    loader = None
    ns: Dict[str, Any] = {'torch': torch, 'np': np}
    try:
        maybe_callable = eval(code_text, ns)  # works if user provides a lambda or a callable
        if callable(maybe_callable):
            loader = maybe_callable
    except Exception:
        loader = None
    if loader is None:
        # Fallback to exec defining def load(path): ...
        ns = {'torch': torch, 'np': np}
        try:
            exec(code_text, ns)
            loader = ns.get('load')
        except Exception as e:
            msg = dmc.Alert(f"Loader code error: {type(e).__name__}: {e}", color='red', variant='light')
            return state, msg, "not loaded", 'gray'
    if not callable(loader):
        msg = dmc.Alert("Provided code did not yield a callable loader.", color='red', variant='light')
        return state, msg, "not loaded", 'gray'

    try:
        model = loader(ckpt_path)
        if hasattr(model, 'eval'):
            model.eval()
        LOADED_MODEL = model
        # Update state
        state.setdefault('model', {})
        state['model'].update({
            'loaded': True,
            'code': code_text,
            'ckpt_path': ckpt_path or '',
            'dim_in': int(dim_in or 2),
        })
        # Add ∇f field if not present
        if not any(f['key'] == 'grad' for f in state['active_fields']):
            state['active_fields'].append(dict(key='grad', label=AVAILABLE_FIELDS['grad']['label'], color='#d62728', stride=state['arrows']['stride'], internal_scale=state['arrows']['internal_scale'], opacity=state['arrows']['opacity']))

        ok = dmc.Alert("Model loaded.", color='green', variant='light')
        return state, ok, "loaded", 'green'
    except Exception as e:
        LOADED_MODEL = None
        msg = dmc.Alert(f"Load failed: {type(e).__name__}: {e}", color='red', variant='light')
        return state, msg, "not loaded", 'gray'


# ==========================
# === Entry point ==========
# ==========================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
