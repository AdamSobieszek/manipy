
# IndependentDistributionTrainerACG
# Matches *multiple* independent 1D target marginals (via your y_distributions.*)
# while keeping the ACG-on-sphere sampler and all diagnostics/visualization goodies.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from IPython.display import display
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
from concurrent.futures import ThreadPoolExecutor, Future


# ---- import your distribution API ----
from deixis.dist.y_distributions import (
    Distribution1D,
    Uniform1D, Normal1D, MixtureOfNormals1D, EmpiricalECDF1D, EmpiricalKDE1D,
    w2_quantile_loss_to_target, cvm_loss_to_target
)


# =============================
# Helpers reused from previous cells
# =============================
def tensor_to_pil_batch(imgs: torch.Tensor, out_size: int = 384) -> list:
    if imgs.ndim != 4:
        raise ValueError("Expected 4D tensor [N,C,H,W]")
    with torch.no_grad():
        N, C, H, W = imgs.shape
        if H != out_size or W != out_size:
            imgs = F.interpolate(imgs, size=(out_size, out_size), mode="bilinear", align_corners=False)
        imgs = torch.clamp((imgs + 1.0) / 2.0, 0.0, 1.0)
        pil_list = []
        for i in range(N):
            arr = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            pil_list.append(Image.fromarray(arr))
        return pil_list

def hds_diversity(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return torch.tensor(1.0, device=z.device, dtype=z.dtype)
    z_norm = z / (z.norm(p=2, dim=1, keepdim=True) + 1e-8)
    sim = z_norm @ z_norm.t()
    n = sim.shape[0]
    tril = torch.tril(sim, diagonal=-1)
    clump = (tril.pow(2).sum() * 2.0) / (n * (n - 1) + 1e-8)
    return (1.0 - clump.sqrt())

def spd_stein_div(S: np.ndarray, T: np.ndarray, eps: float = 1e-6) -> float:
    d = S.shape[0]
    S = S + eps * np.eye(d, dtype=S.dtype)
    T = T + eps * np.eye(d, dtype=T.dtype)
    M = 0.5 * (S + T)
    sign1, logdetM = np.linalg.slogdet(M)
    sign2, logdetS = np.linalg.slogdet(S)
    sign3, logdetT = np.linalg.slogdet(T)
    if sign1 <= 0 or sign2 <= 0 or sign3 <= 0:
        return float(np.linalg.norm(S - T, ord="fro"))
    return float(logdetM - 0.5 * (logdetS + logdetT))

class AsyncConstantDistVisualizer:
    """
    - Shows per-target output histograms with the desired distribution outline,
      plus representative samples and scalar metric trends.
    """
    def __init__(
        self,
        *,
        G: nn.Module,
        device: torch.device,
        z_dim: int,
        sample_size: int = 6,
        target_bins: int = 40,
        target_curve_points: int = 200,
        title: str = "Independent Marginals Training (ACG in Z)"
    ) -> None:
        self.G = G.to(device).eval()
        self.device = device
        self.z_dim = z_dim
        self.sample_size = sample_size
        self.target_bins = int(target_bins)
        self._target_curve_points = int(target_curve_points)
        self._title = title

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._future: Optional[Future] = None
        self._latest_payload: Optional[Any] = None

        self.step_hist: List[int] = []
        self.w2_hist: List[float] = []
        self.cvm_hist: List[float] = []
        self.wstd_hist: List[float] = []
        self.wcov_div_hist: List[float] = []

        self.fig: Optional[go.FigureWidget] = None
        self._target_names: List[str] = []
        self._dist_trace_indices: List[Dict[str, int]] = []
        self._metric_trace_indices: Dict[str, int] = {}
        self._image_trace_idx: Optional[int] = None
        self._ann_idx: Optional[int] = None

        self._ensure_layout([])

    def _ensure_layout(self, target_names: Sequence[str]) -> None:
        names = list(target_names) if target_names else ["Target 1"]
        if self.fig is not None and names == self._target_names:
            return

        self._target_names = names
        n_targets = len(names)

        specs, subplot_titles = [], []
        for idx, name in enumerate(names):
            if idx == 0:
                specs.append([{"type": "xy"}, {"type": "image", "rowspan": n_targets}])
                subplot_titles.extend([f"{name} (current vs target)", "Representative Samples"])
            else:
                specs.append([{"type": "xy"}, None])
                subplot_titles.append(f"{name} (current vs target)")
        specs.append([None, {"type": "xy"}])
        subplot_titles.append("Metrics")
        row_heights = [0.55 / max(1, n_targets)] * n_targets + [0.45]

        fig = go.FigureWidget(make_subplots(
            rows=n_targets + 1,
            cols=2,
            specs=specs,
            column_widths=[0.6, 0.4],
            row_heights=row_heights,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08
        ))

        dist_indices: List[Dict[str, int]] = []
        for idx, name in enumerate(names):
            hist = go.Bar(
                x=[], y=[],
                name=f"{name} current",
                marker=dict(color="#1f77b4"),
                opacity=0.55,
                showlegend=(idx == 0)
            )
            target_line = go.Scatter(
                x=[], y=[],
                mode="lines",
                name=f"{name} target",
                line=dict(color="#d62728", width=2),
                showlegend=(idx == 0)
            )
            fig.add_trace(hist, row=idx + 1, col=1)
            fig.add_trace(target_line, row=idx + 1, col=1)
            dist_indices.append({"hist": len(fig.data) - 2, "target": len(fig.data) - 1})

        fig.add_trace(go.Image(z=None, name="Representative Samples"), row=1, col=2)
        image_idx = len(fig.data) - 1

        metric_indices: Dict[str, int] = {}
        fig.add_trace(go.Scatter(mode="lines+markers", name="W2 Loss (avg)", line=dict(width=2)),
                      row=n_targets + 1, col=2)
        metric_indices["w2"] = len(fig.data) - 1
        fig.add_trace(go.Scatter(mode="lines+markers", name="CVM Loss (avg)", line=dict(width=2, dash="dash")),
                      row=n_targets + 1, col=2)
        metric_indices["cvm"] = len(fig.data) - 1
        fig.add_trace(go.Scatter(mode="lines+markers", name="W-Std (Norm.)", line=dict(width=2)),
                      row=n_targets + 1, col=2)
        metric_indices["wstd"] = len(fig.data) - 1
        fig.add_trace(go.Scatter(mode="lines+markers", name="W-Cov SteinDiv", line=dict(width=2, dash="dot")),
                      row=n_targets + 1, col=2)
        metric_indices["wcov"] = len(fig.data) - 1

        fig.update_layout(
            width=2200,
            height=900 + 140 * n_targets,
            title_text=self._title,
            title_x=0.5,
            showlegend=True,
            legend=dict(x=0.66, y=0.95)
        )
        for idx in range(n_targets):
            fig.update_xaxes(title_text="Value", row=idx + 1, col=1)
            fig.update_yaxes(title_text="Density", row=idx + 1, col=1)
        fig.update_xaxes(title_text="Training Step", row=n_targets + 1, col=2)
        fig.update_yaxes(title_text="Metric Value", row=n_targets + 1, col=2)

        fig.add_annotation(
            xref="x1", yref="y1",
            x=0.02, y=0.98,
            text="",
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="#444",
            borderwidth=1,
            font=dict(size=12, family="Courier New"),
            xanchor="left", yanchor="top"
        )
        ann_idx = len(fig.layout.annotations) - 1

        clear_output(wait=True)
        display(fig)

        self.fig = fig
        self._dist_trace_indices = dist_indices
        self._metric_trace_indices = metric_indices
        self._image_trace_idx = image_idx
        self._ann_idx = ann_idx

        if self.step_hist:
            self._refresh_metric_history()

    def request_update(
        self,
        *,
        step: int,
        B_2x2: np.ndarray,
        z_batch_gpu: torch.Tensor,
        w2_avg: float,
        cvm_avg: float,
        w_std_norm: float,
        wcov_div: Optional[float],
        include_grid: bool,
        per_target_summary: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        if self._future and not self._future.done():
            return
        covdiv_str = "NA" if wcov_div is None else f"{wcov_div:.4f}"
        lines = [
            f"<b>step: {step}</b>",
            f"W2(avg): {w2_avg:.4f}",
            f"CVM(avg): {cvm_avg:.4f}",
            f"W-Std(norm): {w_std_norm:.3f}",
            f"W-Cov(Δ): {covdiv_str}",
        ]
        if per_target_summary:
            lines.append("<br><b>Per-target:</b>")
            for d in per_target_summary:
                name = d.get("name", "y")
                w2 = d.get("w2")
                cvm = d.get("cvm")
                lines.append(f"{name}: W2={w2:.4f}, CVM={cvm:.4f}")
        metrics_text = "<br>".join(lines)

        self._future = self._executor.submit(
            self._prepare_payload,
            step, B_2x2, z_batch_gpu.detach().clone(),
            w2_avg, cvm_avg, w_std_norm, wcov_div,
            metrics_text, include_grid,
            per_target_summary
        )

    def update_plot_if_ready(self) -> bool:
        payload_to_plot = None
        with self._lock:
            if self._latest_payload is not None:
                payload_to_plot = self._latest_payload
                self._latest_payload = None

        if self._future and self._future.done():
            self._future.result()

        if payload_to_plot is None:
            return self._future is not None and self._future.done()

        (step, w2, cvm, wstd, wcov_div, grid_arr, metrics_text, dist_payload) = payload_to_plot
        target_names = [d["name"] for d in dist_payload] if dist_payload else self._target_names
        self._ensure_layout(target_names)

        self.step_hist.append(step)
        self.w2_hist.append(w2)
        self.cvm_hist.append(cvm)
        self.wstd_hist.append(wstd)
        self.wcov_div_hist.append(np.nan if wcov_div is None else wcov_div)

        if not self.fig:
            return True

        with self.fig.batch_update():
            if dist_payload:
                for idx, info in enumerate(dist_payload):
                    traces = self._dist_trace_indices[idx]
                    self.fig.data[traces["hist"]].x = info["hist_x"]
                    self.fig.data[traces["hist"]].y = info["hist_y"]
                    self.fig.data[traces["hist"]].name = f"{info['name']} current"
                    self.fig.data[traces["target"]].x = info["target_x"]
                    self.fig.data[traces["target"]].y = info["target_y"]
                    self.fig.data[traces["target"]].name = f"{info['name']} target"
                for idx in range(len(dist_payload), len(self._dist_trace_indices)):
                    traces = self._dist_trace_indices[idx]
                    self.fig.data[traces["hist"]].x = []
                    self.fig.data[traces["hist"]].y = []
                    self.fig.data[traces["target"]].x = []
                    self.fig.data[traces["target"]].y = []
            if grid_arr is not None and self._image_trace_idx is not None:
                self.fig.data[self._image_trace_idx].z = grid_arr
            self._refresh_metric_history()
            if self._ann_idx is not None:
                self.fig.layout.annotations[self._ann_idx].text = metrics_text

        return True

    def shutdown(self) -> None:
        self._executor.shutdown()

    def _refresh_metric_history(self) -> None:
        if not self.fig or not self._metric_trace_indices:
            return
        self.fig.data[self._metric_trace_indices["w2"]].x = self.step_hist
        self.fig.data[self._metric_trace_indices["w2"]].y = self.w2_hist
        self.fig.data[self._metric_trace_indices["cvm"]].x = self.step_hist
        self.fig.data[self._metric_trace_indices["cvm"]].y = self.cvm_hist
        self.fig.data[self._metric_trace_indices["wstd"]].x = self.step_hist
        self.fig.data[self._metric_trace_indices["wstd"]].y = self.wstd_hist
        self.fig.data[self._metric_trace_indices["wcov"]].x = self.step_hist
        self.fig.data[self._metric_trace_indices["wcov"]].y = self.wcov_div_hist

    def _build_distribution_payload(
        self,
        per_target_summary: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, np.ndarray]]:
        if not per_target_summary:
            return []
        payload: List[Dict[str, np.ndarray]] = []
        for idx, entry in enumerate(per_target_summary):
            payload.append(self._summarize_target_distribution(entry, idx))
        return payload

    def _summarize_target_distribution(
        self,
        entry: Dict[str, Any],
        idx: int
    ) -> Dict[str, np.ndarray]:
        name = entry.get("name", f"y{idx}")
        samples = entry.get("samples") or entry.get("values") or entry.get("current")
        if isinstance(samples, torch.Tensor):
            current_np = samples.detach().cpu().numpy().astype(np.float32).ravel()
        elif samples is not None:
            current_np = np.asarray(samples, dtype=np.float32).ravel()
        else:
            current_np = np.empty(0, dtype=np.float32)

        hist_x = np.array([], dtype=np.float32)
        hist_y = np.array([], dtype=np.float32)
        if current_np.size:
            hist_y, edges = np.histogram(current_np, bins=self.target_bins, density=True)
            hist_x = 0.5 * (edges[:-1] + edges[1:])
            hist_y = hist_y.astype(np.float32)

        target = entry.get("target") or entry.get("target_distribution")
        target_x = np.array([], dtype=np.float32)
        target_y = np.array([], dtype=np.float32)
        if target is not None:
            try:
                if hasattr(target, "_param_device_dtype"):
                    device, dtype = target._param_device_dtype()
                else:
                    device, dtype = torch.device("cpu"), torch.float32
                u = torch.linspace(0.005, 0.995, self._target_curve_points, device=device, dtype=dtype)
                with torch.no_grad():
                    xs = target.icdf(u)
                    logp = target.log_prob(xs)
                target_x = xs.detach().cpu().numpy().astype(np.float32)
                target_y = torch.exp(logp).detach().cpu().numpy().astype(np.float32)
            except Exception:
                target_x = np.array([], dtype=np.float32)
                target_y = np.array([], dtype=np.float32)

        return {
            "name": name,
            "hist_x": hist_x,
            "hist_y": hist_y,
            "target_x": target_x,
            "target_y": target_y,
        }

    def _prepare_payload(
        self,
        step: int,
        _B_2x2: np.ndarray,
        z_gpu: torch.Tensor,
        w2: float,
        cvm: float,
        wstd: float,
        wcov_div: Optional[float],
        metrics_text: str,
        include_grid: bool,
        per_target_summary: Optional[List[Dict[str, Any]]]
    ) -> None:
        dist_payload = self._build_distribution_payload(per_target_summary)
        grid = self._grid(z_gpu) if include_grid else None
        with self._lock:
            self._latest_payload = (
                step, w2, cvm, wstd, wcov_div,
                grid, metrics_text, dist_payload
            )

    def _grid(self, z_gpu: torch.Tensor) -> Optional[np.ndarray]:
        try:
            K = min(self.sample_size, z_gpu.shape[0])
            idx = torch.randint(0, z_gpu.shape[0], (K,), device=z_gpu.device)
            with torch.no_grad():
                ws = self.G.mapping(z_gpu[idx], None)
                imgs_tensor = self.G.synthesis(ws).detach().cpu()
            pil_images = tensor_to_pil_batch(imgs_tensor, out_size=384)
            return np.asarray(self._make_grid(pil_images))
        except Exception as e:
            print(f"[Visualizer] Grid generation failed: {e}")
            return None

    @staticmethod
    def _make_grid(imgs, rows=2):
        if not imgs:
            return None
        w, h = imgs[0].size
        cols = (len(imgs) + rows - 1) // rows
        canvas = Image.new("RGB", (cols * w, rows * h))
        for i, im in enumerate(imgs):
            canvas.paste(im, (i % cols * w, i // cols * h))
        return canvas
