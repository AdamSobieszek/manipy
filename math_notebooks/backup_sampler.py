import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from IPython.display import display, clear_output
import torch
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, List, Dict, Tuple
from PIL import Image
from torchvision.transforms import Compose, Resize
import torchvision.transforms.functional as TF
from scipy.special import softmax
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA

# --- Visualizer and Parameterization classes remain the same ---

class MeanDiscoveryVisualizer:
    """Visualizer for Phase 1: Finding the category's central direction."""
    def __init__(self, z_dim: int = 512):
        self.z_dim = z_dim
        self.fig = go.FigureWidget(make_subplots(
            rows=1, cols=2,
            subplot_titles=("Z-Space Mean Trajectory", "Best Fitness per Generation"),
            column_widths=[0.5, 0.5]
        ))
        self._init_figure()
        self.mean_history = []
        self.fitness_history = []
        self.generation_history = []
        display(self.fig)

    def _init_figure(self):
        for i in range(1, 4):
            self.fig.add_shape(type="circle", xref="x1", yref="y1", x0=-i, y0=-i, x1=i, y1=i, line_color="grey", line_dash="dot", fillcolor="rgba(0,0,0,0)")
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='Mean Path', line=dict(color='orange')), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Current Mean', marker=dict(color='red', size=12)), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='Best Fitness', line=dict(color='green')), row=1, col=2)
        self.fig.update_layout(title_text="Phase 1: Mean Discovery", title_x=0.5, height=500, showlegend=False)
        self.fig.update_xaxes(title_text="Z Dim 0", range=[-5, 5], row=1, col=1)
        self.fig.update_yaxes(title_text="Z Dim 1", range=[-5, 5], scaleanchor="x1", scaleratio=1, row=1, col=1)
        self.fig.update_xaxes(title_text="Generation", row=1, col=2)
        self.fig.update_yaxes(title_text="Fitness (Lower is Better)", type="log", row=1, col=2)

    def update(self, generation: int, mean_vec: np.ndarray, best_fitness: float):
        self.generation_history.append(generation)
        self.mean_history.append(mean_vec[:2].copy())
        self.fitness_history.append(best_fitness)
        mean_path_x, mean_path_y = zip(*self.mean_history) if self.mean_history else ([], [])
        with self.fig.batch_update():
            self.fig.data[0].x, self.fig.data[0].y = mean_path_x, mean_path_y
            self.fig.data[1].x, self.fig.data[1].y = [mean_vec[0]], [mean_vec[1]]
            self.fig.data[2].x, self.fig.data[2].y = self.generation_history, self.fitness_history

class BoundarySeekerVisualizer:
    """Visualizer for Phase 2: Plots the ellipsoid, samples, and eigenvalue spectrum."""
    def __init__(self, p_dim: int = 511):
        self.p_dim = p_dim
        self.fig = go.FigureWidget(make_subplots(
            rows=2, cols=2,
            subplot_titles=(f"Parameter Space Projection", "Failure Eigenvalue Spectrum", "Ellipsoid Volume & Failure Rate"),
            specs=[[{"rowspan": 2}, {}], [None, {"secondary_y": True}]],
            vertical_spacing=0.15
        ))
        self._init_figure()
        self.history = {'gen': [], 'log_det': [], 'fail_rate': []}
        display(self.fig)

    def _init_figure(self):
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Ellipsoid Boundary', line=dict(color='orange', width=2)), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Failed Samples', marker=dict(color='red', size=5, opacity=0.5)), row=1, col=1)
        
        self.fig.add_trace(go.Bar(x=[], y=[], name='Eigenvalues'), row=1, col=2)
        
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Log Volume', line=dict(color='blue')), row=2, col=2)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Failure Rate', line=dict(color='red', dash='dot')), secondary_y=True, row=2, col=2)
        
        self.fig.update_layout(
            title_text="Phase 2: Failure-Driven Boundary Search", title_x=0.5, height=700, showlegend=False,
            yaxis4=dict(title="Failure Rate (%)", overlaying="y3", side="right", range=[0, 100])
        )
        self.fig.update_xaxes(title_text="Dim 0", row=1, col=1)
        self.fig.update_yaxes(title_text="Dim 1", scaleanchor="x1", scaleratio=1, row=1, col=1)
        self.fig.update_xaxes(title_text="Eigenvalue Index", row=1, col=2)
        self.fig.update_yaxes(title_text="Eigenvalue", type="log", row=1, col=2)
        self.fig.update_xaxes(title_text="Generation", row=2, col=2)
        self.fig.update_yaxes(title_text="Log Det(Cov)", row=2, col=2)

    def update(self, generation: int, p_cov: np.ndarray, p_fail_proj: np.ndarray, p_ellipsoid_proj: np.ndarray, fail_rate: float, fail_eigenvalues: np.ndarray, viz_dims_info: str):
        self.history['gen'].append(generation)
        log_det = np.linalg.slogdet(p_cov)[1]
        self.history['log_det'].append(log_det)
        self.history['fail_rate'].append(fail_rate * 100)

        with self.fig.batch_update():
            # Update ellipsoid boundary and sample scatters
            self.fig.data[0].x, self.fig.data[0].y = p_ellipsoid_proj[:, 0], p_ellipsoid_proj[:, 1]
            self.fig.data[1].x = p_fail_proj[:, 0] if p_fail_proj.ndim == 2 and p_fail_proj.shape[0] > 0 else []
            self.fig.data[1].y = p_fail_proj[:, 1] if p_fail_proj.ndim == 2 and p_fail_proj.shape[0] > 0 else []
            
            # Update eigenvalue spectrum
            if fail_eigenvalues is not None:
                self.fig.data[2].x = list(range(len(fail_eigenvalues)))
                self.fig.data[2].y = fail_eigenvalues
            
            # Update metric plots
            self.fig.data[3].x, self.fig.data[3].y = self.history['gen'], self.history['log_det']
            self.fig.data[4].x, self.fig.data[4].y = self.history['gen'], self.history['fail_rate']

            # Dynamically update axis titles and ranges
            self.fig.update_xaxes(title_text=f"Projection Dim 1 ({viz_dims_info})", row=1, col=1)
            self.fig.update_yaxes(title_text=f"Projection Dim 2 ({viz_dims_info})", row=1, col=1)
            
            max_range = np.max(np.abs(p_ellipsoid_proj)) * 1.2 if p_ellipsoid_proj.size > 0 else 0.1
            min_range = 0.01
            final_range = max(max_range, min_range)
            self.fig.update_xaxes(range=[-final_range, final_range], row=1, col=1)
            self.fig.update_yaxes(range=[-final_range, final_range], row=1, col=1)

class Parameterization:
    # This class is assumed to be the final, correct version from previous steps
    def __init__(self, center_vec: np.ndarray):
        self.z_dim = center_vec.shape[0]
        self.p_dim = self.z_dim - 1
        self.north_pole = np.zeros(self.z_dim, dtype=np.float32); self.north_pole[-1] = 1.0
        self.rotation = self._get_rotation_matrix(center_vec); self.inv_rotation = self.rotation.T
    def _get_rotation_matrix(self, v_to: np.ndarray) -> np.ndarray:
        v_from = self.north_pole; v_to_unit = v_to / np.linalg.norm(v_to)
        if np.allclose(v_from, v_to_unit): return np.eye(self.z_dim, dtype=np.float32)
        if np.allclose(v_from, -v_to_unit): R = -np.eye(self.z_dim, dtype=np.float32); R[0, 0] = 1.0; return R
        u = v_from + v_to_unit; u /= np.linalg.norm(u)
        H1 = np.eye(self.z_dim) - 2 * np.outer(u, u); H2 = np.eye(self.z_dim) - 2 * np.outer(v_to_unit, v_to_unit)
        return H2 @ H1
    def map_to_z_space(self, p_vectors: np.ndarray) -> np.ndarray:
        p_vectors = np.atleast_2d(p_vectors)
        polydisk_pts = self._square_to_disk_nd(p_vectors)
        scale_factor = 2.0 / (np.sqrt(self.p_dim / 2.0) + 1e-9)
        disk_pts = polydisk_pts * scale_factor
        hemisphere_pts_aligned = self._inverse_laea_projection(disk_pts)
        return hemisphere_pts_aligned @ self.inv_rotation
    def map_to_parameter_space(self, z_vectors: np.ndarray) -> np.ndarray:
        z_vectors = np.atleast_2d(z_vectors); aligned_z = z_vectors @ self.rotation
        disk_pts = self._laea_projection(aligned_z)
        scale_factor = 2.0 / (np.sqrt(self.p_dim / 2.0) + 1e-9); polydisk_pts = disk_pts / scale_factor
        return self._disk_to_square_nd(polydisk_pts)
    def _laea_projection(self, z_aligned: np.ndarray) -> np.ndarray:
        scale = np.sqrt(2 / (1 + z_aligned[:, -1] + 1e-9)); return z_aligned[:, :-1] * scale[:, np.newaxis]
    def _inverse_laea_projection(self, disk_pts: np.ndarray) -> np.ndarray:
        norm_sq = np.sum(disk_pts**2, axis=1); norm_sq = np.clip(norm_sq, 0, 4)
        z = np.zeros((disk_pts.shape[0], self.z_dim), dtype=np.float32)
        with np.errstate(invalid='ignore'): z[:, :-1] = disk_pts * (np.sqrt(1 - norm_sq / 4))[:, np.newaxis]
        z[:, -1] = 1 - norm_sq / 2; return z
    def _square_to_disk_nd(self, points: np.ndarray) -> np.ndarray:
        out_pts = np.copy(points)
        for i in range(0, self.p_dim - (self.p_dim % 2), 2):
            x, y = points[:, i], points[:, i+1]; r, theta = np.zeros_like(x), np.zeros_like(x)
            mask = np.abs(x) > np.abs(y); r[mask] = x[mask]
            with np.errstate(divide='ignore', invalid='ignore'): theta[mask] = (np.pi / 4.0) * (y[mask] / x[mask])
            r[~mask] = y[~mask]
            with np.errstate(divide='ignore', invalid='ignore'): theta[~mask] = (np.pi / 2.0) - (np.pi / 4.0) * (x[~mask] / y[~mask])
            origin_mask = (np.abs(x) < 1e-9) & (np.abs(y) < 1e-9); theta[origin_mask] = 0
            out_pts[:, i], out_pts[:, i+1] = r * np.cos(theta), r * np.sin(theta)
        return out_pts
    def _disk_to_square_nd(self, points: np.ndarray) -> np.ndarray:
        out_pts = np.copy(points)
        for i in range(0, self.p_dim - (self.p_dim % 2), 2):
            disk_x, disk_y = points[:, i], points[:, i+1]
            r, phi = np.sqrt(disk_x**2 + disk_y**2), np.arctan2(disk_y, disk_x)
            square_u, square_v = np.zeros_like(r), np.zeros_like(r)
            mask_u_dom = (disk_x**2 >= disk_y**2)
            r_u, disk_x_u = r[mask_u_dom], disk_x[mask_u_dom]
            u_u = r_u * np.sign(disk_x_u); u_u[disk_x_u == 0] = 0
            with np.errstate(divide='ignore', invalid='ignore'): v_u = u_u * (4/np.pi) * np.arctan(disk_y[mask_u_dom] / disk_x_u)
            v_u[disk_x_u == 0] = 0; square_u[mask_u_dom], square_v[mask_u_dom] = u_u, v_u
            mask_v_dom = ~mask_u_dom
            r_v, disk_y_v = r[mask_v_dom], disk_y[mask_v_dom]
            v_v = r_v * np.sign(disk_y_v); v_v[disk_y_v == 0] = 0
            with np.errstate(divide='ignore', invalid='ignore'): u_v = v_v * (4/np.pi) * np.arctan(disk_x[mask_v_dom] / disk_y_v)
            u_v[disk_y_v == 0] = 0; square_u[mask_v_dom], square_v[mask_v_dom] = u_v, v_v
            out_pts[:, i], out_pts[:, i+1] = square_u, square_v
        return out_pts

class GeometricBoundarySeeker:
    """
    Finds the boundary of a category in Z-space using a two-phase geometric approach.
    """
    def __init__(self, target_filters: Dict, config: Dict, G: torch.nn.Module, models: Dict, device: torch.device):
        self.config = config; self.G = G.to(device); self.models = {name: model.to(device) for name, model in models.items()}
        self.device = device; self.z_dim = G.mapping.z_dim; self.p_dim = self.z_dim - 1
        self.target_bin = self._parse_filters(target_filters); self.dtype = torch.float32

    def _parse_filters(self, filters: Dict) -> List[Dict]:
        return [{'var': var, 'range': cond['range']} for var, cond in filters.items()]
    
    @torch.no_grad()
    def _calculate_fitness(self, z_batch_gpu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w_batch = self.G.mapping(z_batch_gpu, None)
        w_batch = w_batch[:, 0] if w_batch.ndim == 3 else w_batch
        final_mask = torch.full((w_batch.shape[0],), True, device=self.device)
        compliance_loss = torch.zeros(w_batch.shape[0], device=self.device)
        for condition in self.target_bin:
            var, (low, high) = condition['var'], condition['range']
            preds = self.models[var](w_batch).view(-1) * 100
            final_mask &= (preds >= low) & (preds <= high)
            dist_low = torch.clamp(low - preds, min=0)
            dist_high = torch.clamp(preds - high, min=0)
            compliance_loss += dist_low + dist_high
        
        return compliance_loss, final_mask


    def find_category_center(self) -> np.ndarray:
        """Phase 1: Find the mean direction vector."""
        print("--- Starting Phase 1: Mean Discovery ---")
        cfg = self.config['phase1']
        mean = np.zeros(self.z_dim, dtype=np.float32)
        cov = np.eye(self.z_dim, dtype=np.float32) * cfg['initial_variance']
        vis = MeanDiscoveryVisualizer(self.z_dim)
        successes = []
        success_n=0
        init_lr = 0.5
        pbar = tqdm(range(cfg['generations']), desc="Phase 1")
        for gen in pbar:
            z_cpu = np.random.multivariate_normal(mean, cov*np.cos(gen/cfg['generations']*np.pi*5)**2*(1-gen/cfg['generations']), cfg['population_size']).astype(np.float32)
            z_gpu = torch.from_numpy(z_cpu).to(self.device, self.dtype)
            fitness, _ = self._calculate_fitness(z_gpu)
            fitness_np = fitness.cpu().numpy()
            
            elite_indices = np.argsort(fitness_np)[:cfg['elite_size']]
            lr = init_lr * (1 - gen/cfg['generations'])**0.5    
            if sum(fitness_np==0)<3:
                new_mean = np.mean(z_cpu[elite_indices], axis=0)
                mean = (1 - lr) * mean + lr * new_mean
                successes.append(new_mean)
            else:
                new_mean =np.mean(z_cpu[fitness_np==0], axis=0)
                mean = (1 - lr) * mean + lr * new_mean
            
            if success_n>600:
                new_cov = np.cov(np.concatenate(successes).T-mean.reshape(1,-1), rowvar=False)
                cov = (1 - init_lr) * cov + init_lr * new_cov
                cov = (cov + cov.T) / 2.0
                cov = cov+np.eye(self.z_dim, dtype=np.float32)*1e-6
            best_fitness_val = fitness_np.mean()
            fitness, _ = self._calculate_fitness(torch.from_numpy(mean).to(self.device, self.dtype).unsqueeze(0))
            mean_fitness = fitness.cpu().numpy().mean()

            pbar.set_postfix(best_fitness=f"{best_fitness_val:.2f}", mean_norm=f"{np.linalg.norm(mean):.2f} mean_fitness={mean_fitness:.2f}")
            if gen % 10 == 0:
                vis.update(gen, mean, best_fitness_val)
        
        mu_center = mean / np.linalg.norm(mean)
        print(f"--- Phase 1 Complete. Found center vector. ---")
        return mu_center


    def find_boundary_distribution(self, mu_center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Phase 2: Geometrically grow an ellipsoid using a failure-driven approach."""
        print("\n--- Starting Phase 2: Failure-Driven Boundary Modeling ---")
        cfg = self.config['phase2']
        param = Parameterization(mu_center)
        vis = BoundarySeekerVisualizer(self.p_dim)

        p_mean = np.zeros(self.p_dim)
        zero_mean = np.zeros(self.p_dim)
        p_cov = np.eye(self.p_dim) * cfg['initial_variance']
        
        pca = None
        fail_T_momentum = None
        fail_mean_momentum = None
        pbar = tqdm(range(cfg['generations']), desc="Phase 2 (Failure-Driven)")
        p_fail_cache = []
        for gen in pbar:
            p_gauss = np.random.multivariate_normal(zero_mean, p_cov, cfg['population_size'])
            radii = 1-np.abs(np.random.randn(cfg['population_size'])*0.01)
            p_samples = p_mean + (p_gauss * (radii[:, np.newaxis] / (np.linalg.norm(p_gauss, axis=1, keepdims=True) + 1e-9)))@p_cov
            p_samples = np.clip(p_samples, -0.999, 0.999)
            z_samples = param.map_to_z_space(p_samples)
            z_samples_gpu = torch.from_numpy(z_samples).to(self.device, self.dtype)
            _, success_mask = self._calculate_fitness(z_samples_gpu)
            success_mask = success_mask.cpu().numpy() 
            
            p_fail = p_samples[~success_mask]
            fail_rate = len(p_fail) / cfg['population_size']
            
            fail_eigenvalues = None
            det = np.linalg.slogdet(p_cov)[1]
            if fail_rate < cfg['target_failure_rate_range'][0]:
                p_cov *= cfg['expansion_factor']
                pca = None
            elif fail_rate > cfg['target_failure_rate_range'][1] or (np.abs(p_samples)>1).mean()>0.001:
                 p_cov *= cfg['shrink_factor']
                 pca = None
            else:
                if len(p_fail) > self.p_dim:
                    # p_fail = np.concatenate([p_fail, p_samples[(np.abs(p_samples)>1).mean(axis=1)>0.0]])
                    p_fail_cache.append(p_fail)
                    p_fail = np.concatenate(p_fail_cache)
                    if len(p_fail) > 6:
                        p_fail_cache = p_fail_cache[1:]
                    fail_cov = np.cov(p_fail, rowvar=False)

                    p_fail_mean = np.mean(p_fail, axis=0)
                    fail_eigenvalues, fail_eigenvectors = np.linalg.eigh(fail_cov)
                    regular_eigenvalues, regular_eigenvectors = np.linalg.eigh(p_cov)
                    target_eigenvalue = regular_eigenvalues/np.mean(regular_eigenvalues)*np.mean(fail_eigenvalues)*1.001

                    scaling_factors = np.sqrt(target_eigenvalue / (fail_eigenvalues + 1e-9))
                    S = np.diag(scaling_factors)
                    
                    # Transformation matrix in the standard basis
                    T = fail_eigenvectors @ S @ fail_eigenvectors.T
                    if fail_T_momentum is None:
                        fail_T_momentum = T
                        fail_mean_momentum = p_fail_mean/512
                    else:
                        fail_T_momentum = (1 - 0.5) * fail_T_momentum + 0.5 * T
                        fail_mean_momentum = (1 - 0.5) * fail_mean_momentum + 0.5 * p_fail_mean
                        T = fail_T_momentum
                        p_fail_mean = fail_mean_momentum*0.999
                    
                    p_cov_new = T @ p_cov @ T.T
                    lr = cfg['learning_rate'] * (1 - gen/cfg['generations'])**0.5
                    p_cov = (1 - lr) * p_cov + lr * p_cov_new
                    p_mean = (1 - lr/10) * p_mean - lr/10 * p_fail_mean
                else:
                    p_cov *= cfg['expansion_factor'] # Not enough points for stable cov, expand
                    pca = None

            p_cov = (p_cov + p_cov.T) / 2.0
            if det<1100:
                p_cov = p_cov*1.0001
            pbar.set_postfix(fail_rate=f"{fail_rate*100:.1f}%", vol=f"{det:.2f} wrong={(np.abs(p_samples)>1).mean():.2f} {np.abs(p_samples).max():.2f} mean={np.linalg.norm(p_mean):.2f}")

            if gen % 5 == 0:
                if pca is None and len(p_fail) > 2:
                    pca_viz = PCA(n_components=2).fit(p_fail)
                else:
                    pca_viz = pca
                
                if pca_viz is not None:
                    proj_basis = pca_viz.components_[:2, :].T
                    p_cov_proj = proj_basis.T @ p_cov @ proj_basis
                    p_fail_proj = p_fail @ proj_basis
                    viz_dims_info = "PCA"
                else:
                    proj_basis = np.eye(self.p_dim, 2)
                    p_cov_proj = p_cov[np.ix_([0, 1], [0, 1])]
                    p_fail_proj = p_fail[:, :2] if p_fail.shape[0] > 0 else p_fail
                    viz_dims_info = "Default"
                
                # Project ellipsoid surface for visualization
                p_sphere = np.random.randn(200, self.p_dim)
                p_sphere /= np.linalg.norm(p_sphere, axis=1, keepdims=True)
                try: L = np.linalg.cholesky(p_cov)
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(p_cov)
                    L = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
                p_surface = (L @ p_sphere.T).T
                p_ellipsoid_proj = p_surface @ proj_basis

                vis.update(gen, p_cov, p_fail_proj, p_ellipsoid_proj, fail_rate, fail_eigenvalues, viz_dims_info)

        print(f"--- Phase 2 Complete ---")
        return p_mean, p_cov

    
    # Dummy classes for demonstration
    # device = 'mps'
    # CONFIG = {
    #     'phase1': { 'generations': 50, 'population_size': 2048, 'elite_size': 128, 'initial_variance': 1.0, 'learning_rate': 0.3 },
    #     'phase2': {
    #         'generations': 100,
    #         'population_size': 4096,
    #         'initial_variance': 1e-4,
    #         'learning_rate': 0.1,
    #         'shrink_factor': 0.99999,   # Less aggressive shrinking
    #         'expansion_factor': 1e-8 # Small constant outward push
    #     }
    # }
    # TARGET_FILTERS = { 'gender': {'range': (90, 100)}, 'age': {'range': (85, 100)} }


CONFIG = {
    'phase1': { 'generations':1000, 'population_size': 4096, 'elite_size': 256, 'initial_variance': .5, 'learning_rate': 0.5 },
    'phase2': {
        'generations': 1000,
        'population_size': 4096*5,
        'initial_variance': 20e-1,
        'learning_rate': 0.15,
        'target_failure_rate_range': (0.25, 0.6), # Aim for 15-30% failures
        'expansion_factor': 1.001,
        'shrink_factor': 0.99,
    }
}
TARGET_FILTERS = { 'gender': {'range': (90, 100)}, 'age': {'range': (85, 100)} }

# seeker = GeometricBoundarySeeker(target_filters=TARGET_FILTERS, config=CONFIG, G=G, models=ALL_MODELS, device=device)
# center_vector = seeker.find_category_center()
# final_p_mean, final_p_cov = seeker.find_boundary_distribution(center_vector)

print("\n--- Final Result ---")
print(f"Final Parameter Space Mean (first 5 dims): {final_p_mean[:5]}")
print(f"Final Ellipsoid Volume (log det cov): {np.linalg.slogdet(final_p_cov)[1]:.4f}")

