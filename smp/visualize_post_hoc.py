# Potential-Aligned 2-D Embedding (PA2E) for the CLM-SMP neighborhood on your ratings.csv
# - Builds the CLM-SMP allocation
# - Enumerates 1-exchange neighbors (pairwise swaps between conditions)
# - Computes state vectors y = [R_1..R_C, sig2_1..sig2_C]
# - Builds a 2D embedding aligned to Phi1 and Phi2 sensitivities
# - Exports a CSV with z1,z2,Phi1,Phi2, feasibility/boundary flags, and marks the CLM point

import numpy as np, pandas as pd, itertools, os, math
from scipy import stats as st

# --------------------------
# Load data and define design
# --------------------------
ratings = pd.read_csv('/Users/adamsobieszek/PycharmProjects/_manipy/smp/ratings.csv')
cols = [ratings.columns.get_loc(c) for c in ['Valence_M','origin_M','Ciepło_M','Kompetencja_M']]
X = ratings.iloc[:, cols].to_numpy(dtype=float)
P = X.shape[1]

# 3x3 grid of targets in the first two features; last two controlled target 0
def build_conditions_grid():
    cond = {}
    for a in (-1.0, 0.0, 1.0):
        for b in (-1.0, 0.0, 1.0):
            cond[f"{a},{b}"] = np.array([a, b, 0.0, 0.0], dtype=float)
    return cond

conditions = build_conditions_grid()
cond_keys = list(conditions.keys())
C = len(cond_keys)
n_per_cond = 15

# --------------------------
# CLM-SMP (the greedy construction we used)
# --------------------------
def smp_clm(df, conditions, columns, n_per_cond):
    Xloc = df.iloc[:, columns].to_numpy(dtype=float)
    N, P = Xloc.shape
    cond_keys = list(conditions.keys())
    S_centered = {ck: Xloc - np.asarray(conditions[ck], float).reshape(1,-1) for ck in cond_keys}
    available = np.ones(N, dtype=bool)
    R = {ck: np.zeros(P) for ck in cond_keys}
    chosen = {ck: [] for ck in cond_keys}

    def pool_mean_cov(S, avail_idx):
        S_av = S[avail_idx]
        M = len(S_av)
        if M == 0:
            return np.zeros(S.shape[1]), np.eye(S.shape[1])
        mu = S_av.mean(axis=0)
        if M == 1:
            return mu, np.eye(S.shape[1])
        S0 = S_av - mu
        Sigma = (S0.T @ S0) / (M - 1)
        Sigma_inv = np.linalg.pinv(Sigma, rcond=1e-12)
        return mu, Sigma_inv

    for step in range(n_per_cond):
        for ck in cond_keys:
            if not np.any(available): break
            n_done = len(chosen[ck])
            if n_done >= n_per_cond: continue
            t = n_per_cond - (n_done + 1)
            avail_idx = np.flatnonzero(available)
            mu_U, Sigma_inv = pool_mean_cov(S_centered[ck], avail_idx)
            S_av = S_centered[ck][avail_idx]
            v = S_av + R[ck] + t * mu_U
            w = v @ Sigma_inv
            scores = np.einsum('ij,ij->i', v, w)
            j = int(np.argmin(scores))
            idx = int(avail_idx[j])
            R[ck] += S_centered[ck][idx]
            available[idx] = False
            chosen[ck].append(idx)

    sets = {ck: chosen[ck][:] for ck in cond_keys}
    return sets

sets0 = smp_clm(ratings, conditions, cols, n_per_cond=n_per_cond)

# --------------------------
# State vector y(S) and potentials
# --------------------------
# y = [R_1..R_C, sig2_1..sig2_C]  where R_c in R^P, sig2_c in R^P (unbiased variance of (x-μ_c))
D = 2 * C * P

def state_vector(sets):
    # residuals
    R_blocks = []
    Sigs = []
    for ck in cond_keys:
        mu = conditions[ck]
        idx = sets[ck]
        Xc = X[idx] - mu.reshape(1,-1)
        R = Xc.sum(axis=0)                          # shape (P,)
        # unbiased variance of (x - mu_c) within the group
        # var = sum((x-mu)^2) - m * mean^2, divided by (m-1)
        m = Xc.shape[0]
        if m > 1:
            s = Xc.sum(axis=0)                      # (P,)
            q = (Xc*Xc).sum(axis=0)                 # (P,)
            var = (q - (s*s)/m) / (m - 1)
        else:
            var = np.zeros(P)
        R_blocks.append(R)
        Sigs.append(var)
    R_all = np.concatenate(R_blocks, axis=0)        # shape (C*P,)
    Sig_all = np.concatenate(Sigs, axis=0)          # shape (C*P,)
    y = np.concatenate([R_all, Sig_all], axis=0)    # shape (2*C*P,)
    return y, R_blocks, Sigs

def Phi1(R_blocks):
    return float(sum((R@R) for R in R_blocks))

def Phi2_from_sigblocks(sig_blocks):
    # sig_blocks is list length C of arrays (P,)
    S = np.stack(sig_blocks, axis=0)                # shape (C,P)
    mu = S.mean(axis=0, keepdims=True)              # shape (1,P)
    dev2 = (S - mu)**2
    return float(np.sum(dev2))

def gradients_from_state(R_blocks, sig_blocks):
    # Gradient of Phi1 wrt R: 2R ; wrt sig: 0
    g1_R = np.concatenate([2.0*R for R in R_blocks], axis=0)          # (C*P,)
    g1_S = np.zeros_like(g1_R)                                        # (C*P,)
    g1 = np.concatenate([g1_R, g1_S], axis=0)                         # (2*C*P,)

    # Gradient of Phi2 wrt sig: for each feature v, 2*(sigma_cv - mean_c sigma_cv)
    S = np.stack(sig_blocks, axis=0)                                  # (C,P)
    mu = S.mean(axis=0, keepdims=True)
    g2_S_mat = 2.0*(S - mu)                                           # (C,P)
    g2_S = g2_S_mat.reshape(-1)                                       # (C*P,)
    g2_R = np.zeros_like(g1_R)                                        # (C*P,)
    g2 = np.concatenate([g2_R, g2_S], axis=0)                         # (2*C*P,)
    return g1, g2


# --------------------------
# CLM-SMP greedy
# --------------------------
def smp_clm(df, conditions, columns, n_per_cond):
    Xloc = df.iloc[:, columns].to_numpy(dtype=float)
    N, P = Xloc.shape
    cond_keys = list(conditions.keys())
    S_centered = {ck: Xloc - np.asarray(conditions[ck], float).reshape(1,-1) for ck in cond_keys}
    available = np.ones(N, dtype=bool)
    R = {ck: np.zeros(P) for ck in cond_keys}
    chosen = {ck: [] for ck in cond_keys}

    def pool_mean_cov(S, avail_idx):
        S_av = S[avail_idx]
        M = len(S_av)
        if M == 0:
            return np.zeros(S.shape[1]), np.eye(S.shape[1])
        mu = S_av.mean(axis=0)
        if M == 1:
            return mu, np.eye(S.shape[1])
        S0 = S_av - mu
        Sigma = (S0.T @ S0) / (M - 1)
        Sigma_inv = np.linalg.pinv(Sigma, rcond=1e-12)
        return mu, Sigma_inv

    for step in range(n_per_cond):
        for ck in cond_keys:
            if not np.any(available): break
            n_done = len(chosen[ck])
            if n_done >= n_per_cond: continue
            t = n_per_cond - (n_done + 1)
            avail_idx = np.flatnonzero(available)
            mu_U, Sigma_inv = pool_mean_cov(S_centered[ck], avail_idx)
            S_av = S_centered[ck][avail_idx]
            v = S_av + R[ck] + t * mu_U
            w = v @ Sigma_inv
            scores = np.einsum('ij,ij->i', v, w)
            j = int(np.argmin(scores))
            idx = int(avail_idx[j])
            R[ck] += S_centered[ck][idx]
            available[idx] = False
            chosen[ck].append(idx)

    sets = {ck: chosen[ck][:] for ck in cond_keys}
    return sets

sets0 = smp_clm(ratings, conditions, cols, n_per_cond=n_per_cond)

# --------------------------
# State, potentials, gradients
# --------------------------
D = 2 * C * P

def state_vector(sets):
    R_blocks = []
    Sigs = []
    for ck in cond_keys:
        mu = conditions[ck]
        idx = sets[ck]
        Xc = X[idx] - mu.reshape(1,-1)
        R = Xc.sum(axis=0)
        m = Xc.shape[0]
        if m > 1:
            s = Xc.sum(axis=0)
            q = (Xc*Xc).sum(axis=0)
            var = (q - (s*s)/m) / (m - 1)
        else:
            var = np.zeros(P)
        R_blocks.append(R)
        Sigs.append(var)
    R_all = np.concatenate(R_blocks, axis=0)
    Sig_all = np.concatenate(Sigs, axis=0)
    y = np.concatenate([R_all, Sig_all], axis=0)
    return y, R_blocks, Sigs

def Phi1(R_blocks):
    return float(sum((R@R) for R in R_blocks))

def Phi2_from_sigblocks(sig_blocks):
    S = np.stack(sig_blocks, axis=0)   # (C,P)
    mu = S.mean(axis=0, keepdims=True)
    return float(np.sum((S - mu)**2))

def gradients_from_state(R_blocks, sig_blocks):
    g1_R = np.concatenate([2.0*R for R in R_blocks], axis=0)
    g1_S = np.zeros_like(g1_R)
    g1 = np.concatenate([g1_R, g1_S], axis=0)
    S = np.stack(sig_blocks, axis=0)
    mu = S.mean(axis=0, keepdims=True)
    g2_S_mat = 2.0*(S - mu)
    g2_S = g2_S_mat.reshape(-1)
    g2_R = np.zeros_like(g1_R)
    g2 = np.concatenate([g2_R, g2_S], axis=0)
    return g1, g2

y0, R0, Sig0 = state_vector(sets0)
phi1_0 = Phi1(R0)
phi2_0 = Phi2_from_sigblocks(Sig0)

# --------------------------
# Enumerate all 1-swap neighbors
# --------------------------
def swap_sets(sets, c1, i1, c2, i2):
    new = {ck: sets[ck][:] for ck in sets}
    new[c1][i1], new[c2][i2] = new[c2][i2], new[c1][i1]
    return new

neighbors_info = []
for a in range(C):
    ck1 = cond_keys[a]
    for b in range(a+1, C):
        ck2 = cond_keys[b]
        m1 = len(sets0[ck1]); m2 = len(sets0[ck2])
        for i1 in range(m1):
            for i2 in range(m2):
                Snew = swap_sets(sets0, ck1, i1, ck2, i2)
                yk, Rk, Sigk = state_vector(Snew)
                phi1_k = Phi1(Rk)
                phi2_k = Phi2_from_sigblocks(Sigk)
                neighbors_info.append({
                    "y": yk, "R_blocks": Rk, "Sig_blocks": Sigk,
                    "Phi1": phi1_k, "Phi2": phi2_k
                })

K = 1 + len(neighbors_info)
D = y0.shape[0]
Y = np.zeros((K, D)); Phi1_arr = np.zeros(K); Phi2_arr = np.zeros(K)
G1 = np.zeros((K, D)); G2 = np.zeros((K, D))

Y[0,:] = y0; Phi1_arr[0] = phi1_0; Phi2_arr[0] = phi2_0
g1_0, g2_0 = gradients_from_state(R0, Sig0)
G1[0,:] = g1_0; G2[0,:] = g2_0

for t, rec in enumerate(neighbors_info, start=1):
    Y[t,:] = rec["y"]
    Phi1_arr[t] = rec["Phi1"]
    Phi2_arr[t] = rec["Phi2"]
    g1_t, g2_t = gradients_from_state(rec["R_blocks"], rec["Sig_blocks"])
    G1[t,:] = g1_t; G2[t,:] = g2_t

# --------------------------
# PA2E axes and projection
# --------------------------
A1 = (G1.T @ G1) / K
A2 = (G2.T @ G2) / K
tr1 = float(np.trace(A1)); tr2 = float(np.trace(A2))
A1n = A1 / (tr1 if tr1>0 else 1.0)
A2n = A2 / (tr2 if tr2>0 else 1.0)

w1, V1 = np.linalg.eigh(A1n)
u1 = V1[:, np.argmax(w1)]
P_u1 = np.eye(D) - np.outer(u1, u1)
A2_def = P_u1 @ A2n @ P_u1
w2, V2 = np.linalg.eigh(A2_def)
u2 = V2[:, np.argmax(w2)]
U = np.stack([u1, u2], axis=1)  # D x 2

Y_mean = Y.mean(axis=0)
Z = (Y - Y_mean) @ U

# Diagnostics
rq1 = float(u1.T @ A1n @ u1)
rq2 = float(u2.T @ A2n @ u2)
clm_z = Z[0,:].copy()

# Feasible region (Phi1 <= Phi1(start)); boundary = within 1% relative
phi1_tau = float(Phi1_arr[0])
feasible = Phi1_arr <= phi1_tau + 1e-12
boundary = np.abs(Phi1_arr - phi1_tau) <= (0.01 * max(1.0, phi1_tau))

out = pd.DataFrame({
    "z1": Z[:,0],
    "z2": Z[:,1],
    "Phi1": Phi1_arr,
    "Phi2": Phi2_arr,
    "feasible_Phi1_le_tau": feasible.astype(int),
    "on_boundary_rel1pct": boundary.astype(int),
    "is_start": 0
})
out.loc[0, "is_start"] = 1

csv_path = "/mnt/data/PA2E_design_space.csv"
out.to_csv(csv_path, index=False)

from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("PA2E states (z, potentials, flags)", out)

summary = {
 "embedding_axes_shape": list(U.shape),
 "rayleigh_quotients": {"Phi1_axis": rq1, "Phi2_axis": rq2},
 "clm_point_z": {"z1": float(clm_z[0]), "z2": float(clm_z[1])},
 "phi1_tau": float(phi1_tau),
 "num_states": int(K),
 "num_feasible": int(feasible.sum()),
 "csv": f"[Download CSV]({csv_path})"
}
summary
# Plotly visualizations of the PA2E design space and potentials.
# - Loads the PA2E CSV produced earlier
# - Builds smoothed surfaces for Phi1 and Phi2 on the 2D embedding (z1,z2)
# - Plots: (A) two 3D surfaces; (B) 2D Phi1 boundary with Phi2 coloring; (C) "step" lines from CLM point to feasible Phi2-improving states.

import numpy as np, pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------
# Load the PA2E points
# ------------------------
df = pd.read_csv('/mnt/data/PA2E_design_space.csv')
z1 = df['z1'].to_numpy()
z2 = df['z2'].to_numpy()
Phi1 = df['Phi1'].to_numpy()
Phi2 = df['Phi2'].to_numpy()
feas = df['feasible_Phi1_le_tau'].to_numpy().astype(bool)
bound = df['on_boundary_rel1pct'].to_numpy().astype(bool)
is_start = df['is_start'].to_numpy().astype(bool)

# CLM start point and thresholds
z1_0 = float(z1[is_start][0])
z2_0 = float(z2[is_start][0])
Phi1_tau = float(Phi1[is_start][0])

# ------------------------
# Helper: smooth surfaces on a grid (parameter-free defaults)
# ------------------------
def fit_surface(x, y, v, grid_res=60):
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    Xg, Yg = np.meshgrid(xi, yi)
    # First try 'linear' interpolation; backfill NaNs with 'nearest'
    Vg = griddata((x, y), v, (Xg, Yg), method='linear')
    if np.isnan(Vg).any():
        Vg2 = griddata((x, y), v, (Xg, Yg), method='nearest')
        mask = np.isnan(Vg)
        Vg[mask] = Vg2[mask]
    return Xg, Yg, Vg

Xg, Yg, Phi1g = fit_surface(z1, z2, Phi1, grid_res=70)
_,  _, Phi2g = fit_surface(z1, z2, Phi2, grid_res=70)

# Compute a narrow boundary band for Phi1 == tau (using the grid surface)
# We mark grid cells where Phi1 crosses tau and extract an approximate contour by masking
eps = 1e-6
boundary_mask = np.abs(Phi1g - Phi1_tau) <= 0.01 * max(1.0, Phi1_tau)

# ------------------------
# Plot A: 3D surfaces (Phi1 and Phi2) over the 2D PA2E plane
# ------------------------
figA = make_subplots(
    rows=1, cols=2,
    specs=[[{'type':'surface'}, {'type':'surface'}]],
    subplot_titles=("Primary Potential Φ1 (mean residual energy)",
                    "Secondary Potential Φ2 (dispersion equalization)")
)

surf1 = go.Surface(x=Xg, y=Yg, z=Phi1g, colorbar=dict(title="Φ1"), showscale=True)
surf2 = go.Surface(x=Xg, y=Yg, z=Phi2g, colorbar=dict(title="Φ2"), showscale=True)

figA.add_trace(surf1, row=1, col=1)
figA.add_trace(surf2, row=1, col=2)

# Mark the CLM starting point on both surfaces
figA.add_trace(
    go.Scatter3d(x=[z1_0], y=[z2_0], z=[Phi1_tau],
                 mode='markers', marker=dict(size=4), name="CLM start"),
    row=1, col=1
)
phi2_at_start = float(df.loc[is_start, 'Phi2'].iloc[0])
figA.add_trace(
    go.Scatter3d(x=[z1_0], y=[z2_0], z=[phi2_at_start],
                 mode='markers', marker=dict(size=4), name="CLM start"),
    row=1, col=2
)

figA.update_scenes(xaxis_title="z1", yaxis_title="z2", zaxis_title="Potential")
figA.update_layout(title="Potential-Aligned 2D Space with 3D Potentials")

# ------------------------
# Plot B: 2D Φ1 boundary with Φ2 coloring
# ------------------------
# We'll render a contour line for Φ1 = tau and scatter all states colored by Φ2.
figB = go.Figure()

# Phi1 contour line (from grid surface)
figB.add_trace(go.Contour(
    x=Xg[0, :], y=Yg[:, 0], z=Phi1g,
    contours=dict(start=Phi1_tau, end=Phi1_tau, size=1e-9, showlabels=False),
    showscale=False,
    line=dict(width=2),
    name="Φ1 = τ boundary"
))

# Scatter of all states colored by Φ2
figB.add_trace(go.Scattergl(
    x=z1, y=z2, mode='markers',
    marker=dict(size=5, color=Phi2, showscale=True, colorbar=dict(title="Φ2")),
    name="States (colored by Φ2)",
    text=[f"Φ1={p1:.3f}, Φ2={p2:.3f}" for p1, p2 in zip(Phi1, Phi2)],
    hovertemplate="z1=%{x:.3f}<br>z2=%{y:.3f}<br>%{text}<extra></extra>"
))

# Highlight feasible region (optional overlay of feasible points with open markers)
figB.add_trace(go.Scattergl(
    x=z1[feas], y=z2[feas], mode='markers',
    marker=dict(size=6, symbol='circle-open'),
    name="Feasible Φ1 ≤ τ"
))

# Mark start and best feasible point (min Φ2 among feasible)
idx_best_feas = np.where(feas)[0][np.argmin(Phi2[feas])] if feas.any() else None
if idx_best_feas is not None:
    figB.add_trace(go.Scattergl(
        x=[z1[idx_best_feas]], y=[z2[idx_best_feas]], mode='markers',
        marker=dict(size=9, symbol='x'),
        name="Min Φ2 on feasible"
    ))

figB.add_trace(go.Scattergl(
    x=[z1_0], y=[z2_0], mode='markers',
    marker=dict(size=10),
    name="CLM start"
))

figB.update_layout(
    title="Feasible Boundary (Φ1 = τ) and Dispersion Shape (Φ2)",
    xaxis_title="z1", yaxis_title="z2"
)

# ------------------------
# Plot C: "Possible steps" from CLM start to feasible Φ2-improving states
# We draw line segments from the start to a subset of feasible states with ΔΦ2 < 0.
# To avoid clutter, take up to N_best with largest Φ2 decrease.
# ------------------------
idx_all = np.arange(len(df))
improving = feas & (Phi2 < phi2_at_start)
idx_imp = idx_all[improving]
# rank by Φ2 decrease
dphi2 = phi2_at_start - Phi2[idx_imp]
order = np.argsort(-dphi2)
N_best = int(min(50, len(idx_imp)))  # cap at 50 segments
idx_sel = idx_imp[order[:N_best]] if N_best > 0 else np.array([], dtype=int)

figC = go.Figure()
# Base scatter of feasible states
figC.add_trace(go.Scattergl(
    x=z1, y=z2, mode='markers',
    marker=dict(size=4),
    name="All states"
))
figC.add_trace(go.Scattergl(
    x=z1[feas], y=z2[feas], mode='markers',
    marker=dict(size=6, symbol='circle-open'),
    name="Feasible Φ1 ≤ τ"
))
# Start point
figC.add_trace(go.Scattergl(
    x=[z1_0], y=[z2_0], mode='markers',
    marker=dict(size=10),
    name="CLM start"
))

# Lines from start to selected improving feasible states
for j in idx_sel:
    figC.add_trace(go.Scattergl(
        x=[z1_0, z1[j]], y=[z2_0, z2[j]], mode='lines',
        name="Step", showlegend=False
    ))

# Mark selected endpoints with Φ2 annotations
if len(idx_sel) > 0:
    figC.add_trace(go.Scattergl(
        x=z1[idx_sel], y=z2[idx_sel], mode='markers+text',
        marker=dict(size=7),
        text=[f"ΔΦ2={phi2_at_start - Phi2[j]:.3f}" for j in idx_sel],
        textposition='top center',
        name="Selected improvements"
    ))

figC.update_layout(
    title="Candidate Feasible Steps from CLM Start (ΔΦ2 < 0)",
    xaxis_title="z1", yaxis_title="z2"
)

# Display figures
figA.show()
figB.show()
figC.show()
