#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active-learning rankings for pairwise (binary) face comparisons per dimension.

This version keeps ONLY the best-performing selection family from the synthetic benchmarks:
  ✅ Greedy D-optimal / expected information gain, with SEQUENTIAL covariance updates

Key tweaks vs the earlier draft:
1) Uses an *expected Fisher weight* w = p(1-p) so comparisons that are already
   (predictively) near-certain naturally get ~0 value, without ad-hoc “bad data” checks.
2) Uses an O(N^2) Sherman–Morrison/Kalman update for the covariance P after each selected edge,
   so we can produce a full ranking (a permutation) efficiently.
3) Saves continuous per-edge scores (step-0 + gain-at-selection + cumulative gain),
   and also saves NxN “ranking matrices” (score and greedy step) per dimension.

Outputs per dimension:
- images.csv                           (index -> image_id)
- posterior_mean.npy                   (BT MAP)
- posterior_precision.npy              (Laplace Hessian)
- posterior_cov.npy                    (inverse Hessian)
- edges_step0.csv                      (all pairs + continuous step-0 scores)
- edges_greedy_ranking.csv             (adds greedy step, selection gains, cumulative gains)
- matrix_step0_score.csv / .npy        (NxN matrix of step-0 scores)
- matrix_greedy_step.csv / .npy        (NxN matrix of greedy steps; 0 = not selected in max_steps)

Also:
- comparisons_combined.csv
- edges_all_dimensions.csv             (stack of edges_greedy_ranking across dims)

Run:
  python active_rankings_best.py --sona apka_sona_2x60_twarzy__fixed.csv --ariadna apka_ariadna_2x75_par_twarzy__fixed.csv --out-dir out_active

"""

from __future__ import annotations

import argparse
import ast
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import erf


# =============================================================================
# CONFIG: label aliasing (carry-over)
# =============================================================================

ALIAS_LABEL_MAP = {
    "niegruba": ("gruba", -1),
    "szczupła": ("gruba", -1),

    "niekonserwatywna": ("konserwatywna", -1),
    "niemęska": ("męska", -1),

    "niestara": ("stara", -1),
    "młoda": ("stara", -1),

    "niezarozumiała": ("zarozumiała", -1),
    "skromna": ("zarozumiała", -1),

    "niepochodząca z Bliskiego Wschodu": ("pochodząca z Bliskiego Wschodu", -1),
    "niepochodzącą z Bliskiego Wschodu": ("pochodząca z Bliskiego Wschodu", -1),

    "nieinteligentna (smart)": ("inteligentna (smart)", -1),
    "nieurocza (cute)": ("urocza (cute)", -1),
}


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s)) or "dimension"


# =============================================================================
# PARSING HELPERS (same logic as your Elo script)
# =============================================================================

def collect_pairs(node, out: List[Tuple[str, str, int, str, int]]):
    if isinstance(node, list):
        if (
            len(node) == 2
            and all(isinstance(x, list) and len(x) == 3 for x in node)
            and isinstance(node[0][0], str)
            and isinstance(node[1][0], str)
        ):
            a, b = node
            feature_a, img_a, bin_a = a[0].strip(), str(a[1]).strip(), int(a[2])
            feature_b, img_b, bin_b = b[0].strip(), str(b[1]).strip(), int(b[2])
            out.append((feature_a, img_a, bin_a, img_b, bin_b))
        else:
            for x in node:
                collect_pairs(x, out)


def parse_ratings_cell(s) -> List[Tuple[str, str, int, str, int]]:
    if pd.isna(s):
        return []
    try:
        obj = ast.literal_eval(str(s))
    except Exception:
        return []
    out: List[Tuple[str, str, int, str, int]] = []
    collect_pairs(obj, out)
    return out


def build_pos_neg_mapping(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    rows = df[df["dims"].notna() & df["dims_negated"].notna()]
    if rows.empty:
        raise ValueError("No rows with non-null dims/dims_negated found.")
    dims_pos = ast.literal_eval(rows.iloc[0]["dims"])
    dims_neg = ast.literal_eval(rows.iloc[0]["dims_negated"])

    pairs = []
    seen = set()
    for p, n in zip(dims_pos, dims_neg):
        if (p, n) not in seen:
            seen.add((p, n))
            pairs.append((p, n))
    pos_to_neg = dict(pairs)
    neg_to_pos = {n: p for p, n in pairs}
    return pos_to_neg, neg_to_pos


def resolve_label_to_dimension_and_sign(raw_label: str, pos_to_neg: dict, neg_to_pos: dict):
    raw_label = raw_label.strip()
    if raw_label in pos_to_neg:
        return raw_label, +1
    if raw_label in neg_to_pos:
        return neg_to_pos[raw_label], -1
    if raw_label in ALIAS_LABEL_MAP:
        return ALIAS_LABEL_MAP[raw_label]
    return raw_label, +1


def build_comparisons_for_dataset(
    csv_path: Path,
    dataset_id: str,
    id_col: str = "ResponseId",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    pos_to_neg, neg_to_pos = build_pos_neg_mapping(df)

    records = []
    for row_idx, row in df.iterrows():
        respondent = row.get(id_col, row_idx)

        for col in ("ratings_list", "ratings_list2"):
            if col not in df.columns:
                continue
            comps = parse_ratings_cell(row.get(col))
            for feature, img_a, bin_a, img_b, bin_b in comps:
                base_dim, sign = resolve_label_to_dimension_and_sign(
                    feature, pos_to_neg=pos_to_neg, neg_to_pos=neg_to_pos
                )

                if img_a == img_b:
                    continue

                # bin_a/bin_b: who is "more raw_label"
                if bin_a == 1 and bin_b == 0:
                    winner = "A" if sign == +1 else "B"
                elif bin_a == 0 and bin_b == 1:
                    winner = "B" if sign == +1 else "A"
                else:
                    continue

                records.append(
                    dict(
                        dataset_id=dataset_id,
                        respondent_id=respondent,
                        dimension=base_dim,  # canonical positive pole
                        image_A=str(img_a),
                        image_B=str(img_b),
                        winner=winner,        # relative to positive pole
                        raw_label=feature,
                        raw_col=col,
                    )
                )

    return pd.DataFrame.from_records(records)


# =============================================================================
# BT (Bradley–Terry) Laplace posterior: MAP + precision/cov
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def phi(z: np.ndarray) -> np.ndarray:
    # standard normal CDF using erf
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


@dataclass
class BTLaplaceConfig:
    alpha: float = 1.0      # ridge strength (lambda)
    max_iter: int = 100
    tol: float = 1e-8


def fit_bt_laplace(left: np.ndarray, right: np.ndarray, y: np.ndarray, cfg: BTLaplaceConfig):
    """
    Penalized Bradley–Terry:
      p(y=1) = sigmoid(theta_left - theta_right)
    Returns (images_index, theta_map, precision_H, cov_P)
    """
    images = pd.Index(sorted(set(left).union(set(right))))
    n = len(images)
    idx = {img: i for i, img in enumerate(images)}

    a = np.array([idx[x] for x in left], dtype=int)
    b = np.array([idx[x] for x in right], dtype=int)

    lam = float(cfg.alpha)
    theta = np.zeros(n, dtype=float)

    for _ in range(cfg.max_iter):
        eta = theta[a] - theta[b]
        p = sigmoid(eta)
        w = p * (1.0 - p)
        r = (y - p)

        g = np.zeros(n, dtype=float)
        np.add.at(g, a, r)
        np.add.at(g, b, -r)
        g -= lam * theta

        # Hessian is weighted Laplacian + lam I
        H = np.zeros((n, n), dtype=float)
        np.add.at(H, (a, a), w)
        np.add.at(H, (b, b), w)
        np.add.at(H, (a, b), -w)
        np.add.at(H, (b, a), -w)
        H[np.diag_indices(n)] += lam

        step = np.linalg.solve(H, g)
        theta_new = theta + step
        if np.max(np.abs(step)) < cfg.tol:
            theta = theta_new
            break
        theta = theta_new

    # final precision at MAP
    eta = theta[a] - theta[b]
    p = sigmoid(eta)
    w = p * (1.0 - p)

    H = np.zeros((n, n), dtype=float)
    np.add.at(H, (a, a), w)
    np.add.at(H, (b, b), w)
    np.add.at(H, (a, b), -w)
    np.add.at(H, (b, a), -w)
    H[np.diag_indices(n)] += lam

    P = np.linalg.inv(H)

    # center theta (harmless with ridge; nice for interpretability)
    theta = theta - float(theta.mean())

    return images, theta, H, P


# =============================================================================
# Greedy sequential design (best-performing in the benchmark)
# =============================================================================

@dataclass
class GreedyDesignConfig:
    R: float = 1.0              # "measurement noise" scaling; acts like inverse per-trial precision
    p_clip: float = 1e-6        # numeric clamp for p
    max_steps: int = 0          # 0 => rank ALL candidate edges
    min_p: float = 0.0          # optional hard discard: keep edges with p in [min_p, 1-min_p] (0 disables)
    degree_tiebreak: float = 0.0  # optional mild penalty for already-covered nodes (0 disables)


def diff_variance(P: np.ndarray, i: int, j: int) -> float:
    return float(P[i, i] + P[j, j] - 2.0 * P[i, j])


def kalman_cov_update(P: np.ndarray, i: int, j: int, scale: float):
    """
    Precision update: Lambda <- Lambda + scale * h h^T  with h=(e_i-e_j).
    Cov update (Sherman–Morrison): P <- P - (scale/(1+scale*s)) * (P h)(P h)^T
    where s = h^T P h.
    """
    # Ph = P[:, i] - P[:, j]
    Ph = P[:, i] - P[:, j]
    s = float(Ph[i] - Ph[j])  # equals h^T P h (same as diff_variance)
    denom = 1.0 + scale * s
    if denom <= 0:
        return P, s
    coeff = scale / denom
    P = P - coeff * np.outer(Ph, Ph)
    return P, s


def greedy_rank_edges(
    m: np.ndarray,
    P0: np.ndarray,
    edges_i: np.ndarray,
    edges_j: np.ndarray,
    cfg: GreedyDesignConfig,
):
    """
    Produce a full ranking (permutation) of edges by sequential expected Δlogdet gain.

    Score for candidate edge e=(i,j) at step t:
      s_t = (e_i-e_j)^T P_t (e_i-e_j)
      p_t = Phi( mu / sqrt(s_t + R) )   with mu = (m_i - m_j) fixed
      w_t = p_t (1-p_t)                 expected Fisher weight (certainty downweights itself)
      scale = (w_t / R)
      gain = 0.5 * log(1 + scale * s_t) = marginal Δ(0.5 logdet)
    """
    R = float(cfg.R)
    E = len(edges_i)
    n = len(m)

    mu = m[edges_i] - m[edges_j]  # fixed current mean differences

    # storage
    step = np.zeros(E, dtype=int)             # 0 = not selected within max_steps
    gain_at = np.zeros(E, dtype=float)
    cum_gain_at = np.zeros(E, dtype=float)
    s_at = np.zeros(E, dtype=float)
    p_at = np.zeros(E, dtype=float)
    w_at = np.zeros(E, dtype=float)

    remaining = np.ones(E, dtype=bool)
    P = P0.copy()
    deg = np.zeros(n, dtype=int)

    max_steps = int(cfg.max_steps) if int(cfg.max_steps) > 0 else E
    cum = 0.0

    for t in range(1, max_steps + 1):
        # compute per-edge s under current P (cheap O(E))
        s = (np.diag(P)[edges_i] + np.diag(P)[edges_j] - 2.0 * P[edges_i, edges_j]).astype(float)

        # predictive probit p based on current uncertainty
        # p = Phi(mu / sqrt(s + R))
        denom = np.sqrt(np.maximum(s + R, 1e-12))
        p = phi(mu / denom)
        p = np.clip(p, cfg.p_clip, 1.0 - cfg.p_clip)

        # optional hard discard for too-certain edges
        if cfg.min_p > 0.0:
            ok = (p >= cfg.min_p) & (p <= 1.0 - cfg.min_p)
        else:
            ok = np.ones(E, dtype=bool)

        w = p * (1.0 - p)
        scale = (w / R)  # precision increment factor

        gain = 0.5 * np.log1p(scale * s)

        # optional degree-based tie-breaker: downweight edges whose nodes already covered
        if cfg.degree_tiebreak > 0.0:
            penalty = 1.0 + cfg.degree_tiebreak * (deg[edges_i] + deg[edges_j])
            gain = gain / penalty

        # candidate mask
        cand = remaining & ok
        if not np.any(cand):
            break

        # select best remaining edge
        e_star = int(np.argmax(np.where(cand, gain, -np.inf)))
        best_gain = float(gain[e_star])

        # stop if marginal gain is essentially zero
        if not np.isfinite(best_gain) or best_gain <= 0.0:
            break

        i = int(edges_i[e_star])
        j = int(edges_j[e_star])

        # update covariance with scale_star = w_star/R
        scale_star = float((w[e_star]) / R)
        P, s_star = kalman_cov_update(P, i, j, scale_star)

        # record
        step[e_star] = t
        gain_at[e_star] = best_gain
        cum += best_gain
        cum_gain_at[e_star] = cum
        s_at[e_star] = float(s_star)
        p_at[e_star] = float(p[e_star])
        w_at[e_star] = float(w[e_star])

        remaining[e_star] = False
        deg[i] += 1
        deg[j] += 1

    return dict(
        mu=mu,
        greedy_step=step,
        greedy_gain=gain_at,
        greedy_cum_gain=cum_gain_at,
        greedy_s_at=s_at,
        greedy_p_at=p_at,
        greedy_w_at=w_at,
    )


# =============================================================================
# I/O helpers: candidate edges, observed counts, matrices
# =============================================================================

def all_pairs(n: int):
    i, j = np.triu_indices(n, k=1)
    return i.astype(int), j.astype(int)


def observed_pair_counts(images: pd.Index, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    idx = {img: i for i, img in enumerate(images)}
    a = np.array([idx[x] for x in left], dtype=int)
    b = np.array([idx[x] for x in right], dtype=int)
    u = np.minimum(a, b)
    v = np.maximum(a, b)
    c = Counter(zip(u.tolist(), v.tolist()))
    n = len(images)
    ei, ej = all_pairs(n)
    return np.array([c.get((int(ei[k]), int(ej[k])), 0) for k in range(len(ei))], dtype=int)


def make_matrix_from_edges(n: int, ei: np.ndarray, ej: np.ndarray, values: np.ndarray, fill: float = 0.0):
    M = np.full((n, n), fill, dtype=float)
    M[ei, ej] = values
    M[ej, ei] = values
    np.fill_diagonal(M, fill)
    return M


# =============================================================================
# Per-dimension runner
# =============================================================================

def process_dimension(
    dim: str,
    df_dim: pd.DataFrame,
    out_dir: Path,
    bt_cfg: BTLaplaceConfig,
    gd_cfg: GreedyDesignConfig,
):
    # convert comparisons to BT format
    # winner indicates which is higher on positive pole
    left = df_dim["image_A"].astype(str).values
    right = df_dim["image_B"].astype(str).values
    y = (df_dim["winner"].values == "A").astype(int)

    images, theta, H, P = fit_bt_laplace(left, right, y, bt_cfg)
    n = len(images)

    ei, ej = all_pairs(n)

    # step-0 stats (continuous)
    mu0 = theta[ei] - theta[ej]
    s0 = np.array([diff_variance(P, int(ei[k]), int(ej[k])) for k in range(len(ei))], dtype=float)

    R = float(gd_cfg.R)
    p0 = phi(mu0 / np.sqrt(np.maximum(s0 + R, 1e-12)))
    p0 = np.clip(p0, gd_cfg.p_clip, 1.0 - gd_cfg.p_clip)
    w0 = p0 * (1.0 - p0)
    gain0 = 0.5 * np.log1p((w0 / R) * s0)

    obs_counts = observed_pair_counts(images, left, right)

    dim_dir = out_dir / "dimensions" / safe_filename(dim)
    dim_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"image_index": np.arange(n), "image": images.astype(str)}).to_csv(dim_dir / "images.csv", index=False)
    np.save(dim_dir / "posterior_mean.npy", theta)
    np.save(dim_dir / "posterior_precision.npy", H)
    np.save(dim_dir / "posterior_cov.npy", P)

    edges0 = pd.DataFrame({
        "dimension": dim,
        "edge_id": np.arange(len(ei), dtype=int),
        "i": ei,
        "j": ej,
        "image_i": images[ei].astype(str),
        "image_j": images[ej].astype(str),
        "mu_step0": mu0,
        "s_step0": s0,
        "p_step0": p0,
        "w_step0": w0,
        "gain_step0": gain0,
        "observed_pair_count": obs_counts,
    }).sort_values("gain_step0", ascending=False).reset_index(drop=True)
    edges0.to_csv(dim_dir / "edges_step0.csv", index=False)

    # Greedy sequential ranking (best-performing)
    greedy = greedy_rank_edges(theta, P, ei, ej, gd_cfg)

    edges_rank = pd.DataFrame({
        "dimension": dim,
        "edge_id": np.arange(len(ei), dtype=int),
        "i": ei,
        "j": ej,
        "image_i": images[ei].astype(str),
        "image_j": images[ej].astype(str),

        "mu_step0": mu0,
        "s_step0": s0,
        "p_step0": p0,
        "w_step0": w0,
        "gain_step0": gain0,

        "observed_pair_count": obs_counts,

        "greedy_step": greedy["greedy_step"],
        "greedy_gain_at_selection": greedy["greedy_gain"],
        "greedy_cum_gain_at_selection": greedy["greedy_cum_gain"],
        "greedy_s_at_selection": greedy["greedy_s_at"],
        "greedy_p_at_selection": greedy["greedy_p_at"],
        "greedy_w_at_selection": greedy["greedy_w_at"],
    })

    # Sort by selected step first (ascending), then by step-0 gain
    edges_rank["selected"] = edges_rank["greedy_step"] > 0
    edges_rank = edges_rank.sort_values(
        by=["selected", "greedy_step", "gain_step0"],
        ascending=[False, True, False],
    ).drop(columns=["selected"]).reset_index(drop=True)

    edges_rank.to_csv(dim_dir / "edges_greedy_ranking.csv", index=False)

    # Save NxN ranking matrices
    M_gain0 = make_matrix_from_edges(n, ei, ej, gain0, fill=0.0)
    M_step = make_matrix_from_edges(n, ei, ej, edges_rank.set_index("edge_id").loc[np.arange(len(ei)), "greedy_step"].values, fill=0.0)

    np.save(dim_dir / "matrix_step0_score.npy", M_gain0)
    np.save(dim_dir / "matrix_greedy_step.npy", M_step)

    pd.DataFrame(M_gain0, index=images.astype(str), columns=images.astype(str)).to_csv(dim_dir / "matrix_step0_score.csv")
    pd.DataFrame(M_step, index=images.astype(str), columns=images.astype(str)).to_csv(dim_dir / "matrix_greedy_step.csv")

    return edges_rank


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="out_active_rankings")

    ap.add_argument("--sona", type=str, default="apka_sona_2x60_twarzy__fixed.csv", help="Path to SONA CSV.")
    ap.add_argument("--ariadna", type=str, default="apka_ariadna_2x75_par_twarzy__fixed.csv", help="Path to ARIADNA CSV.")
    # BT posterior config
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--bt-max-iter", type=int, default=100)
    ap.add_argument("--bt-tol", type=float, default=1e-8)

    # Greedy design config
    ap.add_argument("--R", type=float, default=1.0, help="Noise scaling in gain: gain=0.5*log(1+(w/R)*s)")
    ap.add_argument("--max-steps", type=int, default=0, help="0 => rank all pairs; else stop after K selections")
    ap.add_argument("--min-p", type=float, default=0.0, help="Optional hard discard: keep p in [min_p,1-min_p]. 0 disables.")
    ap.add_argument("--degree-tiebreak", type=float, default=0.0, help="Optional penalty weight to spread edges over nodes.")
    ap.add_argument("--p-clip", type=float, default=1e-6)

    ap.add_argument("--limit-dims", type=int, default=0, help="0 => all; else first K dims alphabetically")
    return ap.parse_args()


def main():
    args = parse_args()

    parts = []
    if args.sona:
        parts.append(build_comparisons_for_dataset(Path(args.sona), dataset_id="sona"))
    if args.ariadna:
        parts.append(build_comparisons_for_dataset(Path(args.ariadna), dataset_id="ariadna"))
    if not parts:
        raise ValueError("Provide at least one dataset via --sona and/or --ariadna.")

    comparisons = pd.concat(parts, ignore_index=True)
    if comparisons.empty:
        raise ValueError("No comparisons built.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparisons.to_csv(out_dir / "comparisons_combined.csv", index=False)

    bt_cfg = BTLaplaceConfig(alpha=args.alpha, max_iter=args.bt_max_iter, tol=args.bt_tol)
    gd_cfg = GreedyDesignConfig(
        R=args.R,
        p_clip=args.p_clip,
        max_steps=args.max_steps,
        min_p=args.min_p,
        degree_tiebreak=args.degree_tiebreak,
    )

    dims = sorted(comparisons["dimension"].unique())
    if args.limit_dims and args.limit_dims > 0:
        dims = dims[: int(args.limit_dims)]

    all_edges = []
    for dim in dims:
        df_dim = comparisons[comparisons["dimension"] == dim].copy()
        edges_rank = process_dimension(dim, df_dim, out_dir, bt_cfg, gd_cfg)
        all_edges.append(edges_rank)
        print(f"[{dim}] saved -> {out_dir / 'dimensions' / safe_filename(dim)}")

    edges_all = pd.concat(all_edges, ignore_index=True)
    edges_all.to_csv(out_dir / "edges_all_dimensions.csv", index=False)
    print(f"Saved: {out_dir / 'edges_all_dimensions.csv'}")
    print("Use greedy_step as a ready-to-cut ranking (take first K).")
    print("Use gain_step0 or greedy_gain_at_selection for continuous cutoffs.")


if __name__ == "__main__":
    main()
