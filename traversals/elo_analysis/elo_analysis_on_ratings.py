#!/usr/bin/env python3
"""
Pairwise face-attribute analysis with a robust Bradley–Terry (BTL) fit
(+ optional Elo baseline), designed for binary choices (A vs B) across
many participants and multiple dimensions.

Key outputs:
- Per-dimension image scores on a BT latent scale (theta)
- Elo-like rescaled score (optional, just a monotone transform)
- Comparison counts
- Graph connectivity diagnostics (components per dimension)
- Optional respondent-cluster bootstrap CIs

Assumptions (as in your current pipeline):
- ratings_list / ratings_list2 contain nested list structures parseable via ast.literal_eval
- Each comparison yields exactly one winner (A or B) after your bin logic
- dims / dims_negated define poles for label sign resolution

Usage examples:
  python bt_faces.py \
    --sona apka_sona_2x60_twarzy__fixed.csv \
    --ariadna apka_ariadna_2x75_par_twarzy__fixed.csv \
    --out bt_all_dimensions.csv \
    --bootstrap 300

If you want only BT (recommended), leave --elo disabled.
If you also want Elo, pass --elo and choose K/epochs.
"""

from __future__ import annotations

import argparse
import ast
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# sklearn/scipy are commonly available; if not, the script will give a clear error.
try:
    from sklearn.linear_model import LogisticRegression
except Exception as e:
    raise ImportError(
        "This script requires scikit-learn (sklearn). Install it with:\n"
        "  pip install scikit-learn\n"
        f"Original import error: {e}"
    )

try:
    from scipy import sparse
except Exception:
    sparse = None  # fallback to dense if scipy isn't available


# =============================================================================
# CONFIG (your alias map kept as-is)
# =============================================================================

IMAGE_URL_PREFIX_DEFAULT = "https://langtorch.org/static/gandalf/"

ALIAS_LABEL_MAP = {
    # Body size dimension: canonical label 'gruba'
    "niegruba": ("gruba", -1),
    "szczupła": ("gruba", -1),

    # Conservatism dimension: canonical positive 'konserwatywna'
    "niekonserwatywna": ("konserwatywna", -1),

    # Masculinity dimension: canonical positive 'męska'
    "niemęska": ("męska", -1),

    # Age dimension: canonical positive 'stara'
    "niestara": ("stara", -1),
    "młoda": ("stara", -1),

    # Arrogance / modesty dimension: canonical positive 'zarozumiała'
    "niezarozumiała": ("zarozumiała", -1),
    "skromna": ("zarozumiała", -1),

    # Middle-East origin
    "niepochodząca z Bliskiego Wschodu":  ("pochodząca z Bliskiego Wschodu", -1),
    "niepochodzącą z Bliskiego Wschodu": ("pochodząca z Bliskiego Wschodu", -1),

    # Mis-typed negations in ratings data
    "nieinteligentna (smart)": ("inteligentna (smart)", -1),
    "nieurocza (cute)": ("urocza (cute)", -1),
}


# =============================================================================
# PARSING HELPERS
# =============================================================================

def collect_pairs(node, out: List[Tuple[str, str, int, str, int]]):
    """
    Recursively traverse a nested list 'node' and append tuples:
    (feature_label, image_a, bin_a, image_b, bin_b).
    """
    if isinstance(node, list):
        # Direct comparison: [[feature, img_a, bin_a],[feature, img_b, bin_b]]
        if (
            len(node) == 2
            and all(isinstance(x, list) and len(x) == 3 for x in node)
            and isinstance(node[0][0], str)
            and isinstance(node[1][0], str)
        ):
            a, b = node
            feature_a, img_a, bin_a = a[0].strip(), str(a[1]).strip(), int(a[2])
            feature_b, img_b, bin_b = b[0].strip(), str(b[1]).strip(), int(b[2])
            # If they ever differ, we could reconcile; for now we follow the first.
            feature = feature_a
            out.append((feature, img_a, bin_a, img_b, bin_b))
        else:
            for x in node:
                collect_pairs(x, out)


def parse_ratings_cell(s) -> List[Tuple[str, str, int, str, int]]:
    """
    Parse one Qualtrics ratings_list / ratings_list2 cell into:
      list of (feature, img_a, bin_a, img_b, bin_b) tuples.
    """
    if pd.isna(s):
        return []
    try:
        obj = ast.literal_eval(str(s))
    except Exception:
        return []
    out: List[Tuple[str, str, int, str, int]] = []
    collect_pairs(obj, out)
    return out


# =============================================================================
# POS <-> NEG MAPS FROM dims / dims_negated
# =============================================================================

def build_pos_neg_mapping(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Uses 'dims' and 'dims_negated' columns to construct:
      pos_to_neg: {positive_label -> negative_label}
      neg_to_pos: {negative_label -> positive_label}

    Additionally guards against inconsistent pairings within the file.
    """
    rows = df[df["dims"].notna() & df["dims_negated"].notna()]
    if rows.empty:
        raise ValueError("No rows with non-null dims and dims_negated found.")

    # Use first row as reference
    ref_pos = ast.literal_eval(rows.iloc[0]["dims"])
    ref_neg = ast.literal_eval(rows.iloc[0]["dims_negated"])
    if len(ref_pos) != len(ref_neg):
        raise ValueError("dims and dims_negated lengths differ in reference row.")

    ref_pairs = list(zip(ref_pos, ref_neg))

    # Check that any other non-null row agrees with the reference mapping
    for i in range(1, len(rows)):
        pos_i = ast.literal_eval(rows.iloc[i]["dims"])
        neg_i = ast.literal_eval(rows.iloc[i]["dims_negated"])
        if len(pos_i) != len(neg_i):
            raise ValueError(f"dims and dims_negated lengths differ at row {rows.index[i]}.")
        pairs_i = list(zip(pos_i, neg_i))
        if set(pairs_i) != set(ref_pairs):
            raise ValueError(
                "Inconsistent dims/dims_negated mapping across rows. "
                "This script expects a stable mapping within each dataset."
            )

    # Deduplicate while preserving first occurrence order
    seen = set()
    pairs_unique = []
    for p, n in ref_pairs:
        if (p, n) not in seen:
            seen.add((p, n))
            pairs_unique.append((p, n))

    pos_to_neg = dict(pairs_unique)
    # Guard: no two positives map to different negatives, etc.
    if len(pos_to_neg) != len(pairs_unique):
        raise ValueError("Duplicate positive labels with conflicting negations detected.")

    neg_to_pos = {}
    for p, n in pairs_unique:
        if n in neg_to_pos and neg_to_pos[n] != p:
            raise ValueError("Duplicate negative labels mapping to different positives detected.")
        neg_to_pos[n] = p

    return pos_to_neg, neg_to_pos


def resolve_label_to_dimension_and_sign(
    raw_label: str,
    pos_to_neg: Dict[str, str],
    neg_to_pos: Dict[str, str],
) -> Tuple[str, int]:
    """
    Map a raw feature label to:
      base_dim = canonical positive dimension label
      sign     = +1 if raw_label is positive, -1 if negative
    """
    raw_label = raw_label.strip()

    if raw_label in pos_to_neg:
        return raw_label, +1

    if raw_label in neg_to_pos:
        return neg_to_pos[raw_label], -1

    if raw_label in ALIAS_LABEL_MAP:
        base_dim, sign = ALIAS_LABEL_MAP[raw_label]
        return base_dim, sign

    # Keep behavior similar to your pipeline: treat as its own positive dimension
    return raw_label, +1


# =============================================================================
# BUILD STANDARDIZED COMPARISONS
# =============================================================================

def build_comparisons_for_dataset(
    csv_path: Path,
    dataset_id: str,
    id_col: str = "ResponseId",
) -> pd.DataFrame:
    """
    Standardized comparisons:

      dataset_id, respondent_id, dimension,
      image_A, image_B,
      winner ('A'/'B' means higher on positive pole),
      raw_label, raw_col
    """
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
                    feature,
                    pos_to_neg=pos_to_neg,
                    neg_to_pos=neg_to_pos,
                )

                # bin_a/bin_b tell us who is "more raw_label"
                if bin_a == 1 and bin_b == 0:
                    winner = "A" if sign == +1 else "B"
                elif bin_a == 0 and bin_b == 1:
                    winner = "B" if sign == +1 else "A"
                else:
                    continue  # ties / invalid encoded states

                if img_a == img_b:
                    continue  # ignore degenerate self-comparisons

                records.append(
                    dict(
                        dataset_id=dataset_id,
                        respondent_id=respondent,
                        dimension=base_dim,
                        image_A=img_a,
                        image_B=img_b,
                        winner=winner,
                        raw_label=feature,
                        raw_col=col,
                    )
                )

    return pd.DataFrame.from_records(records)


# =============================================================================
# CONNECTIVITY (Union-Find)
# =============================================================================

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self) -> Dict[str, int]:
        # Map root -> component id (0..k-1)
        roots = {}
        comp_id = 0
        out = {}
        for x in self.parent:
            r = self.find(x)
            if r not in roots:
                roots[r] = comp_id
                comp_id += 1
            out[x] = roots[r]
        return out


def dimension_connectivity(df_dim: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, int]]:
    """
    Returns:
      node_to_component: image -> component_id
      component_sizes: component_id -> size
    """
    uf = UnionFind()
    # ensure all nodes appear
    for img in pd.concat([df_dim["image_A"], df_dim["image_B"]]).unique():
        uf.find(img)

    for a, b in zip(df_dim["image_A"].values, df_dim["image_B"].values):
        uf.union(a, b)

    node_to_comp = uf.components()
    sizes = Counter(node_to_comp.values())
    return node_to_comp, dict(sizes)


# =============================================================================
# ELO (optional baseline)
# =============================================================================

def elo_ratings(
    matches: List[Tuple[str, str]],
    K: float = 32.0,
    base: float = 1500.0,
    epochs: int = 5,
    seed: int = 123,
) -> Tuple[Dict[str, float], Counter]:
    """
    Elo over winner/loser pairs.
    NOTE: counts reflect true match appearances (NOT multiplied by epochs).
    """
    rng = random.Random(seed)
    ratings = defaultdict(lambda: base)

    counts = Counter()
    for w, l in matches:
        counts[w] += 1
        counts[l] += 1

    for _ in range(epochs):
        order = matches[:]
        rng.shuffle(order)
        for winner, loser in order:
            ra = ratings[winner]
            rb = ratings[loser]
            ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            ratings[winner] = ra + K * (1 - ea)
            ratings[loser]  = rb + K * (0 - (1 - ea))

    return dict(ratings), counts


def compute_elo_for_dimension(
    comparisons_df: pd.DataFrame,
    dimension: str,
    K: float = 32.0,
    base: float = 1500.0,
    epochs: int = 5,
    seed: int = 123,
) -> pd.DataFrame:
    subset = comparisons_df[comparisons_df["dimension"] == dimension].copy()
    if subset.empty:
        raise ValueError(f"No comparisons for dimension: {dimension}")

    matches = []
    for _, row in subset.iterrows():
        if row["winner"] == "A":
            winner_img, loser_img = row["image_A"], row["image_B"]
        else:
            winner_img, loser_img = row["image_B"], row["image_A"]
        matches.append((winner_img, loser_img))

    ratings, counts = elo_ratings(matches, K=K, base=base, epochs=epochs, seed=seed)

    out = pd.DataFrame(
        [
            dict(dimension=dimension, image=img, elo=score, comparisons=counts.get(img, 0))
            for img, score in ratings.items()
        ]
    ).sort_values("elo", ascending=False).reset_index(drop=True)

    return out


# =============================================================================
# BRADLEY–TERRY (robust core)
# =============================================================================

@dataclass
class BTConfig:
    # L2 regularization strength alpha (bigger = more shrinkage).
    # sklearn uses C = 1/alpha
    alpha: float = 1.0
    max_iter: int = 5000
    tol: float = 1e-7
    random_state: int = 0


def _build_design_matrix(
    left: np.ndarray,
    right: np.ndarray,
    image_to_idx: Dict[str, int],
):
    """
    Build X where each row has +1 at left image, -1 at right image.
    Returns either scipy CSR or dense ndarray.
    """
    n = len(left)
    p = len(image_to_idx)

    rows = np.arange(n, dtype=int)
    li = np.array([image_to_idx[x] for x in left], dtype=int)
    ri = np.array([image_to_idx[x] for x in right], dtype=int)

    if sparse is not None:
        data = np.concatenate([np.ones(n), -np.ones(n)]).astype(float)
        r_ix = np.concatenate([rows, rows])
        c_ix = np.concatenate([li, ri])
        X = sparse.csr_matrix((data, (r_ix, c_ix)), shape=(n, p))
        return X
    else:
        X = np.zeros((n, p), dtype=float)
        X[rows, li] = 1.0
        X[rows, ri] = -1.0
        return X


def fit_bradley_terry_for_dimension(
    comparisons_df: pd.DataFrame,
    dimension: str,
    cfg: BTConfig,
    bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """
    Fits a penalized Bradley–Terry model for one dimension.

    Model:
      P(A wins over B) = sigmoid(theta_A - theta_B)
    where theta is per-image latent score.

    Returns per-image theta and an Elo-like rescale.
    Optionally adds respondent-cluster bootstrap CIs for theta and Elo-like score.
    """
    df_dim = comparisons_df[comparisons_df["dimension"] == dimension].copy()
    if df_dim.empty:
        raise ValueError(f"No comparisons for dimension: {dimension}")

    # Left/right in original A/B order
    left = df_dim["image_A"].astype(str).values
    right = df_dim["image_B"].astype(str).values
    y = (df_dim["winner"].values == "A").astype(int)

    images = pd.Index(sorted(set(left).union(set(right))))
    image_to_idx = {img: i for i, img in enumerate(images)}

    # connectivity
    node_to_comp, comp_sizes = dimension_connectivity(df_dim)
    comp_id = np.array([node_to_comp[img] for img in images], dtype=int)

    # counts (true appearances)
    counts = Counter(left.tolist()) + Counter(right.tolist())
    counts_arr = np.array([counts.get(img, 0) for img in images], dtype=int)

    X = _build_design_matrix(left, right, image_to_idx)

    C = 1.0 / float(cfg.alpha) if cfg.alpha > 0 else 1e12

    model = LogisticRegression(
        penalty="l2",
        C=C,
        fit_intercept=False,
        solver="saga",
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        random_state=cfg.random_state,
    )
    model.fit(X, y)

    theta = model.coef_.reshape(-1).astype(float)

    # Identifiability: theta is defined up to an additive constant.
    # Center to mean 0 for readability / comparability.
    theta = theta - float(theta.mean())

    # Elo-like rescale (monotone transform)
    # If you interpret theta in natural log-odds, the chess Elo mapping is:
    # elo = base + (400/ln(10))*theta
    ELO_SCALE = 400.0 / math.log(10.0)
    elo_like = 1500.0 + ELO_SCALE * theta

    out = pd.DataFrame(
        {
            "dimension": dimension,
            "image": images.astype(str),
            "theta": theta,
            "elo_like": elo_like,
            "comparisons": counts_arr,
            "component_id": comp_id,
            "component_size": [comp_sizes.get(int(c), 0) for c in comp_id],
            "n_components_dim": len(comp_sizes),
        }
    ).sort_values("theta", ascending=False).reset_index(drop=True)

    # Optional respondent-cluster bootstrap for uncertainty
    if bootstrap and bootstrap > 0:
        rng = np.random.default_rng(bootstrap_seed)
        respondents = df_dim["respondent_id"].values
        unique_resp = np.unique(respondents)

        theta_boot = np.full((bootstrap, len(images)), np.nan, dtype=float)

        # Pre-split rows by respondent for speed
        resp_to_rows = defaultdict(list)
        for i, r in enumerate(respondents):
            resp_to_rows[r].append(i)

        left_all = left
        right_all = right
        y_all = y

        for b in range(bootstrap):
            # sample respondents with replacement
            sampled = rng.choice(unique_resp, size=len(unique_resp), replace=True)

            # collect row indices
            idx = []
            for r in sampled:
                idx.extend(resp_to_rows[r])
            idx = np.array(idx, dtype=int)

            left_b = left_all[idx]
            right_b = right_all[idx]
            y_b = y_all[idx]

            # Some images may be absent in a resample; we still estimate in the full space
            # by building X over the full image set (shrinkage helps).
            X_b = _build_design_matrix(left_b, right_b, image_to_idx)

            m_b = LogisticRegression(
                penalty="l2",
                C=C,
                fit_intercept=False,
                solver="saga",
                max_iter=cfg.max_iter,
                tol=cfg.tol,
                random_state=int(cfg.random_state + b + 1),
            )
            m_b.fit(X_b, y_b)
            th_b = m_b.coef_.reshape(-1).astype(float)
            th_b = th_b - float(th_b.mean())
            theta_boot[b, :] = th_b

        # Percentile intervals (cluster bootstrap)
        lo = np.nanpercentile(theta_boot, 2.5, axis=0)
        hi = np.nanpercentile(theta_boot, 97.5, axis=0)

        out = out.merge(
            pd.DataFrame(
                {
                    "image": images.astype(str),
                    "theta_ci_low": lo,
                    "theta_ci_high": hi,
                    "elo_like_ci_low": 1500.0 + ELO_SCALE * lo,
                    "elo_like_ci_high": 1500.0 + ELO_SCALE * hi,
                }
            ),
            on="image",
            how="left",
        )

    return out


def fit_bradley_terry_all_dimensions(
    comparisons_df: pd.DataFrame,
    alpha: float = 1.0,
    max_iter: int = 5000,
    tol: float = 1e-7,
    random_state: int = 0,
    bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    cfg = BTConfig(alpha=alpha, max_iter=max_iter, tol=tol, random_state=random_state)
    out_all = []
    for dim in sorted(comparisons_df["dimension"].unique()):
        out_dim = fit_bradley_terry_for_dimension(
            comparisons_df,
            dimension=dim,
            cfg=cfg,
            bootstrap=bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
        out_all.append(out_dim)
    return pd.concat(out_all, ignore_index=True)


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(
    sona_path: Optional[Path],
    ariadna_path: Optional[Path],
    out_path: Path,
    url_prefix: str,
    alpha: float,
    bootstrap: int,
    bootstrap_seed: int,
    do_elo: bool,
    elo_K: float,
    elo_epochs: int,
    elo_seed: int,
):
    parts = []
    if sona_path is not None:
        parts.append(build_comparisons_for_dataset(sona_path, dataset_id="sona"))
    if ariadna_path is not None:
        parts.append(build_comparisons_for_dataset(ariadna_path, dataset_id="ariadna"))
    if not parts:
        raise ValueError("No datasets provided.")

    comparisons = pd.concat(parts, ignore_index=True)
    if comparisons.empty:
        raise ValueError("No comparisons were built. Check input files and columns.")

    # BT fit
    bt_all = fit_bradley_terry_all_dimensions(
        comparisons,
        alpha=alpha,
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
    )

    bt_all["url"] = url_prefix + bt_all["image"].astype(str)

    # Optional Elo baseline per dimension (merged by dimension+image)
    if do_elo:
        elo_frames = []
        for dim in sorted(comparisons["dimension"].unique()):
            e = compute_elo_for_dimension(
                comparisons,
                dimension=dim,
                K=elo_K,
                base=1500.0,
                epochs=elo_epochs,
                seed=elo_seed,
            )
            elo_frames.append(e)
        elo_all = pd.concat(elo_frames, ignore_index=True)
        bt_all = bt_all.merge(
            elo_all[["dimension", "image", "elo", "comparisons"]].rename(
                columns={"comparisons": "comparisons_elo"}
            ),
            on=["dimension", "image"],
            how="left",
        )

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bt_all.to_csv(out_path, index=False)

    # Print compact diagnostics
    dims = sorted(comparisons["dimension"].unique())
    print(f"Built comparisons: {comparisons.shape[0]:,} rows, {len(dims)} dimensions")
    print(f"Saved: {out_path}")

    diag = (
        bt_all.groupby("dimension")
        .agg(
            n_images=("image", "nunique"),
            n_comparisons=("comparisons", "sum"),
            n_components=("n_components_dim", "max"),
            min_comp_size=("component_size", "min"),
            max_comp_size=("component_size", "max"),
        )
        .reset_index()
        .sort_values(["n_components", "n_comparisons"], ascending=[False, False])
    )
    print("\nPer-dimension diagnostics (top 10 by #components then #comparisons):")
    print(diag.head(10).to_string(index=False))


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Robust pairwise face-attribute scoring (Bradley–Terry + optional Elo).")
    p.add_argument("--sona", type=str, default="apka_sona_2x60_twarzy__fixed.csv", help="Path to SONA CSV.")
    p.add_argument("--ariadna", type=str, default="apka_ariadna_2x75_par_twarzy__fixed.csv", help="Path to ARIADNA CSV.")
    p.add_argument("--out", type=str, default="bt_all_dimensions.csv", help="Output CSV path.")
    p.add_argument("--url-prefix", type=str, default=IMAGE_URL_PREFIX_DEFAULT, help="Prefix for image URLs.")

    # Bradley–Terry / logistic regression
    p.add_argument("--alpha", type=float, default=1.0, help="L2 regularization strength (bigger = more shrinkage).")
    p.add_argument("--bootstrap", type=int, default=1000, help="Respondent-cluster bootstrap replicates (0 disables).")
    p.add_argument("--bootstrap-seed", type=int, default=0, help="Bootstrap RNG seed.")

    # Optional Elo baseline
    p.add_argument("--elo", action="store_true", help="Also compute Elo baseline and merge into output.")
    p.add_argument("--elo-K", type=float, default=8.0, help="Elo K factor.")
    p.add_argument("--elo-epochs", type=int, default=5000, help="Elo epochs (shuffled passes).")
    p.add_argument("--elo-seed", type=int, default=123, help="Elo shuffle seed.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sona = Path(args.sona) if args.sona else None
    ariadna = Path(args.ariadna) if args.ariadna else None
    out = Path(args.out)

    run_pipeline(
        sona_path=sona,
        ariadna_path=ariadna,
        out_path=out,
        url_prefix=args.url_prefix,
        alpha=args.alpha,
        bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        do_elo=args.elo,
        elo_K=args.elo_K,
        elo_epochs=args.elo_epochs,
        elo_seed=args.elo_seed,
    )