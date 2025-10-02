# response_style.py
# Utilities for response-style metrics, invariants, and randomness probability

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Iterable

# -----------------------------
# Data ingestion / normalization
# -----------------------------

def df_from_nested(
    dim_to_photo_to_rating: Dict[str, Dict[str, Dict[str, List[float] | float]]]
) -> pd.DataFrame:
    """
    Convert nested dict {dim: {pid: {fname: rating or [rating]}}} -> long DataFrame
    with columns: ['dim', 'pid', 'fname', 'rating'].
    """
    rows = []
    for dim, pid_map in dim_to_photo_to_rating.items():
        for pid, f2r in pid_map.items():
            for fname, y in f2r.items():
                if isinstance(y, (list, tuple)):
                    if len(y) == 0: 
                        continue
                    yv = float(y[0])
                else:
                    yv = float(y)
                if 0.0 < yv < 1.0:
                    rows.append((dim, pid, fname, yv))
    df = pd.DataFrame(rows, columns=["dim", "pid", "fname", "rating"])
    return df


def ensure_long_df(
    data: pd.DataFrame | Dict[str, Dict[str, Dict[str, List[float] | float]]]
) -> pd.DataFrame:
    """
    Accept either a long DataFrame or a nested dict and return a clean long DataFrame.
    Required columns for DataFrame: dim, pid, fname, rating
    """
    if isinstance(data, pd.DataFrame):
        for col in ("dim", "pid", "fname", "rating"):
            if col not in data.columns:
                raise ValueError(f"DataFrame missing required column: {col}")
        df = data.copy()
        df = df[(df["rating"] > 0) & (df["rating"] < 1)].copy()
        return df
    elif isinstance(data, dict):
        return df_from_nested(data)
    else:
        raise TypeError("data must be a pandas DataFrame or the nested dict format.")


# -----------------------------
# Core metric helpers
# -----------------------------

def _quantiles(x: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.full_like(u_grid, 0.5, dtype=float)
    # numpy >=1.22: method='linear'; older: interpolation='linear'
    return np.quantile(x, u_grid)

def _js_divergence_hist(a: np.ndarray, b: np.ndarray, nbins: int = 60, eps: float = 1e-8) -> float:
    pa, _ = np.histogram(a, bins=nbins, range=(0.0, 1.0), density=True)
    pb, _ = np.histogram(b, bins=nbins, range=(0.0, 1.0), density=True)
    pa = pa + eps; pb = pb + eps
    pa = pa / pa.sum(); pb = pb / pb.sum()
    m = 0.5 * (pa + pb)
    js = 0.5 * (np.sum(pa * (np.log(pa) - np.log(m))) + np.sum(pb * (np.log(pb) - np.log(m))))
    return float(js)

def _w2_1d(a: np.ndarray, b: np.ndarray, u_grid=None) -> float:
    if u_grid is None:
        u_grid = np.linspace(1e-3, 1 - 1e-3, 200)
    Qa = _quantiles(a, u_grid)
    Qb = _quantiles(b, u_grid)
    return float(np.sqrt(np.mean((Qa - Qb) ** 2)))

def _average_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks

def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size < 3 or y.size < 3:
        return np.nan
    rx = _average_ranks(x); ry = _average_ranks(y)
    rx_c = rx - rx.mean(); ry_c = ry - ry.mean()
    denom = np.sqrt((rx_c ** 2).sum() * (ry_c ** 2).sum())
    if denom <= 0: return np.nan
    return float((rx_c @ ry_c) / denom)

def _ols_slope_intercept_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size < 2: return np.nan, np.nan, np.nan
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    yhat = intercept + slope * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return slope, intercept, r2

# -----------------------------
# Population summaries per item
# -----------------------------

def _item_population_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (dim,fname) compute: n, mean, sd and store sorted values for ECDF/PIT.
    Returns a DataFrame indexed by (dim,fname) with columns: n, mean, sd, values (list of ratings).
    """
    grp = df.groupby(["dim", "fname"])["rating"]
    stats = grp.agg(n="count", mean="mean", sd=lambda s: float(np.std(s, ddof=1) if len(s) > 1 else np.std(s)))
    # keep raw values for PIT
    values = grp.apply(lambda s: np.sort(np.array(s, dtype=float))).rename("values")
    out = stats.join(values)
    return out

def _pit_u(values_sorted: np.ndarray, y: float) -> float:
    """Right-continuous ECDF u=F(y) in (0,1]."""
    n = len(values_sorted)
    if n == 0:
        return 0.5
    # rank of y among sorted values (right)
    r = np.searchsorted(values_sorted, y, side="right")
    return (r / n)

# -----------------------------
# Rated-item invariant transforms
# -----------------------------

def _itemwise_pit_series(df_person: pd.DataFrame, pop_stats: pd.DataFrame) -> np.ndarray:
    """For each row in df_person (dim, fname, rating) compute u_i = F_pop_item(rating)."""
    u = []
    for _, row in df_person.iterrows():
        key = (row["dim"], row["fname"])
        if key in pop_stats.index:
            vals = pop_stats.loc[key, "values"]
            u.append(_pit_u(vals, row["rating"]))
        else:
            u.append(0.5)
    return np.asarray(u, dtype=float)

def _itemwise_zscore_series(df_person: pd.DataFrame, pop_stats: pd.DataFrame, eps=1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Return (z_pop_item, z_person_item) where each is standardized by item mean/sd."""
    mu = []; sd = []; y = []
    for _, row in df_person.iterrows():
        key = (row["dim"], row["fname"])
        if key in pop_stats.index:
            mu_i = float(pop_stats.loc[key, "mean"])
            sd_i = float(pop_stats.loc[key, "sd"])
        else:
            mu_i = 0.5; sd_i = 0.15
        mu.append(mu_i); sd.append(sd_i); y.append(float(row["rating"]))
    mu = np.asarray(mu); sd = np.asarray(sd); y = np.asarray(y)
    z_person = (y - mu) / (sd + eps)
    # For regression x variable, use z of μ_pop (i.e., 0 vector). But for a slope-like invariant,
    # we can regress standardized population means (all zeros) is degenerate. Instead return z-scored y only.
    return z_person, (y - y.mean()) / (y.std(ddof=1) + eps)

# -----------------------------
# Per-participant metrics (with invariants)
# -----------------------------

def participant_metrics_from_df(
    df_long: pd.DataFrame,
    dim_key: str,
    pid: str,
    min_items: int = 10,
    histogram_bins: int = 60,
) -> Optional[Dict[str, float]]:
    """
    Compute metrics for a single participant in a given dimension.
    Returns None if not enough items or no population baseline.
    """
    df_dim = df_long[df_long["dim"] == dim_key]
    df_p = df_dim[df_dim["pid"] == pid].copy()
    if len(df_p) < min_items:
        return None

    # population excluding this pid
    df_pop = df_dim[df_dim["pid"] != pid].copy()
    if df_pop.empty:
        return None

    # per-item population stats
    pop_stats = _item_population_stats(df_pop)

    # build arrays aligned to this participant's items
    fnames = df_p["fname"].values
    y_p = df_p["rating"].astype(float).values

    mu_pop_i = []
    sd_pop_i = []
    pool_all = []
    for f in fnames:
        key = (dim_key, f)
        if key in pop_stats.index:
            mu_pop_i.append(float(pop_stats.loc[key, "mean"]))
            sd_pop_i.append(float(pop_stats.loc[key, "sd"]))
            pool_all.extend(pop_stats.loc[key, "values"].tolist())
        else:
            mu_pop_i.append(np.nan)
            sd_pop_i.append(np.nan)
    mu_pop_i = np.asarray(mu_pop_i, dtype=float)
    sd_pop_i = np.asarray(sd_pop_i, dtype=float)
    y_pop_all = np.asarray(pool_all, dtype=float)

    if y_pop_all.size == 0:
        return None

    # --- Global distances ---
    W2 = _w2_1d(y_p, y_pop_all)
    JS = _js_divergence_hist(y_pop_all, y_p, nbins=histogram_bins)

    # --- Regression & rank correlation ---
    mask = ~np.isnan(mu_pop_i)
    slope = intercept = r2 = np.nan
    rho = np.nan
    if mask.sum() >= 3:
        slope, intercept, r2 = _ols_slope_intercept_r2(mu_pop_i[mask], y_p[mask])
        rho = _spearman_rho(mu_pop_i[mask], y_p[mask])

    # --- Person-fit (continuous outfit/infit) ---
    sd_global = float(np.std(y_pop_all, ddof=1)) if y_pop_all.size > 1 else 0.15
    sd_i = np.where(np.isnan(sd_pop_i) | (sd_pop_i <= 1e-8), sd_global, sd_pop_i)
    mu_i = np.where(np.isnan(mu_pop_i), float(np.mean(y_pop_all)), mu_pop_i)
    r = (y_p - mu_i) / (sd_i + 1e-6)
    outfit = float(np.mean(r ** 2))
    w = 1.0 / ((sd_i + 1e-6) ** 2)
    infit = float(np.sum(w * (r ** 2)) / np.sum(w))

    # --- Style indices ---
    mean_shift = float(np.mean(y_p) - np.mean(y_pop_all))
    extremity_shift = float(np.mean(np.abs(y_p - 0.5)) - np.mean(np.abs(y_pop_all - 0.5)))

    # --- Invariant metrics ---
    # PIT per item → compare to Uniform(0,1)
    u = _itemwise_pit_series(df_p, pop_stats)
    W2_invariant = _w2_1d(u, np.random.uniform(0, 1, size=2048))  # Monte-Carlo uniform reference
    JS_invariant = _js_divergence_hist(np.random.uniform(0, 1, size=4096), u, nbins=histogram_bins)
    extremity_invariant = float(np.mean(np.abs(u - 0.5)) - 0.25)   # E|U-0.5| of Uniform is 0.25

    # z-score residual energy (rated-item-scale invariant)
    z_person, _ = _itemwise_zscore_series(df_p, pop_stats)
    outfit_z = float(np.mean(z_person ** 2))
    infit_z = float(np.mean((z_person ** 2)))  # identical weights after standardization

    return dict(
        pid=pid,
        n_items=int(len(df_p)),
        W2=W2, JS=JS,
        slope_ols=slope, intercept=intercept, r2=r2, spearman_rho=rho,
        outfit=outfit, infit=infit,
        mean_shift=mean_shift, extremity_shift=extremity_shift,
        # invariants
        W2_inv=W2_invariant, JS_inv=JS_invariant, extremity_inv=extremity_invariant,
        outfit_z=outfit_z, infit_z=infit_z
    )

def metrics_dataframe(
    data: pd.DataFrame | Dict[str, Dict[str, Dict[str, List[float] | float]]],
    dim_key: str,
    min_items_per_participant: int = 20,
    histogram_bins: int = 60,
) -> pd.DataFrame:
    """
    Compute metrics (including invariant variants) for all participants in a dimension.
    Returns a DataFrame with one row per participant.
    """
    df = ensure_long_df(data)
    rows = []
    for pid in sorted(df[df["dim"] == dim_key]["pid"].unique()):
        r = participant_metrics_from_df(df, dim_key, pid, min_items=min_items_per_participant, histogram_bins=histogram_bins)
        if r is not None:
            rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["W2", "JS"], ascending=[False, False]).reset_index(drop=True)
    return out

# -----------------------------
# Randomness probability model
# -----------------------------

class RandomnessModel:
    """
    Simple logistic classifier over features [W2, slope_ols, spearman_rho, r2].
    If scikit-learn is installed, uses it; else falls back to L2-regularized
    logistic regression via batch gradient descent.
    """
    def __init__(self):
        self.coef_: Optional[np.ndarray] = None  # shape [4]
        self.intercept_: float = 0.0
        self.feature_names_ = ["W2", "slope_ols", "spearman_rho", "r2"]
        self._use_sklearn = False
        self._sk_model = None
        self._x_mean = None
        self._x_std = None

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._x_mean = X.mean(axis=0, keepdims=True)
            self._x_std = X.std(axis=0, keepdims=True) + 1e-8
        return (X - self._x_mean) / self._x_std

    def fit(self, X: np.ndarray, y: np.ndarray, l2=1.0, max_iter=2000, lr=0.1, seed=123):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != 4:
            raise ValueError("X must be of shape [n_samples, 4] with columns [W2, slope_ols, spearman_rho, r2].")

        # Try scikit-learn
        try:
            from sklearn.linear_model import LogisticRegression
            self._use_sklearn = True
            self._sk_model = LogisticRegression(C=1.0/l2, solver="lbfgs", max_iter=2000)
            Xz = self._standardize(X, fit=True)
            self._sk_model.fit(Xz, y.astype(int))
            self.coef_ = self._sk_model.coef_.reshape(-1)
            self.intercept_ = float(self._sk_model.intercept_.reshape(()))
            return self
        except Exception:
            self._use_sklearn = False

        # Fallback: manual logistic regression with L2
        rng = np.random.default_rng(seed)
        Xz = self._standardize(X, fit=True)
        w = rng.normal(scale=0.01, size=4)
        b = 0.0

        for _ in range(max_iter):
            z = Xz @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            # gradients
            grad_w = (Xz.T @ (p - y)) / len(y) + l2 * w
            grad_b = float(np.mean(p - y))
            w -= lr * grad_w
            b -= lr * grad_b

        self.coef_ = w
        self.intercept_ = b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fit yet.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        Xz = (X - self._x_mean) / self._x_std
        if self._use_sklearn:
            return self._sk_model.predict_proba(Xz)[:, 1]
        else:
            z = Xz @ self.coef_ + self.intercept_
            return 1.0 / (1.0 + np.exp(-z))


def train_randomness_model(
    data: pd.DataFrame | Dict[str, Dict[str, Dict[str, List[float] | float]]],
    dim_key: str,
    min_items_per_participant: int = 20,
    histogram_bins: int = 60,
    # selection of participants to build null via permutations
    selection_W2_max: float = 0.2,
    require_positive_correlation: bool = True,
    n_permutations_per_participant: int = 20,
    seed: int = 123,
) -> Tuple[RandomnessModel, pd.DataFrame]:
    """
    Train a P(random responses) model over features [W2, slope, spearman, r2].
    The positive class is constructed by permuting each selected participant's
    ratings across their rated items (breaking item-response alignment).

    Selection of participants for permutation: those with W2 < selection_W2_max
    and (if require_positive_correlation) max(slope, spearman) > 0.

    Returns (model, training_table) where training_table has columns:
      ['pid', 'label_random', 'W2', 'slope_ols', 'spearman_rho', 'r2'].
    """
    rng = np.random.default_rng(seed)
    df = ensure_long_df(data)
    # compute metrics for all
    base_df = metrics_dataframe(df, dim_key, min_items_per_participant, histogram_bins)

    # select participants to build permutation null
    select_mask = (base_df["W2"] < selection_W2_max)
    if require_positive_correlation:
        pos = ((base_df["slope_ols"] > 0) | (base_df["spearman_rho"] > 0))
        select_mask = select_mask & pos
    selected = base_df[select_mask]["pid"].tolist()

    # helper to recompute metrics after permutation for one participant
    def permuted_metrics_for_pid(pid: str) -> List[Tuple[float, float, float, float]]:
        out: List[Tuple[float, float, float, float]] = []
        df_dim = df[df["dim"] == dim_key]
        df_p = df_dim[df_dim["pid"] == pid].copy()
        if len(df_p) < min_items_per_participant: 
            return out
        df_pop = df_dim[df_dim["pid"] != pid].copy()
        if df_pop.empty:
            return out
        pop_stats = _item_population_stats(df_pop)
        # arrays aligned
        y = df_p["rating"].values.copy()
        mu_pop_i = []
        pool_all = []
        for f in df_p["fname"].values:
            key = (dim_key, f)
            if key in pop_stats.index:
                mu_pop_i.append(float(pop_stats.loc[key, "mean"]))
                pool_all.extend(pop_stats.loc[key, "values"].tolist())
            else:
                mu_pop_i.append(np.nan)
        mu_pop_i = np.asarray(mu_pop_i, dtype=float)
        y_pop_all = np.asarray(pool_all, dtype=float)
        if y_pop_all.size == 0:
            return out

        for _ in range(n_permutations_per_participant):
            yp = y.copy()
            rng.shuffle(yp)
            # recompute features
            W2 = _w2_1d(yp, y_pop_all)
            mask = ~np.isnan(mu_pop_i)
            if mask.sum() >= 3:
                slope, _, r2 = _ols_slope_intercept_r2(mu_pop_i[mask], yp[mask])
                rho = _spearman_rho(mu_pop_i[mask], yp[mask])
            else:
                slope = r2 = rho = np.nan
            out.append((W2, slope, rho, r2))
        return out

    rows = []
    # Originals (label 0)
    for _, r in base_df.iterrows():
        rows.append(dict(
            pid=r["pid"], label_random=0,
            W2=r["W2"], slope_ols=r["slope_ols"], spearman_rho=r["spearman_rho"], r2=r["r2"]
        ))
    # Permuted (label 1)
    for pid in selected:
        feats = permuted_metrics_for_pid(pid)
        for (W2, slope, rho, r2) in feats:
            rows.append(dict(
                pid=pid, label_random=1,
                W2=W2, slope_ols=slope, spearman_rho=rho, r2=r2
            ))

    train_tab = pd.DataFrame(rows).dropna(subset=["W2", "slope_ols", "spearman_rho", "r2"])
    if train_tab.empty:
        raise RuntimeError("No training data for randomness model (check selection / data size).")

    X = train_tab[["W2", "slope_ols", "spearman_rho", "r2"]].values
    y = train_tab["label_random"].values.astype(int)

    model = RandomnessModel().fit(X, y, l2=1.0, max_iter=2000, lr=0.1, seed=seed)
    return model, train_tab

def predict_randomness_probabilities(
    model: RandomnessModel,
    metrics_df: pd.DataFrame
) -> pd.Series:
    """
    Given a fitted RandomnessModel and a metrics_df (from metrics_dataframe),
    return a pandas Series with index=pid and values P(random).
    """
    cols = ["W2", "slope_ols", "spearman_rho", "r2"]
    X = metrics_df[cols].values
    p = model.predict_proba(X)
    return pd.Series(p, index=metrics_df["pid"].values, name="p_random")


# -----------------------------
# Convenience: one-shot pipeline
# -----------------------------

def build_metrics_and_randomness(
    data: pd.DataFrame | Dict[str, Dict[str, Dict[str, List[float] | float]]],
    dim_key: str,
    min_items_per_participant: int = 20,
    histogram_bins: int = 60,
    selection_W2_max: float = 0.2,
    require_positive_correlation: bool = True,
    n_permutations_per_participant: int = 20,
    seed: int = 123,
) -> Tuple[pd.DataFrame, RandomnessModel, pd.Series]:
    """
    End-to-end: compute metrics_df, train randomness model, and return
    per-participant P(random) as a Series aligned to metrics_df.
    """
    df = ensure_long_df(data)
    metrics_df = metrics_dataframe(df, dim_key, min_items_per_participant, histogram_bins)
    model, train_tab = train_randomness_model(
        df, dim_key, min_items_per_participant, histogram_bins,
        selection_W2_max, require_positive_correlation, n_permutations_per_participant, seed
    )
    p_random = predict_randomness_probabilities(model, metrics_df)
    metrics_df = metrics_df.copy()
    metrics_df["p_random"] = p_random.values
    return metrics_df, model, p_random

