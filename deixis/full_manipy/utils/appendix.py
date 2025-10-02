import numpy as np

def drop_indices_to_flatten_distribution(arr, drop_frac=0.5, bins=10, random_state=None):
    """
    Returns indices to keep from a 1D array such that dropping drop_frac of the array
    makes the distribution of values more uniform across bins.

    Parameters:
        arr (np.ndarray): 1D array of values.
        drop_frac (float): Fraction of indices to drop (0 < drop_frac < 1).
        bins (int or sequence): Number of bins or bin edges for histogramming.
        random_state (int or None): For reproducibility.

    Returns:
        keep_indices (np.ndarray): Indices to keep.
        drop_indices (np.ndarray): Indices to drop.
    """
    rng = np.random.default_rng(random_state)
    arr = np.asarray(arr)
    N = len(arr)
    n_drop = int(np.floor(N * drop_frac))

    # Bin the data
    hist, bin_edges = np.histogram(arr, bins=bins)
    bin_ids = np.digitize(arr, bin_edges[:-1], right=True) - 1
    bin_ids = np.clip(bin_ids, 0, len(hist)-1)

    # Compute how many to drop from each bin to flatten the distribution
    target_per_bin = int(np.ceil((N - n_drop) / len(hist)))
    drop_per_bin = hist - target_per_bin
    drop_per_bin = np.clip(drop_per_bin, 0, None)

    # For each bin, randomly select indices to drop
    drop_indices = []
    for b in range(len(hist)):
        idx_in_bin = np.where(bin_ids == b)[0]
        n_to_drop = min(drop_per_bin[b], len(idx_in_bin))
        if n_to_drop > 0:
            drop_idx = rng.choice(idx_in_bin, size=n_to_drop, replace=False)
            drop_indices.append(drop_idx)
    drop_indices = np.concatenate(drop_indices) if drop_indices else np.array([], dtype=int)

    # If we didn't drop enough, randomly drop more
    # if len(drop_indices) < n_drop:
    #     remaining = np.setdiff1d(np.arange(N), drop_indices)
    #     n_extra = n_drop - len(drop_indices)
    #     extra_drop = rng.choice(remaining, size=n_extra, replace=False)
    #     drop_indices = np.concatenate([drop_indices, extra_drop])

    # # If we dropped too many, randomly keep some
    # if len(drop_indices) > n_drop:
    #     drop_indices = rng.choice(drop_indices, size=n_drop, replace=False)

    keep_indices = np.setdiff1d(np.arange(N), drop_indices)
    return keep_indices, drop_indices
