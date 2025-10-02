# data/data_utils.py
import numpy as np
import torch
from scipy.stats import beta

def estimate_beta_parameters(data):
    """Estimate the parameters alpha and beta using method of moments for Beta distribution."""
    # Ensure data is numpy array and filter NaNs
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    data = data[~np.isnan(data)]
    if len(data) < 2: # Need at least 2 points to estimate variance
        return np.nan, np.nan # Cannot estimate

    mean_data = np.mean(data)
    var_data = np.var(data, ddof=1) # Use sample variance

    # Clamp mean and variance to avoid numerical issues at boundaries (0 or 1)
    mean_data = np.clip(mean_data, 1e-6, 1.0 - 1e-6)
    var_data = np.clip(var_data, 1e-9, mean_data * (1 - mean_data) - 1e-9) # Variance must be < mean*(1-mean)

    if var_data <= 0: # Cannot estimate if variance is non-positive
         return np.nan, np.nan

    # Method of moments estimates (MoM)
    common_factor = mean_data * (1 - mean_data) / var_data - 1
    alpha_est = mean_data * common_factor
    beta_est = (1 - mean_data) * common_factor

    # Ensure alpha and beta are positive
    alpha_est = max(alpha_est, 1e-6)
    beta_est = max(beta_est, 1e-6)

    # Optional: Refine with MLE (can be slow, MoM often sufficient)
    # try:
    #     a_mle, b_mle, _, _ = beta.fit(data, alpha_est, beta_est, floc=0, fscale=1)
    #     return a_mle, b_mle
    # except Exception:
    #     # Fallback to MoM if MLE fails
    #     return alpha_est, beta_est
    return alpha_est, beta_est

def bootstrap_tensor(data_tensor):
    """
    Performs bootstrapping on each row of a 2D tensor, ignoring NaN values.

    Args:
        data_tensor (torch.Tensor): A 2D tensor of shape (batch_size, n_examples).

    Returns:
        torch.Tensor: A bootstrapped tensor of the same shape.
    """
    batch_size, n_examples = data_tensor.shape
    bootstrapped_data = torch.empty_like(data_tensor)
    device = data_tensor.device

    for i in range(batch_size):
        valid_data = data_tensor[i, ~torch.isnan(data_tensor[i])] # Filter NaNs for the current row
        num_valid = len(valid_data)

        if num_valid == 0:
             # If all are NaNs, fill bootstrapped row with NaNs
             bootstrapped_data[i].fill_(float('nan'))
             continue

        # Generate random indices with replacement from the valid data indices
        indices = torch.randint(0, num_valid, (n_examples,), device=device)
        bootstrapped_data[i] = valid_data[indices]

    return bootstrapped_data

# --- Age/Bin Utilities from dataset_generation ---
def get_age_bin(age_float, min_age=20, max_age=60, bin_size=2):
    """ Given a floating age, produce an integer bin index. Returns None if outside range."""
    if age_float < min_age or age_float >= max_age:
        return None
    # E.g., [20, 22) -> 0, [22, 24) -> 1, ...
    bin_index = int((age_float - min_age) // bin_size)
    return bin_index

def bin_to_age_range(bin_index, min_age=20, bin_size=2):
    """ For an integer bin index, return the (low, high) EXCLUSIVE range end. """
    low_age = min_age + bin_index * bin_size
    high_age = low_age + bin_size # Exclusive end
    return (low_age, high_age)

# --- Distance Functions for Pairing ---
def custom_cdist(batch1, batch2):
    """ Default coordinate distance: Euclidean """
    # Ensure inputs are float tensors for cdist
    return torch.cdist(batch1.float().reshape(-1, 512),
                       batch2.float().reshape(-1, 512))

def custom_rdist(batch1_ratings, batch2_ratings):
    """ Calculate signed rating difference: rating2 - rating1 """
    b1 = batch1_ratings.view(-1, 1).float() # Ensure float
    b2 = batch2_ratings.view(-1).float() # Ensure float (destination ratings)
    # Result shape: (len(batch1), len(batch2))
    # Positive diff means batch2 rating > batch1 rating
    return b2 - b1

# --- Cost calculation from training loop ---
def calculate_cost_multiplier(x0: torch.Tensor, x1: torch.Tensor, logit_fn, device: torch.device) -> torch.Tensor:
    """
    Calculates the cost multiplier based on the inverse cubic distance normalized by logits.
    Handles potential division by zero and NaN values.
    Args:
        x0: Starting points (batch_size, latent_dim).
        x1: Ending points (batch_size, latent_dim).
        logit_fn: A function that takes latent codes (W) and returns their logits.
        device: The torch device.

    Returns:
        Cost multiplier tensor (batch_size, 1).
    """
    x0, x1 = x0.to(device), x1.to(device)
    epsilon = 1e-9 # Small value to prevent division by zero

    with torch.no_grad():
        logit0 = logit_fn(x0) # Shape (batch_size, 1)
        logit1 = logit_fn(x1) # Shape (batch_size, 1)
        delta_logit = logit1 - logit0

        # Ensure delta_logit is not zero or too small PURELY for the division below
        # If delta_logit is truly zero, cost is technically infinite/undefined.
        # We clamp the denominator away from zero. Cost reflects sensitivity.
        # A multiplier value might be capped or set based on domain logic if logits are identical.
        # Here, clamping to epsilon makes cost very large if logits are same, small otherwise.
        delta_logit_clamped = delta_logit.sign() * torch.clamp(torch.abs(delta_logit), min=epsilon)
        # Alternative: Assign a default large cost if delta_logit is near zero?

        norm_diff = torch.norm(x0 - x1, dim=-1, keepdim=True)
        # Clamp norm_diff away from zero if x0 and x1 are identical
        norm_diff_clamped = torch.clamp(norm_diff, min=epsilon)

        # Cost calculation based on original formula: 1 / ( (norm / delta_logit)**3 )
        # This can be rewritten as (delta_logit / norm)**3
        # cost = (delta_logit_clamped / norm_diff_clamped) ** 3 # Use clamped values
        # Original formula version:
        cost = 1 / torch.clamp(((norm_diff_clamped / delta_logit_clamped) ** 3), min=epsilon)


        # Handle potential NaNs or Infs resulting from edge cases (e.g., if clamping wasn't enough)
        cost = torch.nan_to_num(cost, nan=0.0, posinf=1e9, neginf=-1e9) # Replace NaN with 0, cap infinities

        # Ensure cost has reasonable bounds if needed (e.g., non-negative)
        cost = torch.clamp(cost, min=0.0) # Ensure cost is non-negative


        # Normalize the cost to act as a multiplier (mean=1) across the batch
        cost_sum = cost.sum()
        if cost.numel() > 0 and cost_sum > epsilon: # Check sum is positive and non-zero
            multiplier = cost / cost_sum * cost.numel()
            multiplier = torch.nan_to_num(multiplier, nan=1.0) # Default to 1 if division failed
        else:
            # If sum is zero or empty batch, return ones
            multiplier = torch.ones_like(cost)

    return multiplier.to(device)
