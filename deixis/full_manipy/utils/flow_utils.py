# utils/flow_utils.py
import torch
import numpy as np

def sample_conditional_pt(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Samples points xt along the probability path N(t * x1 + (1 - t) * x0, sigma_t^2).
    The noise level sigma_t seems to depend on t in the original code via (1.1-t)*sigma.

    Args:
        x0: Starting points (batch_size, latent_dim).
        x1: Ending points (batch_size, latent_dim).
        t: Time steps (batch_size,), values between 0 and 1.
        sigma: Base standard deviation of the noise.

    Returns:
        Sampled points xt (batch_size, latent_dim).
    """
    t_reshaped = t.reshape(-1, *([1] * (x0.dim() - 1))) # Shape (bs, 1, ..., 1)
    mu_t = t_reshaped * x1 + (1 - t_reshaped) * x0
    # Noise scaled by (1.1 - t), similar to original code snippet
    # Ensure t is also shaped correctly for this calculation if needed, here using t_reshaped
    sigma_t = 0#(1.1 - t_reshaped) * sigma
    epsilon = torch.randn_like(x0)
    xt = mu_t + sigma_t * epsilon
    return xt

def compute_conditional_vector_field(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Computes the conditional vector field direction U(t|x0, x1) = (x1 - x0).
    Normalization happens *inside* the training loop in the original code.
    This function just returns the difference vector.

    Args:
        x0: Starting points (batch_size, latent_dim).
        x1: Ending points (batch_size, latent_dim).

    Returns:
        Vector field directions (x1 - x0).
    """
    return x1 - x0
    