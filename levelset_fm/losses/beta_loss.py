import torch
import torch.nn as nn
import numpy as np
from data.data_utils import bootstrap_tensor # Import utility

def beta_pdf(x, alpha, beta, epsilon=1e-9):
    """ Calculates the PDF of the Beta distribution, handling edge cases. """
    # Ensure inputs are positive
    alpha = torch.clamp(alpha, min=epsilon)
    beta = torch.clamp(beta, min=epsilon)
    # Ensure x is within (0, 1), clamping slightly away from boundaries
    x = torch.clamp(x, min=epsilon, max=1.0 - epsilon)

    # Log Beta function: log(Gamma(a)) + log(Gamma(b)) - log(Gamma(a+b))
    log_beta_func = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    # Log PDF: (a-1)*log(x) + (b-1)*log(1-x) - log_beta_func
    log_pdf = (alpha - 1) * torch.log(x) + (beta - 1) * torch.log(1 - x) - log_beta_func

    # Clamp log_pdf to avoid extreme values leading to NaN loss
    log_pdf = torch.clamp(log_pdf, min=-100, max=100) # Adjust bounds as needed

    # Return PDF (optional, usually work with log_pdf for NLL)
    # pdf = torch.exp(log_pdf)
    # return pdf
    return log_pdf


class BetaNLLLoss(nn.Module):
    """ Negative Log-Likelihood loss for Beta distribution. """
    def __init__(self, parametrization='a,b', use_bootstrap=True, reduction='mean'):
        """
        Args:
            parametrization (str): 'a,b' or 'loc,disp'.
            use_bootstrap (bool): Whether to bootstrap target data during training.
            reduction (str): 'mean' or 'sum' for the final loss.
        """
        super().__init__()
        self.parametrization = parametrization
        self.use_bootstrap = use_bootstrap
        self.reduction = reduction
        if parametrization not in ['a,b', 'loc,disp']:
            raise ValueError("Invalid parametrization. Choose 'a,b' or 'loc,disp'.")

    def forward(self, params, target_data):
        """
        Calculates Beta NLL loss.
        Args:
            params (torch.Tensor): Predicted parameters (batch, 2).
                                   If 'a,b', represents [alpha, beta].
                                   If 'loc,disp', represents [location, dispersion].
            target_data (torch.Tensor): Target data samples (batch, n_samples), potentially with NaNs.

        Returns:
            torch.Tensor: Calculated NLL loss (scalar).
        """
        # 1. Convert params to alpha, beta
        if self.parametrization == 'loc,disp':
            loc = params[:, :1]  # Shape (batch, 1)
            disp = params[:, 1:] # Shape (batch, 1)
            # Ensure loc is in (0, 1) and disp > 0
            loc = torch.sigmoid(loc) # Apply sigmoid to ensure loc is in (0,1) - adjust if model outputs differently
            disp = torch.exp(disp) + 1e-6 # Apply exp for positivity, add epsilon
            # disp = torch.clamp(disp, min=1e-6) # Alternative: clamp if model outputs positive values directly
            total_ab = disp
            alpha = loc * total_ab
            beta = (1 - loc) * total_ab
        else: # 'a,b' parametrization
            # Assume model outputs values that can be directly used or are activated (e.g., exp)
            alpha = params[:, :1] # Shape (batch, 1)
            beta = params[:, 1:]  # Shape (batch, 1)
            # Add epsilon for stability if needed, depending on model's output activation
            alpha = torch.clamp(alpha, min=1e-6)
            beta = torch.clamp(beta, min=1e-6)


        # 2. Handle target data (bootstrap if training)
        if self.training and self.use_bootstrap:
            target_data_processed = bootstrap_tensor(target_data)
        else:
            target_data_processed = target_data

        # Filter NaNs from target data
        valid_mask = ~torch.isnan(target_data_processed)
        if not valid_mask.any(): # If no valid data points, return 0 loss
            return torch.tensor(0.0, device=params.device)

        # Expand alpha, beta to match target_data shape for calculation
        # target_data_processed shape: (batch, n_samples)
        # alpha, beta shape: (batch, 1) -> expand to (batch, n_samples)
        alpha_expanded = alpha.expand_as(target_data_processed)
        beta_expanded = beta.expand_as(target_data_processed)

        # 3. Calculate log PDF only for valid data points
        log_pdf_values = torch.full_like(target_data_processed, float('nan')) # Initialize with NaN
        log_pdf_values[valid_mask] = beta_pdf(
            target_data_processed[valid_mask],
            alpha_expanded[valid_mask],
            beta_expanded[valid_mask]
        )

        # 4. Calculate NLL
        # Use nanmean/nansum to ignore NaNs introduced where valid_mask was False
        if self.reduction == 'mean':
            # Average NLL per valid data point across the batch
             nll_loss = -torch.nansum(log_pdf_values) / valid_mask.sum() # Sum valid log_pdfs, divide by count
             # Alternative: Mean per sample, then mean per batch
             # nll_per_sample = -torch.nanmean(log_pdf_values, dim=1) # Shape (batch,)
             # nll_loss = torch.mean(nll_per_sample)
        elif self.reduction == 'sum':
            nll_loss = -torch.nansum(log_pdf_values) # Sum over all valid data points
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}. Choose 'mean' or 'sum'.")

        # Handle potential NaN loss if calculations resulted in NaN despite clamps
        if torch.isnan(nll_loss):
             print("Warning: NaN loss detected in BetaNLLLoss.")
             # Decide fallback: return 0, raise error, etc.
             return torch.tensor(0.0, device=params.device, requires_grad=True) # Return 0 loss with grad enabled

        return nll_loss


# --- Mixup Specific Loss ---

def mixup_bootstrapped_samples(y_a, y_b, lam):
    """
    Interpolates bootstrapped samples based on the mixup coefficient lambda using sorting.
    Args:
        y_a (torch.Tensor): First batch of bootstrapped targets [batch_size, n_examples].
        y_b (torch.Tensor): Second batch of bootstrapped targets [batch_size, n_examples].
        lam (torch.Tensor): Mixup coefficients [batch_size,].
    Returns:
        torch.Tensor: Interpolated samples [batch_size, n_examples].
    """
    batch_size, n_examples = y_a.shape
    device = y_a.device
    lam = lam.to(device).view(-1, 1) # Reshape lambda to (batch_size, 1) for broadcasting

    mixed_y = torch.zeros_like(y_a)
    for i in range(batch_size):
        sample_a = y_a[i]
        sample_b = y_b[i]

        # Handle NaNs before sorting
        valid_a_mask = ~torch.isnan(sample_a)
        valid_b_mask = ~torch.isnan(sample_b)

        if not valid_a_mask.any() or not valid_b_mask.any():
             # If either sample is all NaNs, how to mix?
             # Option 1: Return NaNs
             # mixed_y[i].fill_(float('nan'))
             # Option 2: Return the non-NaN sample scaled? (Less standard for mixup)
             # Let's return NaNs for simplicity if mixing isn't possible
             mixed_y[i].fill_(float('nan'))
             continue

        # Sort only the valid parts
        sorted_a, _ = torch.sort(sample_a[valid_a_mask])
        sorted_b, _ = torch.sort(sample_b[valid_b_mask])

        # Interpolate based on quantiles (requires equal number of valid points or approximation)
        # Simple approach: Linearly interpolate between sorted lists. Requires same length after NaN removal.
        # If lengths differ, quantile interpolation is more robust but complex.
        # Assuming bootstrapping provides enough non-NaN points of same length for simplicity here.
        # A robust way handles differing number of valid points via quantile matching.
        len_a = len(sorted_a)
        len_b = len(sorted_b)

        # Simplest case: if lengths match after NaN removal
        if len_a == len_b:
             interpolated_sorted = lam[i] * sorted_a + (1 - lam[i]) * sorted_b
             # Place back into the original tensor structure, handling NaNs
             # Since number of NaNs might differ, this direct placement is tricky.
             # Let's fill the output row with the interpolated values, possibly fewer than n_examples if NaNs existed.
             # Padding with NaNs might be needed if original structure must be preserved.

             # Fill with interpolated values, pad rest with NaN if needed
             result = torch.full((n_examples,), float('nan'), device=device)
             result[:len(interpolated_sorted)] = interpolated_sorted
             mixed_y[i] = result

        else:
           # Quantile interpolation (more robust but complex)
           # - Define target quantiles (e.g., np.linspace(0, 1, n_examples))
           # - Compute quantiles for sorted_a and sorted_b
           # - Interpolate the quantile values: q_mixed = lam * q_a + (1-lam) * q_b
           # - This gives the value at each quantile for the mixed distribution.
           # This is significantly more involved; using the simpler length-match assumption for now.
           print(f"Warning: Different number of valid samples for mixup ({len_a} vs {len_b}). Using simple matching, may be inaccurate.")
           min_len = min(len_a, len_b)
           interpolated_sorted = lam[i] * sorted_a[:min_len] + (1 - lam[i]) * sorted_b[:min_len]
           result = torch.full((n_examples,), float('nan'), device=device)
           result[:min_len] = interpolated_sorted
           mixed_y[i] = result


    return mixed_y


class MixupBetaNLLLoss(nn.Module):
    """ Wrapper for BetaNLLLoss that handles mixup labels. """
    def __init__(self, base_criterion: BetaNLLLoss):
        super().__init__()
        if not isinstance(base_criterion, BetaNLLLoss):
            raise TypeError("base_criterion must be an instance of BetaNLLLoss")
        self.base_criterion = base_criterion
        # Mixup alpha is handled by the Dataset/DataLoader, not the loss function itself

    def forward(self, outputs, y1, y2, lam):
        """
        Args:
            outputs (torch.Tensor): Model predictions (params) for the mixed input x.
            y1 (torch.Tensor): Target data corresponding to the first sample [batch, n_samples].
            y2 (torch.Tensor): Target data corresponding to the second sample [batch, n_samples].
            lam (torch.Tensor): Mixup coefficient [batch,].
        Returns:
            torch.Tensor: Mixup loss.
        """
        # 1. Bootstrap the original targets (optional, based on base_criterion setting)
        # Bootstrapping happens inside base_criterion if enabled and training
        # We need bootstrapped versions *before* mixing them based on lambda

        # If base_criterion uses bootstrapping, we assume y1, y2 are already bootstrapped *or*
        # we bootstrap them here before mixing. Let's bootstrap here for clarity, assuming base loss
        # won't bootstrap again. Requires base_criterion.use_bootstrap = False potentially.
        # OR rely on base_criterion doing it. Let's rely on base_criterion.

        # 2. Mix up the bootstrapped target samples based on lambda
        # Important: Assumes y1, y2 are already potentially bootstrapped by the base loss call below.
        # This mixing strategy might need refinement. The original paper might mix parameters or expected values.
        # Mixing raw bootstrapped samples requires care (e.g., quantile matching).
        # Let's use the provided mixup_bootstrapped_samples function.
        mixed_y = mixup_bootstrapped_samples(y1, y2, lam)

        # 3. Calculate the loss using the model's output (params for mixed input)
        #    and the mixed target data distribution.
        # We need to ensure base_criterion uses the mixed_y correctly.
        # Set base_criterion to not use bootstrapping internally if we mixed bootstrapped data.
        original_bootstrap_setting = self.base_criterion.use_bootstrap
        if self.training: # Only override during training when mixup is active
             # Assume mixup_bootstrapped_samples already handled the effective bootstrapping+mixing
             self.base_criterion.use_bootstrap = False

        loss = self.base_criterion(outputs, mixed_y)

        # Restore original setting
        self.base_criterion.use_bootstrap = original_bootstrap_setting

        return loss








# **19. `requirements.txt`** (More comprehensive)

# ```
# torch>=2.0.0 # Or your specific torch version
# torchvision
# torchaudio
# numpy>=1.20.0
# pandas>=1.3.0
# scipy>=1.7.0
# scikit-learn>=1.0.0
# matplotlib>=3.4.0
# Pillow>=8.4.0
# opencv-python>=4.5.0 # cv2
# einops>=0.4.0
# x-transformers>=1.0.0 # Check specific version if needed
# torchdyn>=1.0.0 # Or torchdiffeq if preferred
# torchcfm>=0.1.0 # Check specific version
# pot>=0.8.0 # Python Optimal Transport library
# tqdm>=4.60.0
# gdown>=4.4.0 # For downloading from Google Drive
# requests>=2.25.0 # For potential downloads
# ipywidgets # For notebook interactivity emulation if needed, perhaps not strict req
# ninja # Often needed for custom CUDA kernels in StyleGAN/torch extensions
# tables # Was imported in notebook, maybe for pandas HDFStore? Include just in case.
# ```