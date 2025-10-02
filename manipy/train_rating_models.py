# train_trust_model.py
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from manipy import config
from manipy.models.rating_models import AlphaBetaRegressor, EnsembleRegressor, load_trust_model_ensemble # Import model definition
from manipy.losses.beta_loss import BetaNLLLoss, MixupBetaNLLLoss, mixup_bootstrapped_samples # Import loss
from .stylegan.utils import get_w_avg, sample_w # Need w_avg and sampling for laplacian reg
from .utils.general_utils import print_gpu_memory
from .data.data_utils import estimate_beta_parameters # Needed? Maybe just ratings needed.

# Need mechanism to load OMI data and W latents
# Assuming 'X' (W latents) and 'ratings' (tensor with user ratings per stimulus) are loaded externally for now.
# Placeholder loading - replace with actual data loading logic
def load_training_data(dim_name=config.TARGET_DIMENSION, device=config.DEVICE):
    """ Placeholder function to load W latents and corresponding ratings. """
    print("Loading training data (Placeholder)...")
    # Example: Load from preprocessed files or generate dummy data
    try:
        # Attempt to load data similar to the 'init' cell in the notebook
        # This part is highly specific to how 'df' and 'ratings' were created.
        # Need to replicate the loading of 'X' and 'ratings' tensor.

        # Simplified placeholder based on notebook snippet:
        # Needs attribute_ratings.csv, attribute_means.csv, coords_wlosses.csv
        coords_file = os.path.join(config.PSYCHGAN_REPO_PATH, 'content/coords_wlosses.csv')
        ratings_file = os.path.join(config.OMI_DATA_DIR, 'attribute_ratings.csv')
        means_file = os.path.join(config.OMI_DATA_DIR,'attribute_means.csv') # Assuming path

        if not os.path.exists(coords_file) or not os.path.exists(ratings_file) or not os.path.exists(means_file):
            raise FileNotFoundError("Required data files not found. Run data preparation steps.")

        df_coords = pd.read_csv(coords_file)
        # Handle potential 'eval' if latents are stored as strings
        try:
            df_coords["dlatents"] = df_coords["dlatents"].apply(eval)
        except: pass # Assume already loaded correctly if eval fails

        df_ratings = pd.read_csv(ratings_file)
        df_means = pd.read_csv(means_file)

        # Filtering based on notebook logic
        df_ratings = df_ratings.loc[df_ratings.stimulus <= 1004] # Original stimuli IDs?
        df_ratings = df_ratings.loc[(df_ratings.attribute == dim_name)]
        selected_stimuli = df_means.loc[df_means.age > 0].stimulus # Select stimuli with age > 0
        df_ratings = df_ratings.loc[df_ratings.stimulus.isin(selected_stimuli)]

        # Check if coords dataframe has corresponding stimuli
        df_coords = df_coords.loc[df_coords.stimulus.isin(selected_stimuli)]

        # Align coords and ratings
        stimuli_intersect = sorted(list(set(df_coords.stimulus) & set(df_ratings.stimulus)))
        df_coords = df_coords.set_index('stimulus').loc[stimuli_intersect].reset_index()
        df_ratings = df_ratings.loc[df_ratings.stimulus.isin(stimuli_intersect)]


        # Group ratings
        grouped_ratings = df_ratings.groupby('stimulus')['rating'].apply(list)
        grouped_ratings = grouped_ratings.loc[stimuli_intersect] # Ensure order matches coords

        # Pad ratings (find max length dynamically)
        max_len = grouped_ratings.apply(len).max()
        print(f"Max ratings per stimulus: {max_len}")
        padded_ratings = grouped_ratings.apply(lambda l: l + [np.nan] * (max_len - len(l)))

        X = np.stack(df_coords['dlatents'].values) # W latents
        ratings_tensor = torch.tensor(padded_ratings.tolist(), dtype=torch.float32, device=device)

        # Normalize ratings (0-100 -> 0-1, plus offset?)
        ratings_tensor = (ratings_tensor + 0.5) / 101.0 # As per notebook
        ratings_tensor = torch.clamp(ratings_tensor, 0.0, 1.0) # Ensure valid range for Beta dist

        print(f"Loaded data: X shape {X.shape}, Ratings shape {ratings_tensor.shape}")
        return torch.tensor(X, dtype=torch.float32, device=device), ratings_tensor

    except Exception as e:
        print(f"Error loading real data (Placeholder): {e}. Generating dummy data.")
        num_dummy = 1000
        X = torch.randn(num_dummy, config.LATENT_DIM_W, device=device)
        # Dummy beta-distributed ratings
        dummy_alpha = torch.rand(num_dummy, 1, device=device) * 5 + 1 # e.g., alpha ~ U(1, 6)
        dummy_beta = torch.rand(num_dummy, 1, device=device) * 5 + 1  # e.g., beta ~ U(1, 6)
        dist = torch.distributions.Beta(dummy_alpha, dummy_beta)
        ratings_tensor = dist.sample((20,)).permute(1, 0) # Sample 20 ratings per item -> (num_dummy, 20)
        # Add some NaNs
        nan_mask = torch.rand_like(ratings_tensor) > 0.8
        ratings_tensor[nan_mask] = float('nan')
        print(f"Generated dummy data: X shape {X.shape}, Ratings shape {ratings_tensor.shape}")
        return X, ratings_tensor


# --- Dataset & DataLoader ---
class LatentRatingsDataset(Dataset):
    """ Simple Dataset for W latents and corresponding raw ratings. """
    def __init__(self, latents, targets):
        # Expects latents (N, 512) and targets (N, num_ratings)
        self.latents = latents
        self.targets = targets

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.targets[idx]

class MixupDataset(Dataset):
    """ Dataset wrapper for applying Mixup augmentation. """
    def __init__(self, X, y, alpha=0.2):
        self.X = X # Latent vectors
        self.y = y # Ratings tensor (N, num_ratings)
        self.alpha = alpha
        if alpha <= 0:
            print("Mixup alpha is 0, mixup will not be applied.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x1, y1 = self.X[idx], self.y[idx]

        if self.alpha > 0 and self.training: # Apply mixup only during training
             # Sample lambda from Beta distribution
             lam = np.random.beta(self.alpha, self.alpha)
             # Choose a random second sample
             idx2 = np.random.randint(len(self.X))
             x2, y2 = self.X[idx2], self.y[idx2]
             # Perform mixup on inputs
             x_mixed = lam * x1 + (1 - lam) * x2
             # Return mixed input and original targets + lambda
             return x_mixed, y1, y2, torch.tensor(lam, dtype=torch.float32)
        else:
             # If not training or alpha=0, return original sample format for MixupLoss
             return x1, y1, y1, torch.tensor(1.0, dtype=torch.float32)

def mixup_collate_fn(batch):
    """ Collates data points from MixupDataset. """
    # batch is a list of tuples: [(x_mixed, y1, y2, lam), ...] or [(x1, y1, y1, 1.0), ...]
    x, y1, y2, lams = zip(*batch)
    x_stacked = torch.stack(x, dim=0)
    y1_stacked = torch.stack(y1, dim=0)
    y2_stacked = torch.stack(y2, dim=0)
    lams_tensor = torch.stack(lams, dim=0)
    return x_stacked, y1_stacked, y2_stacked, lams_tensor

# --- Evaluation ---
@torch.no_grad()
def validate_trust_model(model, val_dataloader, criterion, device, plot=False):
    """ Evaluates the trust model (AlphaBetaRegressor or Ensemble) on the validation set. """
    model.eval()
    total_loss_val = 0
    all_outputs_mean = [] # Store predicted mean rating
    all_targets_mean = [] # Store empirical mean rating from validation data
    all_losses = []

    for x_val, y_val in val_dataloader:
        x_val, y_val = x_val.to(device), y_val.to(device)

        # Get model prediction (requesting 'a,b' for loss, 'mean' for correlation)
        params_pred = model(x_val, output='a,b')
        mean_pred = model(x_val, output='mean') # Shape (batch, 1)

        # Calculate validation loss using the base NLL criterion (no mixup, no bootstrapping usually)
        # Ensure criterion does not bootstrap during validation
        original_bootstrap_setting = False
        if hasattr(criterion, 'use_bootstrap'):
             original_bootstrap_setting = criterion.use_bootstrap
             criterion.use_bootstrap = False # Disable bootstrap for validation

        val_loss = criterion(params_pred, y_val)

        if hasattr(criterion, 'use_bootstrap'): # Restore setting
             criterion.use_bootstrap = original_bootstrap_setting

        total_loss_val += val_loss.item()
        all_losses.append(val_loss.item())

        # Calculate empirical mean from target ratings (ignore NaNs)
        target_mean = torch.nanmean(y_val, dim=1, keepdim=True) # Shape (batch, 1)

        all_outputs_mean.append(mean_pred.cpu().numpy())
        all_targets_mean.append(target_mean.cpu().numpy())

    mean_val_loss = total_loss_val / len(val_dataloader) if len(val_dataloader) > 0 else 0

    # Concatenate predictions and targets for correlation calculation
    if not all_outputs_mean: # Handle empty validation set
        print("Warning: Validation set empty or processing failed.")
        return mean_val_loss, 0.0 # Return 0 correlation

    all_outputs_np = np.concatenate(all_outputs_mean, axis=0).flatten()
    all_targets_np = np.concatenate(all_targets_mean, axis=0).flatten()

    # Filter out NaNs that might occur if a whole row in y_val was NaN
    valid_mask = ~np.isnan(all_targets_np) & ~np.isnan(all_outputs_np)
    if valid_mask.sum() < 2: # Need at least 2 points for correlation
        mean_corr = 0.0
        print("Warning: Not enough valid data points for validation correlation.")
    else:
        try:
            correlation, _ = pearsonr(all_outputs_np[valid_mask], all_targets_np[valid_mask])
            mean_corr = correlation if not np.isnan(correlation) else 0.0
        except ValueError:
             mean_corr = 0.0 # Handle potential errors in pearsonr

    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(all_losses, bins=30)
        plt.title('Validation Loss Distribution')
        plt.xlabel('Loss')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        if valid_mask.sum() >= 2:
             plt.scatter(all_targets_np[valid_mask], all_outputs_np[valid_mask], alpha=0.3)
             plt.plot([0, 1], [0, 1], 'r--', label='Ideal') # Add diagonal line
             plt.title(f'Validation: Predicted Mean vs Target Mean (Corr={mean_corr:.3f})')
             plt.xlabel('Target Mean Rating')
             plt.ylabel('Predicted Mean Rating')
             plt.xlim(0, 1)
             plt.ylim(0, 1)
             plt.grid(True)
             plt.legend()
        else:
             plt.text(0.5, 0.5, "Not enough data for scatter plot", ha='center', va='center')


        plt.tight_layout()
        plt.show()

    return mean_val_loss, mean_corr

# --- Laplacian Regularization ---
def compute_laplacian_regularization(model, x_unlabeled, device):
    """ Computes ||nabla_x f(x)||^2 regularization term. """
    batch_size = x_unlabeled.size(0)
    x_unlabeled = x_unlabeled.detach().to(device).requires_grad_(True) # Detach and require grad

    # We want gradient of a scalar output w.r.t input x.
    # Model outputs (alpha, beta) or mean. Which one to use for regularization?
    # Original code used model(x).sum(). If model outputs (a,b), sum is a+b (related to dispersion).
    # If model outputs mean, sum is sum(mean).
    # Let's assume we regularize the gradient of the *mean* prediction.
    mean_output = model(x_unlabeled, output="mean") # Get mean prediction

    # Sum the mean output to get a single scalar value per batch element (or sum across batch)
    scalar_output = mean_output.sum() # Total sum across batch

    # Calculate gradients of this scalar w.r.t. x_unlabeled
    grads = torch.autograd.grad(
        outputs=scalar_output,
        inputs=x_unlabeled,
        grad_outputs=torch.ones_like(scalar_output), # d(scalar)/d(scalar) = 1
        create_graph=True, # Allow higher-order derivatives if needed elsewhere
        retain_graph=True, # Keep graph for potential subsequent backward passes
        only_inputs=True   # Only compute grad w.r.t specified inputs
    )[0] # Get the gradient tensor (shape matches x_unlabeled)

    # Calculate the squared L2 norm of the gradient for each sample, then average
    # grads shape: (batch_size, latent_dim)
    grad_norm_sq = grads.norm(p=2, dim=1).pow(2) # Shape (batch_size,)
    mean_grad_norm_sq = grad_norm_sq.mean() # Average over batch

    return mean_grad_norm_sq

# --- Training Loop ---
def training_loop_trust_model(
    model, optimizer, scheduler, epochs, train_dataloader, val_dataloader,
    criterion, device, save_path, use_laplacian_reg=False, lambda_laplacian=1e-4
):
    """ Training loop specifically for the trust model (AlphaBetaRegressor). """
    best_val_loss = float('inf')
    best_mean_corr = -float('inf')
    max_mixup_alpha = config.TRUST_MODEL_MIXUP_ALPHA # Get from config

    # Determine if using mixup based on dataloader's collate_fn
    using_mixup = hasattr(train_dataloader, 'collate_fn') and train_dataloader.collate_fn.__name__ == 'mixup_collate_fn'
    if using_mixup:
        print("Using Mixup augmentation.")
        if not isinstance(criterion, MixupBetaNLLLoss):
             raise TypeError("Criterion must be MixupBetaNLLLoss when using mixup dataloader.")
        base_criterion = criterion.base_criterion # For validation
    else:
        print("Not using Mixup augmentation.")
        if not isinstance(criterion, BetaNLLLoss):
             # Allow MixupBetaNLLLoss if alpha=0? No, stick to base criterion if no mixup loader.
             raise TypeError("Criterion must be BetaNLLLoss when not using mixup dataloader.")
        base_criterion = criterion # For validation

    # Directory for saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch_num in tqdm(range(epochs), desc="Training Trust Model"):
        model.train()
        total_loss_train = 0
        laplacian_loss_total = 0

        # Dynamic mixup alpha based on LR (from original notebook)
        current_lr = optimizer.param_groups[0]['lr']
        if using_mixup and hasattr(train_dataloader.dataset, 'alpha'):
            # Adjust dataset's alpha dynamically if attribute exists
            train_dataloader.dataset.alpha = max(0, min(max_mixup_alpha, (current_lr / config.TRUST_MODEL_LR) * max_mixup_alpha))


        for step, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()

            if using_mixup:
                x, y1, y2, lam = batch_data
                x, y1, y2, lam = x.to(device), y1.to(device), y2.to(device), lam.to(device)
                outputs = model(x, output='a,b') # Predict alpha, beta for mixed input
                loss = criterion(outputs, y1, y2, lam) # Mixup loss handles label mixing
            else:
                x, y = batch_data
                x, y = x.to(device), y.to(device)
                outputs = model(x, output='a,b') # Predict alpha, beta
                loss = criterion(outputs, y) # Standard NLL loss

            total_loss = loss

            # --- Laplacian Regularization ---
            if use_laplacian_reg and lambda_laplacian > 0:
                # Sample unlabeled data (random W vectors)
                # Size should be comparable to batch size for stable regularization estimate
                x_unlabeled = sample_w(n=x.size(0), truncation_psi=np.random.uniform(0.7, 1.0), device=device) # Sample random W
                laplacian_reg = compute_laplacian_regularization(model, x_unlabeled, device)
                total_loss = total_loss + lambda_laplacian * laplacian_reg
                laplacian_loss_total += laplacian_reg.item()

            # Backward pass and optimization step
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients
            optimizer.step()
            # Scheduler step (per step or per epoch based on scheduler type)
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                 scheduler.step() # Step per optimizer step if CosineAnnealingLR with T_max=total_steps
                 #scheduler.step(epoch_num + step / len(train_dataloader)) # Alternative: step per epoch fraction
            elif scheduler is not None: # e.g., LinearWarmup might step per step
                 scheduler.step()


            total_loss_train += total_loss.item()

        # --- Validation ---
        # Frequency based on epoch number (less frequent later)
        val_interval = 5 if epoch_num < 50 else 10 if epoch_num < 100 else 25
        if (epoch_num + 1) % val_interval == 0:
            mean_val_loss, mean_corr = validate_trust_model(
                model, val_dataloader, base_criterion, device, # Use base criterion for validation
                plot=((epoch_num + 1) % (val_interval * 4) == 0) # Plot occasionally
            )

            avg_train_loss = total_loss_train / len(train_dataloader) if len(train_dataloader) > 0 else 0
            avg_laplacian_loss = laplacian_loss_total / len(train_dataloader) if use_laplacian_reg and len(train_dataloader) > 0 else 0


            print(f'Epoch {epoch_num + 1}/{epochs}: Train Loss: {avg_train_loss:.4f} '
                  f'(Laplacian: {avg_laplacian_loss:.4f}), Val Loss: {mean_val_loss:.4f}, '
                  f'Val Corr: {mean_corr:.4f}, LR: {current_lr:.6f}')

            # Save model if validation loss improved
            # Note: Original code saved based on best_val_loss being lower, implying minimization.
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_mean_corr = mean_corr # Store corr corresponding to best loss
                torch.save(model.state_dict(), save_path)
                print(f" *** Saved best model (Val Loss: {best_val_loss:.4f}) to {save_path} ***")
            elif mean_corr > best_mean_corr: # Track best correlation separately if needed
                 best_mean_corr = mean_corr
                 # Optionally save best correlation model separately
                 # torch.save(model.state_dict(), save_path.replace('.pt', '_bestcorr.pt'))

    # Load the best model found during training
    try:
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded best model state from {save_path} (Best Val Loss: {best_val_loss:.4f}, Corr: {best_mean_corr:.4f})")
    except Exception as e:
        print(f"Warning: Could not reload best model state from {save_path}: {e}")

    return model, best_val_loss, best_mean_corr


# --- Main Training Execution ---
def train_trust_models_main(
    target_dim_name=config.TARGET_DIMENSION,
    num_ensemble=config.TRUST_MODEL_ENSEMBLE_SIZE,
    epochs=config.TRUST_MODEL_TRAIN_EPOCHS,
    batch_size=config.TRUST_MODEL_BATCH_SIZE,
    lr=config.TRUST_MODEL_LR,
    weight_decay=config.TRUST_MODEL_WEIGHT_DECAY,
    mixup_alpha=config.TRUST_MODEL_MIXUP_ALPHA,
    use_laplacian_reg=config.TRUST_MODEL_LAPLACIAN_LAMBDA > 0,
    lambda_laplacian=config.TRUST_MODEL_LAPLACIAN_LAMBDA,
    device=config.DEVICE,
    save_dir=config.CHECKPOINT_DIR_TRUST
    ):
    """ Trains an ensemble of trust models for the specified dimension. """

    print(f"\n--- Training Trust Model Ensemble for: {target_dim_name} ---")
    print(f"Ensemble size: {num_ensemble}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    print(f"Mixup Alpha: {mixup_alpha}, Laplacian Reg: {use_laplacian_reg} (Lambda: {lambda_laplacian})")
    print_gpu_memory()

    # 1. Load Data
    X_data, y_data = load_training_data(dim_name=target_dim_name, device=device) # y_data is the raw ratings tensor

    best_models = []
    final_val_metrics = [] # Store (loss, corr) for each model

    # 2. Train each model in the ensemble
    for i in range(num_ensemble):
        print(f"\n--- Training Model {i+1}/{num_ensemble} ---")

        # Split data for this model (different random state per model)
        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42 + i)

        # Create datasets and dataloaders
        if mixup_alpha > 0:
             train_dataset = MixupDataset(X_train, y_train, alpha=mixup_alpha)
             train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mixup_collate_fn, num_workers=0) # Use mixup collate
             criterion = MixupBetaNLLLoss(BetaNLLLoss(use_bootstrap=True)) # Mixup loss wraps base NLL
        else:
             # If no mixup, use standard dataset and NLL loss directly
             train_dataset = LatentRatingsDataset(X_train, y_train)
             train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
             criterion = BetaNLLLoss(use_bootstrap=True) # Base NLL loss

        val_dataset = LatentRatingsDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0) # Larger batch size for validation

        # Initialize model
        model = AlphaBetaRegressor(dim=config.TRUST_MODEL_DIM_INTERNAL).to(device)

        # Optimizer and Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Scheduler setup depends on total steps vs epochs. CosineAnnealingLR often uses epochs.
        # T_max is total number of *scheduler steps*. If stepping per epoch, T_max=epochs.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) # Step per epoch
        # If stepping per optimizer step: T_max = epochs * len(train_loader)


        # Define save path for this specific model instance
        # Match naming convention used in loading: model_{dim}_v{3+i}.pt
        model_save_path = os.path.join(save_dir, f"model_{target_dim_name}_v{3+i}.pt")

        # Run training loop
        best_model_instance, best_loss, best_corr = training_loop_trust_model(
            model, optimizer, scheduler, epochs, train_loader, val_loader,
            criterion, device, model_save_path,
            use_laplacian_reg=use_laplacian_reg, lambda_laplacian=lambda_laplacian
        )

        best_models.append(best_model_instance) # Store the best model instance
        final_val_metrics.append((best_loss, best_corr))
        print(f"Model {i+1} finished. Best Val Loss: {best_loss:.4f}, Best Val Corr: {best_corr:.4f}")
        print_gpu_memory()


    print("\n--- Trust Model Training Complete ---")
    avg_loss = np.mean([m[0] for m in final_val_metrics])
    avg_corr = np.mean([m[1] for m in final_val_metrics])
    print(f"Average Best Validation Loss across ensemble: {avg_loss:.4f}")
    print(f"Average Best Validation Correlation across ensemble: {avg_corr:.4f}")

    # Optionally, create and save the final ensemble model state_dict if needed elsewhere
    # ensemble_final = EnsembleRegressor(best_models).cpu() # Create ensemble on CPU
    # torch.save(ensemble_final.state_dict(), os.path.join(save_dir, f"ensemble_{target_dim_name}_final.pt"))
    # print(f"Saved final ensemble state dict for {target_dim_name}.")

    return best_models, final_val_metrics

if __name__ == "__main__":
    # Example: Run training for the target dimension specified in config
    train_trust_models_main(target_dim_name=config.TARGET_DIMENSION, device=config.DEVICE)
