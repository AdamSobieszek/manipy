
# train_flow.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from tqdm import tqdm
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from configs import config
from data.dataset import OTDataset, get_dataloader
from data.data_utils import calculate_cost_multiplier # For dynamic cost calculation
from models.flow_model import VectorFieldTransformer # Flow model definition
from models.trust_model import load_trust_model_ensemble # To load pre-trained trust model
from utils.flow_utils import sample_conditional_pt, compute_conditional_vector_field
from utils.checkpoint_utils import save_flow_checkpoint, load_flow_checkpoint
from utils.general_utils import print_gpu_memory

def train_flow_model():
    """ Main function to train the VectorFieldTransformer flow model. """
    device = config.DEVICE
    print(f"--- Starting Flow Model Training for: {config.TARGET_DIMENSION} ---")
    print(f"Using device: {device}")
    print_gpu_memory()

    # --- Seed for reproducibility ---
    torch.manual_seed(42)
    np.random.seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)

    # --- Load Pre-trained Trust Model ---
    # This is required for conditioning and dynamic cost calculation
    trust_model = load_trust_model_ensemble(
        dim_name=config.TARGET_DIMENSION,
        ensemble_size=config.TRUST_MODEL_ENSEMBLE_SIZE,
        checkpoint_dir=config.CHECKPOINT_DIR_TRUST,
        device=device
    )
    trust_model.eval()
    # Define the logit function using the loaded trust model
    # This function will be passed to calculate_cost_multiplier
    logit_fn = lambda x: trust_model(x, output='logit')
    print("Pre-trained trust model loaded for conditioning.")

    # --- Initialize Flow Model, Optimizer, Scheduler ---
    flow_model = VectorFieldTransformer( # Uses parameters from config
        dim=config.FLOW_MODEL_DIM,
        depth=config.FLOW_MODEL_DEPTH,
        num_heads=config.FLOW_MODEL_NUM_HEADS,
        dim_head=config.FLOW_MODEL_DIM_HEAD,
        num_registers=config.FLOW_MODEL_NUM_REGISTERS,
        dropout=config.FLOW_MODEL_DROPOUT
    ).to(device)

    optimizer = optim.AdamW(
        flow_model.parameters(),
        lr=config.FLOW_LR,
        weight_decay=config.FLOW_WEIGHT_DECAY,
        betas=config.FLOW_BETAS
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.FLOW_SCHEDULER_T_MAX # T_max is total steps or epochs? Assume steps here if T_max large
    )

    # --- Load Checkpoint (if resuming flow model training) ---
    start_epoch = config.FLOW_START_EPOCH
    run_name = config.FLOW_RUN_NAME # Use config default or loaded value
    best_metrics = config.BEST_METRICS.copy() # Get mutable copy

    # Specify path to checkpoint if resuming
    resume_model_path = None # e.g., "flow_checkpoints/attractive_123456_epoch_100.pt"
    if resume_model_path and os.path.exists(resume_model_path):
        print(f"Attempting to resume from checkpoint: {resume_model_path}")
        flow_model_loaded, opt_loaded, sched_loaded, epoch_loaded, metrics_loaded, name_loaded = load_flow_checkpoint(
            model_path=resume_model_path,
            device=device
        )
        if flow_model_loaded:
             flow_model = flow_model_loaded
             optimizer = opt_loaded if opt_loaded else optimizer # Use loaded if successful
             scheduler = sched_loaded if sched_loaded else scheduler
             start_epoch = epoch_loaded
             best_metrics.update(metrics_loaded) # Update best metrics dict
             run_name = name_loaded # Use the run name from the checkpoint
             print(f"Resumed from epoch {start_epoch}. Run name: {run_name}")
        else:
             print("Checkpoint loading failed. Starting from scratch.")
    else:
        print("No valid resume checkpoint specified. Starting training from scratch.")
        run_name = f"{int(time.time())}" # Assign new run name if not resuming


    # --- Prepare Data ---
    print("Loading dataset...")
    dataset = OTDataset(
        dim_name=config.TARGET_DIMENSION,
        dataset_dir=config.DATASET_DIR,
        device=device # OTDataset handles moving W to device if possible
    )
    # DataLoader yielding large batches as per original code structure
    # Ensure data generation created enough pairs
    if len(dataset) < config.FLOW_DATALOADER_BATCH_SIZE:
         print(f"Warning: Dataset ({len(dataset)} pairs) is smaller than dataloader batch size ({config.FLOW_DATALOADER_BATCH_SIZE}).")
         # Consider reducing dataloader batch size or steps per epoch if this happens often
         effective_dl_batch_size = min(len(dataset), config.FLOW_DATALOADER_BATCH_SIZE)
         if effective_dl_batch_size < config.FLOW_BATCH_SIZE_PER_STEP:
             print("Error: Dataset too small for even one step per epoch chunk. Aborting.")
             return
         print(f"Using adjusted dataloader batch size: {effective_dl_batch_size}")
    else:
         effective_dl_batch_size = config.FLOW_DATALOADER_BATCH_SIZE

    # Use standard DataLoader (OTDataset loads pairs, not groups necessarily)
    dataloader = get_dataloader(
         dataset,
         batch_size=effective_dl_batch_size, # Loads one large chunk per iteration
         shuffle=True,
         num_workers=0 # Set > 0 if data loading is bottleneck and W is on CPU
    )
    print("Dataset and DataLoader prepared.")

    # --- Training Loop ---
    print(f"Starting flow training loop from epoch {start_epoch}... Run: {run_name}")
    total_epochs = config.FLOW_TOTAL_EPOCHS
    steps_per_epoch_chunk = config.FLOW_EPOCH_CHUNK_STEPS
    bs_per_step = config.FLOW_BATCH_SIZE_PER_STEP
    accum_steps = config.FLOW_ACCUMULATION_STEPS
    effective_batch_size = bs_per_step * accum_steps

    epoch_losses = [] # Accumulates losses within an epoch chunk for averaging

    for k, (X0_epoch, X1_epoch, Q_epoch, Dr_epoch) in enumerate(tqdm(dataloader, initial=start_epoch, total=total_epochs, desc="Epoch Chunks"), start=start_epoch):
        if k >= total_epochs: break # Stop if we reach target total epochs

        flow_model.train()
        current_epoch_chunk_losses = [] # Losses specific to this chunk's steps

        # Ensure data is on the correct device (OTDataset might keep W on device)
        X0_epoch, X1_epoch, Q_epoch, Dr_epoch = X0_epoch.to(device), X1_epoch.to(device), Q_epoch.to(device), Dr_epoch.to(device)

        # --- Precompute for the epoch chunk ---
        # 1. Calculate dynamic cost multiplier based on trust model logits
        cost_multiplier = calculate_cost_multiplier(X0_epoch, X1_epoch, logit_fn, device)

        # 2. Compute target vector field directions (UNNORMALIZED)
        Ut_epoch = compute_conditional_vector_field(X0_epoch, X1_epoch) # x1 - x0

        # --- Inner loop over steps within the epoch chunk ---
        optimizer.zero_grad() # Zero gradients at the start of the accumulation cycle
        accumulated_loss_val = 0.0 # Tracks loss for logging within accumulation cycle

        # Determine actual number of steps based on loaded data size vs requested steps
        num_samples_in_chunk = X0_epoch.size(0)
        actual_steps = min(steps_per_epoch_chunk, num_samples_in_chunk // bs_per_step)
        if actual_steps == 0 and num_samples_in_chunk > 0:
             actual_steps = 1 # Ensure at least one step if there's data
             bs_per_step = num_samples_in_chunk # Adjust batch size if needed for one step

        for i in range(actual_steps):
            # --- Get mini-batch for this step ---
            start_idx = i * bs_per_step
            end_idx = start_idx + bs_per_step
            x0 = X0_epoch[start_idx:end_idx]
            x1 = X1_epoch[start_idx:end_idx]
            mult = cost_multiplier[start_idx:end_idx]
            ut_unnormalized = Ut_epoch[start_idx:end_idx]

            if x0.shape[0] == 0: continue # Skip if somehow batch is empty

            # --- Normalize target vector field 'ut' ---
            with torch.no_grad():
                 norm_ut = torch.linalg.norm(ut_unnormalized, dim=-1, keepdim=True)
                 # Handle potential zero norm (x0=x1 case) -> target is zero vector
                 zero_norm_mask = norm_ut < 1e-9
                 ut = torch.zeros_like(ut_unnormalized)
                 ut[~zero_norm_mask] = ut_unnormalized[~zero_norm_mask] / norm_ut[~zero_norm_mask]
                 # Quality (multiplier) should be zero where ut had zero norm (original x0=x1)
                 quality = mult
                 quality[zero_norm_mask.squeeze()] = 0.0 # Zero out quality where target vector is undefined/zero

            # --- Sample time t and point xt on the path ---
            t = torch.rand(x0.shape[0], device=device)
            # Sigma scheduling as per original code (random factor squared)
            sigma_val = config.FLOW_SIGMA * (np.random.random()**2) # Base sigma * random factor
            xt = sample_conditional_pt(x0, x1, t, sigma=sigma_val)

            # --- Get rating condition at xt ---
            with torch.no_grad():
                rating_condition = logit_fn(xt) # Shape (batch, 1)

            # --- Predict vector field vt = flow(xt, condition) ---
            vt = flow_model(xt, rating_condition)

            # --- Calculate Loss ---
            # Loss: quality-weighted squared error between predicted vt and target ut
            loss = torch.mean(quality * ((vt - ut) ** 2)) # Use mean, NaNs handled by quality=0

            # Handle potential NaN loss
            if torch.isnan(loss):
                print(f"\nWarning: NaN loss detected at epoch {k}, step {i}. Skipping step.")
                loss_value = np.nan
                # Reset gradients for this step? Or just skip accumulation?
                optimizer.zero_grad() # Reset gradients for safety
                accumulated_loss_val = 0.0 # Reset accumulated loss for this cycle
                continue # Skip accumulation and backward pass
            else:
                loss_value = loss.item()

            # Scale loss for gradient accumulation
            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            accumulated_loss_val += loss_value # Accumulate *unscaled* loss for logging

            # --- Optimizer Step after Accumulation ---
            if (i + 1) % accum_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=config.FLOW_CLIP_GRAD_NORM)

                # Optimizer step
                optimizer.step()

                # Scheduler step (adjust based on scheduler type)
                # CosineAnnealingLR usually steps per optimizer step or per epoch
                scheduler.step() # Step per optimizer step

                # Reset gradients for the next cycle
                optimizer.zero_grad()

                # Log accumulated loss
                step_loss = accumulated_loss_val / accum_steps if not np.isnan(accumulated_loss_val) else np.nan
                current_epoch_chunk_losses.append(step_loss)
                epoch_losses.append(step_loss) # Collect globally for overall average trend
                # tqdm description can be updated here if needed
                # pbar.set_postfix({"Step Loss": f"{step_loss:.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.6f}"})

                accumulated_loss_val = 0.0 # Reset accumulated loss


        # Handle partial accumulation at the end of the inner loop if needed
        if (actual_steps % accum_steps != 0) and accumulated_loss_val > 0:
            print("Processing partial accumulation step at end of epoch chunk.")
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=config.FLOW_CLIP_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            partial_step_loss = accumulated_loss_val / (actual_steps % accum_steps) if not np.isnan(accumulated_loss_val) else np.nan
            if not np.isnan(partial_step_loss):
                 current_epoch_chunk_losses.append(partial_step_loss)
                 epoch_losses.append(partial_step_loss)


        # --- End of Epoch Chunk ---
        mean_chunk_loss = np.nanmean(current_epoch_chunk_losses) if current_epoch_chunk_losses else np.nan
        print(f"\nEpoch Chunk {k} finished. Mean Loss: {mean_chunk_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.8f}")

        # Optional: Add evaluation step here periodically (e.g., every N epochs)
        # to update best_metrics['quality'], etc. for checkpointing 'best' models based on metrics.

        # --- Checkpointing ---
        if (k + 1) % config.FLOW_CHECKPOINT_INTERVAL == 0:
             # Use the average loss from recent steps or the current chunk loss for checkpoint metadata
             checkpoint_loss = mean_chunk_loss if not np.isnan(mean_chunk_loss) else best_metrics.get('loss', float('inf'))

             # Update the 'loss' in best_metrics if current is better before saving
             if checkpoint_loss < best_metrics.get('loss', float('inf')):
                 best_metrics['loss'] = checkpoint_loss

             save_flow_checkpoint(
                 model=flow_model,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 epoch=k, # Save based on epoch chunk number k
                 loss=checkpoint_loss,
                 run_name=run_name,
                 best_metrics=best_metrics, # Pass the potentially updated dict
                 dim_name=config.TARGET_DIMENSION
             )
             print_gpu_memory() # Check memory usage after checkpointing


    print("--- Flow Model Training Finished ---")

if __name__ == "__main__":
    train_flow_model()
