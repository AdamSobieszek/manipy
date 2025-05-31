# utils/checkpoint_utils.py
import torch
import os
import time # Import time
import numpy as np
from models.flow_model import VectorFieldTransformer # Need model definition to instantiate
from configs import config # For default dims etc.

def save_flow_checkpoint(model, optimizer, scheduler, epoch, loss, run_name, best_metrics, path=config.CHECKPOINT_DIR_FLOW, dim_name=config.TARGET_DIMENSION):
    """ Saves flow model state dict and training state separately. """
    os.makedirs(path, exist_ok=True)

    # --- Check if this is the best loss ---
    is_best_loss = loss < best_metrics.get('loss', float('inf'))
    if is_best_loss:
        best_metrics['loss'] = loss # Update the dict tracking best metrics

    # --- Save Model State Dict ---
    model_filename = f"{dim_name}_{run_name}_epoch_{epoch}.pt"
    model_path = os.path.join(path, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Saved flow model state dict to {model_path}")

    # --- Save Training State ---
    state_filename = f"training_state_{dim_name}_{run_name}_epoch_{epoch}.pt"
    state_path = os.path.join(path, state_filename)
    training_state = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss, # Current epoch's average loss
        'best_metrics': best_metrics, # Save the dictionary tracking bests
        'run_name': run_name,
        'config': { # Save key config values for reproducibility check
             'flow_model_depth': config.FLOW_MODEL_DEPTH,
             'flow_model_heads': config.FLOW_MODEL_NUM_HEADS,
             'flow_model_dim_head': config.FLOW_MODEL_DIM_HEAD,
             'flow_model_registers': config.FLOW_MODEL_NUM_REGISTERS,
             'target_dimension': config.TARGET_DIMENSION,
        }
    }
    torch.save(training_state, state_path)
    print(f"Saved training state to {state_path}")

    # --- Save Best Loss Model Separately ---
    if is_best_loss:
        best_model_path = os.path.join(path, f"{dim_name}_{run_name}_best_loss.pt")
        best_state_path = os.path.join(path, f"training_state_{dim_name}_{run_name}_best_loss.pt")
        torch.save(model.state_dict(), best_model_path)
        torch.save(training_state, best_state_path)
        print(f"*** Saved new best loss model to {best_model_path} ***")


def load_flow_checkpoint(model_path, state_path=None, device=config.DEVICE):
    """ Loads flow model and training state from separate files. """

    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}. Cannot load.")
        return None, None, None, 0, config.BEST_METRICS.copy(), f"error_run_{int(time.time())}"

    # --- Determine State Path ---
    if state_path is None:
        # Infer state path from model path conventions used in saving
        base_name = os.path.basename(model_path)
        dir_name = os.path.dirname(model_path)
        # Example inference: replace dim_name with training_state_dim_name
        # This needs to exactly match the saving convention
        state_filename = base_name.replace(f"{config.TARGET_DIMENSION}_", f"training_state_{config.TARGET_DIMENSION}_", 1)
        if "best_loss" in state_filename:
             state_filename = state_filename.replace("_best_loss.pt", "_best_loss.pt", 1) # Ensure consistent ending
        # If model file is like 'dim_run_epoch.pt', state is 'training_state_dim_run_epoch.pt'

        potential_state_path = os.path.join(dir_name, state_filename)

        if os.path.exists(potential_state_path):
            state_path = potential_state_path
            print(f"Inferred state path: {state_path}")
        else:
            # Fallback if inference fails - try simpler replacement
            simple_state_path = model_path.replace(f"{config.TARGET_DIMENSION}_", f"training_state_{config.TARGET_DIMENSION}_")
            if os.path.exists(simple_state_path):
                 state_path = simple_state_path
                 print(f"Using fallback state path: {state_path}")
            else:
                 print(f"Could not automatically find state path for {model_path}. State will not be loaded.")
                 state_path = None # Ensure state_path is None if not found


    # --- Load Model State Dict ---
    print(f"Loading flow model state dict from: {model_path}")
    # Instantiate the model architecture first
    # Important: Ensure these parameters match the saved model!
    # We could store key arch params in the state file for verification
    flow = VectorFieldTransformer(
        dim=config.FLOW_MODEL_DIM,
        depth=config.FLOW_MODEL_DEPTH,
        num_heads=config.FLOW_MODEL_NUM_HEADS,
        dim_head=config.FLOW_MODEL_DIM_HEAD,
        num_registers=config.FLOW_MODEL_NUM_REGISTERS,
        dropout=config.FLOW_MODEL_DROPOUT # Dropout during load doesn't matter as much if eval() follows
    ).to(device)
    try:
        flow.load_state_dict(torch.load(model_path, map_location=device))
        print("Flow model state dict loaded successfully.")
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        print("Model may have incorrect architecture or corrupted file.")
        return None, None, None, 0, config.BEST_METRICS.copy(), f"error_run_{int(time.time())}" # Indicate failure


    # --- Load Training State (if path exists) ---
    optimizer = None
    scheduler = None
    start_epoch = 0
    best_metrics = config.BEST_METRICS.copy() # Start with defaults
    run_name = f"resumed_{int(time.time())}" # Default if loading fails

    if state_path and os.path.exists(state_path):
        print(f"Loading training state from: {state_path}")
        try:
            training_state = torch.load(state_path, map_location=device) # Allow pickle

            # Recreate optimizer and scheduler
            optimizer = torch.optim.AdamW(
                flow.parameters(),
                lr=config.FLOW_LR, # Use config LR as default, state might override below
                weight_decay=config.FLOW_WEIGHT_DECAY,
                betas=config.FLOW_BETAS
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.FLOW_SCHEDULER_T_MAX # Use config T_max
            )

            # Load states
            if 'optimizer_state_dict' in training_state:
                optimizer.load_state_dict(training_state['optimizer_state_dict'])
                # Move optimizer state to correct device (important!)
                for state in optimizer.state.values():
                     for k, v in state.items():
                          if isinstance(v, torch.Tensor):
                               state[k] = v.to(device)
                print("Optimizer state loaded.")
            else: print("Warning: Optimizer state not found in checkpoint.")


            if 'scheduler_state_dict' in training_state:
                # Need to handle potential issues if config mismatch (e.g., T_max changed)
                try:
                    scheduler.load_state_dict(training_state['scheduler_state_dict'])
                    print("Scheduler state loaded.")
                except Exception as e:
                     print(f"Warning: Could not load scheduler state: {e}. Scheduler might reset.")
            else: print("Warning: Scheduler state not found in checkpoint.")


            start_epoch = training_state.get('epoch', -1) + 1 # Resume from next epoch
            # Carefully update best_metrics dictionary
            loaded_best = training_state.get('best_metrics', {})
            best_metrics.update(loaded_best) # Overwrite defaults with loaded values

            run_name = training_state.get('run_name', run_name) # Use loaded run name

            print(f"Resuming training from epoch {start_epoch}.")
            print(f"Loaded best metrics: {best_metrics}")
            print(f"Original run name: {run_name}")

        except Exception as e:
            print(f"Error loading training state from {state_path}: {e}")
            print("Starting optimizer, scheduler, and epoch count from scratch.")
            # Keep the loaded model, but reset training components
            optimizer = torch.optim.AdamW(flow.parameters(), lr=config.FLOW_LR, weight_decay=config.FLOW_WEIGHT_DECAY, betas=config.FLOW_BETAS)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.FLOW_SCHEDULER_T_MAX)
            start_epoch = 0
            best_metrics = config.BEST_METRICS.copy()
            # Keep the loaded run_name or generate new one? Let's keep loaded if available.
            try: run_name = training_state.get('run_name', run_name)
            except: pass # If training_state itself failed to load

    else:
        print("No training state file found or specified. Initializing optimizer and scheduler.")
        optimizer = torch.optim.AdamW(flow.parameters(), lr=config.FLOW_LR, weight_decay=config.FLOW_WEIGHT_DECAY, betas=config.FLOW_BETAS)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.FLOW_SCHEDULER_T_MAX)
        start_epoch = 0
        best_metrics = config.BEST_METRICS.copy()
        # Generate a new run name based on the loaded model's likely name
        try:
            base_model_name = os.path.basename(model_path).replace('.pt','')
            parts = base_model_name.split('_')
            if len(parts) > 2: # Assuming dim_run_epoch format
                 run_name = parts[1]
        except:
             pass # Stick with default if parsing fails


    return flow, optimizer, scheduler, start_epoch, best_metrics, run_name
