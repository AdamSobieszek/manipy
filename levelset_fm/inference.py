
# inference.py
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from configs import config
from models.flow_model import VectorFieldTransformer, RatingODE # Flow model and ODE wrapper
from models.trust_model import load_trust_model_ensemble # To load trust model for conditioning
from utils.checkpoint_utils import load_flow_checkpoint # Function to load flow model checkpoint
from utils.stylegan_utils import sample_w, generate_images_from_w, get_w_avg # For sampling start points and generating images
from utils.visualization import display_images_matplotlib # For displaying results

# ODE Solver (using torchdyn as an example, or simple Euler)
try:
    from torchdyn.core import NeuralODE
    print("Using torchdyn for ODE solving.")
    use_torchdyn = True
except ImportError:
    print("torchdyn not found. Using simple Euler method for ODE solving.")
    use_torchdyn = False


def run_inference():
    """ Runs inference using the trained flow model to manipulate latent vectors. """
    device = config.DEVICE
    print(f"--- Running Inference using Flow Model ---")
    print(f"Using device: {device}")

    # --- Load Models ---
    print("Loading models...")
    # Load the trained Flow Model checkpoint
    # Specify the correct path to your ".pt" file containing the model state dict
    flow_model, _, _, _, _, loaded_run_name = load_flow_checkpoint(
        model_path=config.INFERENCE_FLOW_MODEL_PATH, # Path from config
        device=device
    )
    if flow_model is None:
        print("Failed to load flow model. Aborting.")
        return
    flow_model.eval()
    print(f"Flow model loaded (from run: {loaded_run_name}).")

    # Load the corresponding Trust Model ensemble used during training
    trust_model = load_trust_model_ensemble(
        dim_name=config.TARGET_DIMENSION,
        ensemble_size=config.TRUST_MODEL_ENSEMBLE_SIZE,
        checkpoint_dir=config.CHECKPOINT_DIR_TRUST,
        device=device
    )
    trust_model.eval()
    print("Trust model loaded for conditioning.")

    # --- Prepare Input Data (Starting Latents) ---
    num_samples = config.INFERENCE_BATCH_SIZE
    print(f"Generating {num_samples} starting W vectors...")
    # Sample random starting points in W space
    x0_samples = sample_w(n=num_samples, truncation_psi=0.7, device=device) # Example psi

    # Alternative: Load specific starting W vectors if needed
    # x0_samples = torch.load("path/to/start_latents.pt").to(device)

    # Get initial ratings for reference
    with torch.no_grad():
        initial_ratings_logit = trust_model(x0_samples, output='logit')
        print(f"Initial mean logit: {initial_ratings_logit.mean().item():.3f}")

    # --- Define ODE ---
    # Wrap the flow and trust model in the RatingODE class for the solver
    ode_func = RatingODE(flow_model, trust_model).to(device)

    # --- Perform ODE Integration ---
    print(f"Starting ODE integration ({config.INFERENCE_ODE_STEPS} steps)...")
    t_span = torch.linspace(0, 1, config.INFERENCE_ODE_STEPS + 1, device=device) # Time points from 0 to 1

    w_trajectory = [] # To store intermediate steps if needed

    if use_torchdyn:
        # Use torchdyn NeuralODE solver
        neural_ode = NeuralODE(ode_func, solver='dopri5', sensitivity='adjoint') # Choose solver
        # Note: Adjoint sensitivity might require more memory but is often faster for gradients (not needed for inference itself)
        # For pure inference, 'autograd' sensitivity might be simpler, or rely on no_grad context.
        with torch.no_grad():
            xt_final = neural_ode.trajectory(x0_samples, t_span=t_span)[-1] # Get the final state at t=1
            # traj = neural_ode.trajectory(x0_samples, t_span) # Get full trajectory
            # xt_final = traj[-1]
            # w_trajectory = traj.cpu().numpy() # Store trajectory if needed

    else:
        # Simple Euler Integration (Fallback)
        xt = x0_samples.clone()
        w_trajectory.append(xt.cpu().numpy())
        dt = t_span[1] - t_span[0]
        for i in tqdm(range(config.INFERENCE_ODE_STEPS), desc="Euler Steps"):
            with torch.no_grad():
                 # Use ode_func which wraps flow and trust model
                 vector_field = ode_func(t_span[i], xt) # t is technically not used by this ODE func
                 xt = xt + vector_field * dt
                 w_trajectory.append(xt.cpu().numpy()) # Store step
        xt_final = xt


    print("ODE integration finished.")
    final_latents_w = xt_final # Resulting W vectors at t=1

    # --- Evaluate Results ---
    with torch.no_grad():
        final_ratings_logit = trust_model(final_latents_w, output='logit')
        print(f"Final mean logit: {final_ratings_logit.mean().item():.3f}")
        delta_logit = final_ratings_logit - initial_ratings_logit
        print(f"Mean logit change: {delta_logit.mean().item():.3f}")

    # --- Generate and Display Images ---
    print("Generating images from initial and final W vectors...")
    num_display = min(num_samples, 8) # How many samples to display

    initial_images = generate_images_from_w(x0_samples[:num_display], device=device)
    final_images = generate_images_from_w(final_latents_w[:num_display], device=device)

    print("\nInitial Images:")
    display_images_matplotlib(initial_images, grid=True, rows=1, labels=[f"Logit: {l.item():.2f}" for l in initial_ratings_logit[:num_display]])

    print("\nFinal Images (after flow):")
    display_images_matplotlib(final_images, grid=True, rows=1, labels=[f"Logit: {l.item():.2f}" for l in final_ratings_logit[:num_display]])

    # Optional: Visualize trajectory if stored and using few samples
    # if num_samples <= 1 and w_trajectory:
    #     w_traj_np = np.array(w_trajectory) # Shape (steps+1, batch, dim)
    #     # Project to 2D using PCA or specific components if available
    #     # ... visualization code ...


    # Optional: Save final latents
    # save_path = f"inference_results_{config.TARGET_DIMENSION}_{loaded_run_name}.pt"
    # torch.save(final_latents_w.cpu(), save_path)
    # print(f"Saved final W vectors to {save_path}")


if __name__ == "__main__":
    run_inference()