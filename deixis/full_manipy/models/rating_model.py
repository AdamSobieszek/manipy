
# models/trust_model.py
import torch
import torch.nn as nn
import os

from .. import config
from .layers import Exponent # Import auxiliary layer

class AlphaBetaRegressor(nn.Module):
    """ Predicts Alpha and Beta parameters of a Beta distribution from W latent. """
    def __init__(self, dim=config.TRUST_MODEL_DIM_INTERNAL, output_activation=Exponent(), w_avg=None): # Default to Exp activation
        super().__init__()
        self.w_avg = w_avg #if w_avg is not None else Eget_w_avg().detach() # Get w_avg, keep on its default device initially

        # Define the network structure
        self.network = nn.Sequential(
            nn.Linear(config.LATENT_DIM_W, dim), # Input is W latent (512)
            nn.BatchNorm1d(dim),
            nn.ReLU(),

            nn.Linear(dim, dim * 2),
            nn.BatchNorm1d(dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(dim * 2, dim * 4),
            nn.BatchNorm1d(dim * 4),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(dim * 4, config.LATENT_DIM_W), # Project back to 512
            nn.BatchNorm1d(config.LATENT_DIM_W),
            nn.ReLU(),
            # Additional layer as per original code? Added Linear(512,512)
            nn.Linear(config.LATENT_DIM_W, config.LATENT_DIM_W),
            # Note: Original code had BatchNorm -> ReLU -> Linear(512, 512) *without final activation* before head
        )

        # Head predicts 2 outputs (for alpha, beta)
        self.head = nn.Sequential(nn.Linear(config.LATENT_DIM_W, config.TRUST_MODEL_TARGET_DIM), # Output dim = 2
        )

        # Output activation to ensure positivity (e.g., softplus, exp)
        # Using Exponent layer for direct exp as in original code
        self.output_activation = output_activation if output_activation is not None else nn.Identity()

        # Dictionary for different output modes
        self.output_modes = {
            "a,b": self._get_alpha_beta, # Directly return processed head output (alpha, beta)
            "mean": self._get_mean,       # Calculate mean: alpha / (alpha + beta)
            "logit": self._get_logit       # Calculate logit of the mean
            # Add "mean+var" etc. if needed
        }

    def _get_alpha_beta(self, params):
        # params are already activated (e.g., exponentiated)
        return params # Shape (batch, 2)

    def _get_mean(self, params):
        # params are alpha, beta (shape: batch, 2)
        alpha = params[:, :1]
        beta = params[:, 1:]
        # Avoid division by zero, ensure mean is in (0, 1)
        alpha_beta_sum = torch.clamp(alpha + beta, min=1e-9)
        mean = torch.clamp(alpha / alpha_beta_sum, min=1e-6, max=1.0 - 1e-6)
        return mean # Shape (batch, 1)

    def _get_logit(self, params):
        mean = self._get_mean(params)
        # Logit function expects input in (0, 1) exclusive
        return torch.logit(mean) # Shape (batch, 1)


    def forward(self, x, output="a,b"):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input W latent vectors (batch_size, 512).
            output (str): Specifies the desired output format ('a,b', 'mean', 'logit').
        Returns:
            torch.Tensor: Model output based on the specified mode.
        """
        if output not in self.output_modes:
            raise ValueError(f"Invalid output mode '{output}'. Choose from {list(self.output_modes.keys())}")

        # Ensure w_avg is on the same device as input x
        if not isinstance(self.w_avg, torch.Tensor):
            self.w_avg = torch.tensor(self.w_avg)
            
        w_avg_dev = self.w_avg.to(x.device)

        # Center the input W vector
        x_centered = x - w_avg_dev

        # Pass through the main network
        features = self.network(x_centered)

        # Pass through the prediction head
        params_raw = self.head(features)

        # Apply output activation (e.g., exp to ensure alpha, beta > 0)
        params_activated = self.output_activation(params_raw)

        # Return the desired output format
        return self.output_modes[output](params_activated)


class EnsembleRegressor(nn.Module):
    """ Ensembles multiple models by averaging their outputs. """
    def __init__(self, models, model_kwargs={'output':"a,b"}):
        super().__init__()
        self.models = nn.ModuleList(models)
        if not models:
            raise ValueError("Ensemble must contain at least one model.")
        self.model_kwargs = model_kwargs

    def forward(self, x, *args, **kwargs):
        """ Averages the outputs of all models in the ensemble. """
        # Collect outputs from each model
        all_outputs = [model(x, *args, **(self.model_kwargs | kwargs)) for model in self.models]
        # Stack outputs along a new dimension (dim=0)
        stacked_outputs = torch.stack(all_outputs, dim=0)
        # Compute the mean along the ensemble dimension (dim=0)
        mean_output = torch.mean(stacked_outputs, dim=0)
        return mean_output

# --- Loading Functions ---

# --- Helper Activation Module (Unchanged) ---
class StableExponent(nn.Module):
    """
    A custom activation function that provides a stable alternative to torch.exp().
    It is C1 continuous, transitioning from exp(x) to a linear function at a threshold.
    """
    def __init__(self, threshold: float = 10.0):
        super().__init__()
        self.threshold = threshold
        self.exp_threshold = torch.exp(torch.tensor(self.threshold, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the stable exponential function."""
        # The linear part is defined as f(x) = a*x + b where
        # a = exp(threshold)
        # b = exp(threshold) * (1 - threshold)
        linear_part = self.exp_threshold * (x - self.threshold + 1)
        return torch.where(
            x > self.threshold,
            linear_part,
            torch.exp(x)
        )

# --- Model Definition (Unchanged) ---
class AlphaBetaRegressorNew(nn.Module):
    """
    A neural network that regresses the alpha and beta parameters of a Beta
    distribution from a 512-dimensional latent vector.
    """
    def __init__(self, latent_dim: int = 512, w_avg: torch.Tensor = None, hidden_dim_multiplier: list = [8, 4, 2], dropout_rates: list = [0.1, 0.1, 0.1], exp_threshold: float = 10.0):
        super().__init__()
        if not isinstance(w_avg, torch.Tensor):
            raise TypeError("w_avg must be a PyTorch Tensor.")
        self.register_buffer('w_avg', w_avg)

        layers = []
        in_features = latent_dim
        for i, mult in enumerate(hidden_dim_multiplier):
            out_features = latent_dim * mult
            layers.extend([
                nn.Linear(in_features, out_features),
                # nn.BatchNorm1d(out_features),
                nn.ReLU(),
            ])
            if i < len(dropout_rates):
                layers.append(nn.Dropout(dropout_rates[i]))
            in_features = out_features

        layers.extend([
            nn.Linear(in_features, latent_dim),
            # nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        ])
        
        self.network = nn.Sequential(*layers)
        
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 2),
            StableExponent(threshold=exp_threshold)
        )

        # Dictionary for different output modes
        self.output_modes = {
            "a,b": lambda x: x, # Directly return processed head output (alpha, beta)
            "mean": self._get_mean,       # Calculate mean: alpha / (alpha + beta)
            "logit": self._get_logit       # Calculate logit of the mean
            # Add "mean+var" etc. if needed
        }

    def _get_mean(self, params):
        # params are alpha, beta (shape: batch, 2)
        alpha = params[:, :1]
        beta = params[:, 1:]
        # Avoid division by zero, ensure mean is in (0, 1)
        alpha_beta_sum = torch.clamp(alpha + beta, min=1e-9)
        mean = torch.clamp(alpha / alpha_beta_sum, min=1e-6, max=1.0 - 1e-6)
        return mean # Shape (batch, 1)

    def _get_logit(self, params):
        mean = self._get_mean(params)
        # Logit function expects input in (0, 1) exclusive
        return torch.logit(mean) # Shape (batch, 1)
    
    def forward(self, x: torch.Tensor, output="a,b") -> torch.Tensor:
        x_centered = x - self.w_avg
        features = self.network(x_centered)
        params = self.head(features)
        return self.output_modes[output](params)

def load_trust_model_ensemble(dim_name, ensemble_size, checkpoint_dir, device, dtype=torch.float32, w_avg=None):
    """ Loads the ensemble of AlphaBetaRegressor models for a specific dimension. """
    print(f"Loading trust model ensemble for dimension: {dim_name}")
    if ensemble_size == 0:
        suffix = ""
        for path in [f"ensemble_model_{dim_name}{suffix}.pt", f"ensemble_{dim_name}{suffix}.pt"]:
            ckpt_path = os.path.join(checkpoint_dir, path) # Match naming
            if os.path.exists(ckpt_path):
                model_instance = EnsembleRegressor([MeanRegressor(512,1) for _ in range(8)], model_kwargs={}).to(device)
                model_instance.load_state_dict(torch.load(ckpt_path, map_location=device))
                model_instance.eval()
                return model_instance
        for path in [f"model_new_{dim_name}{suffix}.pt", f"{dim_name}{suffix}.pt"]:
            ckpt_path = os.path.join(checkpoint_dir, path) # Match naming
            if os.path.exists(ckpt_path):
                model_instance = AlphaBetaRegressorNew(w_avg=w_avg).to(device, dtype)
                state_dict = torch.load(ckpt_path, map_location=device)
                model_instance.load_state_dict(state_dict)
                model_instance.eval()
                return model_instance
        
    models = []
    for i in range(1 if ensemble_size == 0 else ensemble_size):
        # Construct path based on naming convention from training script
        # Example: model_trustworthy_v3.pt
        suffix = f"_v{3+i}" if ensemble_size > 0 else ""
        for path in [f"model_{dim_name}{suffix}.pt", f"{dim_name}{suffix}.pt"]:
            ckpt_path = os.path.join(checkpoint_dir, path) # Match naming
            if os.path.exists(ckpt_path):
                model_instance = AlphaBetaRegressor(dim=config.TRUST_MODEL_DIM_INTERNAL, w_avg=w_avg).to(device, dtype)

                try:
                    state_dict = torch.load(ckpt_path, map_location=device)
                    model_instance.load_state_dict(state_dict)
                    print(f" Loaded weights for model {i} from {ckpt_path}")
                except Exception as e:
                    print(f"Warning: Error loading state dict from {ckpt_path}: {e}. Using initialized weights for model {i}.")
                    # Keep the initialized model_instance

                model_instance.eval() # Set to evaluation mode
                models.append(model_instance)

    if not models:
        raise RuntimeError(f"Could not load any models for the trust ensemble '{dim_name}'.")

    ensemble_model = EnsembleRegressor(models).to(device, dtype)
    ensemble_model.eval()
    print("Trust model ensemble loaded.")
    return ensemble_model

class MeanRegressor(nn.Module):
     """ Simple MLP regressor used for control models (age, gender). """
     def __init__(self, latent_dim=config.LATENT_DIM_W, target_dim=1):
         super().__init__()
         # Network structure from the notebook example
         self.network = nn.Sequential(
             nn.Linear(latent_dim, 1024),
             nn.BatchNorm1d(1024),
             nn.ReLU(),
             # nn.Dropout(0.1), # Optional dropout

             nn.Linear(1024, 2048),
             nn.BatchNorm1d(2048),
             nn.ReLU(),
             nn.Dropout(0.2), # From notebook

             nn.Linear(2048, 512),
             nn.BatchNorm1d(512),
             nn.ReLU(),
             nn.Dropout(0.2), # From notebook

             nn.Linear(512, target_dim)
             # No output activation specified for control models in notebook
         )

     def forward(self, x):
         # Control models don't seem to subtract w_avg in the notebook usage
         return self.network(x)


def load_control_models(path_template, control_names, device):
    """ Loads the pre-trained control models (age, gender, etc.). """
    control_models_dict = {}
    print("Loading control models...")
    for name in control_names:
        # Assuming each control model is an ensemble of MeanRegressor
        ensemble_members = []
        # Need to know the ensemble size for control models if they are ensembles
        # Assuming size 8 based on trust model example, adjust if different
        control_ensemble_size = 8
        for i in range(control_ensemble_size):
             # Instantiate the base regressor
             # Target dim is 1 for age, gender etc.
             regressor = MeanRegressor(latent_dim=config.LATENT_DIM_W, target_dim=1).to(device)
             ensemble_members.append(regressor)

        # Create the ensemble
        ensemble = EnsembleRegressor(ensemble_members).to(device)

        # Load the saved state dict for the *entire ensemble*
        ckpt_path = path_template.format(name)
        if os.path.exists(ckpt_path):
            try:
                state_dict = torch.load(ckpt_path, map_location=device)
                ensemble.load_state_dict(state_dict)
                print(f" Loaded control model '{name}' from {ckpt_path}")
            except Exception as e:
                 print(f"Warning: Error loading control model '{name}' state dict from {ckpt_path}: {e}. Using initialized weights.")
        else:
            print(f"Warning: Control model checkpoint not found for '{name}' at {ckpt_path}. Using initialized weights.")

        ensemble.eval() # Set to evaluation mode
        control_models_dict[name] = ensemble # Store by name

    print("Control models loaded.")
    # Return as a list in the order requested, similar to original code if needed
    loaded_list = [control_models_dict.get(name) for name in config.CONTROL_MODEL_NAMES]
    # Check if all were loaded
    if None in loaded_list:
         print("Warning: Not all control models could be loaded.")
         # Decide error handling: raise error or return partial list?
         # Returning potentially incomplete list to match original code behavior
         # Filter out None values?
         loaded_list = [m for m in loaded_list if m is not None]


    # The original code seemed to access them by index: control_models[1] for gender, [2] for age
    # Ensure the order in config.CONTROL_MODEL_NAMES matches this expected indexing if used later
    # Example: config.CONTROL_MODEL_NAMES = ['happy', 'gender', 'age'] -> index 1 is gender, 2 is age
    return loaded_list # Return list [happy_model, gender_model, age_model]



