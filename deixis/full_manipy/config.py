import torch
import time
import os

# --- Device Configuration ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use CUDA if available


# --- Paths ---
# Adjust these paths based on your project structure and where data/models are stored
# Example paths - MODIFY THESE
PROJECT_ROOT = "./" # Or '/workspace/' etc. #/Users/adamsobieszek/PycharmProjects/psychGAN
STYLEGAN_MODEL_PATH = os.path.join(PROJECT_ROOT, "stylegan2-ffhq-1024x1024.pkl")
PSYCHGAN_REPO_PATH = os.path.join(PROJECT_ROOT, "content/psychGAN")
BIN_FILES_DIR = os.path.join(PROJECT_ROOT, "bin_files") # Directory to store generated W/Z bins
DATASET_DIR_BASE = os.path.join(PROJECT_ROOT, "dataset") # Base directory for processed datasets (e.g., dataset_trustworthy)
CHECKPOINT_DIR_TRUST = os.path.join(PROJECT_ROOT, "best_models") # Checkpoints for trust model
CHECKPOINT_DIR_FLOW = os.path.join(PSYCHGAN_REPO_PATH, "flow_checkpoints") # Checkpoints for flow model
OMI_DATA_DIR = os.path.join(PROJECT_ROOT, "content") # Where attribute_ratings.csv etc. might be

# --- Environment Setup ---
# Used by prepare_environment.py or setup functions
GIT_TOKEN = "ghp_YOUR_TOKEN_HERE" # Replace with your token if needed for private repo access

# --- StyleGAN Configuration ---
LATENT_DIM_Z = 512
LATENT_DIM_W = 512
TRUNCATION_PSI_DEFAULT = 0.5

# --- Trust Model Configuration ---
TRUST_MODEL_DIM_BASE = 512 # Input dimension W
TRUST_MODEL_DIM_INTERNAL = 512 # Internal dimension for AlphaBetaRegressor
TRUST_MODEL_TARGET_DIM = 2 # Output dim for Alpha/Beta
TRUST_MODEL_ENSEMBLE_SIZE = 8
TRUST_MODEL_TRAIN_EPOCHS = 300
TRUST_MODEL_BATCH_SIZE = 16
TRUST_MODEL_LR = 0.1e-3
TRUST_MODEL_WEIGHT_DECAY = 0.001
TRUST_MODEL_MIXUP_ALPHA = 0.01
TRUST_MODEL_LAPLACIAN_LAMBDA = 1e-3 # Set to 0 if not using laplacian reg.

# --- Control Model Configuration --- (Pre-trained Gender/Age models)
CONTROL_MODEL_NAMES = ['happy', 'gender', 'age'] # Used to load pre-trained control models
CONTROL_MODEL_PATH_TEMPLATE = os.path.join(PROJECT_ROOT, "final_models/ensemble_{}.pt")

# --- Data Generation Configuration ---
TARGET_DIMENSION = "attractive" # The psychological dimension being modeled (e.g., 'trustworthy', 'attractive')
MIN_AGE = 6
MAX_AGE = 60
BIN_SIZE = 2
N_BINS = (MAX_AGE - MIN_AGE) // BIN_SIZE
MAX_SAMPLES_PER_BIN = 500000 # For initial sampling
MAX_HIGH_SCORE_SAMPLES_PER_BIN = 50000 # For high-score refinement
DATA_GEN_BATCH_SIZE = 200000 # Batch size during sampling W/Z
HIGH_SCORE_CUTOFF = 0.7 # Trust score cutoff for high-score generation
REFINE_ITERATIONS = 15 # Iterations for refining high-score Z vectors
REFINE_TOP_FRACTION = 0.33 # Fraction of top Z vectors to keep per refinement iteration
REFINE_NOISE_STD = 1.0 # Noise added in Z-space during refinement
PAIRING_MAX_AGE_DIFF = 1.0 # Max age difference for pairing
PAIRING_WINDOW_SIZE = 400000 # Window size for finding pairs (+/- this amount) - adjusted based on code
PAIRING_N_MATCHES = 2 # Number of matches per sample
PAIRING_BATCH_SIZE = 512 # Batch size for distance calculation during pairing

# Add derived dataset directory name
DATASET_DIR = f"{DATASET_DIR_BASE}_{TARGET_DIMENSION}2"


# --- Flow Model Configuration ---
FLOW_MODEL_NAME = f"{TARGET_DIMENSION}_flow" # Base name for saving flow model
FLOW_MODEL_DIM = 512
FLOW_MODEL_DEPTH = 8
FLOW_MODEL_NUM_HEADS = 8
FLOW_MODEL_DIM_HEAD = 48
FLOW_MODEL_NUM_REGISTERS = 32
FLOW_MODEL_DROPOUT = 0.1
FLOW_MODEL_CONDITION_DIM = 512 # Change to max 50 !!

# --- Flow Model Training ---
FLOW_SIGMA = 0.01
FLOW_START_EPOCH = 0 # Will be overwritten if loading checkpoint
FLOW_LR = 0.0003
FLOW_WEIGHT_DECAY = 0.05
FLOW_BETAS = (0.9, 0.95)
FLOW_SCHEDULER_T_MAX = 8000 # T_max for CosineAnnealingLR
FLOW_BATCH_SIZE_PER_STEP = 2048*2 + 1024 # Original `bs`
FLOW_ACCUMULATION_STEPS = 1
FLOW_EFFECTIVE_BATCH_SIZE = FLOW_BATCH_SIZE_PER_STEP * FLOW_ACCUMULATION_STEPS
FLOW_EPOCH_CHUNK_STEPS = 100 # Original `steps` variable, number of mini-batches per epoch chunk
FLOW_DATALOADER_BATCH_SIZE = FLOW_BATCH_SIZE_PER_STEP * FLOW_EPOCH_CHUNK_STEPS # How many samples Loader yields at once
FLOW_CLIP_GRAD_NORM = 1.0
FLOW_TOTAL_EPOCHS = 10000 # Example total epochs
FLOW_CHECKPOINT_INTERVAL = 20

# --- Logging & Checkpointing (Flow) ---
FLOW_RUN_NAME = f"{int(time.time())}" # For current run, might be overwritten by loaded checkpoint


# --- Initial Best Metrics (Flow) --- Adjust as needed
# These should track the best during training to save the "best" model checkpoint
BEST_METRICS = {
    'quality': 1.35,
    'delta_rating': 0.5,
    'median': 1.3,
    'loss': float('inf') # Initialize loss high for minimization
}

# --- Inference Configuration ---
INFERENCE_FLOW_MODEL_PATH = os.path.join(CHECKPOINT_DIR_FLOW, f"{TARGET_DIMENSION}_{FLOW_RUN_NAME}_epoch_XXX.pt") # Specify path to trained flow model
# Specify path to the corresponding trust model ensemble used for conditioning
INFERENCE_TRUST_MODEL_PATHS = [os.path.join(CHECKPOINT_DIR_TRUST, f"model_{TARGET_DIMENSION}_v{3+i}.pt") for i in range(TRUST_MODEL_ENSEMBLE_SIZE)]
INFERENCE_BATCH_SIZE = 32
INFERENCE_ODE_STEPS = 100 # Number of steps for solver
# Example target condition for inference - needs defining
INFERENCE_TARGET_RATING_LOGIT = torch.tensor([[2.0]], device=DEVICE) # Example target logit
