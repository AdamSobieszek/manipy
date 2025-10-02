# import os
# import sys
# import pickle
# import torch
# import numpy as np
# import PIL.Image
# from torchvision.transforms import Compose, Resize
# import torchvision.transforms.functional as TF
# from ..configs import config # Import config for paths and settings
# # Assuming setup_stylegan installs/clones necessary repos if run directly or paths are preset
# # Need to import G definition if not using pickle directly
# # Sys path manipulation is tricky, try relative imports if possible, or ensure PYTHONPATH is set

# # Global cache for G and w_avg to avoid reloading repeatedly
# _G_cache = None
# _w_avg_cache = None


# def load_stylegan_G(device=None):
#     """Loads the StyleGAN generator model (G_ema) and its average W vector."""
#     global _G_cache, _w_avg_cache
#     if device is None:
#         device = config.DEVICE

#     if _G_cache is not None and _w_avg_cache is not None:
#         return _G_cache.to(device), _w_avg_cache.to(device)

#     model_path = config.STYLEGAN_MODEL_PATH
#     if not os.path.exists(model_path):
#         # Attempt to download if file missing (requires wget or requests)
#         print(f"StyleGAN model not found at {model_path}. Attempting download...")
#         try:
#             # Example using requests (install requests: pip install requests)
#             import requests
#             url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl"
#             response = requests.get(url, stream=True)
#             response.raise_for_status()
#             with open(model_path, 'wb') as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)
#             print("Download complete.")
#         except Exception as e:
#             print(f"Error downloading StyleGAN model: {e}")
#             print("Please download the model manually and place it at:", model_path)
#             raise FileNotFoundError(f"StyleGAN model file not found: {model_path}") from e

#     print(f"Loading StyleGAN model from {model_path} to {device}...")
#     # Add stylegan3 path if necessary - consider managing PYTHONPATH externally
#     stylegan_repo_path = os.path.join(config.PSYCHGAN_REPO_PATH, 'stylegan3')
#     if stylegan_repo_path not in sys.path:
#          sys.path.append(stylegan_repo_path)
#          print(f"Added {stylegan_repo_path} to sys.path")
         
#     try:
#         # Need dnnlib for pickle loading if using custom classes
#         import dnnlib # noqa
#     except ImportError:
#         print("Warning: dnnlib not found. Ensure stylegan3 requirements are installed and in PYTHONPATH.")


#     with open(model_path, 'rb') as fp:
#         network_dict = pickle.load(fp)
#         if 'G_ema' not in network_dict:
#              raise ValueError("Loaded pickle does not contain 'G_ema'. Check the StyleGAN model file.")
#         G = network_dict['G_ema'].to(device)
#         G.eval() # Set to evaluation mode

#     # Compute w_avg if not directly available (often it is in the pickle)
#     # Or load if saved separately
#     w_avg = G.mapping.w_avg.detach() # Assuming G_ema has w_avg attribute precomputed

#     # Alternative: Compute w_avg manually (optional, usually not needed for G_ema)
#     # print("Computing w_avg...")
#     # with torch.no_grad():
#     #     z_samples = torch.randn([10000, G.mapping.z_dim], device=device)
#     #     w_samples = G.mapping(z_samples, None, truncation_psi=1.0)[:, 0, :] # Get W for first layer
#     #     w_avg = torch.mean(w_samples, dim=0, keepdim=True) # Shape (1, 512)
#     #     # If G expects (1, 18, 512), tile it:
#     #     # w_avg = w_avg.repeat(1, G.mapping.num_ws, 1) # Shape (1, 18, 512)

#     _G_cache = G
#     _w_avg_cache = w_avg

#     print("StyleGAN model and w_avg loaded.")
#     return _G_cache, _w_avg_cache

# def get_w_avg(device=None):
#     """Returns the cached or loaded w_avg."""
#     if _w_avg_cache is None:
#         _, w_avg = load_stylegan_G(device=device)
#         return w_avg
#     if device is None:
#         device = config.DEVICE
#     return _w_avg_cache.to(device)


# @torch.no_grad()
# def sample_w(n, truncation_psi=config.TRUNCATION_PSI_DEFAULT, device=None, seed=None):
#     """Samples N vectors in W space using the StyleGAN mapping network."""
#     if device is None:
#         device = config.DEVICE
#     G, _ = load_stylegan_G(device)
#     print("Warning: sample_w is deprecated. Use stylegan.utils.sample_w instead.")

#     if seed is not None:
#         torch.manual_seed(seed)

#     all_z = torch.randn([n, G.mapping.z_dim], device=device)
#     # Map Z to W, typically taking the first layer's W: w = G.mapping(...)[0, :]
#     # The original code uses [:,0], implying it takes the first layer's W if num_ws > 1
#     # Ensure the output shape is (n, 512)
#     all_w = G.mapping(all_z, None, truncation_psi=truncation_psi) # Shape (n, num_ws, 512)
#     if all_w.ndim == 3 and G.mapping.num_ws > 1:
#          all_w = all_w[:, 0, :] # Take W from the first layer

#     return all_w # Shape (n, 512)


# @torch.no_grad()
# def generate_images_from_w(w_vectors, device=None):
#     """Generates images from W vectors using the StyleGAN synthesis network."""
#     if device is None:
#         device = config.DEVICE
#     G, _ = load_stylegan_G(device)
#     print("Warning: generate_images_from_w is deprecated. Use generate_images_from_w_torch instead.")
#     if w_vectors.ndim == 2: # Shape (n, 512)
#         # StyleGAN synthesis usually expects W in shape (n, num_ws, 512)
#         # Repeat the single W vector across all layers if needed
#         w_vectors = w_vectors.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)
#     elif w_vectors.ndim != 3 or w_vectors.shape[1] != G.mapping.num_ws or w_vectors.shape[2] != config.LATENT_DIM_W:
#          raise ValueError(f"Input W vector shape mismatch. Expected (n, {G.mapping.num_ws}, {config.LATENT_DIM_W}) or (n, {config.LATENT_DIM_W}). Got {w_vectors.shape}")

#     w_vectors = w_vectors.to(device)
#     images_out = G.synthesis(w_vectors) # Output images tensor

#     # Post-process images (normalize to [0, 1])
#     transform = Compose([
#         # Resize(512), # Synthesis usually outputs target size
#         lambda x: torch.clamp((x + 1) / 2, min=0, max=1)
#     ])
#     images_pil = [TF.to_pil_image(transform(im.cpu())) for im in images_out]
#     return images_pil


# def show_faces(target, add=None, subtract=False, plot=True, grid=True, rows=1, labels = None, device=None):
#     """
#     Displays or returns images of faces generated from latent vectors (W space).
#     Uses generate_images_from_w and display_images_matplotlib.

#     Args:
#         target: W vectors (Tensor or np.array) or paths to .npy files. Shape (n, 512) or (n, 18, 512).
#         add: Optional W vector(s) to add.
#         subtract: If True, also subtracts 'add' from 'target'.
#         plot: If True, display images using matplotlib.
#         grid: If True and plot=True, display in a grid.
#         rows: Number of rows for the grid.
#         labels: Optional list of labels for the images.
#         device: Device for PyTorch operations.
#     """
#     from utils.general_utils import listify, read # Avoid circular import if possible
#     from utils.visualization import display_images_matplotlib, create_image_grid

#     if device is None:
#         device = config.DEVICE

#     target_list = listify(target)
#     add_list = listify(add)

#     w_vectors_to_generate = []
#     processed_targets = []

#     for t in target_list:
#         w = read(t, passthrough=True) # Read handles loading npy if path string
#         if isinstance(w, torch.Tensor):
#             w = w.to(device)
#         elif isinstance(w, np.ndarray):
#             w = torch.from_numpy(w).to(device)
#         elif isinstance(w, str) and '.npy' in w: # If read couldn't load, it returns path
#             try:
#                 w = torch.from_numpy(np.load(w)).to(device)
#             except:
#                 print(f"Warning: Could not load numpy file: {w}")
#                 continue # Skip if path not loadable
#         elif isinstance(t, PIL.Image.Image) or (isinstance(t, str) and not '.npy' in t):
#              # Handle direct image inputs or image paths later
#              processed_targets.append(t)
#              continue
#         else:
#              print(f"Warning: Skipping unrecognized target type: {type(t)}")
#              continue

#         # Ensure w has correct shape (n, 512) or (n, 18, 512)
#         # Assuming read function provides W-space vectors
#         processed_targets.append(w) # Store the processed tensor

#     # Prepare combinations +/- add vector
#     final_w_list = []
#     original_indices = [] # Keep track of which original target each final W corresponds to for labels

#     for idx, w_target in enumerate(processed_targets):
#         if not isinstance(w_target, (torch.Tensor, np.ndarray)): continue # Skip non-tensor targets

#         w_target_tensor = w_target.float() # Ensure float

#         w_add = None
#         if add_list[0] is not None:
#              if len(add_list) == len(processed_targets):
#                   add_val = add_list[idx]
#              else:
#                   add_val = add_list[0]

#              w_add = read(add_val, passthrough=True)
#              if isinstance(w_add, torch.Tensor):
#                  w_add = w_add.to(device).float()
#              elif isinstance(w_add, np.ndarray):
#                  w_add = torch.from_numpy(w_add).to(device).float()
#              else:
#                   print(f"Warning: Could not process 'add' value: {add_val}")
#                   w_add = None

#              # Ensure compatible shapes for addition/subtraction
#              # Basic check: if w_target is (N, D) and w_add is (D), unsqueeze w_add
#              if w_add is not None:
#                   if w_target_tensor.ndim == 2 and w_add.ndim == 1 and w_target_tensor.shape[1] == w_add.shape[0]:
#                        w_add = w_add.unsqueeze(0)
#                   elif w_target_tensor.ndim == 3 and w_add.ndim == 2 and w_target_tensor.shape[1:] == w_add.shape:
#                        w_add = w_add.unsqueeze(0)
#                    # Add more shape compatibility checks if needed based on how 'read' works


#         current_w_batch = []
#         if subtract and w_add is not None:
#             try: current_w_batch.append(w_target_tensor - w_add)
#             except RuntimeError as e: print(f"Subtract shape mismatch: {e}")
#         current_w_batch.append(w_target_tensor)
#         if w_add is not None:
#             try: current_w_batch.append(w_target_tensor + w_add)
#             except RuntimeError as e: print(f"Add shape mismatch: {e}")

#         final_w_list.extend(current_w_batch)
#         original_indices.extend([idx] * len(current_w_batch))


#     # Handle image inputs/paths
#     other_images = []
#     image_indices = []
#     for idx, t in enumerate(processed_targets):
#          if isinstance(t, PIL.Image.Image):
#               other_images.append(t)
#               image_indices.append(idx)
#          elif isinstance(t, str) and not '.npy' in t:
#               try:
#                    img = PIL.Image.open(t)
#                    other_images.append(img)
#                    image_indices.append(idx)
#               except:
#                    print(f"Warning: Could not open image file: {t}")

#     # Generate images from W vectors
#     generated_images_pil = []
#     if final_w_list:
#         # Concatenate all W vectors for batch generation
#         try:
#              w_batch = torch.cat(final_w_list, dim=0)
#              generated_images_pil = generate_images_from_w(w_batch, device=device)
#         except Exception as e:
#              print(f"Error during image generation: {e}")


#     # Combine generated images and pre-existing images in the correct order
#     all_images = []
#     gen_idx = 0
#     other_idx = 0
#     temp_combined = {} # Use original index as key

#     for i, w in enumerate(final_w_list):
#          original_target_idx = original_indices[i]
#          if original_target_idx not in temp_combined: temp_combined[original_target_idx] = []
#          if gen_idx < len(generated_images_pil):
#               temp_combined[original_target_idx].append(generated_images_pil[gen_idx])
#               gen_idx += 1

#     for i, img in enumerate(other_images):
#          original_target_idx = image_indices[i]
#          if original_target_idx not in temp_combined: temp_combined[original_target_idx] = []
#          temp_combined[original_target_idx].append(img)

#     # Flatten based on sorted original indices
#     for key in sorted(temp_combined.keys()):
#          all_images.extend(temp_combined[key])


#     # Adjust labels if necessary to match the final number of images
#     final_labels = None
#     if labels is not None:
#          # If labels were provided per input target, expand them
#          if len(labels) == len(processed_targets):
#               expanded_labels = []
#               gen_label_idx = 0
#               other_label_idx = 0
#               for key in sorted(temp_combined.keys()):
#                    num_imgs_for_key = len(temp_combined[key])
#                    original_label = labels[key]
#                    expanded_labels.extend([f"{original_label}_{j}" for j in range(num_imgs_for_key)]) # Append indices for clarity
#               final_labels = expanded_labels
#          else:
#               # If labels provided match final image count directly
#               if len(labels) == len(all_images):
#                    final_labels = labels
#               else:
#                    print("Warning: Label count doesn't match input targets or final images. Ignoring labels.")


#     if plot:
#         display_images_matplotlib(all_images, grid=grid, rows=rows, labels=final_labels)
#     else:
#         return create_image_grid(all_images, rows=rows) if grid else all_images


# # --- Setup Function --- (Optional - can be run from a script)
# def setup_stylegan_environment():
#     """
#     Installs packages, clones repos etc. Tries to be idempotent.
#     NOTE: This assumes a Colab-like or Unix-like environment with git, pip, apt-get, gdown.
#     Modify commands if running on a different OS. Best practice is often manual setup or Docker.
#     """
#     print("Setting up StyleGAN environment...")

#     # Check if in a GPU runtime (basic check)
#     try:
#        if config.DEVICE.type == 'cuda':
#           print("CUDA device detected.")
#        elif config.DEVICE.type == 'mps':
#            print("MPS device detected.")
#        else:
#            print("CPU device detected.")
#     except Exception as e:
#         print(f"Device check failed: {e}")


#     # Install required packages only if needed
#     required_packages = ['einops', 'ninja', 'torch', 'torchvision', 'torchaudio', 'pandas', 'numpy', 'matplotlib', 'Pillow', 'scipy', 'scikit-learn', 'opencv-python', 'tqdm', 'gdown', 'requests', 'ipywidgets', 'x-transformers', 'torchdyn', 'torchcfm', 'ot'] # Added POT library based on usage
#     missing_packages = []
#     for package in required_packages:
#         try:
#              if package == 'ot':
#                   __import__('ot') # POT library namespace is 'ot'
#              elif package == 'opencv-python':
#                   __import__('cv2')
#              elif package == 'x-transformers':
#                   __import__('x_transformers') # Check main module
#              elif package == 'torchcfm':
#                   __import__('torchcfm')
#              else:
#                   __import__(package)
#         except ImportError:
#              missing_packages.append(package)

#     if missing_packages:
#         print(f"Installing missing packages: {missing_packages}")
#         pip_command = f"pip install {' '.join(missing_packages)} --quiet"
#         # Special flags for torch if needed (usually handled by environment)
#         # pip_command += " -f https://download.pytorch.org/whl/torch_stable.html" # Example specific source
#         os.system(pip_command)
#     else:
#         print("Required Python packages seem installed.")


#     # Navigate to a common content directory (adjust if needed)
#     content_dir = os.path.join(config.PROJECT_ROOT, 'content')
#     os.makedirs(content_dir, exist_ok=True)
#     os.chdir(content_dir)
#     print(f"Changed directory to: {os.getcwd()}")



#     # Clone psychGAN repo if not present
#     psychgan_repo_path = config.PSYCHGAN_REPO_PATH
#     if not os.path.exists(psychgan_repo_path):
#         print("Cloning psychGAN repository...")
#         # Use token for private repo if necessary
#         # repo_url = 'github.com/AdamSobieszek/psychGAN' # Public?
#         # os.system(f"git clone https://{config.GIT_TOKEN}@{repo_url} {psychgan_repo_path}") # With token
#         os.system(f"git clone https://github.com/AdamSobieszek/psychGAN {psychgan_repo_path}") # Public clone
#     else:
#         print("psychGAN repository already exists.")

#     # Download OMI data if specified and not present (example)
#     omi_data_zip = os.path.join(config.OMI_DATA_DIR, 'attribute_ratings.zip')
#     omi_data_csv = os.path.join(config.OMI_DATA_DIR, 'attribute_ratings.csv')
#     if not os.path.exists(omi_data_csv):
#         if not os.path.exists(omi_data_zip):
#              # Original command referenced gdown ID 1O79M5F5G3ktmt1-zeccbJf1Bhe8K9ROz for something else?
#              # Assuming a different ID or direct link for attribute ratings if needed
#              print("Downloading OMI attribute ratings (placeholder)...")
#              # os.system('gdown YOUR_GDRIVE_ID_FOR_OMI_RATINGS') # Replace with actual ID/URL
#              # Example: os.system('wget https://example.com/path/to/attribute_ratings.zip')
#         if os.path.exists(omi_data_zip):
#              print("Unzipping OMI attribute ratings...")
#              os.system(f'unzip -o {omi_data_zip} -d {config.OMI_DATA_DIR}') # Unzip to OMI_DATA_DIR
#         else:
#              print("Warning: OMI attribute ratings zip file not found and download placeholder used.")
#     else:
#          print("OMI attribute ratings CSV already exists.")


#     # Download final_models.zip if needed
#     final_models_zip = os.path.join(config.PSYCHGAN_REPO_PATH, "final_models.zip")
#     final_models_dir = os.path.join(config.PROJECT_ROOT, "final_models") # Extracted dir
#     if not os.path.exists(final_models_dir):
#          if not os.path.exists(final_models_zip):
#               print("Downloading final_models.zip...")
#               os.chdir(config.PSYCHGAN_REPO_PATH) # gdown works better in target dir?
#               os.system('gdown 1pPjOd-mx-d-vOw1QR_lpJoJmLAGdkI3W')
#               os.chdir(content_dir) # Go back
#          if os.path.exists(final_models_zip):
#               print("Unzipping final_models.zip...")
#               # Ensure unzip command works, might need 'apt-get install unzip' if in minimal container
#               try:
#                    os.system(f'unzip -o {final_models_zip} -d {config.PROJECT_ROOT}') # Unzip to project root
#                    print(f"Unzipped models to {final_models_dir}")
#               except Exception as e:
#                    print(f"Error unzipping final_models.zip: {e}. Ensure 'unzip' is installed.")
#          else:
#               print("Warning: final_models.zip not found.")
#     else:
#          print("final_models directory already exists.")


#     # Download StyleGAN model (handled by load_stylegan_G now)


#     # Add psychGAN paths to sys.path (less ideal than PYTHONPATH)
#     paths_to_add = [
#         config.PSYCHGAN_REPO_PATH,
#         os.path.join(config.PSYCHGAN_REPO_PATH, 'stylegan3'),
#         # config.PROJECT_ROOT # Maybe not needed
#     ]
#     for p in paths_to_add:
#         if p not in sys.path:
#             sys.path.append(p)
#             print(f"Added to sys.path: {p}")

#     # Go back to project root
#     os.chdir(config.PROJECT_ROOT)
#     print(f"Changed directory back to project root: {os.getcwd()}")

#     print("Environment setup attempt complete.")


# if __name__ == "__main__":
#     setup_stylegan_environment()
#     # Now load models etc.
#     G, w_avg = load_stylegan_G()
#     print("Setup and model loading successful.")
