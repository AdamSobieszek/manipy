# data/dataset_generation.py
import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import time

from full_manipy import config
from data.data_utils import get_age_bin, bin_to_age_range, custom_cdist, custom_rdist # Import local utils
from utils.stylegan_utils import load_stylegan_G, get_w_avg # Need G and w_avg

# --- Functions for Sampling W/Z vectors based on Age/Gender ---

@torch.no_grad()
def sample_dataset_by_age_gender_saveZ(
    G,
    age_model,
    gender_model,
    n_bins,
    min_age=config.MIN_AGE,
    max_age=config.MAX_AGE,
    bin_size=config.BIN_SIZE,
    max_per_bin=config.MAX_SAMPLES_PER_BIN,
    truncation_psi_range=(0.9, 1.1), # Adjusted from original code example
    batch_size=config.DATA_GEN_BATCH_SIZE,
    device=config.DEVICE,
    max_iterations=100,
    output_dir=config.BIN_FILES_DIR
):
    """
    Samples W and Z vectors, bins them by predicted age/gender, and saves chunks to disk.
    Each saved chunk is a dict: {"w": w_tensor, "z": z_tensor}.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting W/Z sampling, saving chunks to: {output_dir}")

    bins_data = { # In-memory buffer for current chunk
        'female': [[] for _ in range(n_bins)],
        'male':   [[] for _ in range(n_bins)]
    }
    chunk_counter = { # Tracks file part number for each bin
        'female': [0] * n_bins,
        'male':   [0] * n_bins
    }

    for iteration in tqdm(range(max_iterations), desc="Sampling W/Z"):
        # Random truncation_psi
        truncation_psi = np.random.uniform(*truncation_psi_range)
        if np.random.random() > 0.5: # Add chance for psi=1.0 as per original code
            truncation_psi = 1.0

        # 1) Generate random Z
        z = torch.randn([batch_size, G.mapping.z_dim], device=device)
        # Map Z -> W
        w = G.mapping(z, None, truncation_psi=truncation_psi)[:, 0, :]  # (batch_size, w_dim)

        # Predict age & gender
        predicted_ages = age_model(w).view(-1)
        predicted_gender = gender_model(w).view(-1)

        # Bin samples
        for i in range(batch_size):
            age_float = predicted_ages[i].item()
            bin_index = get_age_bin(age_float, min_age=min_age, max_age=max_age, bin_size=bin_size)
            if bin_index is None: continue

            gen_val = predicted_gender[i].item()
            if gen_val > 55: gender_str = 'female'
            elif gen_val < 45: gender_str = 'male'
            else: continue # Skip ambiguous gender predictions

            # Append (W[i], Z[i]) pair to the in-memory buffer
            bins_data[gender_str][bin_index].append((
                w[i].unsqueeze(0).cpu(), # Store on CPU to save GPU memory
                z[i].unsqueeze(0).cpu()
            ))

            # Flush bin to disk if buffer exceeds max_per_bin
            if len(bins_data[gender_str][bin_index]) >= max_per_bin:
                w_list = [pair[0] for pair in bins_data[gender_str][bin_index]]
                z_list = [pair[1] for pair in bins_data[gender_str][bin_index]]
                w_tensor = torch.cat(w_list, dim=0)
                z_tensor = torch.cat(z_list, dim=0)

                out_fname = os.path.join(output_dir, f"{gender_str}_agebin_{bin_index}_part_{chunk_counter[gender_str][bin_index]}.pt")
                torch.save({"w": w_tensor, "z": z_tensor}, out_fname)
                # print(f" Saved {w_tensor.size(0)} W/Z pairs to {out_fname} (flush).") # Verbose

                chunk_counter[gender_str][bin_index] += 1
                bins_data[gender_str][bin_index].clear() # Clear buffer

    # Save any remaining data in buffers
    print("Saving remaining samples...")
    for gender_str in ['female', 'male']:
        for b in range(n_bins):
            if bins_data[gender_str][b]:
                w_list = [pair[0] for pair in bins_data[gender_str][b]]
                z_list = [pair[1] for pair in bins_data[gender_str][b]]
                w_tensor = torch.cat(w_list, dim=0)
                z_tensor = torch.cat(z_list, dim=0)
                out_fname = os.path.join(output_dir, f"{gender_str}_agebin_{b}_part_{chunk_counter[gender_str][b]}.pt")
                torch.save({"w": w_tensor, "z": z_tensor}, out_fname)
                print(f" Saved remaining {w_tensor.size(0)} W/Z pairs to {out_fname}.")
                chunk_counter[gender_str][b] += 1 # Increment counter even for leftovers
                bins_data[gender_str][b].clear()

    print("W/Z sampling finished.")
    return chunk_counter # Return final counters which might be needed by refinement

# --- Functions for Refining Z vectors for High Trust Scores ---

@torch.no_grad()
def refine_z_batch_for_bin(
    z_init,               # Initial Z batch for this bin (on CPU)
    G, trust_model, age_model, gender_model, # Models
    gender_str, bin_index, # Target bin identifiers
    min_age=config.MIN_AGE, bin_size=config.BIN_SIZE,
    noise_std=config.REFINE_NOISE_STD,
    top_fraction=config.REFINE_TOP_FRACTION,
    iterations=config.REFINE_ITERATIONS,
    device=config.DEVICE
):
    """ Iteratively refines Z vectors to maximize trust score while staying in the target age/gender bin. """

    if z_init.numel() == 0: return None, None # Skip if initial Z is empty

    z_current = z_init.to(device)
    initial_size = z_current.size(0)
    age_low, age_high_exclusive = bin_to_age_range(bin_index, min_age=min_age, bin_size=bin_size)
    # Make age comparison inclusive of upper bound based on original code logic (<= age_high)
    age_high_inclusive = age_high_exclusive - (1e-6) # Approx inclusive upper bound

    for it in range(iterations):
        if z_current.size(0) == 0: break # Stop if no vectors are left

        # 1) Map Z -> W
        w_current = G.mapping(z_current, None, truncation_psi=1.0)[:, 0, :]

        # 2) Evaluate trust score
        trust_vals = trust_model(w_current, "mean").view(-1) # Assuming "mean" gives the score

        # 3) Predict age & gender to confirm bin membership
        predicted_ages = age_model(w_current).view(-1)
        predicted_gender = gender_model(w_current).view(-1)

        # Gender mask
        if gender_str == 'female': gender_mask = (predicted_gender > 55)
        else: gender_mask = (predicted_gender < 45)
        # Age mask (inclusive range)
        age_mask = (predicted_ages >= age_low) & (predicted_ages < age_high_exclusive) # Match get_age_bin logic
        bin_mask = gender_mask & age_mask

        if not bin_mask.any(): # If none remain in the bin
            z_current = z_current[:0] # Empty tensor
            break

        z_in_bin = z_current[bin_mask]
        trust_in_bin = trust_vals[bin_mask]

        # 4) Keep only top fraction based on trust score
        n_in_bin = z_in_bin.size(0)
        keep_count = int(max(1, top_fraction * n_in_bin))

        sorted_indices = torch.argsort(trust_in_bin, descending=True)
        top_indices = sorted_indices[:keep_count]
        z_top = z_in_bin[top_indices]

        # 5) Replicate survivors and add noise in Z-space
        if z_top.size(0) == 0: # Should not happen if keep_count >= 1, but safety check
             z_current = z_current[:0]
             break

        tile_times = (initial_size // keep_count) + 1 # How many times to repeat top Zs
        z_replicated = z_top.repeat(tile_times, 1)[:initial_size] # Fill back to original size

        z_noised = z_replicated + torch.randn_like(z_replicated) * noise_std
        z_current = z_noised
        # print(f" Iter {it}: Kept {keep_count}/{n_in_bin}, Trust range: {trust_in_bin.min():.3f}-{trust_in_bin.max():.3f}") # Debug

    # Final check: After iterations, ensure the remaining z_current maps to W in the bin
    if z_current.size(0) == 0: return None, None

    w_final = G.mapping(z_current, None, truncation_psi=1.0)[:, 0, :]
    ages_final = age_model(w_final).view(-1)
    genders_final = gender_model(w_final).view(-1)

    if gender_str == 'female': final_gender_mask = (genders_final > 55)
    else: final_gender_mask = (genders_final < 45)
    final_age_mask = (ages_final >= age_low) & (ages_final < age_high_exclusive)
    final_mask = final_gender_mask & final_age_mask

    if not final_mask.any(): return None, None

    w_kept = w_final[final_mask].cpu() # Return survivors on CPU
    z_kept = z_current[final_mask].cpu()

    return w_kept, z_kept

@torch.no_grad()
def generate_high_score_data(
    G, trust_model, age_model, gender_model, # Models
    input_dir=config.BIN_FILES_DIR, output_dir=config.BIN_FILES_DIR, # Dirs
    min_age=config.MIN_AGE, max_age=config.MAX_AGE, bin_size=config.BIN_SIZE,
    refine_iterations=config.REFINE_ITERATIONS,
    refine_top_fraction=config.REFINE_TOP_FRACTION,
    refine_noise_std=config.REFINE_NOISE_STD,
    high_score_prefix=f"{config.TARGET_DIMENSION}_high", # Prefix for output files
    max_chunks_per_bin=None, # Limit chunks to process per bin (optional)
    device=config.DEVICE
):
    """
    Loads W/Z chunks from input_dir, refines Z for high trust scores using
    refine_z_batch_for_bin, and saves the resulting high-score W/Z pairs
    to output_dir with the specified prefix.
    """
    n_bins = (max_age - min_age) // bin_size
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting high-score refinement, output prefix: '{high_score_prefix}'")

    output_chunk_counter = { # Tracks output file part numbers
        'female': [0] * n_bins,
        'male':   [0] * n_bins
    }

    for gender_str in ['female', 'male']:
        for b in tqdm(range(n_bins), desc=f"Refining {gender_str} bins"):
            input_pattern = os.path.join(input_dir, f"{gender_str}_agebin_{b}_part_*.pt")
            input_files = sorted(glob.glob(input_pattern))

            if not input_files: continue # Skip if no input files for this bin

            chunks_processed = 0
            for fname in input_files:
                if max_chunks_per_bin is not None and chunks_processed >= max_chunks_per_bin:
                    print(f"Reached max chunks ({max_chunks_per_bin}) for {gender_str} bin {b}.")
                    break
                try:
                    data = torch.load(fname, map_location='cpu')
                    if "z" not in data:
                         print(f"Warning: File {fname} does not contain 'z' tensor. Skipping.")
                         continue
                    z_chunk = data["z"]
                except Exception as e:
                    print(f"Error loading {fname}: {e}. Skipping.")
                    continue

                # Refine the Z vectors in this chunk
                w_kept, z_kept = refine_z_batch_for_bin(
                    z_init=z_chunk, G=G, trust_model=trust_model, age_model=age_model,
                    gender_model=gender_model, gender_str=gender_str, bin_index=b,
                    min_age=min_age, bin_size=bin_size, noise_std=refine_noise_std,
                    top_fraction=refine_top_fraction, iterations=refine_iterations, device=device
                )

                # Save refined W/Z if any survived
                if w_kept is not None and z_kept is not None and w_kept.size(0) > 0:
                    out_fname = os.path.join(output_dir, f"{high_score_prefix}_{gender_str}_agebin_{b}_part_{output_chunk_counter[gender_str][b]}.pt")
                    torch.save({"w": w_kept, "z": z_kept}, out_fname)
                    # print(f"  Saved refined {w_kept.size(0)} W/Z pairs to {out_fname}") # Verbose
                    output_chunk_counter[gender_str][b] += 1
                # else:
                #      print(f"  No samples survived refinement for chunk {os.path.basename(fname)}") # Debug

                chunks_processed += 1

    print("High-score refinement finished.")
    return output_chunk_counter

# --- Functions for Loading Data and Finding Pairs ---

@torch.no_grad()
def load_and_stack_by_age(
    gender_str, n_bins,
    input_dir=config.BIN_FILES_DIR, # Where W/Z files are
    high_score_prefix=f"{config.TARGET_DIMENSION}_high", # Prefix of high-score files
    limit_per_bin=None, # Max samples total per bin (normal + high)
    device=config.DEVICE, # Device to load final tensors onto
    base_model=None # Trust model needed to calculate 'y' values
):
    """
    Loads both normal and high-score W vectors for a given gender across all age bins,
    concatenates them, optionally limits samples per bin, calculates trust scores ('y'),
    sorts by predicted age, and returns the stacked tensors.
    """
    print(f"Loading and stacking data for {gender_str}...")
    if base_model is None:
         raise ValueError("'base_model' (trust model) must be provided to calculate y values.")
    base_model.eval()
    base_model.to(device) # Ensure trust model is on correct device

    # Need age model to sort final data
    from full_manipy.models.rating_models import load_control_models # Assuming control models load function exists
    control_models = load_control_models(config.CONTROL_MODEL_PATH_TEMPLATE, config.CONTROL_MODEL_NAMES, device)
    age_model = control_models[2] # Assuming index 2 is age

    all_w_bin = [] # List to hold tensors for each bin
    all_ages_bin = [] # List for predicted ages (for sorting)
    all_y_bin = [] # List for trust scores

    for b in tqdm(range(n_bins), desc=f"Loading {gender_str} bins"):
        bin_w_chunks = []

        # 1. Load normal W chunks
        pattern_normal = os.path.join(input_dir, f"{gender_str}_agebin_{b}_part_*.pt")
        normal_files = sorted(glob.glob(pattern_normal))
        for fname in normal_files:
            try:
                # Load only 'w' tensor to save memory initially
                chunk_w = torch.load(fname, map_location='cpu')["w"]
                bin_w_chunks.append(chunk_w)
            except Exception as e:
                print(f"Warning: Error loading normal chunk {fname}: {e}")

        # 2. Load high-score W chunks
        pattern_high = os.path.join(input_dir, f"{high_score_prefix}_{gender_str}_agebin_{b}_part_*.pt")
        high_files = sorted(glob.glob(pattern_high))
        for fname in high_files:
            try:
                chunk_w = torch.load(fname, map_location='cpu')["w"]
                bin_w_chunks.append(chunk_w)
            except Exception as e:
                print(f"Warning: Error loading high-score chunk {fname}: {e}")

        if not bin_w_chunks: continue # Skip bin if no data found

        # Concatenate all W vectors for the current bin
        w_bin_tensor_cpu = torch.cat(bin_w_chunks, dim=0)

        # Optional: Limit samples per bin
        if limit_per_bin is not None and w_bin_tensor_cpu.size(0) > limit_per_bin:
            perm = torch.randperm(w_bin_tensor_cpu.size(0))[:limit_per_bin]
            w_bin_tensor_cpu = w_bin_tensor_cpu[perm]

        if w_bin_tensor_cpu.size(0) == 0: continue # Skip if bin becomes empty after limit

        # Move to device to calculate age and trust score (y)
        w_bin_tensor_dev = w_bin_tensor_cpu.to(device).float() # Ensure float for models

        # Predict age and trust score ('y') for this bin's data
        ages_pred = age_model(w_bin_tensor_dev).view(-1)
        y_pred = base_model(w_bin_tensor_dev, "mean").view(-1) # Assuming "mean" gives score

        # Append results (back on CPU to save GPU memory if needed, or keep on device if sufficient memory)
        all_w_bin.append(w_bin_tensor_dev) # Keep W on device for pairing
        all_ages_bin.append(ages_pred)     # Keep Age on device
        all_y_bin.append(y_pred)         # Keep Y on device

        # print(f" Bin {b}: Loaded {w_bin_tensor_dev.size(0)} samples.") # Debug

    if not all_w_bin: return None, None, None # No data loaded

    # Stack results from all bins
    stacked_w = torch.cat(all_w_bin, dim=0)
    stacked_ages = torch.cat(all_ages_bin, dim=0)
    stacked_y = torch.cat(all_y_bin, dim=0)

    # Sort the stacked tensors by predicted age
    sort_indices = torch.argsort(stacked_ages)
    sorted_w = stacked_w[sort_indices]
    sorted_ages = stacked_ages[sort_indices]
    sorted_y = stacked_y[sort_indices]

    print(f"Finished loading for {gender_str}. Total samples: {sorted_w.size(0)}")
    return sorted_w, sorted_ages, sorted_y


@torch.no_grad()
def find_optimal_pairs_batched(
    coords, ratings, # Sorted tensors on device
    cdist_fn=custom_cdist, rdist_fn=custom_rdist, # Distance functions
    batch_size=config.PAIRING_BATCH_SIZE,
    k_matches=config.PAIRING_N_MATCHES,
    window_size=config.PAIRING_WINDOW_SIZE # Search window size
):
    """ Finds optimal transport pairs using batched distance calculations. """
    N = coords.size(0)
    device = coords.device
    if N == 0: return []
    print(f"Finding optimal pairs for {N} samples...")

    pairs = [] # List to store (idx_low_rating, idx_high_rating, quality, rating_diff)

    for start_idx in tqdm(range(0, N, batch_size), desc="Finding Pairs"):
        end_idx = min(start_idx + batch_size, N)
        batch_coords = coords[start_idx:end_idx]
        batch_ratings = ratings[start_idx:end_idx]

        # Define the search window relative to the current batch
        window_start = max(0, start_idx - window_size)
        window_end = min(N, end_idx + window_size)
        window_coords = coords[window_start:window_end]
        window_ratings = ratings[window_start:window_end]

        if window_coords.numel() == 0: continue

        # Calculate distances within the window
        # cdist shape: (batch_size, window_size)
        # rdist shape: (batch_size, window_size)
        cdist = cdist_fn(batch_coords, window_coords)
        rdist = rdist_fn(batch_ratings, window_ratings) # rating_win - rating_batch

        # Cost: distance / rating_difference. We want pairs where rating increases (rdist > 0).
        # Cost is lower for small coordinate distance and large positive rating difference.
        cost = torch.full_like(rdist, float('inf'))
        valid_rdist_mask = rdist > 1e-9 # Only consider pairs where rating increases significantly
        # Clamp cdist to avoid division by zero if coords are identical
        cdist_clamped = torch.clamp(cdist, min=1e-9)
        cost[valid_rdist_mask] = cdist_clamped[valid_rdist_mask] / rdist[valid_rdist_mask]

        # Mask out self-comparisons and invalid areas
        # Calculate the offset of the batch within the window
        batch_offset_in_window = start_idx - window_start
        for i in range(len(batch)):
             idx_in_window = batch_offset_in_window + i
             if 0 <= idx_in_window < cost.shape[1]:
                  cost[i, idx_in_window] = float('inf') # Mask self-comparison


        # Find top k matches (lowest cost) for each sample in the batch
        num_candidates = cost.shape[1]
        actual_k = min(k_matches, num_candidates)
        if actual_k <= 0: continue

        top_costs, top_indices_in_window = torch.topk(cost, k=actual_k, dim=1, largest=False)

        # Store the pairs: (idx_low_rating, idx_high_rating, quality, rating_diff)
        for i in range(len(batch)):
            global_idx_batch = start_idx + i # Index of the current sample (lower rating)
            for k in range(actual_k):
                cost_val = top_costs[i, k].item()
                if cost_val == float('inf'): continue # Skip invalid matches

                idx_in_window = top_indices_in_window[i, k].item()
                global_idx_match = window_start + idx_in_window # Index of the matched sample (higher rating)

                # Ensure it's not a self-match and rating actually increased
                if global_idx_match != global_idx_batch and ratings[global_idx_match] > ratings[global_idx_batch]:
                     quality = 1.0 / cost_val # Quality = 1 / cost = rdist / cdist
                     rating_diff = (ratings[global_idx_match] - ratings[global_idx_batch]).item()
                     pairs.append((global_idx_batch, global_idx_match, quality, rating_diff))


    # Optional: Sort pairs by quality or remove duplicates if needed
    # pairs.sort(key=lambda x: x[2], reverse=True) # Sort by quality descending
    print(f"Found {len(pairs)} potential pairs.")
    return pairs


# --- Main Dataset Generation Orchestration ---

def run_dataset_generation_pipeline(dim_name=config.TARGET_DIMENSION, device=config.DEVICE):
    """Orchestrates the full dataset generation process."""

    print(f"--- Starting Dataset Generation Pipeline for: {dim_name} ---")
    start_time = time.time()

    # 1. Load Models needed for generation
    print("Loading models...")
    G, _ = load_stylegan_G(device=device)
    from full_manipy.models.rating_models import load_trust_model_ensemble, load_control_models # Need these functions
    trust_model = load_trust_model_ensemble(config.TARGET_DIMENSION, config.TRUST_MODEL_ENSEMBLE_SIZE, config.CHECKPOINT_DIR_TRUST, device=device)
    control_models = load_control_models(config.CONTROL_MODEL_PATH_TEMPLATE, config.CONTROL_MODEL_NAMES, device)
    age_model = control_models[2]
    gender_model = control_models[1]
    print("Models loaded.")

    # 2. Sample initial W/Z distribution, binned by age/gender
    print("\n--- Step 1: Sampling Initial W/Z ---")
    # chunk_counters_sampling = sample_dataset_by_age_gender_saveZ(
    #     G=G, age_model=age_model, gender_model=gender_model,
    #     n_bins=config.N_BINS, min_age=config.MIN_AGE, max_age=config.MAX_AGE,
    #     bin_size=config.BIN_SIZE, max_per_bin=config.MAX_SAMPLES_PER_BIN,
    #     batch_size=config.DATA_GEN_BATCH_SIZE, device=device,
    #     max_iterations=150, # Adjust iterations as needed
    #     output_dir=config.BIN_FILES_DIR
    # )
    # print("Initial W/Z sampling complete.")
    print("Skipping initial sampling assuming files exist")

    # 3. Refine Z vectors for high trust scores
    print("\n--- Step 2: Refining Z for High Trust Scores ---")
    # generate_high_score_data(
    #     G=G, trust_model=trust_model, age_model=age_model, gender_model=gender_model,
    #     input_dir=config.BIN_FILES_DIR, output_dir=config.BIN_FILES_DIR,
    #     min_age=config.MIN_AGE, max_age=config.MAX_AGE, bin_size=config.BIN_SIZE,
    #     refine_iterations=config.REFINE_ITERATIONS, # Use config iterations
    #     refine_top_fraction=config.REFINE_TOP_FRACTION,
    #     refine_noise_std=config.REFINE_NOISE_STD,
    #     high_score_prefix=f"{dim_name}_high",
    #     device=device
    # )
    print("Skipping refignement stage assumes they exist")
    print("High-score refinement complete.")

    # 4. Load, Stack, and Find Pairs
    print("\n--- Step 3: Loading Data and Finding Pairs ---")
    os.makedirs(config.DATASET_DIR, exist_ok=True) # Ensure target dataset directory exists

    for gender in ['male', 'female']:
        print(f"\nProcessing {gender} data...")
        # Load W, Age, Y (trust score) sorted by Age
        stacked_w, stacked_ages, stacked_y = load_and_stack_by_age(
            gender_str=gender, n_bins=config.N_BINS,
            input_dir=config.BIN_FILES_DIR,
            high_score_prefix=f"{dim_name}_high",
            limit_per_bin=None, # Load all available after sampling/refinement limits
            device=device,
            base_model=trust_model # Pass trust model to calculate Y
        )

        if stacked_w is None or stacked_w.size(0) == 0:
            print(f"No data found for {gender}. Skipping pairing.")
            continue

        # Save the stacked W, Age, Y before pairing (useful for inspection/reuse)
        stacked_data_path = os.path.join(config.DATASET_DIR, f"{dim_name}_{gender}.pt")
        torch.save({"w": stacked_w.cpu(), "age": stacked_ages.cpu(), "y": stacked_y.cpu()}, stacked_data_path)
        print(f"Saved stacked W/Age/Y for {gender} to {stacked_data_path}")

        # Find optimal pairs
        pairs = find_optimal_pairs_batched(
            coords=stacked_w, ratings=stacked_y, # Use trust score (y) as rating
            cdist_fn=custom_cdist, rdist_fn=custom_rdist, # Use defined distance functions
            batch_size=config.PAIRING_BATCH_SIZE,
            k_matches=config.PAIRING_N_MATCHES,
            window_size=config.PAIRING_WINDOW_SIZE
        )

        # Save the found pairs
        pairs_path = os.path.join(config.DATASET_DIR, f"{gender}_pairs.pt")
        # Saving pairs as list of tuples: (idx1, idx2, quality, rating_diff)
        # Convert indices to standard Python ints if needed for broader compatibility, though tensors okay too.
        # Example saving format: list of tuples
        torch.save(pairs, pairs_path)
        print(f"Saved {len(pairs)} pairs for {gender} to {pairs_path}")

        # Clear memory
        del stacked_w, stacked_ages, stacked_y, pairs
        torch.cuda.empty_cache()

    end_time = time.time()
    print(f"\n--- Dataset Generation Pipeline Completed in {(end_time - start_time)/60:.2f} minutes ---")
