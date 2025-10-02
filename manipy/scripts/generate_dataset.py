# import sys
# import os
# # Ensure the project root is in the Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from data.dataset_generation import run_dataset_generation_pipeline
# from manipy import config
# from utils.general_utils import print_gpu_memory

# if __name__ == "__main__":
#     print(f"Using device: {config.DEVICE}")
#     print_gpu_memory()

#     # Run the full pipeline using settings from config.py
#     run_dataset_generation_pipeline(
#         dim_name=config.TARGET_DIMENSION,
#         device=config.DEVICE
#     )

#     print("Dataset generation script finished.")
#     print_gpu_memory()
# ```


# data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import glob
from tqdm import tqdm

from manipy import config

class OTDataset(Dataset):
    def __init__(self,
                 dim_name=config.TARGET_DIMENSION,
                 dataset_dir=config.DATASET_DIR, # Directory with {gender}_pairs.pt and {dim}_{gender}.pt
                 device=config.DEVICE,
                 load_limit_pairs=None # Optional limit on number of pairs per gender
                ):
        """
        Loads pre-computed W vectors and optimal pairs for flow matching.

        Assumes dataset_dir contains:
        - '{dim_name}_male.pt': {'w': tensor, 'age': tensor, 'y': tensor}
        - '{dim_name}_female.pt': {'w': tensor, 'age': tensor, 'y': tensor}
        - 'male_pairs.pt': list of (idx1_male, idx2_male, quality, rating_diff)
        - 'female_pairs.pt': list of (idx1_female, idx2_female, quality, rating_diff)
        """
        super().__init__()
        self.dim_name = dim_name
        self.dataset_dir = dataset_dir
        self.device = device
        self.all_w = []
        self.all_pairs_info = [] # Store as (idx1_global, idx2_global, quality, rating_diff)

        print(f"Loading OTDataset from: {self.dataset_dir}")

        offset = 0
        for gender in ['male', 'female']:
            w_data_path = os.path.join(self.dataset_dir, f"{self.dim_name}_{gender}.pt")
            pairs_path = os.path.join(self.dataset_dir, f"{gender}_pairs.pt")

            if not os.path.exists(w_data_path) or not os.path.exists(pairs_path):
                print(f"Warning: Data or pairs file missing for {gender} in {self.dataset_dir}. Skipping.")
                continue

            try:
                # Load W vectors (load to CPU first, then maybe move subset to GPU if needed/possible)
                w_data = torch.load(w_data_path, map_location='cpu')
                w_gender = w_data['w'].to(dtype=torch.float32) # Ensure float32
                n_gender = w_gender.size(0)
                print(f" Loaded {n_gender} W vectors for {gender}.")
                self.all_w.append(w_gender)

                # Load pairs for this gender
                pairs_gender = torch.load(pairs_path) # list of (idx1, idx2, quality, rating_diff)
                if load_limit_pairs is not None and len(pairs_gender) > load_limit_pairs:
                     # Optionally limit the number of pairs loaded
                     print(f" Limiting pairs for {gender} to {load_limit_pairs} from {len(pairs_gender)}.")
                     # Simple random sample or just take the first N? Let's take first N for consistency.
                     pairs_gender = pairs_gender[:load_limit_pairs]

                print(f" Loaded {len(pairs_gender)} pairs for {gender}.")

                # Adjust indices by current offset and add to global list
                for idx1, idx2, quality, rating_diff in pairs_gender:
                     # Ensure indices are within bounds of loaded W vectors for this gender
                     if idx1 < n_gender and idx2 < n_gender:
                          self.all_pairs_info.append((
                                idx1 + offset,
                                idx2 + offset,
                                float(quality),       # Ensure float
                                float(rating_diff)    # Ensure float
                          ))
                     else:
                          print(f"Warning: Pair indices ({idx1}, {idx2}) out of bounds for {gender} (size {n_gender}). Skipping pair.")


                # Update offset for the next gender
                offset += n_gender

            except Exception as e:
                print(f"Error loading data for {gender}: {e}")

        if not self.all_w:
            raise RuntimeError(f"No data loaded. Check dataset directory: {self.dataset_dir}")

        # Concatenate all W vectors (keep on CPU for now, move to GPU in __getitem__ or batch)
        self.all_w = torch.cat(self.all_w, dim=0).to(dtype=torch.float32) # Final W tensor on CPU
        self.num_pairs = len(self.all_pairs_info)
        self.num_samples = self.all_w.size(0)

        print(f"Dataset loaded: {self.num_samples} total samples, {self.num_pairs} pairs.")

        # Transfer W to target device if memory allows, otherwise handle in __getitem__
        try:
             self.all_w = self.all_w.to(self.device)
             print(f"Moved all W vectors ({self.all_w.shape}) to {self.device}.")
             self.w_on_device = True
        except RuntimeError: # Likely CUDA OOM
             print(f"Warning: Could not move all W vectors to {self.device}. Will load in __getitem__.")
             self.w_on_device = False


    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        """ Returns (X0, X1, COST, Dr) for the indexed pair. """
        idx1_global, idx2_global, quality, rating_diff = self.all_pairs_info[idx]

        if self.w_on_device:
            x0 = self.all_w[idx1_global]
            x1 = self.all_w[idx2_global]
        else:
            # Load from CPU tensor on demand
            x0 = self.all_w[idx1_global].to(self.device)
            x1 = self.all_w[idx2_global].to(self.device)

        # Cost is related to quality (quality = 1 / cost ?) Need clarification from usage.
        # Assuming COST in the training loop is derived dynamically.
        # Let's pass 'quality' as COST placeholder for now.
        cost_placeholder = torch.tensor(quality, dtype=torch.float32)

        # Dr seems to be delta_rating
        dr = torch.tensor(rating_diff, dtype=torch.float32)

        return x0, x1, cost_placeholder, dr

# --- DataLoader Function ---
def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    """Creates a standard DataLoader."""
    # Pin memory True is good if using GPU and num_workers > 0
    pin_memory = (num_workers > 0) and (config.DEVICE.type == 'cuda')
    # Prefetch factor helps overlap data loading with GPU computation
    prefetch_factor = 2 if pin_memory else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers= (num_workers > 0) # Keep workers alive between epochs
    )
