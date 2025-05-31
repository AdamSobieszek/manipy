import numpy as np
import pandas as pd
import torch
import PIL.Image
from io import BytesIO
import IPython.display
import locale
import math
import os
import time

def listify(x):
    """Converts a single element or a pandas DataFrame/Series to a list."""
    if isinstance(x, (list, pd.DataFrame, pd.Series)):
        return list(x)
    return [x]

def read(target, passthrough=True):
    """Transforms a path or array of coordinates into a standard format."""
    if target is None: return 0
    if isinstance(target, PIL.Image.Image): return None # Skip images
    if isinstance(target, str):
        try: target = np.load(target)
        except: return target if passthrough else None # Return path if load fails and passthrough
    # Check for typical W space shapes or passthrough
    if isinstance(target, (np.ndarray, torch.Tensor)):
        if len(target.shape) == 3 and target.shape[1:] == (18, 512): # (1, 18, 512)
             return target
        if len(target.shape) == 2 and target.shape[0] == 18 and target.shape[1] == 512: # (18, 512)
             return target
        if len(target.shape) == 2 and target.shape[1] == 512: # (N, 512) - Tile needs context, maybe handle elsewhere
             # Avoid tiling here, could be ambiguous. Assume target is already in correct W format needed by caller.
             return target
        if len(target.shape) == 1 and target.shape[0] == 512: # (512,)
             # Needs context for tiling (18,1). Handle in calling function if needed.
             return target # Return as is, caller decides tiling

    return target if passthrough else None # Return original if no transformation applied and passthrough

def dot_product(x, y):
    """Computes the normalized dot product of two vectors."""
    x = np.load(x) if isinstance(x, str) else x
    y = np.load(y) if isinstance(y, str) else y
    # Assuming shape might be [something, 512] or just [512]
    x_norm = x[-1] if len(x.shape) > 1 else x
    y_norm = y[-1] if len(y.shape) > 1 else y
    x_norm = x_norm / np.linalg.norm(x_norm)
    y_norm = y_norm / np.linalg.norm(y_norm)
    return np.dot(x_norm, y_norm)

def print_gpu_memory():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    # Print memory for each GPU device
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i} Memory Summary:")
        print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1e6:.2f} MB")
        print(f"  Reserved:  {torch.cuda.memory_reserved(i) / 1e6:.2f} MB")
        # print(torch.cuda.memory_summary(device=i, abbreviated=True)) # More detailed

def set_locale():
    try:
        locale.getpreferredencoding = lambda: "UTF-8"
        print("Preferred encoding set to UTF-8.")
    except Exception as e:
        print(f"Could not set preferred encoding: {e}")

# Call it once on import if needed globally
set_locale()
