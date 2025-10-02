import os, re
import numpy as np
import pandas as pd
import torch

class BinnedDataset(Dataset):
    """
    A memory-efficient dataset that hot-loads pre-binned data files.

    Given a gender and age, it loads the corresponding '{gender}_{age}.pt' file,
    which is small and memory-friendly. It holds only one bin in memory at a time.
    """
    def __init__(self, binned_data_dir):
        self.binned_data_dir = binned_data_dir
        self.w = torch.empty(0)
        self.age = torch.empty(0)
        self.y = torch.empty(0)
        self.current_bin = None
        print(f"BinnedStreamingDataset initialized for directory: {binned_data_dir}")

    def load_bin(self, gender, age):
        """
        Loads the data for a specific gender and integer age bin into memory.

        Returns:
            bool: True if the bin was loaded successfully, False otherwise.
        """
        bin_id = (gender, int(age))
        if self.current_bin == bin_id:
            return True # Bin is already loaded

        fp = os.path.join(self.binned_data_dir, f"{gender}_{int(age)}.pt")

        if not os.path.exists(fp):
            fp = os.path.join(self.binned_data_dir, f"{gender}_age_{int(age)}.pt")
            print(fp)
        if not os.path.exists(fp):
            fp = os.path.join(self.binned_data_dir, f"{gender}_agebin_{int(age)}_part_0.pt")
        if not os.path.exists(fp):
            print(f"[BinnedStreamingDataset] file not found: {fp}")
            return False

        try:
            data = torch.load(fp, map_location='cpu')
            print(data)
            if isinstance(data, torch.Tensor):
              data = {'w':data}
            print(data)
            # Here we could select indices based on the y values to achieve uniform distribution
            indices = np.arange(len(data['w']))
            np.random.shuffle(indices)
            indices = torch.from_numpy(indices)#[:40_000]
            if "y" in data:
              data['w'] = data['w'][indices].to(torch.float32)
              data['age'] = data['age'][indices]
              data['y'] = data['y'][indices]
              self.w = data['w']
              self.age = data['age']
              self.y = data['y']
            else:
              data['w'] = data['w'][indices].to(torch.float32)
              data['age'] = None
              self.w = data['w']
              self.age = torch.zeros(len(data['w']))
              self.y = torch.zeros(len(data['w']))
            self.current_bin = bin_id
            print(f"Successfully loaded bin {bin_id} with {len(self.w)} samples.")
            return True
        except Exception as e:
            print(f"Error loading bin {bin_id} from {fp}: {e}")
            self.w, self.age, self.y = [torch.empty(0)] * 3
            self.current_bin = None
            return False

    def get_full_bin_data(self):
        """Returns all data from the currently loaded bin."""
        if self.current_bin is None:
            return None, None, None
        return self.w, self.age, self.y

    def __len__(self):
        return self.w.shape[0]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range for the currently loaded bin")
        return self.w[idx], self.age[idx], self.y[idx]
BINNED_DATA_PATH = '/content/drive/Shareddrives/[PsychGAN]/[1] Flow_datasets/dataset_trustworthy3/face_coordinate_bins'
BINNED_DATA_PATH = '/content/drive/Shareddrives/[PsychGAN]/bins'
# Assume: simple_dataset = BinnedDataset(BINNED_DATA_PATH)
simple_dataset = BinnedDataset(BINNED_DATA_PATH)

# Discover available bins from filenames like "male_21.pt"
all_files = set(os.listdir(BINNED_DATA_PATH))
def ages_for_gender(gender: str):
    pat = re.compile(rf'^{gender}_(\d+)\.pt$')
    pat2 = re.compile(rf'^{gender}_agebin_(\d+)_part_0\.pt$')
    return sorted({int(m.group(1)) for f in all_files if (m := pat.match(f))}) or sorted({int(m.group(1)) for f in all_files if (m := pat2.match(f))})

genders = ['male', 'female']
ages_by_gender = {g: ages_for_gender(g) for g in genders}
print(all_files, ages_by_gender
      )
def predict_batched(model, w: torch.Tensor, batch_size: int = 8192) -> np.ndarray:
    """
    Run model(w) in batches, returning a 1D numpy array of predictions.
    Uses model's own device if it's an nn.Module; otherwise stays on CPU.
    """
    if isinstance(model, torch.nn.Module):
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device('cpu')
        was_training = model.training
        model.eval()
    else:
        dev = torch.device('cpu')
        was_training = None

    outs = []
    with torch.no_grad():
        for i in range(0, w.shape[0], batch_size):
            x = w[i:i+batch_size].to(dev).float()
            yhat = model(x)
            if isinstance(yhat, (tuple, list)):
                yhat = yhat[0]
            outs.append(yhat.detach().float().cpu().numpy().reshape(-1))

    if isinstance(model, torch.nn.Module) and was_training:
        model.train()

    return np.concatenate(outs, axis=0) if outs else np.empty((0,), dtype=np.float32)

rows = []
model_keys = list(ALL_MODELS.keys())
@torch.no_grad()
def save_diverse_by_unit_age(
    G,
    w: torch.Tensor,
    age_t: torch.Tensor,
    y: Optional[torch.Tensor],
    gender: str,
    out_dir: str,
    k: int,
    *,
    start_idx: int = 0,
    selection: str = "qr",        # or "greedy"
    batch_size: int = 30,
) -> Tuple[int, int]:
    """
    Minimal helper:
      - splits (w, age_t[, y]) into unit-age bins (floor(age) == i),
      - selects up to k diverse rows per age bin,
      - synthesizes images with G and saves jpg + .pt sidecar.

    Returns (next_start_idx, images_written).
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "/coords"),exist_ok=True)
    device = next(G.parameters()).device if isinstance(G, torch.nn.Module) else torch.device("cpu")
    if isinstance(G, torch.nn.Module):
        was_training = G.training
        G.eval()
    else:
        was_training = None

    # Accept [N,D] (preferred). If [N,num_ws,D], use the first w.
    if w.ndim == 3 and w.shape[1] >= 1:
        w = w[:, 0, :]
    assert w.ndim == 2, "w must be [N, D] or [N, num_ws, D]"

    # Move to device, clean ages
    w0 = w.to(device).float()
    a0 = age_t.to(device).float()
    valid = torch.isfinite(a0)
    base_idx = torch.arange(w0.shape[0], device=device)[valid]
    wv = w0[valid]
    av = a0[valid]
    y0 = y if (y is not None and y.shape[0] == w.shape[0]) else None

    if wv.shape[0] == 0:
        if isinstance(G, torch.nn.Module) and was_training:
            G.train()
        return start_idx, 0

    ages_int = torch.floor(av).to(torch.int64)
    unique_ages = torch.unique(ages_int, sorted=True)
    gender_ch = "m" if str(gender).lower().startswith("m") else "f"
    num_ws = int(getattr(getattr(G, "synthesis", G), "num_ws", 18))

    images_written = 0

    for age_i in unique_ages.tolist():
        rel_idx = torch.where(ages_int == age_i)[0]                # indices into filtered set
        if rel_idx.numel() == 0:
            continue
        sub_w = wv.index_select(0, rel_idx)
        kk = min(int(k), int(sub_w.shape[0]))
        if kk <= 0:
            continue

        sel_local = _select_diverse_rows(sub_w, k=kk, method=selection)
        if not sel_local:
            continue
        sel_local = torch.as_tensor(sel_local, device=device, dtype=torch.long)

        sel_w = sub_w.index_select(0, sel_local)
        orig_idx = base_idx.index_select(0, rel_idx).index_select(0, sel_local)  # back to original row ids

        from PIL import Image
        import numpy as np

        for i in range(0, sel_w.shape[0], batch_size):
            batch = sel_w[i:i+batch_size]
            ws = batch.unsqueeze(1).repeat(1, num_ws, 1)           # [B, num_ws, D]
            imgs = G.synthesis(ws, noise_mode="const")             # [B,C,H,W], ~[-1,1]
            imgs = torch.clamp((imgs + 1.0) * 0.5, 0.0, 1.0)
            imgs8 = (imgs.detach().cpu() * 255.0).round().to(torch.uint8)

            B = imgs8.shape[0]

            # ---- Stage 2: Parallel batch encode with the fixed recipe ----
            fmt, scale, q = ("webp", 0.25, 50)
            tasks = []
            with ProcessPoolExecutor(max_workers=8) as ex:
                for b in range(B):
                    stem = f"{start_idx}_{gender_ch}_{int(age_i)}"
                    img_path = os.path.join(out_dir, f"{stem}.jpg")
                    # Convert imgs8[b] (C,H,W) torch tensor to PIL Image
                    img_arr = imgs8[b].permute(1, 2, 0).numpy()  # (H, W, C)
                    pil_img = Image.fromarray(img_arr)
                    tasks.append(ex.submit(_encode_im_to_path, pil_img, img_path, fmt, scale, q))

                    oi = int(orig_idx[i + b].item())
                    payload = {
                        "w": batch[b].detach().cpu(),
                        "age_int": int(age_i),
                        "gender": gender_ch,
                        "orig_index": oi,
                        "sel_rank_in_agebin": i + b,
                    }
                    if y0 is not None and 0 <= oi < y0.shape[0]:
                        payload["y"] = y0[oi].detach().cpu()
                    torch.save(payload, os.path.join(out_dir, "/coords", f"{stem}.pt"))

                    start_idx += 1
                    images_written += 1



                ok = 0
                for fut in as_completed(tasks):
                    try:
                    result = fut.result()
                    ok += 1
                    except Exception as e:
                    print("[ERR]", e)

    if isinstance(G, torch.nn.Module) and was_training:
        G.train()
    return start_idx, images_written
images_written = 0
img_idx = 0
OUT_DIR = '/content/drive/Shareddrives/[PsychGAN]/age_test'
k = 32
w_avg = w_avg.to('cuda:0')
with torch.no_grad():
  for gender in genders:
      for age in ages_by_gender.get(gender, []):
          if not simple_dataset.load_bin(gender, age):
              continue
          w, age_t, y = simple_dataset.get_full_bin_data()
          w = w.to('cuda:0')
          if w is None or w.numel() == 0:
              continue
          w = (w-w_avg)*0.9+w_avg
          age_t = ALL_MODELS['age'](w)
          # plt.hist(age_t.cpu().reshape(-1).numpy())
          # plt.show()

          for age in [int(age_t.mean()-0.5), int(age_t.mean()+0.5)]:
            _w,_age_t = w[age_t.to(torch.int).view(-1)==age,:],age_t[age_t.to(torch.int)==age]
            img_idx, inc = save_diverse_by_unit_age(G, _w, _age_t, None, gender, OUT_DIR, 225, start_idx=0)
            images_written += inc