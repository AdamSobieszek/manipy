import pickle
import numpy as np
import torch


def pad_list(l, max_len):
    return l[:max_len] + [np.nan] * (max_len - len(l[:max_len]))
    
def prepare_data2(config: dict, verbose=False, device='mps', dtype=torch.float32, return_imgs=False, data=None):
    if data is None:
        photo_to_coords, dim_to_photo_to_ratings = load_psychGAN_data()
    else:
        photo_to_coords, dim_to_photo_to_ratings = data
    xs, ys = photo_to_coords, dim_to_photo_to_ratings[config['data']['attribute_dim']]
    _imgs = set(ys.keys()).intersection(set(xs.keys()))#-set('638.jpg')
    if config['data'].get('imgs', None) is not None:
        imgs = _imgs.intersection(set(config['data']['imgs']))
        if len(imgs)<len(config['data']['imgs']):
            print(f"Warning: {len(config['data']['imgs'])-len(imgs)} images not found in psychGAN dataset")
    else:
        imgs = sorted(_imgs)

    X = torch.stack([xs[k] for k in imgs]).to(device, dtype)
    max_len = int(max(len(ys[k]) for k in imgs) * config['data'].get('rating_oversampling_factor', 1.0))
    y = torch.tensor([pad_list(ys[k], max_len) for k in imgs], device=device, dtype=dtype)
    n = y.shape[1] - y.isnan().sum(dim=1)
    sample_size_weights = (n.reshape(-1)).to(torch.float32)
    sample_size_weights = sample_size_weights / sample_size_weights.mean()
    # from sklearn.neighbors import KernelDensity
    # kde = KernelDensity(kernel='gaussian', bandwidth=.5).fit(torch.nanmean(y, dim=1).cpu().numpy().reshape(-1,1))
    # log_dens = kde.score_samples(torch.nanmean(y, dim=1).cpu().numpy().reshape(-1,1))
    # weights = (np.exp(-log_dens))
    # weights = torch.tensor(weights/np.sum(weights)*len(n), device=DEVICE, dtype=DTYPE)

    # pressure = torch.sigmoid(torch.nanmean(y, dim=1, keepdim=True))*(1-torch.sigmoid(torch.nanmean(y, dim=1, keepdim=True)))
    # pressure = pressure / pressure.sum()*len(n)
    # weights = sample_size_weights*pressure
    # weights = weights**2
    # weights = weights / weights.sum()*len(n)
    if return_imgs:
        return X, y, sample_size_weights, imgs
    else:
        return X, y, sample_size_weights


def load_psychGAN_data():
    psychGAN_path = "/Users/adamsobieszek/PycharmProjects/psychGAN/"
    with open(psychGAN_path+"photo_to_coords.pkl", "rb") as f:
        photo_to_coords = pickle.load(f)
    with open(psychGAN_path+"dim_to_photo_to_ratings.pkl", "rb") as f:
        dim_to_photo_to_ratings = pickle.load(f)


    return photo_to_coords, dim_to_photo_to_ratings



