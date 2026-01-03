"""Utilities for loading latent priors and covariances for interpolation."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import torch


@dataclass(frozen=True)
class InterpolationData:
    """Container holding latents, covariances, and (optional) ratings."""

    photo_to_coords: Dict[str, torch.Tensor]
    photo_to_covariance: Dict[str, torch.Tensor]
    dim_to_photo_to_rating: Dict[str, Dict[str, float]]


def load_interp_data(base_dir = "/Users/adamsobieszek/PycharmProjects/PsychGAN/") -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, Dict[str, float]]
]:
    """
    Load latent coordinates, per-image covariance estimates, and ratings.

    Returns:
        (photo_to_coords, photo_to_covariance, dim_to_photo_to_rating)
    """

    # df = pd.read_csv(
    #     base_dir+"content/coords_wlosses.csv"
    # ).iloc[:, 2:-2]



    base_dir = Path(base_dir)

    cov_path = base_dir / "covs_2.pt"
    p2c_path = base_dir / "newer_photo_to_coords.pt"
    d2p2r_path = base_dir / "dim_to_photo_to_ratings2.pkl"
    with open(d2p2r_path, "rb") as f:
        dim_to_photo_to_ratings = pickle.load(f)
    print(dim_to_photo_to_ratings)
    # ws = torch.load(ws_path, map_location="cpu")
    # names = ws["order"]
    # w = ws["w"]
    # print(len(names))
    # cov = torch.load(cov_path, map_location="cpu")
    # print([*cov.keys()])
    # names_cov = cov["order"]
    # cov = cov["cov"]
    # print(len(names_cov))

    new_photo_to_w: Dict[str, torch.Tensor] = {}
    photo_to_covariance: Dict[str, torch.Tensor] = {}
    # for i, nm in enumerate(names):
    #     new_photo_to_w[nm] = w[i].view(-1)
    # for i, nm in enumerate(names_cov):
    #     photo_to_covariance[nm] = cov[i]

    photo_to_coords = torch.load(p2c_path, map_location="cpu")

    photo_to_coords = {**photo_to_coords, **new_photo_to_w}
    keys = set(sorted(photo_to_coords.keys())).intersection(set(dim_to_photo_to_ratings["age"].keys()))
    photo_to_coords = {k: torch.tensor(photo_to_coords[k]).cpu().flatten() for k in keys}
    photo_to_covariance = {k: torch.tensor(photo_to_covariance[k]).cpu().reshape(512,512) for k in keys} if photo_to_covariance else None
    if photo_to_covariance is None:
        return photo_to_coords, dim_to_photo_to_ratings
    return photo_to_coords, photo_to_covariance, dim_to_photo_to_ratings


def load_interp_dataclass(base_dir = "/Users/adamsobieszek/PycharmProjects/PsychGAN/") -> InterpolationData:
    """Convenience wrapper returning :class:`InterpolationData`."""

    data = load_interp_data(base_dir=base_dir)
    if len(data) == 2:
        return InterpolationData(
            photo_to_coords=data[0],
            photo_to_covariance={},
            dim_to_photo_to_rating=data[1],
        )
    else:
        return InterpolationData(
            photo_to_coords=data[0],
            photo_to_covariance=data[1],
            dim_to_photo_to_rating=data[2],
        )


def subset_priors(
    data: InterpolationData,
    filenames: Iterable[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Restrict priors to a subset of filenames (matching `Path(name).name`).
    """

    names = {Path(name).name for name in filenames}
    coords = {k: v for k, v in data.photo_to_coords.items() if k in names}
    covs = {k: v for k, v in data.photo_to_covariance.items() if k in names}
    return coords, covs


__all__ = [
    "InterpolationData",
    "load_interp_data",
    "load_interp_dataclass",
    "subset_priors",
]

