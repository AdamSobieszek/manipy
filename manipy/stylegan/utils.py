#@title utils
import numpy as np
import PIL.Image

import IPython.display
from PIL import Image, ImageDraw
from math import ceil
from io import BytesIO
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
import torchvision.transforms.functional as TF
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFont
import urllib.request
import functools
import io

import requests
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
from .core import setup_stylegan
from ..visualization import get_font, add_label_to_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define activation functions and their default parameters
activation_funcs = {
    'linear': {
        'func': lambda x, alpha: x,
        'def_alpha': 0,
        'def_gain': 1,
    },
    'relu': {
        'func': F.relu,
        'def_alpha': 0,
        'def_gain': np.sqrt(2),
    },
    'lrelu': {
        'func': lambda x, alpha: F.leaky_relu(x, negative_slope=alpha),
        'def_alpha': 0.2,
        'def_gain': np.sqrt(2),
    },
    'tanh': {
        'func': torch.tanh,
        'def_alpha': 0,
        'def_gain': 1,
    },
    'sigmoid': {
        'func': torch.sigmoid,
        'def_alpha': 0,
        'def_gain': 1,
    },
    'elu': {
        'func': F.elu,
        'def_alpha': 0,
        'def_gain': 1,
    },
    'selu': {
        'func': F.selu,
        'def_alpha': 0,
        'def_gain': 1,
    },
    'softplus': {
        'func': F.softplus,
        'def_alpha': 0,
        'def_gain': 1,
    },
    'swish': {
        'func': lambda x, alpha: x * torch.sigmoid(x),
        'def_alpha': 0,
        'def_gain': np.sqrt(2),
    },
}

def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None):
    """
    Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. It supports
    first and second order gradients.

    Args:
        x (torch.Tensor): Input activation tensor. Can be of any shape.
        b (torch.Tensor, optional): Bias vector, or `None` to disable. Must be a
            1D tensor of the same type as `x`. The shape must be known, and it
            must match the dimension of `x` corresponding to `dim`.
        dim (int, optional): The dimension in `x` corresponding to the elements
            of `b`. Ignored if `b` is not specified. Default is 1.
        act (str, optional): Name of the activation function to evaluate, or
            `"linear"` to disable. Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`,
            `"sigmoid"`, `"swish"`, etc. See `activation_funcs` for a full list.
            `None` is not allowed. Default is `'linear'`.
        alpha (float, optional): Shape parameter for the activation function, or
            `None` to use the default. Default is `None`.
        gain (float, optional): Scaling factor for the output tensor, or `None`
            to use default. See `activation_funcs` for the default scaling of
            each activation function. If unsure, consider specifying 1. Default is `None`.
        clamp (float, optional): Clamp the output values to `[-clamp, +clamp]`, or
            `None` to disable the clamping (default). Default is `None`.

    Returns:
        torch.Tensor: Tensor of the same shape and datatype as `x`.
    """
    if act is None:
        raise ValueError("`act` cannot be None. Use 'linear' to disable activation.")
    if act not in activation_funcs:
        raise ValueError(f"Unsupported activation function '{act}'. "
                         f"Supported functions are: {list(activation_funcs.keys())}")
    
    # Retrieve activation function and default parameters
    act_spec = activation_funcs[act]
    act_func = act_spec['func']
    alpha = float(alpha) if alpha is not None else act_spec['def_alpha']
    gain = float(gain) if gain is not None else act_spec['def_gain']

    # Add bias if provided
    if b is not None:
        if not isinstance(b, torch.Tensor):
            raise TypeError(f"Bias `b` must be a torch.Tensor, got {type(b)}")
        if b.ndim != 1:
            raise ValueError(f"Bias `b` must be a 1D tensor, got shape {b.shape}")
        if not (0 <= dim < x.ndim):
            raise ValueError(f"Dimension `dim`={dim} is out of range for input tensor with {x.ndim} dimensions")
        if b.shape[0] != x.shape[dim]:
            raise ValueError(f"Bias `b` has shape {b.shape}, which does not match the size of dimension {dim} in `x` ({x.shape[dim]})")
        
        # Reshape bias for broadcasting
        reshape_dims = [1] * x.ndim
        reshape_dims[dim] = -1
        b_reshaped = b.view(reshape_dims)
        x = x + b_reshaped

    # Apply activation function
    if act != 'linear':
        x = act_func(x, alpha=alpha)

    # Apply gain
    if gain != 1:
        x = x * gain

    # Apply clamping
    if clamp is not None:
        if clamp < 0:
            raise ValueError(f"Clamp value must be non-negative, got {clamp}")
        x = x.clamp(-clamp, clamp)

    return x

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()
    
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'



def listify(x):
    """
    Converts a single element or a pandas DataFrame/Series to a list.
    If the input is already a list, it returns the input unmodified.

    Args:
        x: The input to be listified.

    Returns:
        list: A list of the input elements.
    """
    if isinstance(x, (list, pd.DataFrame, pd.Series)):
        return list(x)
    return [x]

def display_image(image_array, format='png', jpeg_fallback=True):
    """
    Displays an image in IPython.

    Args:
        image_array: A numpy array representing the image.
        format: The format of the image to display.
        jpeg_fallback: Whether to fall back to JPEG if the image is too large.

    Returns:
        The IPython.display object.
    """
    image_array = np.asarray(image_array, dtype=np.uint8)
    str_file = BytesIO()
    PIL.Image.fromarray(image_array).save(str_file, format)
    im_data = str_file.getvalue()
    try:
        return IPython.display.display(IPython.display.Image(im_data))
    except IOError as e:
        if jpeg_fallback and format != 'jpeg':
            print(f'Warning: image was too large to display in format "{format}"; trying jpeg instead.')
            return display_image(image_array, format='jpeg')
        else:
            raise

def create_image_grid(images, scale=1, rows=1):
    """
    Creates a grid of images.

    Args:
        images: A list of PIL.Image objects.
        scale: The scale factor for each image.
        rows: The number of rows in the grid.

    Returns:
        A single PIL.Image object containing the grid of images.
    """
    w, h = images[0].size
    w, h = int(w * scale), int(h * scale)
    height = rows * h
    cols = ceil(len(images) / rows)
    width = cols * w
    canvas = PIL.Image.new('RGBA', (width, height), 'white')
    for i, img in enumerate(images):
        img = img.resize((w, h), PIL.Image.ANTIALIAS)
        canvas.paste(img, (w * (i % cols), h * (i // cols)))
    return canvas

def dot_product(x, y):
    """
    Computes the normalized dot product of two vectors.

    Args:
        x, y: The vectors to compute the dot product of. Can be file paths or numpy arrays.

    Returns:
        The normalized dot product of x and y.
    """
    x = np.load(x) if isinstance(x, str) else x
    y = np.load(y) if isinstance(y, str) else y
    x_norm = x[1] if len(x.shape) > 1 else x
    y_norm = y[1] if len(y.shape) > 1 else y
    return np.dot(x_norm / np.linalg.norm(x_norm), y_norm / np.linalg.norm(y_norm))

def reshape_to_18(target, is_w=False, truncation_psi=1, _G=None):
    global G
    if _G is None and 'G' in globals():
        _G = G
    if len(target.shape) == 1:
        target = target.view(1,-1)

    
    if len(target.shape) == 2:
        target = target.view(-1, 1, 512)


    if target.shape[1] != 18:
        target = target.repeat(1,18,1)

    if not is_w and _G is not None:
        norms_target = target.norm(dim=-1).mean(dim=-1)
        is_z = norms_target>20
        if any(is_z):
            target[is_z] = _G.mapping(target[is_z].mean(dim=1), None, truncation_psi=truncation_psi)

    return target


def read(target, passthrough=True, device='mps', is_w=False, truncation_psi=1, _G=None):
    """
    Transforms a path or array of coordinates into a standard format.

    Args:
        target: A path to the coordinate file or a numpy array.
        passthrough: If True, returns the target if it cannot be transformed.

    Returns:
        Transformed target or original target based on passthrough.
    """
    global G
    if _G is None and 'G' in globals():
        _G = G
    device = device or _G.mapping.w_avg.device
    
    if target is None:
        return 0
    if isinstance(target, PIL.Image.Image):
        return None
    
    if isinstance(target, str):
        if target.endswith('.npy'):
            return torch.tensor(np.load(target), map_location=device)
        elif target.endswith('.pt'):
            return torch.load(target, map_location=device)
        elif target.endswith('.jpg') or target.endswith('.png') or target.endswith('.jpeg'):
            return (target, PIL.Image.open(target) )       
        else:
            raise ValueError(f"Unknown file type: {target}")
        
    if isinstance(target, np.ndarray):
        target = torch.tensor(target, device=device)
    if not isinstance(target, torch.Tensor):
        raise ValueError(f"Unknown target type: {type(target)}")
    

    if not len(target.shape) == 3 or target.shape[0] != 18:
        target = reshape_to_18(target, is_w=is_w, truncation_psi=truncation_psi, _G=_G)


    return target

def show_faces(target, add=None, subtract=True, plot=True, grid=True, rows=1, labels = None, device='mps', is_w=False, verbose=False, truncation_psi=1, _G=None):
    """
    Displays or returns images of faces generated from latent vectors.

    Args:
        target: Latent vectors or paths to images. Can be a string, np.array, or list thereof.
        add: Latent vector to add to the target. Can be None, np.array, or list thereof.
        subtract: If True, subtracts 'add' from 'target'.
        plot: If True, plots the images using matplotlib.
        grid: If True, displays images in a grid.
        rows: Number of rows in the grid.
        device: Device for PyTorch operations.
        G: The StyleGAN generator model.

    Returns:
        PIL images or None, depending on the 'plot' argument.
    """
    global G
    
    if _G is None and 'G' in globals():
        _G = G
    device = device or _G.mapping.w_avg.device

    transform = Compose([
        Resize(512),
        lambda x: torch.clamp((x + 1) / 2, min=0, max=1)
    ])

    target, add, subtract = listify(target), listify(add), listify(subtract)

    to_generate = [read(t, False, device, is_w=is_w, truncation_psi=truncation_psi) for t in target if read(t, False) is not None]
    loading_dict, to_generate = {t[0]: t[1] for t in to_generate if isinstance(t, tuple)}, [t for t in to_generate if not isinstance(t, tuple)]
    target = [loading_dict.get(t, t) for t in target]


    if add[0] is not None:
        to_generate_add = [t + read(v, False, device, is_w=True) for t in to_generate for v in add]
        to_generate_sub = [t - read(v, False, device, is_w=True) for t in to_generate for v in add]
        to_generate = [m for pair in zip(to_generate_sub, to_generate, to_generate_add) for m in pair] if subtract else [m for pair in zip(to_generate, to_generate_add) for m in pair]

    other = [PIL.Image.open(t) for t in target if isinstance(t, str) and not '.npy' in t and not '.pt' in t]
    other += [t for t in target if isinstance(t, PIL.Image.Image)]
    for im in target:
        try:
            other += [TF.to_pil_image(transform(im))]
        except:
            pass

    images_pil = []
    if len(to_generate) > 0:
        if _G is None:
            raise ValueError("G is not set")
        
        to_generate = torch.cat(to_generate, dim=0).to(device)
        
        with torch.no_grad():
            assert len(to_generate)<=32, "Too many faces to generate"
            images = _G.synthesis(to_generate.view(-1, 18, 512)).cpu()
            images_pil = [TF.to_pil_image(transform(im)) for im in images]

    images_pil += [(t) for t in other]

    if plot:
        display_images(images_pil, grid, rows, labels=labels)
    else:
        return create_image_grid(images_pil, rows=rows) if grid else images_pil



class G_context:
    """A context manager to temporarily set the global G object."""
    def __init__(self, g_instance):
        # Store the G instance you want to use
        self.g_instance = g_instance
        self.original_g = None

    def __enter__(self):
        """Called when entering the 'with' block."""
        global G
        if 'G' in globals():
            # Save the original G so we can restore it later
            self.original_g = G
        # Set the global G to our new instance
        G = self.g_instance
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Called when exiting the 'with' block."""
        global G
        # Restore the original G
        G = self.original_g


def display_images(images, grid, rows, labels):
    """
    Helper function to display images using matplotlib, with optional labels on each image.

    Args:
        images: A list of PIL.Image objects.
        grid: If True, displays images in a grid.
        rows: Number of rows in the grid.
        labels: List of labels for each image; if provided, labels will be added to images.
    """
    if labels:
        images = [add_label_to_image(im.copy(), lbl) for im, lbl in zip(images, labels)]

    if grid and len(images) > 1:
        cols = (len(images) + rows - 1) // rows  # Compute number of columns needed
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axs = axs.flatten()  # Flatten the array of axes for easier iteration
        for idx, (im, ax) in enumerate(zip(images, axs)):
            ax.imshow(im)
            ax.axis('off')  # Hide axes
        plt.tight_layout()
        plt.show()
    else:
        for idx, im in enumerate(images):
            plt.figure(figsize=(5, 5))
            plt.imshow(im)
            plt.axis('off')
            plt.show()

def sample_w(n, truncation_psi=1, device=None, **kwargs):
    """Samples N latent vectors (w) that pass certain filter criteria."""
    global G
    device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    try:
        _G = G
        assert _G is not None, "G is not set. Use the 'G_context' manager. or pass G as an argument."
    except:
        G = kwargs.get("G", kwargs.get("_G", None))

    if G is None:
        class MockGenerator(nn.Module):
            mapping = MappingNetwork(512,0,512,18).to(device).load_state_dict(torch.load("/Users/adamsobieszek/PycharmProjects/_manipy/mapping_network/map (1).pt", weights_only=True, map_location=device))
        
        G = MockGenerator()
        print("'G' is not set. Use the 'G_context' manager. or pass G as an argument.")

    if G is None:
        raise ValueError("G is not set. Use the 'G_context' manager. or pass G as an argument.")
    _G = G
    device = device or _G.mapping.w_avg.device

    with torch.no_grad():
        all_z = torch.randn([n, 512], device=device)
        all_w = _G.mapping(all_z, None, truncation_psi=truncation_psi)[:, 0]
        return all_w