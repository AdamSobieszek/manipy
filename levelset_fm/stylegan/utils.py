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


def add_label_to_image(image, label, position=(10, 10), font_size=20):
    """
    Adds a label with a black stroke to an image at the specified position.

    Args:
        image: PIL.Image object.
        label: Text to add to the image.
        position: Tuple specifying the position to place the text.
        font_size: Size of the font.

    Returns:
        PIL.Image object with text added.
    """
    draw = ImageDraw.Draw(image)

    # You can use a system font or a bundled .ttf file
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, font_size)

    # Get the bounding box for the text
    bbox = draw.textbbox(position, label, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Adjust position based on the text height
    position = (position[0], position[1] - text_height*.5)

    # Outline (stroke) parameters
    stroke_width = 2
    stroke_fill = "black"

    # Draw text with outline
    draw.text(position, label, font=font, fill="white", stroke_width=stroke_width, stroke_fill=stroke_fill, textlength = text_width)

    return image



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

def sample_w(n, truncation_psi=1, device=None, _G=None):
    """Samples N latent vectors (w) that pass certain filter criteria."""
    global G
    if _G is None and 'G' in globals():
        _G = G
    device = device or _G.mapping.w_avg.device

    with torch.no_grad():
        all_z = torch.randn([n, 512], device=device)
        all_w = _G.mapping(all_z, None, truncation_psi=truncation_psi)[:, 0]
        return all_w