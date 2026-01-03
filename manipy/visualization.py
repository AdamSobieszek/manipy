import numpy as np
import PIL.Image
from PIL import Image, ImageDraw, ImageFont
from math import ceil
from io import BytesIO
import matplotlib.pyplot as plt
import os
import cv2
import IPython.display
import torch
import torchvision.transforms.functional as TF

# Attempt to find a system font, fallback to DejaVuSans if available with OpenCV
def get_font(font_size=20):
    font_path = None
    try:
        # Try common system paths (adjust for your OS if needed)
        possible_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Linux
            "/Library/Fonts/Arial.ttf", # macOS
            "C:/Windows/Fonts/arial.ttf", # Windows
            os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf') # OpenCV path
        ]
        for path in possible_paths:
            if os.path.exists(path):
                font_path = path
                break
        if font_path:
            return ImageFont.truetype(font_path, font_size)
        else:
            print("Warning: Could not find default font. Using PIL default.")
            return ImageFont.load_default() # PIL's fallback
    except Exception as e:
        print(f"Warning: Error loading font: {e}. Using PIL default.")
        return ImageFont.load_default()


def display_image_ipython(image_array, format='png', jpeg_fallback=True):
    """Displays an image in IPython."""
    image_array = np.asarray(image_array, dtype=np.uint8)
    str_file = BytesIO()
    PIL.Image.fromarray(image_array).save(str_file, format)
    im_data = str_file.getvalue()
    try:
        return IPython.display.display(IPython.display.Image(im_data))
    except IOError as e:
        if jpeg_fallback and format != 'jpeg':
            print(f'Warning: image was too large to display in format "{format}"; trying jpeg instead.')
            return display_image_ipython(image_array, format='jpeg')
        else:
            raise

def create_image_grid(images, scale=1.0, rows=1):
    """ Creates a grid of images. """
    if not images: return None
    w, h = images[0].size
    w, h = int(w * scale), int(h * scale)
    cols = ceil(len(images) / rows)
    height = rows * h
    width = cols * w
    canvas = PIL.Image.new('RGBA', (width, height), 'white')
    for i, img in enumerate(images):
        try:
            img_resized = img.resize((w, h), PIL.Image.Resampling.LANCZOS) # Updated resampling method
            canvas.paste(img_resized, (w * (i % cols), h * (i // cols)))
        except Exception as e:
            print(f"Error resizing/pasting image {i}: {e}")
    return canvas


def add_label_to_image(image, label, position=(10, 10), font_size=20):
    """Adds a label with a black stroke to an image."""
    draw = ImageDraw.Draw(image)
    max_line_length = max(max(len(line) for line in label.split('\n')), 10)
    scale_factor = 1000/max_line_length/21.0
    font = get_font(int(font_size*scale_factor))

    # Get text size using textbbox
    try:
         # Use textbbox for better accuracy if available
         bbox = draw.textbbox(position, label, font=font)
         text_width = bbox[2] - bbox[0]
         text_height = bbox[3] - bbox[1]
         # Adjust position slightly (optional, based on how bbox works)
         draw_pos = (position[0], position[1])
    except AttributeError:
         # Fallback for older PIL versions
         text_width, text_height = draw.textsize(label, font=font)
         draw_pos = position


    # Outline (stroke) parameters
    stroke_width = 2
    stroke_fill = "black"

    # Draw text outline by drawing in offset positions
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            if dx*dx + dy*dy >= stroke_width*stroke_width: continue # circular stroke
            draw.text((draw_pos[0]+dx, draw_pos[1]+dy), label, font=font, fill=stroke_fill)

    # Draw the main text over the outline
    draw.text(draw_pos, label, font=font, fill="white")

    return image

def display_images_matplotlib(images, grid=True, rows=1, labels=None, title=None):
    """Displays images using matplotlib, with optional grid and labels."""
    if not images:
        print("No images to display.")
        return

    if labels:
        if len(labels) != len(images):
            print("Warning: Number of labels does not match number of images. Skipping labels.")
            labels = None
        else:
            # Add labels directly to copies of the images
            images_to_display = [add_label_to_image(im.copy(), lbl) for im, lbl in zip(images, labels)]
    else:
        images_to_display = images

    if grid and len(images_to_display) > 1:
        cols = (len(images_to_display) + rows - 1) // rows
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False) # Ensure axs is always 2D
        axs_flat = axs.flatten()
        for idx, (im, ax) in enumerate(zip(images_to_display, axs_flat)):
            ax.imshow(im)
            ax.axis('off')
        # Hide unused subplots
        for idx in range(len(images_to_display), len(axs_flat)):
             axs_flat[idx].axis('off')

        if title:
            fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95] if title else None) # Adjust layout for suptitle
        plt.show()
    else:
        for idx, im in enumerate(images_to_display):
            plt.figure(figsize=(5, 5))
            if labels: plt.title(labels[idx])
            plt.imshow(im)
            plt.axis('off')
            plt.show()

