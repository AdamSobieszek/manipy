#@title core
import os
import pickle
import torch
import torchvision.transforms as transforms
import sys

def whatever():
  print("whatever")

def setup_stylegan():
    """
    Sets up the StyleGAN environment by installing required packages, cloning the necessary GitHub repository,
    mounting Google Drive (if in Colab), downloading additional files, and loading the StyleGAN model.

    Returns:
        G (torch.nn.Module): The StyleGAN generator model.
        face_w (torch.Tensor): A tensor of sample latent vectors.
    """
    # Check if in a GPU runtime
    _ = torch.ones(1).to('mps')
    # Check and install required packages
    required_packages = ['einops', 'ninja']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            os.system(f"pip install --upgrade {package} -f https://download.pytorch.org/whl/torch_stable.html --quiet")

    # Change directory to content
    os.chdir('/Users/adamsobieszek/PycharmProjects/psychGAN/content')

    # Download additional files
    os.system('gdown 1O79M5F5G3ktmt1-zeccbJf1Bhe8K9ROz')
    if not os.path.exists('content/omi'):
        os.system('git clone https://github.com/jcpeterson/omi')
        os.system('unzip content/omi/attribute_ratings.zip')

    repo_url = 'github.com/AdamSobieszek/psychGAN'
    repo_path = 'content/psychGAN'
    token = "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"
    if not os.path.exists(repo_path):
        os.system(f"git clone https://{token}@{repo_url}")

    # Add necessary paths to sys.path
    sys.path.append('psychGAN/stylegan3')
    sys.path.append('/Users/adamsobieszek/PycharmProjects/psychGAN/content')
    sys.path.append('/Users/adamsobieszek/PycharmProjects/psychGAN/content/psychGAN')
    sys.path.append('/Users/adamsobieszek/PycharmProjects/psychGAN')
    os.chdir('/Users/adamsobieszek/PycharmProjects/psychGAN')

    # Download the StyleGAN model file if not present
    model_path = "stylegan2-ffhq-1024x1024.pkl"
    if not os.path.exists(model_path):
        os.system('wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl')

    # Load the StyleGAN model
    device = torch.device('mps')
    with open(model_path, 'rb') as fp:
        G = pickle.load(fp)['G_ema'].to(device)

    # Compute the average latent vector
    all_z = torch.randn([1, G.mapping.z_dim], device=device)
    face_w = G.mapping(all_z, None, truncation_psi=0.5)

    return G, face_w, device
