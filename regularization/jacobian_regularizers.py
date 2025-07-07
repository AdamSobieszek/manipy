# jacobian_regularizers.py

import torch
import torch.nn as nn
from torch.func import vmap, jacrev

# --- 1. The Autoencoder Model ---

class Autoencoder(nn.Module):
    """A standard MLP-based autoencoder."""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# --- 1. The Autoencoder Model ---

class Autoencoder(nn.Module):
    """A standard MLP-based autoencoder."""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# --- 2. The Regularization Functions ---

def loss_off_diagonal(autoencoder: Autoencoder, x: torch.Tensor) -> torch.Tensor:
    """
    (CORRECTED) Penalizes the off-diagonal elements of the decoder's Gram Matrix (J^T J).
    
    This encourages the columns of the Jacobian to be orthogonal, meaning the
    effects of different latent variables on the output are independent. This is a
    sound way to enforce disentanglement for non-square Jacobians.
    """
    z = autoencoder.encoder(x)
    
    compute_jac = jacrev(autoencoder.decoder)
    jacobians = vmap(compute_jac)(z) # Shape: [batch_size, input_dim, latent_dim]

    # Transpose Jacobians to get shape [batch_size, latent_dim, input_dim]
    jacobians_t = jacobians.transpose(1, 2)

    # Compute the Gram matrix J^T @ J for each item in the batch.
    # The result is a batch of square matrices, shape: [batch_size, latent_dim, latent_dim]
    gram_matrix = torch.bmm(jacobians_t, jacobians)

    # Get the diagonal part of each Gram matrix
    gram_diag = torch.diag_embed(torch.diagonal(gram_matrix, dim1=-2, dim2=-1))
    
    # Calculate the squared Frobenius norm of the off-diagonal elements
    off_diag_norm_sq = torch.linalg.matrix_norm(gram_matrix - gram_diag, ord='fro') ** 2
    
    return torch.mean(off_diag_norm_sq)


def loss_contractive(autoencoder: Autoencoder, x: torch.Tensor) -> torch.Tensor:
    """
    (VERIFIED) Contractive Autoencoder (CAE) loss.
    
    Penalizes the Frobenius norm of the encoder's Jacobian. This norm is
    well-defined for rectangular matrices and works correctly as is.
    """
    compute_jac = jacrev(autoencoder.encoder)
    jacobians = vmap(compute_jac)(x) # Shape: [batch_size, latent_dim, input_dim]
    
    # Calculate the squared Frobenius norm of the entire Jacobian
    frobenius_norm_sq = torch.linalg.matrix_norm(jacobians, ord='fro') ** 2
    
    return torch.mean(frobenius_norm_sq)


def loss_nuclear_norm(autoencoder: Autoencoder, x: torch.Tensor) -> torch.Tensor:
    """
    (VERIFIED) Penalizes the nuclear norm of the decoder's Jacobian.
    
    The nuclear norm (sum of singular values) is well-defined for any
    rectangular matrix and works correctly as is.
    """
    z = autoencoder.encoder(x)
    
    compute_jac = jacrev(autoencoder.decoder)
    jacobians = vmap(compute_jac)(z) # Shape: [batch_size, input_dim, latent_dim]
    
    # The nuclear norm is the sum of singular values
    nuclear_norm = torch.linalg.matrix_norm(jacobians, ord='nuc')
    
    return torch.mean(nuclear_norm)


def loss_l1_sparse(autoencoder: Autoencoder, x: torch.Tensor) -> torch.Tensor:
    """
    (VERIFIED) Penalizes the L1 norm of the latent code itself.
    
    This regularization operates on the latent vector, not a Jacobian matrix,
    so it is unaffected by the matrix shape issues.
    """
    z = autoencoder.encoder(x)
    
    # Compute the L1 norm for each latent vector in the batch
    l1_norm = torch.linalg.norm(z, ord=1, dim=1)
    
    return torch.mean(l1_norm)