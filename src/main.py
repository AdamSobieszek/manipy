import torch
import torch.nn as nn
from torch.distributions.beta import Beta

import os
def download_data():
    """
    Sets up the StyleGAN environment by installing required packages, cloning the necessary GitHub repository,
    mounting Google Drive (if in Colab), downloading additional files, and loading the StyleGAN model.

    Returns:
        G (torch.nn.Module): The StyleGAN generator model.
        face_w (torch.Tensor): A tensor of sample latent vectors.
    """
    # Download additional files
    os.system('gdown 1O79M5F5G3ktmt1-zeccbJf1Bhe8K9ROz')
    if not os.path.exists('omi'):
        os.system('git clone https://github.com/jcpeterson/omi')
        os.system('unzip omi/attribute_ratings.zip')

    repo_url = 'github.com/AdamSobieszek/psychGAN'
    repo_path = 'psychGAN'
    token = "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"
    if not os.path.exists(repo_path):
        os.system(f"git clone https://{token}@{repo_url}")

    return

class BetaZGradientApproximator(nn.Module):
    def __init__(self, epsilon=0.1):
        """
        Initialize the Beta CDF Gradient Approximator.

        Args:
            epsilon (float): Parameter to balance numerical accuracy and stability.
                             Default is 0.1 as specified in the original text.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, z, alpha, beta):
        """
        Compute the approximation of dz/dalpha and dz/dbeta.

        Args:
            z (torch.Tensor): The value at which to evaluate the CDF.
            alpha (torch.Tensor): The alpha parameter of the Beta distribution.
            beta (torch.Tensor): The beta parameter of the Beta distribution.

        Returns:
            tuple: (dz/dalpha, dz/dbeta)
        """
        # Compute dz/dalpha
        dz_dalpha = self._compute_gradient(z, alpha, beta)

        # Use symmetry to compute dz/dbeta
        dz_dbeta = self._compute_gradient(1 - z, beta, alpha)

        return dz_dalpha, dz_dbeta

    def _compute_gradient(self, z, alpha, beta):
        """
        Compute the gradient (dz/dalpha or dz/dbeta) based on the input region.
        """
        sigma = self._compute_sigma(alpha, beta)
        mean = alpha / (alpha + beta)

        # Determine which approximation to use
        use_singularity_approx = torch.abs(z - mean) <= self.epsilon * sigma

        # Compute both approximations
        main_approx = self._main_approximation(z, alpha, beta)
        singularity_approx = self._near_singularity_approximation(z, alpha, beta)

        # Use torch.where to select the appropriate approximation
        return torch.where(use_singularity_approx, singularity_approx, main_approx)

    def _compute_sigma(self, alpha, beta):
        """
        Compute the standard deviation of the Beta distribution.
        """
        return torch.sqrt(alpha * beta) / ((alpha + beta) * torch.sqrt(alpha + beta + 1))

    def _main_approximation(self, z, alpha, beta):
        """
        Compute the main approximation for dz/dalpha (or dz/dbeta with appropriate input).
        """
        A = self._compute_A(z, alpha, beta)
        B = self._compute_B(z, alpha, beta)
        S_term = self._compute_S_term(alpha, beta)

        numerator = z * (1 - z) * (A + torch.log(alpha / (z * (alpha + beta))) * B)
        denominator = torch.sqrt(2 * alpha * beta / (alpha + beta)) * S_term

        return numerator / denominator

    def _compute_A(self, z, alpha, beta):
        """Compute the A term in the main approximation."""
        numerator = beta * (2 * alpha ** 2 * (1 - z) + alpha * beta * (1 - z) + beta ** 2 * z)
        denominator = torch.sqrt(2 * alpha * beta) * (alpha + beta) ** (3 / 2) * (alpha * (1 - z) - beta * z) ** 2
        return numerator / denominator

    def _compute_B(self, z, alpha, beta):
        """Compute the B term in the main approximation."""
        term1 = torch.sqrt(2 * alpha * beta / (alpha + beta)) / (alpha * (1 - z) - beta * z)
        term2 = 0.5 * (alpha * torch.log(alpha / ((alpha + beta) * (1 - z))) +
                       beta * torch.log(beta / ((alpha + beta) * z))) ** (-3 / 2)
        return term1 + term2

    def _compute_S_term(self, alpha, beta):
        """Compute the S term in the main approximation."""
        return torch.exp(torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta))

    def _near_singularity_approximation(self, z, alpha, beta):
        """
        Compute the near-singularity approximation for dz/dalpha (or dz/dbeta with appropriate input).
        """
        H = 8 * alpha ** 4 * (135 * beta - 11) * (1 - z)
        I = alpha ** 3 * beta * (453 - 455 * z + 1620 * beta * (1 - z))
        J = 3 * alpha ** 2 * beta ** 2 * (180 * beta - 90 * z + 59)
        K = alpha * beta ** 3 * (20 * z * (27 * beta + 16) + 43) + 47 * beta ** 4 * z

        numerator = (12 * alpha + 1) * (12 * beta + 1) * (H + I + J + K)
        denominator = 12960 * alpha ** 3 * beta ** 2 * (alpha + beta) ** 2 * (12 * alpha + 12 * beta + 1)

        return numerator / denominator


class BetaCDFGradientCalculator(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.z_gradient_approximator = BetaZGradientApproximator(epsilon)

    def forward(self, z, alpha, beta):
        """
        Compute the gradients of the Beta CDF with respect to alpha and beta.

        Args:
            z (torch.Tensor): The value at which to evaluate the CDF.
            alpha (torch.Tensor): The alpha parameter of the Beta distribution.
            beta (torch.Tensor): The beta parameter of the Beta distribution.

        Returns:
            tuple: (dF/dalpha, dF/dbeta)
        """
        # Calculate implicit gradients
        dz_dalpha, dz_dbeta = self.z_gradient_approximator(z, alpha, beta)

        # Calculate PDF
        pdf = self._beta_pdf(z, alpha, beta)

        # Calculate CDF gradients
        dF_dalpha = -pdf * dz_dalpha
        dF_dbeta = -pdf * dz_dbeta

        return dF_dalpha, dF_dbeta

    def _beta_pdf(self, z, alpha, beta):
        """
        Compute the probability density function of the Beta distribution.
        """
        return torch.exp(Beta(alpha, beta).log_prob(z))


def gradient_vector_field(calculator, x_max, num_points=20):
    """
    Generate a gradient vector field for the Beta CDF.

    Args:
        calculator (BetaCDFGradientCalculator): The gradient calculator.
        x_max (float): The maximum value for alpha and beta.
        num_points (int): Number of points in each dimension of the grid.

    Returns:
        tuple: (alpha_grid, beta_grid, dF_dalpha_grid, dF_dbeta_grid)
    """
    alpha = torch.linspace(0.1, x_max, num_points)
    beta = torch.linspace(0.1, x_max, num_points)
    alpha_grid, beta_grid = torch.meshgrid(alpha, beta, indexing='ij')

    z_values = torch.linspace(0.5, 0.50001, 1)  # Sample z values

    dF_dalpha_grid = torch.zeros_like(alpha_grid)
    dF_dbeta_grid = torch.zeros_like(beta_grid)

    for z in z_values:
        dF_dalpha, dF_dbeta = calculator(z.expand_as(alpha_grid), alpha_grid, beta_grid)
        dF_dalpha_grid += dF_dalpha
        dF_dbeta_grid += dF_dbeta

    dF_dalpha_grid /= len(z_values)
    dF_dbeta_grid /= len(z_values)

    return alpha_grid, beta_grid, dF_dalpha_grid, dF_dbeta_grid

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode
# Usage example
if __name__ == "__main__":
    calculator = BetaZGradientApproximator()
    alpha_grid, beta_grid, dF_dalpha_grid, dF_dbeta_grid = gradient_vector_field(calculator, x_max=10, num_points=20)

    print("Alpha grid shape:", alpha_grid.shape)
    print("Beta grid shape:", beta_grid.shape)
    print("dF/dalpha grid shape:", dF_dalpha_grid.shape)
    print("dF/dbeta grid shape:", dF_dbeta_grid.shape)

    # You can now use these grids to plot the vector field
    # For example, using matplotlib:

    plt.figure(figsize=(10, 8))
    plt.quiver(alpha_grid.numpy(), beta_grid.numpy(),
               dF_dalpha_grid.numpy(), dF_dbeta_grid.numpy(),
               angles='xy', scale_units='xy', scale=0.1)
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Beta CDF Gradient Vector Field')
    plt.show()
    plt.pause(100)  # Pause for 1 second to keep the window open