import torch
import torch.nn as nn
from torch.distributions.beta import Beta

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

        This is used to determine the boundaries between approximation regions.
        """
        return torch.sqrt(alpha * beta) / ((alpha + beta) * torch.sqrt(alpha + beta + 1))

    def _main_approximation(self, z, alpha, beta):
        """
        Compute the main approximation for dz/dalpha (or dz/dbeta with appropriate input).

        This approximation is derived from the Lugannani-Rice approximation and is
        accurate for alpha > 6 and beta > 6, away from the singularity.
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
        return term1 + term2  # Note: Original text had ±, we use + as it's more stable

    def _compute_S_term(self, alpha, beta):
        """Compute the S term in the main approximation."""
        return torch.exp(torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta))

    def _near_singularity_approximation(self, z, alpha, beta):
        """
        Compute the near-singularity approximation for dz/dalpha (or dz/dbeta with appropriate input).

        This approximation is used when |z - alpha/(alpha+beta)| <= epsilon * sigma.
        """
        H = 8 * alpha ** 4 * (135 * beta - 11) * (1 - z)
        I = alpha ** 3 * beta * (453 - 455 * z + 1620 * beta * (1 - z))
        J = 3 * alpha ** 2 * beta ** 2 * (180 * beta - 90 * z + 59)
        K = alpha * beta ** 3 * (20 * z * (27 * beta + 16) + 43) + 47 * beta ** 4 * z

        numerator = (12 * alpha + 1) * (12 * beta + 1) * (H + I + J + K)
        denominator = 12960 * alpha ** 3 * beta ** 2 * (alpha + beta) ** 2 * (12 * alpha + 12 * beta + 1)

        return numerator / denominator



class BetaCDFGradientApproximator(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.implicit_gradient_approximator = BetaZGradientApproximator(epsilon)

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
        dz_dalpha, dz_dbeta = self.implicit_gradient_approximator(z, alpha, beta)

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

    def _beta_function(self, alpha, beta):
        """
        Compute the Beta function B(alpha, beta).
        """
        return torch.exp(torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta))

