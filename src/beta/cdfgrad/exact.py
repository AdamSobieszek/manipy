import torch
import torch.nn as nn


class PreciseBetaCDF(nn.Module):
    def __init__(self, max_iterations=1000, epsilon=1e-15):
        super().__init__()
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def forward(self, z, alpha, beta):
        """
        Compute the CDF of the Beta distribution.

        Args:
            z (torch.Tensor): The value at which to evaluate the CDF.
            alpha (torch.Tensor): The alpha parameter of the Beta distribution.
            beta (torch.Tensor): The beta parameter of the Beta distribution.

        Returns:
            torch.Tensor: The CDF value.
        """
        # Ensure z is clamped between 0 and 1
        z = torch.clamp(z, 0, 1)

        # Handle edge cases
        cdf = torch.where(z == 0, torch.zeros_like(z),
                          torch.where(z == 1, torch.ones_like(z),
                                      self._incomplete_beta(z, alpha, beta)))

        return cdf

    def _incomplete_beta(self, x, a, b):
        """
        Compute the incomplete beta function using Lentz's algorithm.
        """
        # Initialize variables
        f = torch.zeros_like(x)
        c = f + 1
        d = 1 / (1 - (a + b) * x / (a + 1))

        for i in range(self.max_iterations):
            m = i // 2
            numerator = torch.where(i % 2 == 0,
                                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
                                    m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)))

            d = 1 / (1 + numerator * d)
            f = f * d
            c = c * d

            if torch.all(torch.abs(f - 1) < self.epsilon):
                break

        # Compute the final result
        beta_x = torch.exp(torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b) +
                           a * torch.log(x) + b * torch.log(1 - x))
        return beta_x * f / a

    def cdf_gradients(self, z, alpha, beta):
        """
        Compute the gradients of the Beta CDF with respect to alpha and beta.

        Args:
            z (torch.Tensor): The value at which to evaluate the CDF.
            alpha (torch.Tensor): The alpha parameter of the Beta distribution.
            beta (torch.Tensor): The beta parameter of the Beta distribution.

        Returns:
            tuple: (dF/dalpha, dF/dbeta)
        """
        z = z.requires_grad_(True)
        alpha = alpha.requires_grad_(True)
        beta = beta.requires_grad_(True)

        cdf = self.forward(z, alpha, beta)

        # Compute gradients
        dF_dalpha, dF_dbeta = torch.autograd.grad(cdf.sum(), [alpha, beta], create_graph=True)

        return dF_dalpha, dF_dbeta


# Example usage:
if __name__ == "__main__":
    precise_beta_cdf = PreciseBetaCDF()

    z = torch.tensor([0.3, 0.5, 0.7])
    alpha = torch.tensor([2.0, 5.0, 10.0])
    beta = torch.tensor([2.0, 5.0, 10.0])

    cdf = precise_beta_cdf(z, alpha, beta)
    print("CDF values:", cdf)

    dF_dalpha, dF_dbeta = precise_beta_cdf.cdf_gradients(z, alpha, beta)
    print("dF/dalpha:", dF_dalpha)
    print("dF/dbeta:", dF_dbeta)