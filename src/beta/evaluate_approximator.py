import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as scipy_beta
from cdfgrad import PreciseBetaCDF, BetaCDFGradientApproximator

class BetaCDFGradientExperiment:
    def __init__(self, num_samples=1000, seed=42):
        self.num_samples = num_samples
        self.seed = seed
        self.precise_calculator = PreciseBetaCDF()
        self.approx_calculator = BetaCDFGradientApproximator()

    def generate_samples(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Sample alpha and beta using log-uniform distribution
        log_alpha = torch.rand(self.num_samples) * 4 - 2  # range: 10^-2 to 10^2
        log_beta = torch.rand(self.num_samples) * 4 - 2
        alpha = 10 ** log_alpha
        beta = 10 ** log_beta

        # Sample z using beta distribution
        z = torch.tensor([scipy_beta.rvs(a, b) for a, b in zip(alpha, beta)])

        return z, alpha, beta

    def compute_errors(self, z, alpha, beta):
        # Compute gradients using precise method
        precise_dalpha, precise_dbeta = self.precise_calculator.cdf_gradients(z, alpha, beta)

        # Compute gradients using approximation method
        approx_dalpha, approx_dbeta = self.approx_calculator(z, alpha, beta)

        # Compute relative errors
        error_alpha = torch.abs((approx_dalpha - precise_dalpha) / precise_dalpha)
        error_beta = torch.abs((approx_dbeta - precise_dbeta) / precise_dbeta)

        return error_alpha, error_beta

    def classify_regions(self, z, alpha, beta):
        # Classify points into different approximation regions
        sigma = torch.sqrt(alpha * beta) / ((alpha + beta) * torch.sqrt(alpha + beta + 1))
        xi = z * (1 - z) * (alpha + beta)

        region1 = (z <= 0.5) & (xi < 2.5)
        region2 = (z > 0.5) & (xi < 0.75)
        region3 = (alpha > 6) & (beta > 6) & (torch.abs(z - alpha / (alpha + beta)) > 0.1 * sigma)
        region4 = ~(region1 | region2 | region3)

        return region1, region2, region3, region4

    def run_experiment(self):
        z, alpha, beta = self.generate_samples()
        error_alpha, error_beta = self.compute_errors(z, alpha, beta)
        region1, region2, region3, region4 = self.classify_regions(z, alpha, beta)

        results = {
            'z': z.numpy(),
            'alpha': alpha.numpy(),
            'beta': beta.numpy(),
            'error_alpha': error_alpha.numpy(),
            'error_beta': error_beta.numpy(),
            'region1': region1.numpy(),
            'region2': region2.numpy(),
            'region3': region3.numpy(),
            'region4': region4.numpy()
        }

        return results

    def plot_results(self, results):
        plt.figure(figsize=(12, 10))

        # Plot error_alpha
        plt.subplot(2, 1, 1)
        self._scatter_plot(results, 'error_alpha', 'Alpha Gradient Error')

        # Plot error_beta
        plt.subplot(2, 1, 2)
        self._scatter_plot(results, 'error_beta', 'Beta Gradient Error')

        plt.tight_layout()
        plt.show()

    def _scatter_plot(self, results, error_key, title):
        for i, region in enumerate(['region1', 'region2', 'region3', 'region4']):
            mask = results[region]
            plt.scatter(results['alpha'][mask], results['beta'][mask],
                        c=np.log10(results[error_key][mask]),
                        cmap='viridis', alpha=0.7, s=20)

        plt.xscale('log')
        plt.yscale('log')
        plt.colorbar(label='Log10 Relative Error')
        plt.xlabel('Alpha')
        plt.ylabel('Beta')
        plt.title(title)

# Run the experiment
if __name__ == "__main__":
    experiment = BetaCDFGradientExperiment(num_samples=10000)
    results = experiment.run_experiment()
    experiment.plot_results(results)