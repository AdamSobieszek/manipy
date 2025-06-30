import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import minimize

import numpy as np


def interpolate_distributions(observations1, observations2, t):
    """
    Interpolate between two sets of observations based on their order.

    Args:
    observations1 (list or array): First set of observations
    observations2 (list or array): Second set of observations
    t (float): Interpolation parameter between 0 and 1

    Returns:
    numpy.ndarray: Interpolated samples
    """
    # Convert inputs to numpy arrays and sort them
    sorted_obs1 = np.sort(observations1)
    sorted_obs2 = np.sort(observations2)

    # Determine the number of samples in the interpolated distribution
    n_samples = max(len(sorted_obs1), len(sorted_obs2))

    # Create evenly spaced quantiles
    quantiles = np.linspace(0, 1, n_samples)

    # Interpolate both distributions to have the same number of points
    interp_obs1 = np.interp(quantiles, np.linspace(0, 1, len(sorted_obs1)), sorted_obs1)
    interp_obs2 = np.interp(quantiles, np.linspace(0, 1, len(sorted_obs2)), sorted_obs2)

    # Perform the interpolation
    interpolated_samples = (1 - t) * interp_obs1 + t * interp_obs2

    return interpolated_samples

def fit_beta(data):
    def neg_log_likelihood(params):
        return -np.sum(beta.logpdf(data, params[0], params[1]))

    result = minimize(neg_log_likelihood, [1, 1], method='L-BFGS-B', bounds=[(1e-5, None), (1e-5, None)])
    return result.x


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.ion()  # Turn on interactive mode
    observations1 = np.random.beta(1, 1, 1000)
    observations2 = np.random.beta(5, 2, 1000)

    # Create subplots for different interpolation strengths
    interpolation_strengths = [0, 0.25, 0.5, 0.75, 1]
    fig, axes = plt.subplots(1, len(interpolation_strengths), figsize=(20, 4))
    fig.suptitle('Corrected Bootstrap Interpolation with Fitted Beta Distributions')

    for i, t in enumerate(interpolation_strengths):
        interpolated_samples = interpolate_distributions(observations1, observations2, t)

        # Plot histogram
        axes[i].hist(interpolated_samples, bins=30, density=True, alpha=0.7, color='skyblue')

        # Fit and plot beta distribution
        a, b = fit_beta(interpolated_samples)
        x_beta = np.linspace(0, 1, 100)
        y_beta = beta.pdf(x_beta, a, b)
        axes[i].plot(x_beta, y_beta, color='red', linestyle='--', label=f'Beta({a:.2f}, {b:.2f})')

        axes[i].set_title(f't = {t}, a+b = {a + b:.2f}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

    plt.pause(100)