
# models/layers.py
import torch
import torch.nn as nn
import math

class TimestepEmbedding(nn.Module):
    """ Sinusoidal embedding for timestep or rating values. """
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension ({dim}) must be even.")

        half_dim = self.dim // 2
        # Denominator term: 10000^(-2k/d)
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim)
        self.register_buffer('freqs', freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embeds a batch of time or scalar values.
        Args:
            t (torch.Tensor): Input tensor of shape (batch_size,) or (batch_size, 1).
        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, dim).
        """
        # Ensure t has shape (batch_size, 1) for broadcasting
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        # Calculate arguments for sin and cos: t * freqs
        args = t * self.freqs # Broadcasting works: (B, 1) * (D/2,) -> (B, D/2)

        # Concatenate sin and cos embeddings
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # Shape (B, D)
        return embedding


class SigmoExpo(nn.Module):
    """Applies sigmoid to first half and exp to second half of last dimension."""
    def forward(self, input):
        assert input.size(-1) % 2 == 0, "Last dimension must be even"
        split_size = input.size(-1) // 2
        x1, x2 = torch.split(input, split_size, dim=-1)
        out1 = torch.sigmoid(x1)
        out2 = torch.exp(x2)
        return torch.cat([out1, out2], dim=-1)

class Exponent(nn.Module):
    """Applies exp element-wise."""
    def forward(self, input):
        return torch.exp(input)
