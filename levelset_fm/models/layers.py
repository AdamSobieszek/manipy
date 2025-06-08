# models/layers.py
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from ..utils.stylegan_utils import get_w_avg

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
        self.register_buffer('freqs', freqs, persistent=False)

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



class MeanRegressor(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(0.1),  # Removed feature dropout

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, target_dim),
            # Consider an output activation here if appropriate for your target data
        )

    def forward(self, x):
        return self.network(x)


class EnsembleRegressor(nn.Module):
    def __init__(self, models, scale=1):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.scale = scale


    def forward(self, x, *args,**kwargs):
        outputs = [model(x, *args,**kwargs) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)*self.scale


class Flow(nn.Module):
    def __init__(self, latent_dim=512, output_dim=512, time_varying=True):
        super().__init__()
        self.w_avg = w_avg


        self.head = nn.Sequential(
            nn.Linear(512, output_dim),
        )
        self.network = nn.Sequential(
            nn.Linear(latent_dim + (1 if time_varying else 0), 512),
            nn.BatchNorm1d(512),
            nn.SELU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.SELU(),


            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.SELU(),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.SELU(),
            nn.Dropout(0.2),

            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.SELU(),
        )

    def forward(self, x):
        x = x - self.w_avg
        x = self.network(x)
        x = self.head(x)
        return x
    


import torch
import torch.nn as nn
import torch.nn.functional as F

class Exponent(nn.Module):
    def forward(self, input):
        # Ensure the last dimension has length 2
        assert input.size(-1) == 2, "The last dimension must have length 2"

        # Split the input tensor along the last dimension
        x1, x2 = input.chunk(2, dim=-1)

        # Apply sigmoid to the first part and exponential to the second part
        out1 = torch.exp(x1)
        out2 = torch.exp(x2)

        # Concatenate the results along the last dimension
        return torch.cat([out1, out2], dim=-1)



class AlphaBetaRegressor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w_avg = get_w_avg().detach()


        self.head = nn.Sequential(
            nn.Linear(512, 2),
        )
        self.network = nn.Sequential(
            nn.Linear(512, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),

            nn.Linear(dim, dim*2),
            nn.BatchNorm1d(dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(dim*2, dim*4),
            nn.BatchNorm1d(dim*4),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(dim*4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.output = "a,b"  # "mean", "mean+var"
        self.equations = {
            "a,b": lambda x: x,
            "mean": self.mean,
            "mean+var": lambda x: self.mean(x)
        }

    def forward(self, x, output=None):
        if output is None:
            output = self.output
        x = x - self.w_avg
        x = self.network(x)
        x = self.head(x)
        x = Exponent()(x)
        return self.equations[output](x)

    def mean(self, x):
        return (x[:, 0]/(x[:, 0]+x[:, 1])).view(-1, 1)
    


# Model Definition
import torch
from torch import nn

class SigmoExpo(nn.Module):
    def forward(self, input):
        # Ensure the last dimension has length 2
        assert input.size(-1) == 2, "The last dimension must have length 2"

        # Split the input tensor along the last dimension
        x1, x2 = input.chunk(2, dim=-1)

        # Apply sigmoid to the first part and exponential to the second part
        out1 = torch.exp(x1)
        out2 = torch.exp(x2)

        # Concatenate the results along the last dimension
        return torch.cat([out1, out2], dim=-1)



# Custom Dataset
class LatentDataset(Dataset):
    def __init__(self, latents, targets):
        self.latents = torch.tensor(latents, dtype=torch.float32).to(device)
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.targets[idx]
