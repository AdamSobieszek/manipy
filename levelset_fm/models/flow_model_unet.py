# models/flow_model.py
import torch
import torch.nn as nn
from typing import Dict, Any
from einops import rearrange, repeat, pack, unpack
from x_transformers import Attention, FeedForward, RMSNorm
from x_transformers.x_transformers import RotaryEmbedding
from torch.amp import autocast
import math

from ..configs import config
from ..utils.stylegan_utils import get_w_avg
from ..models.layers import TimestepEmbedding # Import needed layer
import torch
import numpy as np
from torch import nn
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import math
from typing import Sequence

# --- Helper Modules ---


class SinusoidalPosEmb(nn.Module):
    """ Sinusoidal Positional Embedding for conditioning (e.g., on ratings). """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of ratings, shape (B,) or (B, 1).
        Returns:
            torch.Tensor: Embedded tensor, shape (B, dim).
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Ensure x is 2D for broadcasting
        emb = x.view(x.shape[0], -1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResnetBlock(nn.Module):
    """ 
    A ResNet block using FiLM (Feature-wise Linear Modulation) for conditioning.
    This is more expressive than simple additive conditioning.
    """
    def __init__(self, dim_in: int, dim_out: int, *, cond_dim: int):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.SiLU(),
            nn.Linear(dim_in, dim_out)
        )
        
        # FiLM generator: projects conditioning vector to a scale and shift parameter
        self.film_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim_out * 2) # Outputs gamma (scale) and beta (shift)
        )

        self.block2 = nn.Sequential(
            nn.LayerNorm(dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
        
        self.res_conn = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        
        # Generate scale and shift from the conditioning embedding
        gamma, beta = self.film_proj(cond_emb).chunk(2, dim=-1)
        
        # Apply FiLM: h_out = h_in * (1 + gamma) + beta
        h = h * (1 + gamma) + beta
        
        h = self.block2(h)
        return h + self.res_conn(x)

# --- Main U-Net Model ---

class VectorFieldUNetLegacy(nn.Module):
    """
    An enhanced U-Net architecture to predict a vector field v(xt, rating_condition)
    in a 512-dim latent space. It uses FiLM for conditioning and a self-attention
    bottleneck to better capture global relationships in the latent vector.

    ---
    Pipeline Improvement Note: Batch Averaging for Variance Reduction
    ---
    The provided research paper suggests a powerful technique to reduce the high
    variance of the training target, which is not an architectural change but a
    *training pipeline* change. It involves computing a variance-reduced target
    vector by averaging over a minibatch of OT pairs.

    Instead of computing the target `V_tangential` from a single pair (x0, x1),
    you can compute it from a minibatch of M pairs `{(x0_i, x1_i), i=1..M}`.
    
    The new conditional vector field `u_t(x | z_bar)` for a batch of pairs `z_bar` is:
    u_t(x | z_bar) = sum_i[u_t(x | z_i) * p_t(x | z_i)] / sum_i[p_t(x | z_i)]
    
    Where:
    - `z_i = (x0_i, x1_i)` is the i-th pair in the minibatch.
    - `u_t(x | z_i) = x1_i - x0_i` is the straight-line path for that pair.
    - `p_t(x | z_i)` is the likelihood of `x` under the Gaussian path for that pair.
      `p_t(x|z_i) = exp(-||x - (t*x1_i + (1-t)*x0_i)||^2 / (2*sigma^2))`.

    Implementation in your training loop:
    
    1. For a batch of `B` interpolated points `xt_batch`, also sample a minibatch
       of `M` OT pairs `(x0_m_batch, x1_m_batch)` from your `MinibatchOTPairer`.
    2. For each `xt` in `xt_batch`, calculate its `M` likelihoods against the `M` paths.
    3. Compute the weighted average `sigma_avg` for each `xt`.
    4. Calculate the final target: `target = sigma_avg - project(sigma_avg, grad_f)`.
    
    This will produce a much more stable training target, directly addressing the
    variance issue.
    """
    def __init__(
        self,
        rating_model= None,
        model_dim: int = 512,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        condition_dim: int = 128,
        num_attn_heads: int = 8,
        add_rating_gradient: bool = True,
        orthogonal_gradient_projection: bool = True,
    ):
        """
        Initializes the VectorFieldUNet.
        Args:
            model_dim*2 (int): Dimension of the input/output latent vectors.
            dim_mults (Sequence[int]): Divisors for feature dimensions at each U-Net level.
            condition_dim (int): Dimension of the embedded rating condition vector.
            num_attn_heads (int): Number of heads for the self-attention bottleneck.
        """
        super().__init__()
        self.add_rating_gradient = add_rating_gradient
        self.orthogonal_gradient_projection = orthogonal_gradient_projection
        self.model_dim = model_dim
        self.rating_model = [rating_model.eval()]
        # --- Rating Embedding ---
        self.rating_emb = nn.Sequential(
            SinusoidalPosEmb(condition_dim),
            nn.Linear(condition_dim, condition_dim * 4),
            nn.GELU(),
            nn.Linear(condition_dim * 4, condition_dim)
        )
        
        # --- U-Net Architecture ---
        hidden_dims = [model_dim*2] + [model_dim*2 // m for m in dim_mults]
        
        # --- Encoder ---
        self.downs = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            dim_in = hidden_dims[i]
            dim_out = hidden_dims[i+1]
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, cond_dim=condition_dim),
                ResnetBlock(dim_in, dim_in, cond_dim=condition_dim),
                nn.Linear(dim_in, dim_out)
            ]))

        # --- Bottleneck with Self-Attention ---
        bottleneck_dim = hidden_dims[-1]
        self.bottleneck = nn.ModuleList([
            ResnetBlock(bottleneck_dim, bottleneck_dim, cond_dim=condition_dim),
            nn.MultiheadAttention(embed_dim=bottleneck_dim, num_heads=num_attn_heads, batch_first=True),
            ResnetBlock(bottleneck_dim, bottleneck_dim, cond_dim=condition_dim)
        ])
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)

        # --- Decoder ---
        self.ups = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            dim_in_upsample = hidden_dims[i]
            dim_out_upsample = hidden_dims[i-1]
            resnet_in_dim = dim_out_upsample * 2
            self.ups.append(nn.ModuleList([
                nn.Linear(dim_in_upsample, dim_out_upsample),
                ResnetBlock(resnet_in_dim, dim_out_upsample, cond_dim=condition_dim),
                ResnetBlock(resnet_in_dim, dim_out_upsample, cond_dim=condition_dim),
            ]))
            
        # --- Final Projection ---
        self.final_resnet = ResnetBlock(model_dim* 4, model_dim, cond_dim=condition_dim)
        self.final_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim)
        )
        
        nn.init.zeros_(self.final_proj[-1].weight)
        nn.init.zeros_(self.final_proj[-1].bias)
        self.w_avg = get_w_avg()

        self.input_proj = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim*2),
            nn.GELU(),
            nn.Linear(self.model_dim*2, self.model_dim)
        )

    def forward(self, x: torch.Tensor, ratings: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the enhanced VectorFieldUNet.
        """
        if x.ndim == 3: x = x.squeeze(1)
        assert x.ndim == 2 and x.shape[1] == self.model_dim, f"Input tensor x must have shape (B, {self.model_dim})"

        if ratings is None:
            with torch.no_grad():
                ratings = self.rating_model[0](x)

        with torch.enable_grad():
            xt_detached = (x).clone().detach().requires_grad_(True)
            rating_output = self.rating_model[0](xt_detached) # Get logit output

            # Sum for scalar loss to get gradient w.r.t. xt
            rating_output_sum = torch.sum(rating_output)
            rating_output_sum.backward()

            xt_grad = xt_detached.grad.detach() # Gradient of rating score w.r.t. xt

        # Add gradient to the predicted vector field
        grad_norm = xt_grad.norm(dim=-1, keepdim=True)
        xt_grad = xt_grad / grad_norm**2
        
        x_orig = x 
        x_centered = x - self.w_avg
        x_proj = self.input_proj(x_orig)
        cond_emb = self.rating_emb(ratings)
        
        # 1. Encoder Path
        residuals = []
        h = torch.cat([x_proj, (x_centered if not self.add_rating_gradient else xt_grad)], dim=-1)
        for block1, block2, downsample in self.downs:
            h = block1(h, cond_emb)
            residuals.append(h)
            h = block2(h, cond_emb)
            residuals.append(h)
            h = downsample(h)

        # 2. Bottleneck Path
        h = self.bottleneck[0](h, cond_emb)
        
        # Self-attention block
        h_norm = self.bottleneck_norm(h)
        # Reshape for multihead attention: (B, L, E) where L=1 for us
        attn_in = h_norm.unsqueeze(1)
        attn_out, _ = self.bottleneck[1](attn_in, attn_in, attn_in)
        # Add residual and reshape back
        h = h + attn_out.squeeze(1)
        
        h = self.bottleneck[2](h, cond_emb)

        # 3. Decoder Path
        for upsample, block1, block2 in self.ups:
            h = upsample(h)
            h = torch.cat([h, residuals.pop()], dim=-1)
            h = block1(h, cond_emb)
            h = torch.cat([h, residuals.pop()], dim=-1)
            h = block2(h, cond_emb)

        # 4. Final Projection
        h = torch.cat([h, x_centered, xt_grad], dim=-1)
        h = self.final_resnet(h, cond_emb)
        vector_field_pred = self.final_proj(h)
        if self.orthogonal_gradient_projection:
                normed_gradient = xt_grad * grad_norm
                vector_field_pred = vector_field_pred - normed_gradient * torch.sum(normed_gradient * vector_field_pred, dim=-1, keepdim=True)
            
        if self.add_rating_gradient:
            vector_field_pred = vector_field_pred/grad_norm + xt_grad
        
        return vector_field_pred


import torch
import torch.nn as nn
import math
from typing import Sequence, Optional

# --- Helper Modules ---

class SinusoidalPosEmb(nn.Module):
    """ 
    Sinusoidal Positional Embedding for conditioning (e.g., on ratings).
    This module is unchanged from the original as it is effective.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.view(x.shape[0], -1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class VectorResBlock(nn.Module):
    """
    A residual block for 1D vectors that uses FiLM (Feature-wise Linear Modulation)
    for conditioning. It replaces convolutional layers with linear layers.

    :param in_channels: The number of input channels (features).
    :param out_channels: The number of output channels (features).
    :param cond_channels: The number of channels in the conditioning vector.
    :param dropout: The dropout rate.
    """
    def __init__(self, in_channels: int, out_channels: int, *, cond_channels: int, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_layers = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, out_channels),
        )

        # FiLM generator: projects conditioning vector to a scale and shift parameter
        self.cond_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_channels, out_channels * 2),
        )

        self.out_layers = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_channels, out_channels)
        )

        # Initialize the final linear layer to be zero
        nn.init.zeros_(self.out_layers[-1].weight)
        nn.init.zeros_(self.out_layers[-1].bias)

        # Skip connection to match output dimensions
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        :param x: an [N x C] Tensor of features.
        :param cond: an [N x cond_channels] Tensor of conditioning embeddings.
        :return: an [N x out_channels] Tensor of outputs.
        """
        h = self.in_layers(x)
        
        # Generate FiLM scale and shift from the conditioning embedding
        scale, shift = self.cond_layers(cond).chunk(2, dim=1)
        
        # Apply FiLM: h_out = h_in * (1 + scale) + shift
        h = h * (1 + scale) + shift
        
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h

class VectorAttentionBlock(nn.Module):
    """
    An attention block for 1D vectors.

    :param channels: The number of input and output channels (features).
    :param num_heads: The number of attention heads.
    """
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(channels)
        # We use MultiheadAttention which is more standard and efficient
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: an [N x C] Tensor of features.
        :return: an [N x C] Tensor of outputs.
        """
        # MultiheadAttention expects (N, L, E) where L is sequence length.
        # For our single vector, L=1.
        h = x.unsqueeze(1)
        h = self.norm(h)
        attn_output, _ = self.attention(h, h, h)
        attn_output = self.proj_out(attn_output)
        
        # Add residual connection
        return x + attn_output.squeeze(1)


# --- Main U-Net Model ---

class VectorFieldUNet(nn.Module):
    """
    A 1D U-Net architecture for predicting a vector field in a latent space.
    This model avoids convolutions and uses linear layers, making it suitable
    for single latent vectors. It preserves the critical logic of incorporating
    the gradient of a rating function to guide the vector field.

    The architecture is inspired by modern UNet designs, using ResBlocks with
    FiLM conditioning and a central self-attention bottleneck.
    """
    def __init__(
        self,
        rating_model: nn.Module,
        model_dim: int = 512,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        condition_dim: int = 128,
        num_res_blocks: int = 2,
        num_attn_heads: int = 8,
        dropout: float = 0.1,
        add_rating_gradient: bool = True,
        orthogonal_gradient_projection: bool = True,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.add_rating_gradient = add_rating_gradient
        self.orthogonal_gradient_projection = orthogonal_gradient_projection
        
        # We store the rating model in a list to prevent its parameters
        # from being registered as part of this module's parameters.
        self.rating_model = [rating_model.eval()]
        # A pre-computed average vector, e.g., from StyleGAN's w_avg
        self.register_buffer('w_avg', torch.zeros(model_dim)) 

        # --- Conditioning Embedding ---
        self.rating_emb = nn.Sequential(
            SinusoidalPosEmb(condition_dim),
            nn.Linear(condition_dim, condition_dim * 4),
            nn.GELU(),
            nn.Linear(condition_dim * 4, condition_dim)
        )

        # --- Network Architecture ---
        # Determine the channel dimensions for each level of the U-Net
        channels = [model_dim] + [model_dim * m for m in dim_mults]
        
        # The input to the UNet will be the concatenation of the centered vector
        # and the rating gradient.
        init_dim = model_dim * 2
        
        self.input_proj = nn.Linear(init_dim, channels[0])

        # --- Encoder (Down-sampling path) ---
        self.downs = nn.ModuleList([])
        for i in range(len(channels) - 1):
            dim_in = channels[i]
            dim_out = channels[i+1]
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(VectorResBlock(dim_in, dim_in, cond_channels=condition_dim, dropout=dropout))
            # Downsample by reducing feature dimension with a linear layer
            block.append(nn.Linear(dim_in, dim_out))
            self.downs.append(block)

        # --- Bottleneck ---
        bottleneck_dim = channels[-1]
        self.middle_block = nn.ModuleList([
            VectorResBlock(bottleneck_dim, bottleneck_dim, cond_channels=condition_dim, dropout=dropout),
            VectorAttentionBlock(bottleneck_dim, num_heads=num_attn_heads),
            VectorResBlock(bottleneck_dim, bottleneck_dim, cond_channels=condition_dim, dropout=dropout)
        ])

        # --- Decoder (Up-sampling path) ---
        self.ups = nn.ModuleList([])
        for i in range(len(channels) - 1, 0, -1):
            dim_in = channels[i]
            dim_out = channels[i-1]
            block = nn.ModuleList()
            # Upsample by increasing feature dimension with a linear layer
            block.append(nn.Linear(dim_in, dim_out))
            # The input to the ResBlock will be the concatenation of the upsampled
            # vector and the skip connection from the encoder.
            for _ in range(num_res_blocks):
                 block.append(VectorResBlock(dim_out * 2, dim_out, cond_channels=condition_dim, dropout=dropout))
            self.ups.append(block)

        # --- Final Projection ---
        self.final_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )
        
        # Initialize final layer to zero for stable starting
        nn.init.zeros_(self.final_proj[-1].weight)
        nn.init.zeros_(self.final_proj[-1].bias)

    def forward(self, x: torch.Tensor, ratings: torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass for the VectorFieldUNet.

        :param x: Input tensor (latent vector), shape (B, model_dim).
        :param ratings: Conditioning ratings, shape (B,).
        :return: Predicted vector field, shape (B, model_dim).
        """
        if x.ndim == 3:
            x = x.squeeze(1)
        assert x.ndim == 2 and x.shape[1] == self.model_dim, \
            f"Input tensor x must have shape (B, {self.model_dim})"
        
        if ratings is None:
            with torch.no_grad():
                ratings = self.rating_model[0](x)

        # 1. CRITICAL: Calculate the gradient of the rating score wrt the input vector x
        with torch.enable_grad():
            xt_detached = x.clone().detach().requires_grad_(True)
            ratings_output = self.rating_model[0](xt_detached)
            ratings_output_sum = torch.sum(ratings_output)
            ratings_output_sum.backward()
            xt_grad = xt_detached.grad.detach()

        grad_norm = xt_grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # 2. Prepare inputs and conditioning
        cond_emb = self.rating_emb(ratings)
        x_centered = x - self.w_avg
        
        # Concatenate inputs for the network
        h = torch.cat([x_centered, xt_grad / grad_norm**2], dim=-1)
        h = self.input_proj(h)

        # 3. Encoder Path
        residuals = []
        for resnet_blocks in self.downs:
            for resnet_block in resnet_blocks[:-1]:
                h = resnet_block(h, cond_emb)
                residuals.append(h)
            downsample = resnet_blocks[-1]
            h = downsample(h)

        # 4. Bottleneck
        for block in self.middle_block:
            if isinstance(block, VectorAttentionBlock):
                h = block(h)
            else:
                h = block(h, cond_emb)

        # 5. Decoder Path
        for resnet_blocks in self.ups:
            upsample = resnet_blocks[0]
            h = upsample(h)
            # Concatenate with skip connection
            for j, resnet_block in enumerate(resnet_blocks[1:]):
                h = torch.cat([h, residuals.pop()], dim=-1)
                h = resnet_block(h, cond_emb)

        # 6. Final Projection
        vector_field_pred = self.final_proj(h)

        # 7. CRITICAL: Preserve the gradient modification logic
        if self.orthogonal_gradient_projection:
            # Project the prediction to be orthogonal to the gradient direction
            normed_gradient = xt_grad / grad_norm
            proj_component = torch.sum(normed_gradient * vector_field_pred, dim=-1, keepdim=True)
            vector_field_pred = vector_field_pred - normed_gradient * proj_component
            
        if self.add_rating_gradient:
            # Add the scaled gradient to the final prediction
            vector_field_pred = vector_field_pred/grad_norm + xt_grad / grad_norm**2
        
        return vector_field_pred
    
    