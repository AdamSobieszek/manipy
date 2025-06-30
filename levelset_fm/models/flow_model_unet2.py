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

import torch
import torch.nn as nn
import math
from typing import Sequence, Optional

# --- Helper Modules (Largely Unchanged) ---

class SinusoidalPosEmb(nn.Module):
    """
    [KEPT] Sinusoidal Positional Embedding for conditioning. No changes needed.
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

class ResBlock(nn.Module):
    """
    [ENHANCED] Renamed from VectorResBlock for clarity.
    A residual block for 1D vectors that uses FiLM (Feature-wise Linear Modulation)
    for conditioning on a time-like embedding (e.g., from the rating scalar).

    :param in_channels: The number of input channels (features).
    :param out_channels: The number of output channels (features).
    :param cond_channels: The number of channels in the conditioning vector for FiLM.
    :param dropout: The dropout rate.
    """
    def __init__(self, in_channels: int, out_channels: int, *, cond_channels: Optional[int], dropout: float = 0.0):
        super().__init__()
        
        self.in_layers = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, out_channels),
        )

        # FiLM generator: projects conditioning vector to a scale and shift parameter
        self.cond_layers = None
        if cond_channels is not None:
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

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.in_layers(x)
        
        if self.cond_layers is not None and cond is not None:
            scale, shift = self.cond_layers(cond).chunk(2, dim=1)
            h = h * (1 + scale) + shift
        
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    [NEW] A SOTA attention block that can perform self- or cross-attention.
    This replaces the less effective VectorAttentionBlock.

    :param channels: The number of channels in the input query vector.
    :param num_heads: The number of attention heads.
    :param context_dim: The dimension of the context vector for cross-attention.
                        If None, performs self-attention.
    """
    def __init__(self, channels: int, num_heads: int = 8, context_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        
        self.norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        
        context_dim = context_dim if context_dim is not None else channels
        self.to_kv = nn.Linear(context_dim, channels * 2)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param x: an [N x C] Tensor of features (query).
        :param context: an optional [N x context_dim] Tensor for cross-attention.
        :return: an [N x C] Tensor of outputs.
        """
        h_ = self.norm(x)
        q = self.to_q(h_)
        
        context = context if context is not None else h_
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Reshape for MultiheadAttention: (N, L, E) where L=1 for our single vector
        q, k, v = map(lambda t: t.unsqueeze(1), (q, k, v))

        attn_output, _ = self.attention(q, k, v)
        attn_output = self.proj_out(attn_output.squeeze(1))
        
        return x + attn_output

# --- Main U-Net Model ---

class VectorFieldUNet(nn.Module):
    """
    [ENHANCED] A modern, SOTA U-Net architecture for predicting a vector field.
    This model uses ResBlocks with FiLM conditioning and powerful cross-attention
    blocks to deeply condition the network on the rating and its gradient.
    """
    def __init__(
        self,
        rating_model: nn.Module,
        model_dim: int = 512,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        condition_dim: int = 512,
        num_res_blocks: int = 2,
        num_attn_heads: int = 8,
        dropout: float = 0.1,
        add_rating_gradient: bool = True,
        orthogonal_gradient_projection: bool = True,
    ):
        super().__init__()
        self.model_dim = model_dim
        
        # [KEPT] Your core logic for handling the rating model and final output
        self.add_rating_gradient = add_rating_gradient
        self.orthogonal_gradient_projection = orthogonal_gradient_projection
        self.rating_model = [rating_model.eval()]
        self.register_buffer('w_avg', torch.zeros(model_dim)) 

        # --- Conditioning Embeddings ---
        # For the scalar rating value (used in FiLM)
        self.rating_emb = nn.Sequential(
            SinusoidalPosEmb(condition_dim),
            nn.Linear(condition_dim, condition_dim * 4),
            nn.GELU(),
            nn.Linear(condition_dim * 4, condition_dim)
        )

        # [NEW] For the rating gradient vector (used in Cross-Attention)
        self.grad_cond_dim = model_dim # The gradient has the same dimension as the model
        self.gradient_emb = nn.Sequential(
            nn.Linear(self.grad_cond_dim, self.grad_cond_dim * 4),
            nn.GELU(),
            nn.Linear(self.grad_cond_dim * 4, self.grad_cond_dim)
        )
        
        # --- Network Architecture ---
        dims = [model_dim, *map(lambda m: model_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.input_proj = nn.Linear(model_dim, dims[0])

        # --- Encoder (Down-sampling path) ---
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            block = nn.ModuleList([
                ResBlock(dim_in, dim_in, cond_channels=condition_dim, dropout=dropout),
                ResBlock(dim_in, dim_in, cond_channels=condition_dim, dropout=dropout),
                AttentionBlock(dim_in, num_heads=num_attn_heads, context_dim=self.grad_cond_dim),
                nn.Linear(dim_in, dim_out) if not is_last else nn.Identity()
            ])
            self.downs.append(block)

        # --- Bottleneck ---
        mid_dim = dims[-1]//2
        self.middle_block = nn.ModuleList([
            ResBlock(mid_dim, mid_dim, cond_channels=condition_dim, dropout=dropout),
            AttentionBlock(mid_dim, num_heads=num_attn_heads, context_dim=self.grad_cond_dim),
            ResBlock(mid_dim, mid_dim, cond_channels=condition_dim, dropout=dropout)
        ])

        # --- Decoder (Up-sampling path) ---
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[:-1])):
            is_last = ind >= (len(in_out)-1)
            block = nn.ModuleList([
                # The input to the ResBlock will have the skip connection concatenated
                ResBlock(dim_out*2, dim_out, cond_channels=condition_dim, dropout=dropout),
                ResBlock(dim_out, dim_out, cond_channels=condition_dim, dropout=dropout),
                AttentionBlock(dim_out, num_heads=num_attn_heads, context_dim=self.grad_cond_dim),
                nn.Linear(dim_out, dim_in) if not is_last else nn.Identity()
            ])
            self.ups.append(block)

        # --- Final Projection ---
        self.final_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )
        nn.init.zeros_(self.final_proj[-1].weight)
        nn.init.zeros_(self.final_proj[-1].bias)

    def forward(self, x: torch.Tensor, ratings: torch.Tensor=None) -> torch.Tensor:
        if x.ndim == 3: x = x.squeeze(1)
        assert x.ndim == 2 and x.shape[1] == self.model_dim

        # [KEPT] Step 1: Critical gradient calculation logic is unchanged.
        if ratings is None:
            with torch.no_grad():
                ratings = self.rating_model[0](x)
        
        with torch.enable_grad():
            xt_detached = x.clone().detach().requires_grad_(True)
            ratings_output = self.rating_model[0](xt_detached)
            # Use sum for a scalar, essential for .backward()
            ratings_output_sum = torch.sum(ratings_output)
            ratings_output_sum.backward()
            xt_grad = xt_detached.grad.detach()

            grad_norm = xt_grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # [ENHANCED] Step 2: Prepare separate conditioning embeddings
        # Rating conditioning for FiLM in ResBlocks
        c_rating = self.rating_emb(ratings) 
        # Gradient conditioning for Cross-Attention in AttentionBlocks
        # We normalize the input gradient for stability
        c_grad = self.gradient_emb(xt_grad) # / grad_norm

        # [ENHANCED] Step 3: Network Pass
        # Initial projection of centered input vector
        h = self.input_proj(x - self.w_avg)
        residuals = [h]

        # Encoder
        for i_down, (res1, res2, attn, downsample) in enumerate(self.downs):
            h = res1(h, c_rating)
            h = res2(h, c_rating)
            h = attn(h, context=c_grad)
            residuals.append(h)
            h = downsample(h)

        # Bottleneck
        h = self.middle_block[0](h, c_rating)
        h = self.middle_block[1](h, context=c_grad)
        h = self.middle_block[2](h, c_rating)

        # Decoder
        for i_up, (res1, res2, attn, upsample) in enumerate(self.ups):
            h = torch.cat((h, residuals.pop()), dim=1)
            h = res1(h, c_rating)
            h = res2(h, c_rating)
            h = attn(h, context=c_grad)
            h = upsample(h) # This is an identity on the last layer
        # Final residual connection
        h = h + residuals.pop()
        
        vector_field_pred = self.final_proj(h)

        # [KEPT] Step 4: Critical gradient modification logic is unchanged.
        if self.orthogonal_gradient_projection:
            normed_gradient = xt_grad / grad_norm
            proj_component = torch.sum(normed_gradient * vector_field_pred, dim=-1, keepdim=True)
            vector_field_pred = vector_field_pred - normed_gradient * proj_component
            
        if self.add_rating_gradient:
            vector_field_pred = (vector_field_pred + (xt_grad / grad_norm))
        
        return vector_field_pred/grad_norm