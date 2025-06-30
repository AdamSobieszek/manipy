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

class VectorFieldModel(torch.nn.Module):
    """
    Neural network that transforms points in R^512 to R^512.
    Acts as a vector field for flow-based models.
    """
    
    def __init__(self, input_dim=513, hidden_dims=[1024, 1024, 1024], output_dim=512, 
                 activation=torch.nn.ReLU(), dropout_rate=0.1, center_w_avg=None):
        """
        Initialize the vector field transformer network.
        
        Args:
            input_dim: Dimension of input vectors (default: 512)
            hidden_dims: List of hidden layer dimensions (default: [1024, 1024])
            output_dim: Dimension of output vectors (default: 512)
            activation: Activation function to use (default: ReLU)
            dropout_rate: Dropout probability for regularization (default: 0.1)
        """
        super(VectorFieldModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            layers.append(torch.nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        
        self.network = torch.nn.Sequential(*layers)
        self.center_w_avg = center_w_avg
    
    def forward(self, x, rating=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 512)
            
        Returns:
            Transformed tensor of shape (batch_size, 512)
        """
        if self.center_w_avg is not None:
            x = x - self.center_w_avg.to(x.device)
        return self.network(torch.cat((x, rating), dim=1)) if rating is not None else self.network(x)
        # Concatenate rating to input\

class VectorFieldModel2(torch.nn.Module):
    """
    Neural network that transforms points in R^512 to R^512.
    Acts as a vector field for flow-based models.
    """
    
    def __init__(self, rating_model, input_dim=513, hidden_dims=[1024, 1024, 1024], output_dim=512, 
                 activation=torch.nn.ReLU(), dropout_rate=0.1, add_rating_gradient=False, center_w_avg=None):
        """
        Initialize the vector field transformer network.
        
        Args:
            input_dim: Dimension of input vectors (default: 512)
            hidden_dims: List of hidden layer dimensions (default: [1024, 1024])
            output_dim: Dimension of output vectors (default: 512)
            activation: Activation function to use (default: ReLU)
            dropout_rate: Dropout probability for regularization (default: 0.1)
        """
        super(VectorFieldModel2, self).__init__()
        self.rating_model = [rating_model]
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            layers.append(torch.nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        
        self.network = torch.nn.Sequential(*layers)
        self.add_rating_gradient = add_rating_gradient
        self.center_w_avg = center_w_avg
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 512)
            
        Returns:
            Transformed tensor of shape (batch_size, 512)
        """

        if self.center_w_avg is not None:
            x = x - self.center_w_avg.to(x.device)

        v = self.network(torch.cat((x, self.rating_model[0](x).view(-1,1)), dim=1))
        # --- Subtract Trust Model Gradient (as in original code) ---
        if self.add_rating_gradient:
            try:
                with torch.enable_grad():
                    xt_detached = (x +self.center_w_avg.to(x.device)).clone().detach().requires_grad_(True)
                    rating_output = self.rating_model[0](xt_detached, output='logit') # Get logit output

                    # Sum for scalar loss to get gradient w.r.t. xt
                    rating_output_sum = torch.sum(rating_output)
                    rating_output_sum.backward()

                    xt_grad = xt_detached.grad.detach() # Gradient of rating score w.r.t. xt

                #norm_raw = torch.linalg.norm(v, dim=-1, keepdim=True)
                ## Only normalize if norm is greater than 1
                #v = v/norm_raw**0.5/torch.clamp(norm_raw**0.5, min=1)
                ## Add gradient to the predicted vector field
                # xt_grad = xt_grad / torch.linalg.norm(xt_grad, dim=-1, keepdim=True) * (1-torch.clamp(norm_raw**2, max=0.99))**0.5
                v = v + xt_grad/torch.linalg.norm(xt_grad, dim=-1, keepdim=True)**2

                return v

            except Exception as e:
                print(f"Warning: Failed to subtract rating gradient: {e}. Returning raw prediction.")
                # Fallback: return the raw prediction, maybe normalized
                #norm_raw = torch.linalg.norm(v, dim=-1, keepdim=True)
                return v #/ torch.clamp(norm_raw, min=1e-9)
        else:
             return v 
        return 
        # Concatenate rating to input\


class VectorFieldTransformer(nn.Module):
    """ Transformer model to predict the vector field v(xt, rating_condition). """
    def __init__(
        self,
        rating_model,
        dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        dim_head: int = 64,
        num_registers: int = 32,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_rotary: bool = True,
        use_flash_attention: bool = True,
        add_rating_gradient: bool = False,
        condition_dim: int = 512, # Dimension of the rating embedding,
        project: bool = False,
        normalize: bool = True,
        sigmoid_on_rating: bool = False,
        vector_field_dim: int = None,
    ):
        super().__init__()

        # Set matmul precision
        torch.set_float32_matmul_precision('high')

        if vector_field_dim is None:
            self.vector_field_dim = dim
        else:
            self.vector_field_dim = vector_field_dim
        self.dim = dim
        self.depth = depth
        self.condition_dim = condition_dim
        self.add_rating_gradient =add_rating_gradient
        self.rating_model = [rating_model]
        self.project = project
        self.sigmoid_on_rating = sigmoid_on_rating
        # Input projections
        self.proj_in = nn.Linear(self.vector_field_dim, self.dim)

        self.rating_proj = nn.Sequential(
            TimestepEmbedding(self.condition_dim),
            nn.Linear(self.condition_dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, self.dim)
        )


        # Registers
        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.zeros(num_registers, self.dim))
        nn.init.normal_(self.registers, std=0.02)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(dim_head) if use_rotary else None

        # Transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RMSNorm(dim),
                Attention(
                    dim=dim,
                    heads=num_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    flash=use_flash_attention,
                    gate_value_heads=True,      # Enable gating of value heads
                    softclamp_logits=False,
                    zero_init_output=True       # Better training stability
                ),
                RMSNorm(dim),
                FeedForward(
                    dim=dim,
                    mult=mlp_ratio,
                    dropout=dropout,
                    glu=True
                )
            ]))

        self.final_norm = RMSNorm(dim)
        self.to_vector_field = nn.Linear(dim, self.vector_field_dim)
        self.normalize = normalize
        self.w_avg = get_w_avg().detach()

    def to_bf16(self):
        """Converts all parameters to bfloat16"""
        self.bfloat16()
        return self

    def forward(
        self,
        x: torch.Tensor,
        ratings: torch.Tensor = None
    ) -> torch.Tensor:
        batch = x.shape[0]

        if ratings is None:
            ratings = self.rating_model[0](x, output='logit')


        # Project input and rating
        x2 = x.clone().detach().requires_grad_()
        x = x - self.w_avg.to(x.device)
        h = self.proj_in(x)
        if self.sigmoid_on_rating:
            ratings = torch.sigmoid(ratings)
        rating_emb = self.rating_proj(ratings)

        # Combine input features with rating embedding
        h = h + rating_emb

        # Add registers for global context
        registers = repeat(self.registers, 'r d -> b r d', b=batch)
        h, ps = pack([registers, h], 'b * d')

        # Get rotary embeddings if used
        rotary_pos_emb = None
        if self.rotary_emb is not None:
            rotary_pos_emb = self.rotary_emb.forward_from_seq_len(h.shape[1])

        # Process through transformer layers
        for norm1, attn, norm2, ff in self.layers:
            # Pre-norm attention
            attn_in = norm1(h)
            h_attn = attn(attn_in, rotary_pos_emb=rotary_pos_emb)
            h = h + h_attn

            # Pre-norm feedforward
            ff_in = norm2(h)
            h_ff = ff(ff_in)
            h = h + h_ff

        # Unpack registers and get main output
        _, h = unpack(h, ps, 'b * d')

        # Final normalization and projection
        h = self.final_norm(h)
        vector_field = self.to_vector_field(h)
        if self.add_rating_gradient:
            with torch.enable_grad():
                r = self.rating_model[0](x2, output='mean')
                torch.sum(torch.logit(r)).backward()
                x_grad = x2.grad.detach()
            
            vector_field = (vector_field - x_grad)


        if self.project:
            with torch.no_grad():
                normed_x_grad = x_grad / (x_grad.norm(dim=1, keepdim=True) + 1e-8)

            dot_products = torch.bmm(vector_field.unsqueeze(1), normed_x_grad.unsqueeze(2)).squeeze(-1)
            projection = dot_products * normed_x_grad
            vector_field = vector_field - projection

        if self.normalize:
            return vector_field/vector_field.norm(dim=-1,keepdim=True).clamp(min=1e-9)
        else:
            return vector_field


class VectorFieldTransformer3(VectorFieldTransformer):
    def __init__(self,
        rating_model,
        dim: int = config.FLOW_MODEL_DIM,
        depth: int = config.FLOW_MODEL_DEPTH,
        num_heads: int = config.FLOW_MODEL_NUM_HEADS,
        dim_head: int = config.FLOW_MODEL_DIM_HEAD,
        num_registers: int = config.FLOW_MODEL_NUM_REGISTERS,
        mlp_ratio: int = 4, # Standard ratio
        dropout: float = config.FLOW_MODEL_DROPOUT,
        use_rotary: bool = True, # Use rotary embeddings
        use_flash_attention: bool = True, # Use flash attention if available
        condition_dim: int = config.FLOW_MODEL_CONDITION_DIM, # Dimension of the rating embedding,
        add_rating_gradient: bool = True,
        normalize: bool = False,
        sigmoid_on_rating: bool = False,
        vector_field_dim: int = None,
    ):
        super().__init__(
            rating_model=rating_model,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            dim_head=dim_head,
            num_registers=num_registers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_rotary=use_rotary,
            use_flash_attention=use_flash_attention,
            condition_dim=condition_dim,
            add_rating_gradient=False,
            normalize=normalize,
            sigmoid_on_rating=sigmoid_on_rating,
            vector_field_dim=vector_field_dim
        )
        self.add_rating_gradient = add_rating_gradient
        self.normalize = normalize
    
    def forward(self, xt, rating_cond=None):
        if self.add_rating_gradient:
                with torch.enable_grad():
                    xt_detached = (xt).clone().detach().requires_grad_(True)
                    rating_output = self.rating_model[0](xt_detached, output='logit') # Get logit output

                    # Sum for scalar loss to get gradient w.r.t. xt
                    rating_output_sum = torch.sum(rating_output)
                    rating_output_sum.backward()

                    xt_grad = xt_detached.grad.detach() # Gradient of rating score w.r.t. xt

        vector_field_pred = super().forward(xt, rating_cond)
        if self.add_rating_gradient:
            xt_grad = xt_grad / xt_grad.norm(dim=-1, keepdim=True)**2
            vector_field_final = vector_field_pred + xt_grad

            return vector_field_final
        else:
             return vector_field_pred 


class VectorFieldTransformer2(VectorFieldTransformer):
    def __init__(self,
        rating_model,
        dim: int = config.FLOW_MODEL_DIM,
        depth: int = config.FLOW_MODEL_DEPTH,
        num_heads: int = config.FLOW_MODEL_NUM_HEADS,
        dim_head: int = config.FLOW_MODEL_DIM_HEAD,
        num_registers: int = config.FLOW_MODEL_NUM_REGISTERS,
        mlp_ratio: int = 4, # Standard ratio
        dropout: float = config.FLOW_MODEL_DROPOUT,
        use_rotary: bool = False, # Use rotary embeddings
        use_flash_attention: bool = True, # Use flash attention if available
        condition_dim: int = config.FLOW_MODEL_CONDITION_DIM, # Dimension of the rating embedding,
        add_rating_gradient: bool = True # Flag to control this behavior
    ):
        super().__init__(
            rating_model,
            dim,
            depth,
            num_heads,
            dim_head,
            num_registers,
            mlp_ratio,
            dropout,
            use_rotary,
            use_flash_attention,
            condition_dim,
            add_rating_gradient
        )

    def forward(self, xt, rating_cond=None):
        vector_field_pred = super().forward(xt, rating_cond)
        # --- Subtract Trust Model Gradient (as in original code) ---
        if self.add_rating_gradient:
            try:
                with torch.enable_grad():
                    xt_detached = xt.clone().detach().requires_grad_(True)
                    rating_output = self.rating_model[0](xt_detached) # Get logit output

                    # Sum for scalar loss to get gradient w.r.t. xt
                    rating_output_sum = torch.sum(rating_output)
                    rating_output_sum.backward()

                    xt_grad = xt_detached.grad.detach() # Gradient of rating score w.r.t. xt

                # if not self.training:
                #     vector_field_pred = vector_field_pred / vector_field_pred.norm(dim=-1).median()*0.3
                norm_raw = torch.linalg.norm(vector_field_pred, dim=-1, keepdim=True)
                # Only normalize if norm is greater than 1
                vector_field_pred = vector_field_pred/torch.clamp(norm_raw, min=1)
                # Add gradient to the predicted vector field
                xt_grad = xt_grad / torch.linalg.norm(xt_grad, dim=-1, keepdim=True) * (1-torch.clamp(norm_raw**2, max=0.99))**0.5
                vector_field_final = vector_field_pred + xt_grad

                # Normalize the final vector field
                norm = torch.linalg.norm(vector_field_final, dim=-1, keepdim=True)
                vector_field_normalized = vector_field_final / torch.clamp(norm, min=1e-9) # Avoid division by zero

                return vector_field_normalized

            except Exception as e:
                print(f"Warning: Failed to subtract rating gradient: {e}. Returning raw prediction.")
                # Fallback: return the raw prediction, maybe normalized
                norm_raw = torch.linalg.norm(vector_field_pred, dim=-1, keepdim=True)
                return vector_field_pred / torch.clamp(norm_raw, min=1e-9)
        else:
             # If not subtracting gradient, just return the prediction (optionally normalized)
            #  norm_raw = torch.linalg.norm(vector_field_pred, dim=-1, keepdim=True)
            #  norm_raw = torch.clamp(norm_raw, min=1e-9)
             return vector_field_pred #/ norm_raw


class RatingODE(nn.Module):
    """ Wraps the flow model and trust model for ODE integration during inference. """
    def __init__(self, flow_model, rating_model=None, kwargs = {"output":"logit"}, reverse=False):
        super().__init__()
        self.flow = flow_model
        self.flow.eval() # Ensure flow model is in eval mode
        self.rating = rating_model
        if self.rating is not None:
            self.rating.eval() # Ensure trust model is in eval mode
        self.kwargs = kwargs
        self.reverse = reverse

    @torch.no_grad() # ODE solver step should not compute gradients normally
    def forward(self, t, x): # torchdyn/torchdiffeq expect forward(t, x) signature
        """
        Predicts the vector field dx/dt = v(x, rating(x)) at time t.
        Args:
            t (torch.Tensor): Current time (scalar or batch). Not directly used by this model but part of ODE signature.
            x (torch.Tensor): Current state (latent vectors) (batch_size, dim).
        Returns:
            torch.Tensor: Predicted vector field dx/dt (batch_size, dim).
        """
        # Get current rating/logit using the trust model
        # Assuming trust model predicts logit directly or via "logit" mode
        # Ensure input x matches what trust model expects (e.g., W space)
        if self.rating is not None:
            rating_condition = self.rating(x, **self.kwargs) # Shape (batch, 1)
            # Get vector field prediction from the flow model using current state x and rating
            # Assuming flow model's forward is flow(xt, rating_cond)
            vector_field = self.flow(x, rating_condition)
        else:
            vector_field = self.flow(x)

        return vector_field*(-1 if self.reverse else 1)

class AdaptiveRatingODE(nn.Module):
    """ Wraps the flow model and trust model for ODE integration during inference. """
    def __init__(self, flow_model, delta_y=1):
        super().__init__()
        self.flow = flow_model
        self.rating = flow_model.rating_model[0]
        self.flow.eval() # Ensure flow model is in eval mode
        self.rating.eval() # Ensure trust model is in eval mode
        self.initial_rating = None 
        self.delta_y = delta_y

    @torch.no_grad() # ODE solver step should not compute gradients normally
    def forward(self, t, x): # torchdyn/torchdiffeq expect forward(t, x) signature
        """
        Predicts the vector field dx/dt = v(x, rating(x)) at time t when t in [0,1].
        Args:
            t (torch.Tensor): Current time (scalar or batch). Not directly used by this model but part of ODE signature.
            x (torch.Tensor): Current state (latent vectors) (batch_size, dim).
        Returns:
            torch.Tensor: Predicted vector field dx/dt (batch_size, dim).
        """
        vector_field = self.flow(x)
        if t==0:
            rating_value = self.rating(x) # Shape (batch, 1)
            self.initial_rating = rating_value

        vector_field = vector_field*self.delta_y


        # Get vector field prediction from the flow model using current state x and rating
        # Assuming flow model's forward is flow(xt, rating_cond)
        return vector_field


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
    """ A ResNet block with conditioning, using Linear layers for non-spatial data. """
    def __init__(self, dim_in: int, dim_out: int, *, cond_dim: int):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.SiLU(),
            nn.Linear(dim_in, dim_out)
        )
        
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim_out)
        )

        self.block2 = nn.Sequential(
            nn.LayerNorm(dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
        
        self.res_conn = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x) + self.cond_proj(cond_emb)
        h = self.block2(h)
        return h + self.res_conn(x)

# --- Main U-Net Model ---

class VectorFieldUNet(nn.Module):
    """
    A U-Net architecture to predict a vector field v(xt, rating_condition) in a 512-dim latent space.
    It replaces standard convolutions with Linear layers to operate on non-spatial vector data,
    making it suitable for isotropic latent spaces like StyleGAN's W-space.
    """
    def __init__(
        self,
        model_dim: int = 512,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        condition_dim: int = 128
    ):
        """
        Initializes the VectorFieldUNet.
        Args:
            model_dim (int): The dimension of the input and output latent vectors (e.g., 512 for StyleGAN W-space).
            dim_mults (Sequence[int]): Divisors for feature dimensions at each level of the U-Net.
                                       For example, (1, 2, 4, 8) with model_dim=512 will create levels with
                                       dimensions [512, 256, 128, 64].
            condition_dim (int): The dimension of the embedded rating condition vector.
        """
        super().__init__()
        self.model_dim = model_dim
        
        # --- Rating Embedding ---
        # Projects the scalar rating into a high-dimensional conditioning vector.
        self.rating_emb = nn.Sequential(
            SinusoidalPosEmb(condition_dim),
            nn.Linear(condition_dim, condition_dim * 4),
            nn.GELU(),
            nn.Linear(condition_dim * 4, condition_dim)
        )
        
        # --- U-Net Architecture ---
        # Determine dimensions for each level of the U-Net
        hidden_dims = [model_dim] + [model_dim // m for m in dim_mults]
        
        # --- Encoder ---
        # Progressively reduces the feature dimension.
        self.downs = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            dim_in = hidden_dims[i]
            dim_out = hidden_dims[i+1]
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, cond_dim=condition_dim),
                ResnetBlock(dim_in, dim_in, cond_dim=condition_dim),
                nn.Linear(dim_in, dim_out) # Downsampling layer
            ]))

        # --- Bottleneck ---
        # The central part of the U-Net with the lowest feature dimension.
        bottleneck_dim = hidden_dims[-1]
        self.bottleneck = nn.ModuleList([
            ResnetBlock(bottleneck_dim, bottleneck_dim, cond_dim=condition_dim),
            ResnetBlock(bottleneck_dim, bottleneck_dim, cond_dim=condition_dim)
        ])

        # --- Decoder ---
        # Progressively increases the feature dimension, incorporating encoder features via skip connections.
        self.ups = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            dim_in_upsample = hidden_dims[i]
            dim_out_upsample = hidden_dims[i-1]
            # Input to the ResNet block will be the upsampled vector concatenated with the residual from the encoder.
            resnet_in_dim = dim_out_upsample * 2
            self.ups.append(nn.ModuleList([
                nn.Linear(dim_in_upsample, dim_out_upsample), # Upsampling layer
                ResnetBlock(resnet_in_dim, dim_out_upsample, cond_dim=condition_dim),
                ResnetBlock(resnet_in_dim, dim_out_upsample, cond_dim=condition_dim),
            ]))
            
        # --- Final Projection ---
        # Final block to process the full-resolution features and project to the output vector field.
        # It takes the concatenated output from the last decoder stage and the original input vector.
        self.final_proj = nn.Sequential(
            ResnetBlock(model_dim * 2, model_dim, cond_dim=condition_dim),
            nn.Linear(model_dim, model_dim)
        )
        
        # Initialize the final projection layer to output zeros at the start of training.
        # This is a common practice for stability, ensuring the model starts by predicting no change.
        nn.init.zeros_(self.final_proj[-1].weight)
        nn.init.zeros_(self.final_proj[-1].bias)

    def forward(self, x: torch.Tensor, ratings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the VectorFieldUNet.
        Args:
            x (torch.Tensor): Input latent vectors `xt`, shape (B, 512).
            ratings (torch.Tensor): Scalar ratings `f(xt)`, shape (B,) or (B, 1).
        Returns:
            torch.Tensor: Predicted tangential vector field `V_tangential`, shape (B, 512).
        """
        # Ensure input x is of the correct shape (B, model_dim)
        if x.ndim == 3: x = x.squeeze(1)
        assert x.ndim == 2 and x.shape[1] == self.model_dim, f"Input tensor x must have shape (B, {self.model_dim})"
        
        x_orig = x # Keep original for final global skip connection

        # 1. Get rating conditioning vector
        cond_emb = self.rating_emb(ratings)
        
        # 2. Encoder Path
        residuals = []
        h = x
        for block1, block2, downsample in self.downs:
            h = block1(h, cond_emb)
            residuals.append(h)
            
            h = block2(h, cond_emb)
            residuals.append(h)
            
            h = downsample(h)

        # 3. Bottleneck Path
        h = self.bottleneck[0](h, cond_emb)
        h = self.bottleneck[1](h, cond_emb)

        # 4. Decoder Path
        for upsample, block1, block2 in self.ups:
            h = upsample(h)
            
            # Concatenate with residual from the corresponding encoder level (skip connection)
            h = torch.cat([h, residuals.pop()], dim=-1)
            h = block1(h, cond_emb)
            
            # Concatenate with the second residual from the corresponding encoder level
            h = torch.cat([h, residuals.pop()], dim=-1)
            h = block2(h, cond_emb)

        # 5. Final Projection
        # Concatenate with the original input vector for a final global residual connection.
        # This helps the model learn deviations from the identity.
        h = torch.cat([h, x_orig], dim=-1)
        
        return self.final_proj(h)

