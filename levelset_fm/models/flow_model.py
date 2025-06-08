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
                 activation=torch.nn.ReLU(), dropout_rate=0.1):
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
    
    def forward(self, x, rating):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 512)
            
        Returns:
            Transformed tensor of shape (batch_size, 512)
        """
        return self.network(torch.cat((x, rating), dim=1))
        # Concatenate rating to input\

class VectorFieldModel2(torch.nn.Module):
    """
    Neural network that transforms points in R^512 to R^512.
    Acts as a vector field for flow-based models.
    """
    
    def __init__(self, rating_model, input_dim=513, hidden_dims=[1024, 1024, 1024], output_dim=512, 
                 activation=torch.nn.ReLU(), dropout_rate=0.1, add_rating_gradient=False):
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
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 512)
            
        Returns:
            Transformed tensor of shape (batch_size, 512)
        """
        v = self.network(torch.cat((x, self.rating_model[0](x).view(-1,1)), dim=1))
        # --- Subtract Trust Model Gradient (as in original code) ---
        if self.add_rating_gradient:
            try:
                with torch.enable_grad():
                    xt_detached = x.clone().detach().requires_grad_(True)
                    rating_output = self.rating_model[0](xt_detached) # Get logit output

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
        super().__init__()

        # Try setting matmul precision if needed and supported
        try: torch.set_float32_matmul_precision('high')
        except: pass # Ignore if not supported
        # Store rating model without registering as parameter
        rating_model.eval()
        self.rating_model = [rating_model]
        self.dim = dim
        self.depth = depth
        self.w_avg = get_w_avg().detach() # Load w_avg

        # Input projection for xt
        self.proj_in = nn.Linear(dim, dim) # W latent dim to model dim

        # Projection/Embedding for the rating condition
        # Using TimestepEmbedding adapted for scalar rating/logit input
        self.rating_proj = nn.Sequential(
            TimestepEmbedding(condition_dim),
            nn.Linear(condition_dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

        # Learnable registers
        self.num_registers = num_registers
        if num_registers > 0:
            self.registers = nn.Parameter(torch.zeros(num_registers, dim))
            nn.init.normal_(self.registers, std=0.02) # Initialize registers
        else:
            self.registers = None

        # Rotary embeddings for position encoding (optional)
        self.rotary_emb = RotaryEmbedding(dim=dim_head) if use_rotary else None

        # Transformer Layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RMSNorm(dim), # Pre-norm attention
                Attention(
                    dim=dim,
                    heads=num_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    flash=use_flash_attention,
                    # Additional options from original code:
                    # gate_value_heads=True, # Gating can sometimes help
                    # softclamp_logits=False, # Typically false
                    # zero_init_output=True # Can improve stability
                ),
                RMSNorm(dim), # Pre-norm feedforward
                FeedForward(
                    dim=dim,
                    mult=mlp_ratio,
                    dropout=dropout,
                    glu=True # Use GLU activation
                )
            ]))

        # Final output layers
        self.final_norm = RMSNorm(dim)
        self.to_vector_field = nn.Linear(dim, dim) # Output matches input W dimension

        # --- Part from Original Code: Subtracting Trust Model Gradient ---
        # This requires the trust model during forward pass.
        # It makes the flow model dependent on the specific trust model structure.
        # Consider if this is desired, or if the target vector 'ut' should already account for this.
        # For now, implementing as in the original 'forward'.
        self.add_rating_gradient = add_rating_gradient # Flag to control this behavior
        print(f"Flow model initialized. Subtract trust gradient: {self.add_rating_gradient}")


    def forward(
        self,
        xt: torch.Tensor,      # Latent vector at time t (batch, 512)
        rating_cond: torch.Tensor = None # Rating condition (batch, 1) or (batch,)
    ) -> torch.Tensor:
        """ Forward pass of the flow model. """
        batch_size = xt.shape[0]
        device = xt.device

        if rating_cond is None:
            rating_cond = self.rating_model[0](xt)

        # --- Input Processing ---
        # Center xt around w_avg
        xt_centered = xt #- self.w_avg.to(device)
        # Project xt features
        h = self.proj_in(xt_centered)

        # Embed rating condition
        # Ensure rating_cond has shape (batch,) or (batch, 1) before embedding
        if rating_cond.ndim > 1 and rating_cond.shape[1] > 1:
             rating_cond = rating_cond[:, 0] # Take first element if more than one dim provided

        rating_emb = self.rating_proj(rating_cond) # Shape (batch, dim)

        # Combine input features with rating embedding (simple addition)
        h = h + rating_emb

        # Add registers if used
        tokens = [h]
        if self.registers is not None:
            registers = repeat(self.registers, 'r d -> b r d', b=batch_size)
            tokens.insert(0, registers) # Prepend registers

        # Pack tokens for transformer input
        h_packed, ps = pack(tokens, 'b * d') # Shape (batch, num_tokens, dim)
        num_tokens = h_packed.shape[1]

        # --- Rotary Embeddings (Optional) ---
        rotary_pos_emb = None
        if self.rotary_emb is not None:
            # Generate rotary embeddings based on sequence length
            rotary_pos_emb = self.rotary_emb.forward_from_seq_len(seq_len=num_tokens)

        # --- Transformer Layers ---
        for norm1, attn, norm2, ff in self.layers:
            # Attention block
            h_res = h_packed
            h_norm1 = norm1(h_packed)
            attn_out = attn(h_norm1, rotary_pos_emb=rotary_pos_emb)
            h_packed = h_res + attn_out # Residual connection

            # FeedForward block
            h_res = h_packed
            h_norm2 = norm2(h_packed)
            ff_out = ff(h_norm2)
            h_packed = h_res + ff_out # Residual connection

        # Unpack tokens, discard registers
        unpacked_tokens = unpack(h_packed, ps, 'b * d')
        h_out = unpacked_tokens[-1] # Get the output corresponding to the original xt input

        # --- Final Output ---
        h_final = self.final_norm(h_out)
        vector_field_pred = self.to_vector_field(h_final) # Predicted raw vector field
        return vector_field_pred


        # # --- Subtract Trust Model Gradient (as in original code) ---
        # if self.add_rating_gradient:
        #     try:
        #         with torch.enable_grad():
        #             xt_detached = xt.clone().detach().requires_grad_(True)
        #             rating_output = self.rating_model[0](xt_detached) # Get logit output

        #             # Sum for scalar loss to get gradient w.r.t. xt
        #             rating_output_sum = torch.sum(rating_output)
        #             rating_output_sum.backward()

        #             xt_grad = xt_detached.grad.detach() # Gradient of rating score w.r.t. xt

        #         # if not self.training:
        #         #     vector_field_pred = vector_field_pred / vector_field_pred.norm(dim=-1).median()*0.3
        #         norm_raw = torch.linalg.norm(vector_field_pred, dim=-1, keepdim=True)
        #         # Only normalize if norm is greater than 1
        #         vector_field_pred = vector_field_pred/torch.clamp(norm_raw, min=1)
        #         # Add gradient to the predicted vector field
        #         xt_grad = xt_grad / torch.linalg.norm(xt_grad, dim=-1, keepdim=True) * (1-torch.clamp(norm_raw**2, max=0.99))**0.5
        #         vector_field_final = vector_field_pred + xt_grad

        #         # Normalize the final vector field
        #         norm = torch.linalg.norm(vector_field_final, dim=-1, keepdim=True)
        #         vector_field_normalized = vector_field_final / torch.clamp(norm, min=1e-9) # Avoid division by zero

        #         return vector_field_normalized

        #     except Exception as e:
        #         print(f"Warning: Failed to subtract rating gradient: {e}. Returning raw prediction.")
        #         # Fallback: return the raw prediction, maybe normalized
        #         norm_raw = torch.linalg.norm(vector_field_pred, dim=-1, keepdim=True)
        #         return vector_field_pred / torch.clamp(norm_raw, min=1e-9)
        # else:
        #      # If not subtracting gradient, just return the prediction (optionally normalized)
        #     #  norm_raw = torch.linalg.norm(vector_field_pred, dim=-1, keepdim=True)
        #     #  norm_raw = torch.clamp(norm_raw, min=1e-9)
        #      return vector_field_pred #/ norm_raw

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
        project: bool = False
    ):
        super().__init__()

        # Set matmul precision
        torch.set_float32_matmul_precision('high')

        self.dim = dim
        self.depth = depth
        self.add_rating_gradient =add_rating_gradient
        self.rating_model = [rating_model]
        self.project = project

        # Input projections
        self.proj_in = nn.Linear(dim, dim)

        self.rating_proj = nn.Sequential(
            TimestepEmbedding(dim),
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )


        # Registers
        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.zeros(num_registers, dim))
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
        self.to_vector_field = nn.Linear(dim, dim)

        self.w_avg = get_w_avg().detach()

    def to_bf16(self):
        """Converts all parameters to bfloat16"""
        self.bfloat16()
        return self

    def forward(
        self,
        x: torch.Tensor,
        ratings: torch.Tensor
    ) -> torch.Tensor:
        batch = x.shape[0]

        # Project input and rating
        x2 = x.clone().detach().requires_grad_()
        x = x - self.w_avg.to(x.device)
        h = self.proj_in(x)
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
        if not self.add_rating_gradient:
            return vector_field/vector_field.norm(dim=-1,keepdim=True)

        with torch.enable_grad():
            r = self.rating_model[0](x2,'mean')
            torch.sum(torch.logit(r)).backward()
            x_grad = x2.grad.detach()

        if self.project:
            with torch.no_grad():
                normed_x_grad = x_grad / (x_grad.norm(dim=1, keepdim=True) + 1e-8)

            dot_products = torch.bmm(vector_field.unsqueeze(1), normed_x_grad.unsqueeze(2)).squeeze(-1)
            projection = dot_products * normed_x_grad
            vector_field = vector_field - projection
            vector_field = (vector_field - x_grad)
            return vector_field/vector_field.norm(dim=-1,keepdim=True)


        vector_field = (vector_field - x_grad)
        return vector_field/vector_field.norm(dim=-1,keepdim=True)

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
    def __init__(self, flow_model, rating_model, kwargs = {"output":"logit"}):
        super().__init__()
        self.flow = flow_model
        self.rating = rating_model
        self.flow.eval() # Ensure flow model is in eval mode
        self.rating.eval() # Ensure trust model is in eval mode
        self.kwargs = kwargs
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
        rating_condition = self.rating(x, **self.kwargs) # Shape (batch, 1)

        # Get vector field prediction from the flow model using current state x and rating
        # Assuming flow model's forward is flow(xt, rating_cond)
        vector_field = self.flow(x, rating_condition)
        return vector_field

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
