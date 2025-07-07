import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define activation functions and their default parameters
activation_funcs = {
    'linear': {
        'func': lambda x, alpha: x,
        'def_alpha': 0,
        'def_gain': 1,
    },
    'relu': {
        'func': F.relu,
        'def_alpha': 0,
        'def_gain': np.sqrt(2),
    },
    'lrelu': {
        'func': lambda x, alpha: F.leaky_relu(x, negative_slope=alpha),
        'def_alpha': 0.2,
        'def_gain': np.sqrt(2),
    },
    'tanh': {
        'func': torch.tanh,
        'def_alpha': 0,
        'def_gain': 1,
    },
    'sigmoid': {
        'func': torch.sigmoid,
        'def_alpha': 0,
        'def_gain': 1,
    },
    'elu': {
        'func': F.elu,
        'def_alpha': 0,
        'def_gain': 1,
    },
    'selu': {
        'func': F.selu,
        'def_alpha': 0,
        'def_gain': 1,
    },
    'softplus': {
        'func': F.softplus,
        'def_alpha': 0,
        'def_gain': 1,
    },
    'swish': {
        'func': lambda x, alpha: x * torch.sigmoid(x),
        'def_alpha': 0,
        'def_gain': np.sqrt(2),
    },
}

def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None):
    """
    Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. It supports
    first and second order gradients.

    Args:
        x (torch.Tensor): Input activation tensor. Can be of any shape.
        b (torch.Tensor, optional): Bias vector, or `None` to disable. Must be a
            1D tensor of the same type as `x`. The shape must be known, and it
            must match the dimension of `x` corresponding to `dim`.
        dim (int, optional): The dimension in `x` corresponding to the elements
            of `b`. Ignored if `b` is not specified. Default is 1.
        act (str, optional): Name of the activation function to evaluate, or
            `"linear"` to disable. Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`,
            `"sigmoid"`, `"swish"`, etc. See `activation_funcs` for a full list.
            `None` is not allowed. Default is `'linear'`.
        alpha (float, optional): Shape parameter for the activation function, or
            `None` to use the default. Default is `None`.
        gain (float, optional): Scaling factor for the output tensor, or `None`
            to use default. See `activation_funcs` for the default scaling of
            each activation function. If unsure, consider specifying 1. Default is `None`.
        clamp (float, optional): Clamp the output values to `[-clamp, +clamp]`, or
            `None` to disable the clamping (default). Default is `None`.

    Returns:
        torch.Tensor: Tensor of the same shape and datatype as `x`.
    """
    if act is None:
        raise ValueError("`act` cannot be None. Use 'linear' to disable activation.")
    if act not in activation_funcs:
        raise ValueError(f"Unsupported activation function '{act}'. "
                         f"Supported functions are: {list(activation_funcs.keys())}")
    
    # Retrieve activation function and default parameters
    act_spec = activation_funcs[act]
    act_func = act_spec['func']
    alpha = float(alpha) if alpha is not None else act_spec['def_alpha']
    gain = float(gain) if gain is not None else act_spec['def_gain']

    # Add bias if provided
    if b is not None:
        if not isinstance(b, torch.Tensor):
            raise TypeError(f"Bias `b` must be a torch.Tensor, got {type(b)}")
        if b.ndim != 1:
            raise ValueError(f"Bias `b` must be a 1D tensor, got shape {b.shape}")
        if not (0 <= dim < x.ndim):
            raise ValueError(f"Dimension `dim`={dim} is out of range for input tensor with {x.ndim} dimensions")
        if b.shape[0] != x.shape[dim]:
            raise ValueError(f"Bias `b` has shape {b.shape}, which does not match the size of dimension {dim} in `x` ({x.shape[dim]})")
        
        # Reshape bias for broadcasting
        reshape_dims = [1] * x.ndim
        reshape_dims[dim] = -1
        b_reshaped = b.view(reshape_dims)
        x = x + b_reshaped

    # Apply activation function
    if act != 'linear':
        x = act_func(x, alpha=alpha)

    # Apply gain
    if gain != 1:
        x = x * gain

    # Apply clamping
    if clamp is not None:
        if clamp < 0:
            raise ValueError(f"Clamp value must be non-negative, got {clamp}")
        x = x.clamp(-clamp, clamp)

    return x

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()
    
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mapping = MappingNetwork(512,0,512,18).to(device)
    mapping.load_state_dict(torch.load("map (1).pt", weights_only=True, map_location=device))
    class gen:
        mapping = mapping

    G = gen()
    print(G.mapping.w_avg.shape)