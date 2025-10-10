import os

import torch
from torch import nn
from torch.autograd import Function
# from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
# fused = load(
#     'fused',
#     sources=[
#         # os.path.join(module_path, 'fused_bias_act.cpp'),
#         # os.path.join(module_path, 'fused_bias_act_kernel.cu'),
#     ],
# )


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        # grad_input = grad_output * dLeakyReLU(out)
        grad_input = grad_output.clone()
        grad_input[out > 0] *= scale
        grad_input[out <= 0] *= scale * negative_slope

        dim = [0]
        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))
        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        # Second derivative for leaky relu is zero almost everywhere, so just propagate gradgrad_input
        gradgrad_out = gradgrad_input.clone()
        gradgrad_out[out > 0] *= ctx.scale
        gradgrad_out[out <= 0] *= ctx.scale * ctx.negative_slope

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale
        )

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
