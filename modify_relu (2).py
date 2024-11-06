import torch.nn.functional as F
import torch.nn as nn
import torch

# Experimented with multiple values of tau for Modified ReLU Function
# Experimented with SoftPlus, ModifiedReLU works better

class ModifiedReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x,  = ctx.saved_tensors
        # x2 = x * x
        grad_input = grad_output.clone()
        i1 = (x < 0)
        i2 = x >= 0
        xi1 = x[i1]
        xi2 = x[i2]
        n1, n2 = xi1.numel(), xi2.numel()
        assert n1 + n2 == x.numel()

        tau = 1e-3
        if n1 > 0:
            xi12 = xi1 * xi1
            new_v = xi1 / torch.sqrt(xi12 + tau) + 1
            grad_input[i1] = grad_input[i1] * new_v
        if n2 > 0:
            xi22 = xi2 * xi2
            new_v = xi2 / torch.sqrt(xi22 + tau)
            grad_input[i2] = grad_input[i2] * new_v

        return grad_input


class ModifiedReLU(nn.Module):
    def forward(self, input):
        return ModifiedReLUFunction.apply(input)


def replace_relu_with_modifiedrelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) or isinstance(child, nn.ReLU6):
            setattr(module, name, ModifiedReLU())
        else:
            replace_relu_with_modifiedrelu(child)