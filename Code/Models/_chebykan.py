import torch
from torch import nn
from torchvision import datasets, transforms
import torch
import torch.nn as nn


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=4, initialization = torch.nn.init.xavier_normal_):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        #nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        initialization(self.cheby_coeffs)
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


class ChebyKAN(nn.Module):
    def __init__(self, degree = 4, layer_sizes = [784, 256, 128, 128], output_size = 10, activation = nn.ReLU(), dropout = 0.2):
        super(ChebyKAN, self).__init__()

        #self.units = [3072, 256, 256, 256, 256, 256]
        self.units = layer_sizes
        #self.output_layer  = nn.Linear(self.units[-1], output_size)
        self.output_layer = ChebyKANLayer(self.units[-1], output_size, degree = degree)

        self.module_list = nn.ModuleList( [ChebyKANLayer(self.units[i], self.units[i+1], degree = degree) for i in range(len(self.units)-1)])
        self.f3 = nn.Dropout(p=dropout)
        self.act2 = activation
        
    def forward(self, data):
        x = data
        output = []
        for module in self.module_list:
            x_ = module(x.detach())
            x = module(x)
            output.append(x_)
        #x = self.f3(x)
        x_ = self.act2(self.output_layer(x.detach()))
        x = self.act2(self.output_layer(x))
        output.append(x_)
        return x, output