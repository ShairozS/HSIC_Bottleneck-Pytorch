import torch
from torch import nn
from torchvision import datasets, transforms

### MODEL ###
# Construct a ChebyKAN for MNIST

import torch
import torch.nn as nn


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
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


class MNISTChebyKAN2(nn.Module):
    def __init__(self, degree = 4):
        super(MNISTChebyKAN2, self).__init__()

        #self.units = [3072, 256, 256, 256, 256, 256]
        self.units = [784, 256, 128, 128]
        self.output_layer  = nn.Linear(self.units[-1], 10)

        self.module_list = nn.ModuleList( [ChebyKANLayer(self.units[i], self.units[i+1], degree = degree) for i in range(len(self.units)-1)])
        self.f3 = nn.Dropout(p=0.2)
        self.act2 = nn.ReLU()
        
        #self.chebykan1 = ChebyKANLayer(28*28, 256, degree)
        #self.ln1 = nn.LayerNorm(256) # To avoid gradient vanishing caused by tanh
        #self.chebykan2 = ChebyKANLayer(256, 128, degree)
        #self.ln2 = nn.LayerNorm(128)
        #self.chebykan3 = ChebyKANLayer(128, 128, degree)
        #self.ln3 = nn.LayerNorm(128)
        #self.output_layer = ChebyKANLayer(128, 10)
        
        #self.module_list = nn.ModuleList( [self.chebykan1, self.chebykan2, self.chebykan3])
        
    def forward(self, data):
        x = data
        output = []
        for module in self.module_list:
            x_ = module(x.detach())
            x = module(x)
            output.append(x_)
        x = self.f3(x)
        x_ = self.act2(self.output_layer(x.detach()))
        x = self.act2(self.output_layer(x))
        output.append(x_)
        return x, output

class MNISTChebyKAN(nn.Module):
    def __init__(self, degree = 4):
        super(MNISTChebyKAN, self).__init__()
        self.chebykan1 = ChebyKANLayer(28*28, 32, degree)
        self.ln1 = nn.LayerNorm(32) # To avoid gradient vanishing caused by tanh
        self.chebykan2 = ChebyKANLayer(32, 16, degree)
        self.ln2 = nn.LayerNorm(16)
        self.chebykan3 = ChebyKANLayer(16, 10, degree)
        
        self.module_list = nn.ModuleList( [self.chebykan1, self.chebykan2, self.chebykan3])
        
    def forward(self, x):
        xorig = x
        output = []
        for module in self.module_list:
            y = module(x).detach()
            if y.shape[0]==128:
                output.append(y)
        x = x.view(-1, 28*28)  # Flatten the images
        x = self.chebykan1(x)
        x = self.ln1(x)
        x = self.chebykan2(x)
        x = self.ln2(x); 
        x = self.chebykan3(x)
        output.append(x)
        return x, output