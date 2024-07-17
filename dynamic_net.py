# -*- coding: utf-8 -*-
"""type_of_neurons.py

Author: Mayank Kunwar
Find Me: https://in.linkedin.com/in/mayankkunwar93
"""

# linear module

import torch
from torch import nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, input):
        return input @ self.weight + self.bias

class DynamicNet(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.linears = nn.ModuleList(
            [MyLinear(4, 4) for _ in range(num_layers)]
            )
        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'lrelu': nn.LeakyReLU()
        })
        self.final = MyLinear(4, 1)

    def forward(self, x, act):
        for linear in self.linears:
            x = linear(x)
        x = self.activations[act](x)
        x = self.final(x)
        return x

dynamic_net = DynamicNet(3)
sample_input = torch.randn(4)
output = dynamic_net(sample_input, 'relu')
output







