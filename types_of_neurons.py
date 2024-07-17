# -*- coding: utf-8 -*-
"""type_of_neurons.ipynb

Author: Mayank Kunwar
Find Me: https://in.linkedin.com/in/mayankkunwar93
"""

import torch
from torch import nn

input = torch.randn(4)
print(input)

# ReLU output
output_relu = nn.ReLU()(input)
print(output_relu)

# Sigmoid output
output_sigmoid = nn.Sigmoid()(input)
print(output_sigmoid)

# Tanh output
output_tanh = nn.Tanh()(input)
print(output_tanh)

# Softmax output
output_softmax = nn.Softmax()(input)
print(output_softmax)

# Leaky bReLU output
output_lrelu = nn.LeakyReLU()(input)
print(output_lrelu)

# Softplus output
output_softplus = nn.Softplus()(input)
print(output_softplus)



