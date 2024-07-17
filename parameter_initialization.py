# -*- coding: utf-8 -*-
"""parameter_initialization.py

Author: Mayank Kunwar
Find Me: https://in.linkedin.com/in/mayankkunwar93
"""

# initialization

def random_initialization(layer):
    torch.nn.init.uniform_(layer.weight, -0.1, 0.1)
    torch.nn.init.zeros_(layer.bias)

def xavier_initialization(layer):
    torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.zeros_(layer.bias)

def he_initialization(layer):
    torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    torch.nn.init.zeros_(layer.bias)

def orthogonal_initialization(layer):
    torch.nn.init.orthogonal_(layer.weight)
    torch.nn.init.zeros_(layer.bias)

import torch
from torch import nn

class MyNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(4, 3)
    self.layer_2 = nn.Linear(3, 2)

  def forward(self, x):
    x = nn.ReLU()(self.layer_1(x))
    x = self.layer_2(x)
    return x

net = MyNet()

# initialization
xavier_initialization(net.layer_1)
he_initialization(net.layer_2)

# printing initialized weights
print(net.layer_1.weight)
print(net.layer_2.weight)







