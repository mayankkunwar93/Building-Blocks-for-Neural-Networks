# -*- coding: utf-8 -*-
"""optimizers.py

Author: Mayank Kunwar
Find Me: https://in.linkedin.com/in/mayankkunwar93
"""

import torch
from torch import nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, input):
        return input @ self.weight + self.bias

import torch.optim as optim

model = MyLinear(10, 2)
inputs = torch.randn(100, 10)
targets = torch.randn(100, 2)
loss_function = nn.MSELoss()

optimizer_gd = optim.SGD(model.parameters(), lr=0.001)
optimizer_momentum = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer_adagrad = optim.Adagrad(model.parameters(), lr=0.001)
optimizer_adadelta = optim.Adadelta(model.parameters(), lr=0.001)
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.001)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
optimizer_adamax = optim.Adamax(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer_adam.zero_grad()  # Clear previous gradients
    outputs = model(inputs)      # Forward pass
    loss = loss_function(outputs, targets)  # Compute loss
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    loss.backward()              # Backward pass
    optimizer_adam.step()        # Update weights







