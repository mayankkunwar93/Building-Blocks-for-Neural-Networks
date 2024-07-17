# -*- coding: utf-8 -*-
"""loss_functions.py

Author: Mayank Kunwar
Find Me: https://in.linkedin.com/in/mayankkunwar93
"""

import torch

output = torch.tensor([0.5, 0.7])
target = torch.tensor([1.0, 0.0])

from torch import nn

# Mean Squared Error
mse_loss = nn.MSELoss()
loss = mse_loss(output, target)
print('MSE Loss:', loss.item())

# Mean Absolute Error
mae_loss = nn.L1Loss()
loss = mae_loss(output, target)
print('MAE Loss:', loss.item())

# Binary Cross-Entropy Loss
bce_loss = nn.BCELoss()
loss = bce_loss(output, target)
print('BCE Loss:', loss.item())

# Categorical Cross-Entropy Loss
ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(output, target)
print('CE Loss:', loss.item())
def hinge_loss(output, target):
   return torch.mean(torch.clamp(1 - output * target, min=0))

# Hinge loss
loss = hinge_loss(output, target)
print('Hinge Loss:', loss.item())

# Kullback-Leibler Divergence (KL Divergence)
kl_loss = nn.KLDivLoss(reduction='batchmean')
loss = kl_loss(output, target)
print('KL Divergence Loss:', loss.item())







