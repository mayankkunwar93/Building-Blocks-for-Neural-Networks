# -*- coding: utf-8 -*-
"""loss_functions.py

Author: Mayank Kunwar
Find Me: https://in.linkedin.com/in/mayankkunwar93
"""

import torch

def mean_squared_error(output, target):
    """
    Compute Mean Squared Error (MSE) loss between output and target.

    Args:
    - output (torch.Tensor): Predicted output tensor.
    - target (torch.Tensor): Target tensor.

    Returns:
    - torch.Tensor: Computed MSE loss.
    """
    return torch.mean((output - target)**2)

def mean_absolute_error(output, target):
    """
    Compute Mean Absolute Error (MAE) loss between output and target.

    Args:
    - output (torch.Tensor): Predicted output tensor.
    - target (torch.Tensor): Target tensor.

    Returns:
    - torch.Tensor: Computed MAE loss.
    """
    return torch.mean(torch.abs(output - target))

def binary_cross_entropy(output, target):
    """
    Compute Binary Cross-Entropy (BCE) loss between output and target.

    Args:
    - output (torch.Tensor): Predicted output tensor (sigmoid logits).
    - target (torch.Tensor): Binary target tensor (0 or 1).

    Returns:
    - torch.Tensor: Computed BCE loss.
    """
    return torch.mean(-(target * torch.log(output) + (1 - target) * torch.log(1 - output)))

def categorical_cross_entropy(output, target):
    """
    Compute Categorical Cross-Entropy (CE) loss between output and target.

    Args:
    - output (torch.Tensor): Predicted output tensor (softmax logits).
    - target (torch.Tensor): Categorical target tensor (class indices).

    Returns:
    - torch.Tensor: Computed CE loss.
    """
    return torch.mean(-torch.log(output.gather(1, target.view(-1, 1))))

def hinge_loss(output, target):
    """
    Compute Hinge loss between output and target.

    Args:
    - output (torch.Tensor): Predicted output tensor.
    - target (torch.Tensor): Target tensor (-1 or 1).

    Returns:
    - torch.Tensor: Computed Hinge loss.
    """
    return torch.mean(torch.clamp(1 - output * target, min=0))

def kl_divergence(output, target):
    """
    Compute Kullback-Leibler (KL) Divergence loss between output and target.

    Args:
    - output (torch.Tensor): Predicted output tensor (log-probabilities).
    - target (torch.Tensor): Target tensor (probabilities).

    Returns:
    - torch.Tensor: Computed KL Divergence loss.
    """
    return torch.mean(target * (torch.log(target) - output))



















