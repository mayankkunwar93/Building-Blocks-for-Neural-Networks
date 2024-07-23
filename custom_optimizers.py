# -*- coding: utf-8 -*-
"""loss_functions.py

Author: Mayank Kunwar
Find Me: https://in.linkedin.com/in/mayankkunwar93
"""

import torch

class SGD:
    def __init__(self, params, lr, momentum=0):
        """
        Initialize SGD optimizer.

        Args:
        - params (iterable): Iterable of parameters to optimize.
        - lr (float): Learning rate.
        - momentum (float, optional): Momentum factor (default: 0).

        Attributes:
        - params (list): List of model parameters.
        - lr (float): Learning rate.
        - momentum (float): Momentum factor.
        - velocity (list): List to store velocity for each parameter.

        Notes:
        - Initializes velocity to zero for each parameter.
        """
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(param) for param in self.params]

    def step(self):
        """
        Perform a single optimization step with SGD.

        Notes:
        - Updates parameters based on current gradient and velocity.
        """
        for param, vel in zip(self.params, self.velocity):
            vel.mul_(self.momentum).add_(param.grad, alpha=self.lr)
            param.data.add_(-vel)

    def zero_grad(self):
        """
        Zero out gradients for all optimized parameters.

        Notes:
        - Detaches gradients if they exist and sets them to zero.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

class Adagrad:
    def __init__(self, params, lr):
        """
        Initialize Adagrad optimizer.

        Args:
        - params (iterable): Iterable of parameters to optimize.
        - lr (float): Learning rate.

        Attributes:
        - params (list): List of model parameters.
        - lr (float): Learning rate.
        - epsilon (float): Small constant to avoid division by zero.
        - sum_sq_grad (list): List to accumulate squared gradients for each parameter.

        Notes:
        - Initializes sum of squared gradients to zero for each parameter.
        """
        self.params = list(params)
        self.lr = lr
        self.epsilon = 1e-10  # small constant to avoid division by zero
        self.sum_sq_grad = [torch.zeros_like(param) for param in self.params]

    def step(self):
        """
        Perform a single optimization step with Adagrad.

        Notes:
        - Updates parameters based on accumulated squared gradients.
        """
        for param, sum_sq_grad in zip(self.params, self.sum_sq_grad):
            sum_sq_grad.add_(param.grad ** 2)
            std = sum_sq_grad.sqrt().add_(self.epsilon)
            param.data.addcdiv_(-self.lr, param.grad, std)

    def zero_grad(self):
        """
        Zero out gradients for all optimized parameters.

        Notes:
        - Detaches gradients if they exist and sets them to zero.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

class Adadelta:
    def __init__(self, params, lr, rho=0.9, eps=1e-6):
        """
        Initialize Adadelta optimizer.

        Args:
        - params (iterable): Iterable of parameters to optimize.
        - lr (float): Learning rate.
        - rho (float, optional): Decay rate (default: 0.9).
        - eps (float, optional): Small constant to avoid numerical instability (default: 1e-6).

        Attributes:
        - params (list): List of model parameters.
        - lr (float): Learning rate.
        - rho (float): Decay rate.
        - eps (float): Small constant to avoid numerical instability.
        - sum_sq_grad (list): List to accumulate squared gradients for each parameter.
        - sum_sq_delta (list): List to accumulate squared deltas for each parameter.

        Notes:
        - Initializes sum of squared gradients and deltas to zero for each parameter.
        """
        self.params = list(params)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.sum_sq_grad = [torch.zeros_like(param) for param in self.params]
        self.sum_sq_delta = [torch.zeros_like(param) for param in self.params]

    def step(self):
        """
        Perform a single optimization step with Adadelta.

        Notes:
        - Updates parameters based on accumulated squared gradients and deltas.
        """
        for param, sum_sq_grad, sum_sq_delta in zip(self.params, self.sum_sq_grad, self.sum_sq_delta):
            sum_sq_grad.mul_(self.rho).addcmul_(1 - self.rho, param.grad, param.grad)
            std = sum_sq_delta.add(self.eps).sqrt_()
            delta = param.grad.mul(std).div(sum_sq_grad.add(self.eps).sqrt_())
            param.data.add_(-self.lr, delta)
            sum_sq_delta.mul_(self.rho).addcmul_(1 - self.rho, delta, delta)

    def zero_grad(self):
        """
        Zero out gradients for all optimized parameters.

        Notes:
        - Detaches gradients if they exist and sets them to zero.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

class RMSprop:
    def __init__(self, params, lr, alpha=0.99, eps=1e-8):
        """
        Initialize RMSprop optimizer.

        Args:
        - params (iterable): Iterable of parameters to optimize.
        - lr (float): Learning rate.
        - alpha (float, optional): Smoothing constant (default: 0.99).
        - eps (float, optional): Small constant to avoid division by zero (default: 1e-8).

        Attributes:
        - params (list): List of model parameters.
        - lr (float): Learning rate.
        - alpha (float): Smoothing constant.
        - eps (float): Small constant to avoid division by zero.
        - avg_sq_grad (list): List to accumulate smoothed squared gradients for each parameter.

        Notes:
        - Initializes average of squared gradients to zero for each parameter.
        """
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.avg_sq_grad = [torch.zeros_like(param) for param in self.params]

    def step(self):
        """
        Perform a single optimization step with RMSprop.

        Notes:
        - Updates parameters based on accumulated squared gradients and smoothing constant.
        """
        for param, avg_sq_grad in zip(self.params, self.avg_sq_grad):
            avg_sq_grad.mul_(self.alpha).addcmul_(1 - self.alpha, param.grad, param.grad)
            std = avg_sq_grad.sqrt().add_(self.eps)
            param.data.addcdiv_(-self.lr, param.grad, std)

    def zero_grad(self):
        """
        Zero out gradients for all optimized parameters.

        Notes:
        - Detaches gradients if they exist and sets them to zero.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Initialize Adam optimizer.

        Args:
        - params (iterable): Iterable of parameters to optimize.
        - lr (float): Learning rate.
        - beta1 (float, optional): Exponential decay rate for the first moment estimates (default: 0.9).
        - beta2 (float, optional): Exponential decay rate for the second moment estimates (default: 0.999).
        - eps (float, optional): Small constant to avoid division by zero (default: 1e-8).

        Attributes:
        - params (list): List of model parameters.
        - lr (float): Learning rate.
        - beta1 (float): Exponential decay rate for the first moment estimates.
        - beta2 (float): Exponential decay rate for the second moment estimates.
        - eps (float): Small constant to avoid division by zero.
        - m (list): List to store first moment estimates for each parameter.
        - v (list): List to store second moment estimates for each parameter.
        - t (int): Time step counter.

        Notes:
        - Initializes first and second moment estimates and time step counter to zero.
        """
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [torch.zeros_like(param) for param in self.params]
        self.v = [torch.zeros_like(param) for param in self.params]
        self.t = 0

    def step(self):
        """
        Perform a single optimization step with Adam.

        Notes:
        - Updates parameters based on first and second moment estimates and learning rate.
        """
        self.t += 1
        for param, m, v in zip(self.params, self.m, self.v):
            m.mul_(self.beta1).add_(1 - self.beta1, param.grad)
            v.mul_(self.beta2).addcmul_(1 - self.beta2, param.grad, param.grad)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            param.data.addcdiv_(-self.lr, m_hat, v_hat.sqrt().add_(self.eps))

    def zero_grad(self):
        """
        Zero out gradients for all optimized parameters.

        Notes:
        - Detaches gradients if they exist and sets them to zero.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

class Adamax:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Initialize Adamax optimizer.

        Args:
        - params (iterable): Iterable of parameters to optimize.
        - lr (float): Learning rate.
        - beta1 (float, optional): Exponential decay rate for the first moment estimates (default: 0.9).
        - beta2 (float, optional): Exponential decay rate for the second moment estimates (default: 0.999).
        - eps (float, optional): Small constant to avoid division by zero (default: 1e-8).

        Attributes:
        - params (list): List of model parameters.
        - lr (float): Learning rate.
        - beta1 (float): Exponential decay rate for the first moment estimates.
        - beta2 (float): Exponential decay rate for the second moment estimates.
        - eps (float): Small constant to avoid division by zero.
        - m (list): List to store first moment estimates for each parameter.
        - u (list): List to store exponentially weighted infinity norm for each parameter.
        - t (int): Time step counter.

        Notes:
        - Initializes first moment estimates, exponentially weighted infinity norm,
          and time step counter to zero.
        """
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [torch.zeros_like(param) for param in self.params]
        self.u = [torch.zeros_like(param) for param in self.params]
        self.t = 0

    def step(self):
        """
        Perform a single optimization step with Adamax.

        Notes:
        - Updates parameters based on first moment estimates, exponentially weighted
          infinity norm, and learning rate.
        """
        self.t += 1
        for param, m, u in zip(self.params, self.m, self.u):
            m.mul_(self.beta1).add_(1 - self.beta1, param.grad)
            u = torch.max(self.beta2 * u, torch.abs(param.grad))
            param.data.add_(-self.lr / (1 - self.beta1 ** self.t), m / u)

    def zero_grad(self):
        """
        Zero out gradients for all optimized parameters.

        Notes:
        - Detaches gradients if they exist and sets them to zero.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()



















