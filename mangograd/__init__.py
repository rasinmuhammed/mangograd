"""A readable, NumPy-backed tensor autograd engine.

Scalar `Value` is kept for the micrograd-style walkthrough. Everything else
is built on `Tensor`, which is vectorised and what you want in practice.
"""

from .engine import Value
from .loss import CrossEntropyLoss, MSELoss
from .nn import BatchNorm, Dropout, Linear, MLP, Module
from .optim import SGD, Adam, Optimizer
from .tensor import Tensor

__version__ = "0.1.0"

__all__ = [
    "Value",
    "Tensor",
    "Module",
    "Linear",
    "MLP",
    "Dropout",
    "BatchNorm",
    "Optimizer",
    "SGD",
    "Adam",
    "MSELoss",
    "CrossEntropyLoss",
]
