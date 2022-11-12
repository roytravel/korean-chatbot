import torch.nn as nn
from torch import Tensor
import numpy as np

class LeakyReLU(nn.Module):
    def __init__(self) -> None:
        super(LeakyReLU, self).__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return np.maximum(0.01 * x, x)


class ReLU(nn.Module):
    def __init__(self) -> None:
        super(ReLU, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return np.maximum(0, x)


class ELU(nn.Module):
    def __init__(self) -> None:
        super(ELU, self).__init__()
    
    def forward(self, x: Tensor, alpha: float) -> Tensor:
        return (x > 0) * x + (x <= 0) * (alpha * (np.exp(x) - 1))
        

class Tanh(nn.Module):
    def __init__(self) -> None:
        super(Tanh, self).__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


class Sigmoid(nn.Module):
    def __init__(self) -> None:
        super(Sigmoid, self).__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + np.exp(-x))