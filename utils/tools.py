import sys

import numpy as np
import torch
from math import floor, log10, inf

from torch import Tensor


def selective_flatten(tensor: Tensor, idx: int = 0) -> Tensor:
    """Flattens 2D-tensors and selects one element (default: first) of 3D-tensor before flattening it"""
    if (dimension := tensor.dim()) < 2:
        return tensor
    elif dimension == 2:
        return torch.flatten(tensor)
    elif dimension == 3:
        return torch.flatten(tensor[idx])
    elif dimension > 3:
        sys.exit('Input tensor has too many dimensions')


def round_to_significant(x: float, significant_figures: int = 3) -> float:
    try:
        digits = significant_figures - int(floor(log10(abs(x)))) - 1
        return round(x, digits)
    except OverflowError:
        return np.Inf


def data_only(d_in):
    if type(d_in) == list or type(d_in) == tuple:
        return d_in[0]
    else:
        return d_in


def correlation_coefficient(x: Tensor, y: Tensor) -> Tensor:
    assert x.shape == y.shape, f'input shapes {x.shape=} and {y.shape=} do not match'

    res_x = x - torch.mean(x, dim=-1, keepdim=True)
    res_y = y - torch.mean(y, dim=-1, keepdim=True)

    num = torch.mean(res_x * res_y, dim=-1)
    root1 = torch.sqrt(torch.mean(res_x**2, dim=-1))
    root2 = torch.sqrt(torch.mean(res_y**2, dim=-1))

    return num / (root1 * root2)


class EarlyStopping:
    def __init__(self, patience: int = 7):
        self.patience = patience
        self.patience_counter = 0
        self.best_loss = inf
        self.best_state_dict = None

    def __call__(self, loss: float, model: torch.nn.Module) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_state_dict = model.state_dict()
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
            else:
                return False
