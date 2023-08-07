from matplotlib import pyplot as plt
import torch

from utils.tools import selective_flatten

from torch import Tensor
from typing import Optional

plt.rcParams['figure.dpi'] = 600


def plot_sample(
        historic_data: Tensor,
        forecast_data: Tensor,
        historic_time: Optional[Tensor] = None,
        forecast_time: Optional[Tensor] = None,
        historic_reference: Optional[Tensor] = None,
        forecast_reference: Optional[Tensor] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        fontsize: int = 15,
        dpi: int = 600
) -> (plt.Figure, plt.Axes):
    if (historic_time is None) != (forecast_time is None):
        raise ValueError(f'Either both or neither of historic_time ({"None" if historic_time is None else "not None"}) '
                         f'and label_time ({"None" if forecast_time is None else "not None"}) should be provided')
    if historic_time is None:
        # According to previous check both are therefore None
        historic_time = torch.arange(len(historic_data))
        forecast_time = torch.arange(len(historic_data), len(forecast_data) + len(historic_data))

    fig, ax = plt.subplots(dpi=dpi, figsize=(6, 4))

    ax.plot(historic_time, selective_flatten(historic_data))
    ax.plot(forecast_time, selective_flatten(forecast_data))
    if historic_reference is not None:
        ax.plot(historic_time, selective_flatten(historic_reference))
    if forecast_reference is not None:
        ax.plot(forecast_time, selective_flatten(forecast_reference))

    ax.tick_params(labelsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)

    return fig, ax


def plot_losses(train_loss: Tensor, valid_loss: Tensor, truncate: int = 0) -> (plt.Figure, plt.Axes):
    fig, ax = plt.subplots()
    train_epoch = torch.arange(len(train_loss)) / len(train_loss) * len(valid_loss)
    valid_epoch = torch.arange(1, len(valid_loss) + 1)
    ax.plot(train_epoch[truncate:], train_loss[truncate:].detach(), "r")
    ax.plot(valid_epoch[truncate:], valid_loss[truncate:].detach(), "b")
    ax.set_yscale("log")
    plt.show()
    return fig, ax
