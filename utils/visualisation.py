from matplotlib import pyplot as plt
import torch
from os import path

from utils.tools import selective_flatten

from torch import Tensor
from typing import Optional, List

if path.isdir('./HAIcolours'):
    from HAIcolours import gray, blue
else:
    blue = 'tab:blue'
    gray = 'tab:gray'

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


def plot_quantiles(quantiles: Tensor, reference: Tensor = None,
                   quantile_labels: Optional[List[str]] = None,
                   idx: int = 0,
                   dpi: int = 600,
                   residual_plot: bool = False) \
        -> (plt.Figure, plt.Axes):

    quantile_nr = quantiles.shape[-1]
    median = (quantile_nr - 1) / 2

    fig, ax = plt.subplots(dpi=dpi, figsize=(6, 4))

    if quantile_nr % 2 == 1:
        median = selective_flatten(quantiles[..., int(median)])
        prediction_x = torch.arange(median.shape[-1])

        # plot median
        ax.plot(prediction_x, median, color=blue, label='Prediction')

    for n in range(int(quantile_nr / 2)):
        lower_bound = selective_flatten(quantiles[..., n])
        upper_bound = selective_flatten(quantiles[..., -(n +1)])
        prediction_x = torch.arange(lower_bound.shape[-1])

        ax.fill_between(prediction_x, lower_bound, upper_bound, color=blue, edgecolor=None, alpha=(1. / quantile_nr))

    if reference is not None:
        reference = selective_flatten(reference, idx)
        ax.plot(prediction_x, reference, color=gray, label='Reference')
    ax.set_xlim([0, prediction_x[-1]])

    x_label = "" if not residual_plot else "Residual "
    x_label += "Consumption [GWh]"
    ax.set_ylabel(x_label)
    ax.set_xlabel('Time [d]')
    ticks = torch.arange(0, len(prediction_x) + 1, 24)
    labels = range(len(ticks))
    plt.xticks(ticks, labels)

    #ax.legend()
    ax.grid(False)
    plt.show()
    return fig, ax


def plot_calibration(predicted: Tensor, target: Tensor, dpi: Optional[int] = 600) -> (plt.Figure, plt.Axes):
    plt.rc('font', size=4)
    fig, ax = plt.subplots(dpi=dpi)#, figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k', linewidth=1, transform=ax.transAxes)

    ax.plot(target, predicted, blue)

    ticks = torch.arange(0, 1.1, 0.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.set_ylabel('Theoretical quantiles')
    ax.set_xlabel('Predicted quantiles')

    plt.show()
    return fig, ax
