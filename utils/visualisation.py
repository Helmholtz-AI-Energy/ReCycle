from matplotlib import pyplot as plt
import torch

from torch import Tensor
from typing import Optional


def plot_sample(
        historic_data: Tensor,
        forecast_data: Tensor,
        historic_time: Optional[Tensor] = None,
        forecast_time: Optional[Tensor] = None,
        historic_reference: Optional[Tensor] = None,
        forecast_reference: Optional[Tensor] = None,
) -> (plt.Figure, plt.Axes):
    if (historic_time is None) != (forecast_time is None):
        raise ValueError(f'Either both or neither of historic_time ({"None" if historic_time is None else "not None"}) '
                         f'and label_time ({"None" if forecast_time is None else "not None"}) should be provided')
    if historic_time is None:
        # According to previous check both are therefore None
        historic_time = torch.arange(len(historic_data))
        forecast_time = torch.arange(len(historic_data), len(forecast_data) + len(historic_data))

    fig, ax = plt.subplots()
    ax.plot(historic_time, historic_data)
    ax.plot(forecast_time, forecast_data)
    if historic_reference is not None:
        ax.plot(historic_time, historic_reference)
    if forecast_reference is not None:
        ax.plot(forecast_time, forecast_reference)

    return fig, ax
