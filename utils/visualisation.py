from matplotlib import pyplot as plt
import torch

from torch import Tensor
from typing import Optional


def plot_sample(
        historic_data: Tensor,
        label_data: Tensor,
        historic_time: Optional[Tensor] = None,
        label_time: Optional[Tensor] = None,
) -> (plt.Figure, plt.Axes):
    if (historic_time is None) != (label_time is None):
        raise ValueError(f'Either both or neither of historic_time ({"None" if historic_time is None else "not None"}) '
                         f'and label_time ({"None" if label_time is None else "not None"}) should be provided')
    if historic_time is None:
        # According to previous check both are therefore None
        historic_time = torch.arange(len(historic_data))
        label_time = torch.arange(len(historic_data), len(label_data) + len(historic_data))

    fig, ax = plt.subplots()
    ax.plot(historic_time, historic_data)
    ax.plot(label_time, label_data)

    return fig, ax
