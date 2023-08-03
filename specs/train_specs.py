from dataclasses import dataclass

from typing import Optional, Type, Tuple
from torch.nn.modules.loss import _Loss, L1Loss
from torch.optim import Optimizer, Adam


__all__ = {
    'TrainSpec'
}


@dataclass
class TrainSpec:
    """
    Training parameter specification

    :param float learning_rate: training learning rate
    :param int batch_size: dataloader batch size
    :param int epochs: maximum number of training epochs
    :param int patience: early stopping patience
    :param _Loss loss: loss function as torch.Module
    :param Type[Optimizer] optimizer: torch.Optimizer for training
    :param dict optimizer_args: additional arguments to be provided to the optimizer
    :param bool profiling: enables profiling for the training loop
    """
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 200
    patience: int = 20
    loss: _Loss = L1Loss()
    optimizer: Type[Optimizer] = Adam
    optimizer_args: Optional[dict] = None
    profiling: bool = False

    def clean(self) -> None:
        """This function ensures usability of the instance by replacing possible None values appropriately"""
        self.optimizer_args = self.optimizer_args or {}
