from dataclasses import dataclass

from typing import Optional


@dataclass
class ActionSpec:
    """
    Spec for high level operations

    :param bool train: if True train the model
    :param bool plot_loss: if True plot loss after training, irrelevant if train is False

    :param bool save: if True save the model to save_path after training
    :param Optional[str] save_path: directory to save the model to if save is True (required in that case)
    :param bool load: if True load the model from load_path instead of initializing
    :param Optional[str] load_path: directory to load from if load is True, defaults to save_path if not provided

    :param bool test: if True perform evaluation on the test set
    :param bool plot_prediction: if True plot sample predictions from the test set

    :param bool hyper_optimization_interrupt: interrupts process after training and returns the validation loss for
        hyperparameter optimization, prevents plot_loss, save and test from taking effect
    """
    train: bool = True
    plot_loss: bool = True

    save: bool = True
    save_path: Optional[str] = './saved_models/'
    load: bool = False
    load_path: Optional[str] = None

    test: bool = True
    plot_prediction: bool = False

    hyper_optimization_interrupt: bool = False

    def check_validity(self) -> None:
        if self.save:
            assert self.save_path is not None, 'No directory to save to provided'
        if self.load:
            assert (self.save_path is not None) or (self.load_path is not None), 'No directory to load from provided'
        if self.hyper_optimization_interrupt:
            assert self.train, 'Returning validation loss for hyperparameter optimization requires train = True'
