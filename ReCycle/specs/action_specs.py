from dataclasses import dataclass

from typing import Optional


@dataclass
class ActionSpec:
    """
    Spec for high level operations

    :param bool train: if True train the model
    :param bool test: if True perform evaluation on the test set
    :param bool infer: if True run inference
    :param bool hpo: if True run hyperparameter search
    :param bool plot_loss: if True plot loss after training, irrelevant if train is False

    :param Optional[str] save_path: directory to save the model to
    :param Optional[str] load_path: directory to load from checkpoint from. If None no checkpoint is loaded
    :param Optional[str] input_path: path to load inference input history from
    :param bool hyper_optimization_interrupt: interrupts process after training and returns the validation loss for
        hyperparameter optimization, prevents plot_loss, save and test from taking effect
    """

    train: bool = True
    test: bool = True
    infer: bool = False
    hpo: bool = False

    save_path: Optional[str] = "./saved_models/"
    load_path: Optional[str] = None
    input_path: Optional[str] = None
    log_path: Optional[str] = "./"

    hyper_optimization_interrupt: bool = False

    # TODO directory and file path checks
    def check_validity(self) -> None:
        if self.hyper_optimization_interrupt:
            assert (
                self.train
            ), "Returning validation loss for hyperparameter optimization requires train = True"
        if self.infer:
            raise NotImplementedError()
            assert not self.train and not self.test
            assert self.input_path is not None
        if self.hpo:
            raise NotImplementedError()
