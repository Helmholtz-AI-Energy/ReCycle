from dataclasses import dataclass

from typing import Optional


@dataclass
class ActionSpec:
    train: bool = True,
    plot_loss: bool = True,

    save: bool = True,
    save_path: str = './saved_models/',
    load: bool = False,
    load_path: str = None,

    test: bool = True,
    plot_prediction: bool = False,

