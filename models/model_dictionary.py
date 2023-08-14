from .oneshot_transformer import OneshotTransformer
from .mlp import MultiLayerPerceptron

from typing import Type
from .model import Model

model_dict = dict(
    Transformer=OneshotTransformer,
    MLP=MultiLayerPerceptron
)


def get_model_class(name: str) -> Type[Model]:
    return model_dict[name]
