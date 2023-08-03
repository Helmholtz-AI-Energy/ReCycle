from .oneshot_transformer import OneshotTransformer

from typing import Type
from .model import Model

model_dict = dict(
    Transformer=OneshotTransformer
)


def get_model_class(name: str) -> Type[Model]:
    return model_dict[name]
