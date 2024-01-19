from dataclasses import dataclass
import torch
from torch import nn

from typing import Optional, Type, TypeVar, Union, List
from torch import Tensor
from ..data.embeddings import FullEmbedding, select_embedding

import logging

logger = logging.getLogger(__name__)

__all__ = {"ModelSpec", "TransformerSpec", "MLPSpec"}

SPEC = TypeVar("SPEC")


@dataclass
class ModelSpec:
    """
    Standard format and base class for model specifications

    :param str model_name: name of the model (mostly for labeling)
    :param int historic_window: length of the historic window
    :param int forecast_window: length of the forecast window
    :param int features_per_step: number of features of the time series, metadata_column_names features excluded

    :param int d_model: input dimension of the model after embedding
    :param FullEmbedding embedding: embedding that combines data and metadata_column_names and projects to d_model
    :param float dropout: dropout ratio
    :param bool residual_input: if True modify input to be residual
    :param bool residual_forecast: if True modify output to be residual
    :param Optional[Union[List[int], Tensor]] custom_quantiles: Specifies a list of quantile values
    :param Optional[int] quantiles: specifies number of prediction quantiles, if not None
    :param bool assume_symmetric_quantiles: halves number of quantiles by assuming they are symmetric around median

    :param torch.device device: GPU or CPU to use, if None autodetection is used
    """

    historic_window: int
    forecast_window: int
    features_per_step: int

    model_name: str

    meta_features: Optional[int] = None
    d_model: int = None
    embedding: FullEmbedding = None
    dropout: float = 0.0
    residual_input: bool = True
    residual_forecast: bool = True
    custom_quantiles: Optional[Union[List[int], Tensor]] = None
    quantiles: Optional[int] = None
    assume_symmetric_quantiles: bool = False

    device: torch.device = None

    def check_validity(self) -> None:
        assert self.embedding.input_features[0] == self.features_per_step, (
            f"Primary embedding input features ({self.embedding.input_features[0]} do not match "
            f"data features ({self.features_per_step}))"
        )
        assert self.embedding.input_features[1] == self.meta_features, (
            f"Secondary embedding input features ({self.embedding.input_features[1]}) do not match "
            f"meta features ({self.meta_features}))"
        )
        if self.custom_quantiles is not None:
            assert self.quantiles == len(
                self.custom_quantiles
            ), f"{self.custom_quantiles=} should reflect number of custom quantiles ({len(self.custom_quantiles)})"

    @classmethod
    def from_embedding_name(
        cls: SPEC,
        *args,
        embedding: str,
        features_per_step: int,
        meta_features: Optional[int] = None,
        embedding_args: Optional[dict] = None,
        **kwargs,
    ) -> SPEC:
        """Accepts the same arguments as __init__, but builds the embedding from a name and optionally arguments"""
        embedding = select_embedding(
            name=embedding,
            input_features=features_per_step,
            meta_features=meta_features,
            embedding_args=embedding_args,
        )
        return cls(
            *args,
            features_per_step=features_per_step,
            meta_features=meta_features,
            embedding=embedding,
            **kwargs,
        )


@dataclass
class TransformerSpec(ModelSpec):
    """
    Spec for a transformer model (models/oneshot_transformer.py)

    :param bool meta_token: if False use 0 filled template as decoder input
    :param int nheads: number of attention heads
    :param int num_encoder_layers: number of encoder layers
    :param int num_decoder_layers: number of decoder layers
    :param int dim_feedforward: dimension of the feedforward layer after attention
    :param bool malformer: if True the slightly more customizable malformer variant is used
    :param int d_hidden: only used if malformer id True, forces decoder output to given dimension instead of d_model
    """

    model_name = "Transformer"
    meta_token: bool = True
    nheads: int = 1
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    dim_feedforward: int = 96
    malformer: bool = False
    d_hidden: int = None


class MLPSpec(ModelSpec):
    """
    Spec for a multilayer perceptron model (models/mpl.py)
    """

    model_name = "MLP"
    nr_of_hidden_layers: Optional[int] = 8
    hidden_layers: Optional[List[int]] = None
    non_linearity: nn.Module = nn.Tanh

    def check_validity(self) -> None:
        super().check_validity()
        if self.hidden_layers is None:
            assert (
                self.nr_of_hidden_layers is not None
            ), "Either hidden_layers or nr_of_hidden_layers must be provided"
        elif self.nr_of_hidden_layers is not None:
            logger.warning(
                f"{self.nr_of_hidden_layers=} is ignored since {self.hidden_layers=} is provided"
            )


model_spec_dict = dict(
    Transformer=TransformerSpec,
    MLP=MLPSpec,
)


def get_model_spec(name: str) -> Type[ModelSpec]:
    if name in model_spec_dict:
        return model_spec_dict[name]
    else:
        logger.warning("No matching model spec found, using default spec.")
        return ModelSpec
