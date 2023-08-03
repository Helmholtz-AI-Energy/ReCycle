from dataclasses import dataclass
import torch

from typing import Optional, Type, TypeVar, Union, List
from torch import Tensor
from data.embeddings import FullEmbedding, select_embedding

import logging
logger = logging.getLogger(__name__)

__all__ = {
    'ModelSpec',
    'TransformerSpec'
}

SPEC = TypeVar('SPEC')


@dataclass
class ModelSpec:
    """
    Standard format and base class for model specifications

    :param str model_name: name of the model (mostly for labeling)
    :param int historic_window: length of the historic window
    :param int forecast_window: length of the forecast window
    :param int features_per_step: number of features of the time series, metadata features excluded

    :param int d_model: input dimension of the model after embedding
    :param FullEmbedding embedding: embedding that combines data and metadata and projects to d_model
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

    d_model: int = None
    embedding: FullEmbedding = None
    residual_input: bool = True
    residual_forecast: bool = True
    custom_quantiles: Optional[Union[List[int], Tensor]] = None
    quantiles: Optional[int] = None
    assume_symmetric_quantiles: bool = False

    device: torch.device = None

    def check_validity(self) -> None:
        assert self.embedding.output_features == self.d_model,\
            (f'Embedding features ({self.embedding.output_features}) do not match'
             f'model input features ({self.d_model})')
        assert self.embedding.input_features == self.features_per_step,\
            (f'Embedding input features ({self.embedding.input_features} do not match '
             f'data features ({self.features_per_step})')

    @classmethod
    def from_embedding_name(
            cls: SPEC, *args,
            features_per_step: int,
            d_model: int,
            embedding: str,
            embedding_args: Optional[dict] = None,
            **kwargs
    ) -> SPEC:
        """Accepts the same arguments as __init__, but builds the embedding from a name and optionally arguments"""
        embedding = select_embedding(
            name=embedding,
            input_features=features_per_step,
            embedding_args=embedding_args
        )
        return cls(*args, features_per_step=features_per_step, d_model=d_model, embedding=embedding, **kwargs)


@dataclass
class TransformerSpec(ModelSpec):
    """
    Specs for a transformer model

    :param bool meta_token: if False use 0 filled template as decoder input
    :param int nheads: number of attention heads
    :param int num_encoder_layers: number of encoder layers
    :param int num_decoder_layers: number of decoder layers
    :param int dim_feedforward: dimension of the feedforward layer after attention
    :param float dropout: dropout ratio
    :param bool malformer: if True the slightly more customizable malformer variant is used
    :param int d_hidden: only used if malformer id True, forces decoder output to given dimension instead of d_model
    """
    model_name: str = 'Transformer'
    meta_token: bool = True
    nheads: int = 1
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    dim_feedforward: int = 96
    dropout: float = 0.
    malformer: bool = False
    d_hidden: int = None


model_spec_dict = dict(
    Transformer=TransformerSpec
)


def get_model_spec(name: str) -> Type[ModelSpec]:
    if name in model_spec_dict:
        return model_spec_dict[name]
    else:
        logger.warning('No matching model spec found, using default spec.')
        return ModelSpec
