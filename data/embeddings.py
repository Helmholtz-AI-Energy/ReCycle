import torch
from torch import nn
from math import log

# from data.normalizer import InformerTimeNorm

from typing import Optional, Type, Union
from torch import Tensor


import logging

logger = logging.getLogger(__name__)


def sine_positional_encoding(max_pos: int, d_model: int, freq: float = 1e-4) -> Tensor:
    position = torch.arange(max_pos)
    freqs = freq ** (
        2 * torch.div(torch.arange(d_model), 2, rounding_mode="floor") / d_model
    )
    pos_enc = position.reshape(-1, 1) * freqs.reshape(1, -1)
    pos_enc[:, ::2] = torch.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = torch.sin(pos_enc[:, 1::2])
    return pos_enc


class BasicEmbedding(nn.Module):
    """Base class for all embeddings, not the same as nn.BasicEmbedding"""

    def __init__(
        self, input_features: int, output_features: int, device: torch.device = None
    ):
        super(BasicEmbedding, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.device = device

    def to(self, device: Optional[torch.device] = None, *args, **kwargs):
        super(BasicEmbedding, self).to(*args, device=device, **kwargs)

        if device is not None:
            self.device = device


class VoidEmbedding(BasicEmbedding):
    def __init__(
        self, input_features: int, output_features: int = 0, device: torch.device = None
    ):
        assert output_features is 0, f"Void embedding always returns 0 output features"
        super(VoidEmbedding, self).__init__(input_features, output_features, device)

    def forward(self, input_sequence: Tensor) -> Tensor:
        return torch.empty(0, device=self.device)


class IdentityEmbedding(BasicEmbedding):
    def __init__(
        self,
        input_features: int,
        output_features: int = None,
        device: torch.device = None,
    ):
        output_features = output_features or input_features
        assert (
            input_features == output_features
        ), f"For identity embedding {input_features=} must equal {output_features=}"
        super(IdentityEmbedding, self).__init__(input_features, output_features, device)

    @staticmethod
    def forward(input_sequence: Tensor) -> Tensor:
        return input_sequence


class LinearEmbedding(BasicEmbedding):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        device: torch.device = None,
        nonlinearity: nn.Module = nn.Tanh(),
    ):
        super(LinearEmbedding, self).__init__(input_features, output_features, device)

        self.linear = nn.Linear(input_features, output_features, device=device)
        self.nonlinear = nonlinearity

    def forward(self, input_sequence: Tensor) -> Tensor:
        output = self.nonlinear(self.linear(input_sequence))
        return output


class ConvolutionalEmbedding(BasicEmbedding):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        device: torch.device = None,
        kernel_size: int = 3,
        padding: int = 1,
        padding_mode: str = "circular",
        **kwargs,
    ):
        super(ConvolutionalEmbedding, self).__init__(
            input_features, output_features, device
        )

        self.conv = nn.Conv1d(
            in_channels=input_features,
            out_channels=output_features,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            device=device,
            **kwargs,
        )

    def forward(self, input_sequence: Tensor) -> Tensor:
        # Channels are used for features
        trans = torch.transpose(input_sequence, -2, -1)
        # Convolve
        output = self.conv(trans).transpose(-2, -1)
        return output


# Augmentations add a feature dimension to highlight structures like sequence ordering or periodicity (e.g. weeks)
class LinearAugmentation(BasicEmbedding):
    def __init__(
        self,
        input_features: int,
        output_features: int = None,
        device: torch.device = None,
        max_len: int = 100,
    ):
        output_features = output_features or input_features + 1
        assert output_features == (
            input_features + 1
        ), f"For linear augmentation {input_features=}+1 must equal {output_features=}"

        super(LinearAugmentation, self).__init__(
            input_features, output_features, device
        )
        pe = (
            torch.true_divide(torch.arange(max_len), max_len - 1)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        self.register_buffer("pe", pe)  # TODO: Why save this with state dict?

    def forward(self, input_sequence: Tensor) -> Tensor:
        pe = self.pe[:, : input_sequence.shape[1], :].expand(
            input_sequence.shape[0], -1, -1
        )
        output = torch.cat((input_sequence, pe), dim=2)
        return output


class BasePositionalEncoding(BasicEmbedding):
    def __int__(
        self,
        input_features: int,
        output_features: int = None,
        device: torch.device = None,
        max_pos: int = 200,
        freq: float = 1e-4,
    ):
        output_features = output_features or input_features
        assert (
            output_features == input_features
        ), "Output features mut equal input features"
        super(BasePositionalEncoding, self).__int__(
            input_features, output_features, device
        )

        positional_encoding = sine_positional_encoding(max_pos, input_features, freq)
        self.register_buffer(
            "positional_encoding", positional_encoding, persistent=False
        )

    def forward(self, input_sequence: Tensor) -> Tensor:
        seq_len = input_sequence.shape[-2]
        if input_sequence.dim() == 3:
            batch_size = input_sequence.shape[0]
            positional_code = self.positional_encoding[:seq_len].expand(
                batch_size, -1, -1
            )
            return positional_code
        elif input_sequence.dim() == 2:
            positional_code = self.positional_encoding[:seq_len]
            return positional_code
        else:
            raise TypeError(
                f"Invalid input sequence shape!"
                f"Input sequence dimension {input_sequence.dim()} should be 2 or 3"
            )


class PeriodicAugmentation(BasicEmbedding):
    def __init__(
        self,
        input_features: int,
        output_features: int = None,
        device: torch.device = None,
        period: int = 24,
        offset: int = 0,
        max_len: int = 100,
    ):
        output_features = output_features or input_features + 1
        assert output_features == (
            input_features + 1
        ), f"For periodic augmentation {input_features=}+1 must equal {output_features=}"
        super(PeriodicAugmentation, self).__init__(
            input_features, output_features, device
        )
        pe_period = torch.true_divide(torch.arange(period), period).roll(offset, 0)
        period_nr = (
            max_len // period + 1
        )  # poor man's round up division to see how many periods cover max_length
        pe = pe_period.repeat(period_nr).unsqueeze(0).unsqueeze(-1)
        self.register_buffer("pe", pe)  # Why save this with state dict?

    def forward(self, input_sequence: Tensor) -> Tensor:
        pe = self.pe[:, : input_sequence.shape[1], :].expand(
            input_sequence.shape[0], -1, -1
        )
        output = torch.cat((input_sequence, pe), dim=2)
        return output


# Embeddings from Informer (Zhou et al 2020)
class InformerPositionalEmbedding(BasicEmbedding):
    def __init__(
        self, input_features: int, output_features: int, device: torch.device = None
    ):
        super(InformerPositionalEmbedding, self).__init__(
            input_features, output_features, device
        )

        max_length = 5000
        dtype = torch.float
        pe = torch.zeros(
            (max_length, output_features), requires_grad=False, device=self.device
        )
        position = torch.arange(max_length, dtype=dtype).unsqueeze(dim=1)
        # Compute positional encoding in log space
        div_term = torch.exp(
            torch.arange(0, output_features, step=2, dtype=dtype)
            * -(log(2 * max_length))
            / output_features
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(dim=0)
        self.register_buffer("pe", pe)

    def forward(self, input_sequence: Tensor) -> Tensor:
        return self.pe[:, : input_sequence.shape[1]]


class InformerConvolutionalEmbedding(BasicEmbedding):
    def __init__(
        self, input_features: int, output_features: int, device: torch.device = None
    ):
        super(InformerConvolutionalEmbedding, self).__init__(
            input_features, output_features, device
        )

        self.conv = nn.Conv1d(
            in_channels=input_features,
            out_channels=output_features,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            device=device,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, input_sequence: Tensor) -> Tensor:
        # Channels are used for features
        trans = torch.transpose(input_sequence, 1, 2)
        # Convolve
        output = self.conv(trans)
        return torch.transpose(output, 1, 2)


class InformerFixedEmbedding(BasicEmbedding):
    def __init__(
        self, input_features: int, output_features: int, device: torch.device = None
    ):
        super(InformerFixedEmbedding, self).__init__(
            input_features, output_features, device
        )
        max_length = 5000
        dtype = torch.float
        pe = torch.zeros(
            (max_length, output_features),
            requires_grad=False,
            dtype=dtype,
            device=self.device,
        )
        position = torch.arange(input_features, dtype=dtype).unsqueeze(dim=1)
        # Compute positional encoding in log space
        div_term = torch.exp(
            torch.arange(0, output_features, step=2, dtype=dtype)
            * -(log(2 * max_length))
            / output_features
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.embedding = nn.Embedding(input_features, output_features)
        self.embedding.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, input_sequence: Tensor) -> Tensor:
        return self.embedding(input_sequence).detach()


# class InformerTimeFeatureEmbedding(BasicEmbedding):
#     def __init__(self, features_per_step: int, output_features: int, device: torch.device = None):
#         super(InformerTimeFeatureEmbedding, self).__init__(features_per_step, output_features, device)
#         time_features = 4
#         self.normalization = InformerTimeNorm()
#         self.embed = nn.Linear(time_features, output_features)
#
#     def forward(self, input_sequence: Tensor) -> Tensor:
#         time_data = self.normalization.normalize(input_sequence)
#         # All time data has the features [year, month, day, hour, minute, holiday]. year and holiday are not used here
#         return self.embed(time_data[..., 1:5])


class FullEmbedding(nn.Module):
    """Base class for all embeddings combining multiple inputs"""

    def __init__(
        self,
        data_embedding: BasicEmbedding,
        metadata_embedding: BasicEmbedding,
        output_features: int,
    ):
        assert (
            data_embedding.device == metadata_embedding.device
        ), f"Both embeddings must be on same device"
        super(FullEmbedding, self).__init__()

        input_features = (
            data_embedding.input_features,
            metadata_embedding.input_features,
        )
        device = data_embedding.device

        self.input_features = input_features
        self.output_features = output_features
        self.device = device

        self.data_embedding = data_embedding
        self.metadata_embedding = metadata_embedding

    def forward(
        self, input_sequence: Tensor, metadata: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError

    def to(self, device: Optional[torch.device] = None, *args, **kwargs):
        super(FullEmbedding, self).to(*args, device=device, **kwargs)

        if device is not None:
            self.device = device


class AdditiveFullEmbedding(FullEmbedding):
    def __init__(
        self, data_embedding: BasicEmbedding, metadata_embedding: BasicEmbedding
    ):
        if metadata_embedding.output_features != 0:
            assert (
                data_embedding.output_features == metadata_embedding.output_features
            ), (
                f"For additive embedding combination output_features of both Embeddings"
                f"({data_embedding.output_features}, {metadata_embedding.output_features}) must match"
            )
        output_features = data_embedding.output_features

        super(AdditiveFullEmbedding, self).__init__(
            data_embedding, metadata_embedding, output_features
        )

    def forward(
        self, input_sequence: Tensor, metadata: Optional[Tensor] = None
    ) -> Tensor:
        # If metadata_dict is None, metadata_embedding should be void or identity
        # If metadata_dict is None or VoidEmbedding is used, the resulting None is replaced with 0
        if self.metadata_embedding.output_features == 0:
            return self.data_embedding(input_sequence)
        else:
            embedded_metadata = self.metadata_embedding(metadata)
            embedded_data = self.data_embedding(input_sequence)
            return embedded_data + embedded_metadata


class ConcatFullEmbedding(FullEmbedding):
    def __init__(
        self, data_embedding: BasicEmbedding, metadata_embedding: BasicEmbedding
    ):
        output_features = (
            data_embedding.output_features + metadata_embedding.output_features
        )

        super(ConcatFullEmbedding, self).__init__(
            data_embedding, metadata_embedding, output_features
        )

    def forward(
        self, input_sequence: Tensor, metadata: Optional[Tensor] = None
    ) -> Tensor:
        embedded_metadata = self.metadata_embedding(metadata)
        embedded_data = self.data_embedding(input_sequence)
        # print(embedded_data.shape, embedded_metadata.shape) # common debug check
        return torch.cat([embedded_data, embedded_metadata], dim=-1)


class InputLinearAugmentation(ConcatFullEmbedding):
    def __init__(
        self,
        input_features: int,
        output_features: int = None,
        device: torch.device = None,
        max_len: int = 100,
    ):
        data_embedding = LinearAugmentation(
            input_features, output_features, device, max_len
        )
        metadata_embedding = VoidEmbedding(input_features, device=device)

        super(InputLinearAugmentation, self).__init__(
            data_embedding, metadata_embedding
        )


class MetadataAugmentation(ConcatFullEmbedding):
    # Straight concat of data and metadata_dict
    def __init__(
        self, input_features: int, metadata_features: int, device: torch.device = None
    ):
        data_embedding = IdentityEmbedding(input_features, device=device)
        metadata_embedding = IdentityEmbedding(metadata_features, device=device)

        super(MetadataAugmentation, self).__init__(data_embedding, metadata_embedding)


class DataOnly(ConcatFullEmbedding):
    # Ignore metadata_dict use straight data
    def __init__(self, input_features: int, device: torch.device = None):
        data_embedding = IdentityEmbedding(input_features, device=device)
        metadata_embedding = VoidEmbedding(input_features, device=device)

        super(DataOnly, self).__init__(data_embedding, metadata_embedding)


def select_embedding(
    name: str,
    input_features: int,
    meta_features: Optional[int] = None,
    embedding_args: Optional[dict] = None,
) -> FullEmbedding:
    if name is None or (name == "default" and meta_features is None):
        data_embedding = IdentityEmbedding(input_features)
        meta_embedding = VoidEmbedding(input_features)
        return ConcatFullEmbedding(data_embedding, meta_embedding)
    elif name == "default":
        data_embedding = IdentityEmbedding(input_features)
        meta_embedding = IdentityEmbedding(meta_features)
        return ConcatFullEmbedding(data_embedding, meta_embedding)


# ------------------------------------------------------------------------------------------------------------------
# class InformerEmbedding(BasicEmbedding):
#    def __init__(self, data_features: int, output_features: int, device: torch.device = None,
#                 time_features: int = 6, dropout: float = 0.1):
#        super(InformerEmbedding, self).__init__(data_features, output_features, device)
#
#        self.value_embedding = InformerConvolutionalEmbedding(self.features_per_step, self.output_features, self.device)
#        self.position_embedding = InformerPositionalEmbedding(self.features_per_step, self.output_features, self.device)
#        self.temporal_embedding = InformerTimeFeatureEmbedding(self.features_per_step, self.output_features, self.device)
#
#        self.dropout = nn.Dropout(dropout)
#
#    def forward(self, input_sequence: Tensor, input_times: Tensor) -> Tensor:
#        output_sequence = self.value_embedding(input_sequence) + self.position_embedding(input_sequence)\
#                          + self.temporal_embedding(input_times)
#        return self.dropout(output_sequence)
