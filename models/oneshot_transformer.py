import torch
from torch import nn

from models.model import Model
from specs.model_specs import TransformerSpec

from typing import Optional
from torch import Tensor

import logging
logger = logging.getLogger(__name__)


class OneshotTransformer(Model):
    def __init__(self, model_spec: TransformerSpec):
        super(OneshotTransformer, self).__init__(model_spec=model_spec)
        self.d_hid = model_spec.d_hid or model_spec.dim_feedforward

        # Adjust for number of heads
        self.d_model *= model_spec.nheads

        if not model_spec.malformer:
            self.transformer = nn.Transformer(
                d_model=self.d_model,
                nhead=model_spec.nheads,
                num_encoder_layers=model_spec.num_encoder_layers,
                num_decoder_layers=model_spec.num_decoder_layers,
                dim_feedforward=model_spec.dim_feedforward,
                dropout=model_spec.dropout,
                batch_first=True,
                device=self.device)
        else:
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=model_spec.nheads,
                dim_feedforward=self.d_hid,
                dropout=model_spec.dropout,
                batch_first=True,
                device=self.device)
            encoder = nn.TransformerEncoder(encoder_layers, model_spec.num_encoder_layers)
            decoder_layers = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=model_spec.nheads,
                dim_feedforward=model_spec.dim_feedforward,
                dropout=model_spec.dropout,
                batch_first=True,
                device=self.device)
            decoder = nn.TransformerDecoder(decoder_layers, model_spec.num_decoder_layers)
            self.transformer = nn.Transformer(
                d_model=self.d_model,
                nhead=model_spec.nheads,
                custom_encoder=encoder,
                custom_decoder=decoder,
                batch_first=True,
                device=self.device)

        if self.d_model == self.output_features:
            self.out = nn.Identity()
        else:
            linear_layer = nn.Linear(self.d_model, self.output_features, device=self.device)
            dropout_layer = nn.Dropout(model_spec.dropout)
            self.out = nn.Sequential(dropout_layer, linear_layer)

        d_in = self.embedding.output_features
        if self.d_model == d_in:
            self.decoder_in = nn.Identity()
            self.encoder_in = nn.Identity()
        else:
            linear_layer1 = nn.Linear(d_in, self.d_model, device=self.device)
            dropout_layer1 = nn.Dropout(model_spec.dropout)
            self.decoder_in = nn.Sequential(dropout_layer1, linear_layer1)

            linear_layer2 = nn.Linear(d_in, self.d_model, device=self.device)
            dropout_layer2 = nn.Dropout(model_spec.dropout)
            self.encoder_in = nn.Sequential(dropout_layer2, linear_layer2)

        self.meta_token = model_spec.meta_token

    def _init_decoder_input(self, batch_size: int, output_metadata: Optional[Tensor] = None) -> Tensor:
        """Returns the correct zero vector for decoder input"""
        if not self.meta_token:
            return torch.zeros((batch_size, self.forecast_window, self.input_features), device=self.device)#.squeeze(0)
        else:
            return output_metadata

    def process(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        encoder_input = self.encoder_in(encoder_input)
        decoder_input = self.decoder_in(decoder_input)
        transformed = self.transformer(encoder_input, decoder_input)
        return self.out(transformed)

    def forward(self, input_sequence: Tensor, batch_size: int, input_metadata: Optional[Tensor] = None,
                decoder_metadata: Optional[Tensor] = None, forecast_pslp: Optional[Tensor] = None,
                reference: Optional[Tensor] = None) -> Tensor:
        # if forecast_pslp.dim() != 3:
        #     forecast_pslp = None

        decoder_input = self._init_decoder_input(batch_size, forecast_pslp)
        decoder_input = self.embedding(decoder_input, decoder_metadata)

        result = self.process(input_sequence, decoder_input)
        return result
