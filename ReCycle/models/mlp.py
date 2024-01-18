import torch
from torch import nn

from models.model import Model
from specs.model_specs import MLPSpec

from typing import Optional
from torch import Tensor

import logging

logger = logging.getLogger(__name__)


class MultiLayerPerceptron(Model):
    def __init__(self, model_spec: MLPSpec) -> None:
        super(MultiLayerPerceptron, self).__init__(model_spec=model_spec)
        self.input_features = self.historic_window * self.embedding.output_features
        self._output_features = self.forecast_window * self.output_features

        layers = []

        if model_spec.hidden_layers is not None:
            in_dims = [self.input_features] + model_spec.hidden_layers
            out_dims = model_spec.hidden_layers + [self._output_features]
            for dim_in, dim_out in zip(in_dims, out_dims):
                layers.extend(
                    [
                        nn.Linear(dim_in, dim_out, device=self.device),
                        nn.Dropout(p=model_spec.dropout),
                        model_spec.non_linearity(),
                    ]
                )

            # Discard final dropout non-linearity
            layers = layers[:-2]
        else:
            if model_spec.nr_of_hidden_layers == 0:
                layers.append(
                    nn.Linear(
                        self.input_features, self._output_features, device=self.device
                    )
                )
            else:
                layers.extend(
                    [
                        nn.Linear(
                            self.input_features, self.d_model, device=self.device
                        ),
                        nn.Dropout(p=model_spec.dropout),
                        model_spec.non_linearity(),
                    ]
                )

                for _ in range(model_spec.nr_of_hidden_layers - 1):
                    layers.extend(
                        [
                            nn.Linear(self.d_model, self.d_model, device=self.device),
                            nn.Dropout(p=model_spec.dropout),
                            model_spec.non_linearity(),
                        ]
                    )

                layers.append(
                    nn.Linear(self.d_model, self._output_features, device=self.device)
                )
        self.layers = nn.Sequential(*layers)

    def process(self, input_sequence: Tensor) -> Tensor:
        return self.layers(input_sequence)

    def forward(
        self,
        input_sequence: Tensor,
        batch_size: int,
        input_metadata: Optional[Tensor] = None,
        decoder_metadata: Optional[Tensor] = None,
        forecast_rhp: Optional[Tensor] = None,
        reference: Optional[Tensor] = None,
    ) -> Tensor:
        model_input = torch.flatten(input_sequence, start_dim=1)
        output_sequence = self.process(model_input)

        return output_sequence.unflatten(
            -1, (self.forecast_window, self.output_features)
        )
