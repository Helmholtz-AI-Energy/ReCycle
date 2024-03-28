from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ...data.embeddings import DataOnly
from ...specs.model_specs import ModelSpec


class ReCycleTorchModel(nn.Module):
    """
    Base class for all ReCycle PyTorch models
    """

    def __init__(self, model_spec: ModelSpec) -> None:
        super(ReCycleTorchModel, self).__init__()
        model_spec.check_validity()

        self.model_name = model_spec.model_name
        self.device = model_spec.device or self._get_device()

        # embedding interface
        if model_spec.embedding is None:
            self.embedding = DataOnly(model_spec.features_per_step, device=self.device)
        else:
            self.embedding = model_spec.embedding
            self.embedding.to(self.device)

        # Store IO specs
        self.features_per_step = model_spec.features_per_step

        self.d_model = model_spec.d_model
        self.historic_window = model_spec.historic_window
        self.forecast_window = model_spec.forecast_window
        self.nr_of_quantiles = model_spec.quantiles
        self.output_features = (
            model_spec.quantiles or 1
        ) * model_spec.features_per_step

        # Residual forecasting specifications
        self.residual_input = model_spec.residual_input
        self.residual_forecast = model_spec.residual_forecast

    @staticmethod
    def _get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(
        self,
        input_sequence: Tensor,
        batch_size: int,
        input_day_metadata: Optional[Tensor] = None,
        decoder_day_metadata: Optional[Tensor] = None,
        input_metadata: Optional[Tensor] = None,
        decoder_metadata: Optional[Tensor] = None,
        forecast_rhp: Optional[Tensor] = None,
        reference: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError

    def _select_metadata(
        self,
        input_sequence: Tensor,
        input_rhp: Tensor,
        input_day_metadata: Tensor,
        forecast_rhp: Tensor,
        decoder_day_metadata: Tensor,
        input_metadata: Optional[Tensor] = None,
        decoder_metadata: Optional[Tensor] = None,
        cat_index: Optional[Tensor] = None,
        reference: Optional[Tensor] = None,
    ) -> dict:
        selected = {
            "input_day_metadata": input_day_metadata,
            "decoder_day_metadata": decoder_day_metadata,
            "forecast_rhp": forecast_rhp,
        }

        if input_metadata is not None:
            selected["input_metadata"] = input_metadata
        if decoder_metadata is not None:
            selected["decoder_metadata"] = decoder_metadata

        return selected

    # TODO interface, dataframes?
    def predict(
        self,
        input_sequence: Tensor,
        input_rhp: Tensor,
        input_day_metadata: Tensor,
        forecast_rhp: Tensor,
        decoder_day_metadata: Tensor,
        cat_index: Optional[Tensor] = None,
        reference: Optional[Tensor] = None,
        input_metadata: Optional[Tensor] = None,
        decoder_metadata: Optional[Tensor] = None,
        raw: bool = False,
    ) -> Tensor:
        # protection from label abuse, but allows use of methods like teacher forcing
        if not self.training:
            assert reference is None, "Evaluation cannot use labels for prediction"

        # this allows for advanced metadata_column_names selection and modification by redefining _select_metadata in the subclass
        metadata = self._select_metadata(
            input_sequence,
            input_rhp,
            input_day_metadata,
            forecast_rhp,
            decoder_day_metadata,
            input_metadata,
            decoder_metadata,
            cat_index,
            reference,
        )

        if self.residual_input:
            input_sequence -= input_rhp

        batch_size = 1 if input_sequence.dim() < 3 else input_sequence.shape[0]
        if input_metadata is None:
            input_sequence = self.embedding(input_sequence, input_day_metadata)
        else:
            input_sequence = self.embedding(
                input_sequence, input_day_metadata, input_metadata
            )

        prediction = self(input_sequence, batch_size, **metadata, reference=reference)

        # fold feature dimension into (feature dimension, quantile dimension) if required
        if self.nr_of_quantiles is not None:
            prediction = prediction.unflatten(-1, [-1, self.nr_of_quantiles])
            forecast_rhp = forecast_rhp.unsqueeze(-1).expand(
                *forecast_rhp.shape, self.nr_of_quantiles
            )

        if raw:
            return prediction

        if self.residual_forecast:
            prediction = prediction + forecast_rhp

        return prediction

    def train_epoch(
        self, train_dataloader: DataLoader, criterion: Module, optimizer: Optimizer
    ) -> Tensor:
        self.train()

        total_loss = torch.zeros(len(train_dataloader), device=self.device)
        for n, batch in enumerate(train_dataloader):
            # Predict, evaluate and track loss
            prediction = self.predict(*batch)

            # Calculate loss
            reference = batch[6]
            loss = criterion(prediction, reference)
            total_loss[n] = loss.clone()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss

    def valid_epoch(self, valid_dataloader: DataLoader, criterion: Module) -> Tensor:
        self.eval()

        with torch.no_grad():
            total_loss = torch.tensor(0.0, device=self.device)
            for batch in valid_dataloader:
                # Predict, evaluate and track loss
                if len(batch) == 7:
                    prediction = self.predict(*batch[:-1])  # Evaluation may not use labels
                else:
                    prediction = self.predict(*batch[:6], None, *batch[7:])
                reference = batch[6]
                loss = criterion(prediction, reference)
                total_loss += loss.clone()

        return total_loss / len(valid_dataloader)
