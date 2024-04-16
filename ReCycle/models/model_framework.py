from typing import Union
import dataclasses
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils.error_metrics import apply_error_metric
from ..utils.tools import EarlyStopping

from ..globals import predefined_models_dict
from ..data.dataset import ResidualDataset
from ..specs import ModelSpec, DatasetSpec, TrainSpec
from ..specs.dataset_specs import ResidualDatasetSpec
from ..data.data_cleaner import clean_dataframe

# Profiling
import cProfile
import pstats
import io

# Logging
import logging

logger = logging.getLogger(__name__)


class ReCycleForecastModel:
    """Wrapper class that saves and distributes all relevant features for the benchmark."""

    def __init__(
        self, model_spec: ModelSpec, dataset_spec: DatasetSpec, train_spec: TrainSpec
    ):
        self.model_spec = model_spec
        self.model_name = model_spec.model_name
        self.train_spec = train_spec
        self.train_spec.clean()

        assert isinstance(
            dataset_spec, ResidualDatasetSpec
        ), "ResidualDataset requires ResidualDatasetSpec"
        dataset_spec.check_validity()
        self.datasets = ResidualDataset.from_csv(dataset_spec=dataset_spec)
        self.dataset_spec = dataset_spec
        self.dataset_name = dataset_spec.data_spec.file_name

        # Define run mode, 0 is train, 1 is eval 2 is testing. Used for gradient tracking and dataset selection
        self._mode = 0

        # Initialize model
        model_class = predefined_models_dict[model_spec.model_name]
        logger.info(f"Using {model_spec.model_name} model")
        self.model = model_class(model_spec=model_spec)
        # This is unnecessary for properly coded models, but some do not manage to do this themselves
        self.model.to(model_spec.device)

    # TODO if a run is started with test, there is really no need to load training and validation datasets
    def mode(self, mode: str) -> None:
        """Sets model mode and flag for dataset selection used to select the dataset"""
        if mode in ["train", 0]:
            self.model.train()
            self._mode = 0
        elif mode in ["valid", 1]:
            self.model.eval()
            self._mode = 1
        elif mode in ["test", 2]:
            self.model.eval()
            self._mode = 2
        else:
            raise ValueError(f"Trying to set invalid mode {mode}")

    def dataset(self) -> ResidualDataset:
        return self.datasets[self._mode]

    def train_model(self, train_df: pd.DataFrame = None, valid_df: pd.DataFrame = None):
        """Train self.model on self.train_set for several epochs, also obtains the validation loss and returns both"""
        # define training method
        criterion = self.train_spec.loss
        optimizer = self.train_spec.optimizer(
            self.model.parameters(),
            lr=10**self.train_spec.log_learning_rate,
            **self.train_spec.optimizer_args,
        )

        # get datasets
        if train_df is None:
            train_set = self.datasets[0]
        else:
            train_df = clean_dataframe(train_df)
            train_set = ResidualDataset(data=train_df, dataset_spec=self.dataset_spec)

        if valid_df is None:
            valid_set = self.datasets[1]
        else:
            valid_df = clean_dataframe(valid_df)
            valid_set = ResidualDataset(data=valid_df, dataset_spec=self.dataset_spec)

        train_dataloader = DataLoader(
            train_set, batch_size=self.train_spec.batch_size, shuffle=True, drop_last=True
        )
        valid_dataloader = DataLoader(
            valid_set, batch_size=self.train_spec.batch_size, shuffle=True
        )
        assert len(train_dataloader) > 0
        assert len(valid_dataloader) > 0

        train_loss = torch.zeros(self.train_spec.epochs, len(train_dataloader))
        valid_loss = torch.zeros(self.train_spec.epochs)

        early_stopping = EarlyStopping(patience=self.train_spec.patience)

        if self.train_spec.profiling:
            # profiling
            pr = cProfile.Profile()
            pr.enable()

        for n in range(self.train_spec.epochs):
            if n % 10 == 0:
                logger.info(f"Starting epoch {n+1}")

            # Training phase
            train_loss[n] = self.model.train_epoch(
                train_dataloader, criterion, optimizer
            )

            # Validation phase
            valid_loss[n] = self.model.valid_epoch(valid_dataloader, criterion)
            if n % 10 == 0:
                logger.info(f"Training loss: {train_loss[n,-1].item()}")
                logger.info(f"Validation loss: {valid_loss[n]}")

            if early_stopping(valid_loss[n], self.model):
                logger.info(f"Early stopping after epoch {n+1}")
                break

        # reset to best model
        self.model.load_state_dict(early_stopping.best_state_dict)
        best_loss = early_stopping.best_loss

        if self.train_spec.profiling:
            pr.disable()
            s = io.StringIO()
            sortby = "tottime"  # SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

        return train_loss.flatten(), valid_loss, best_loss

    def predict(
        self,
        input_df,
        output_path: Path = None,
        denormalize: bool = True,
        raw: bool = False,
    ) -> (Tensor, Tensor, Tensor):
        """

        Used to make sample forecasts from specified set, if raw=True residuals are returned, if used
        Make forecast from input time series dataframe input_df.
        Assumes input is a dataset and takes the final input in the sequence.
        If "denormalize":
        If "raw":

        """
        self.mode = "test"
        inputset_spec = dataclasses.replace(self.dataset_spec)
        inputset_spec.forecast_window = 0
        inputset_spec.reduce = None
        dataset = type(self.dataset())(data=input_df, dataset_spec=inputset_spec)
        loader = DataLoader(dataset, batch_size=1, drop_last=False, shuffle=False)
        # TODO pick last sample in loader better
        for sample in loader:
            pass

        self.model.eval()
        with torch.no_grad():
            prediction = self.model.predict(*sample[:-1])

            if denormalize:
                cat_data = sample[
                    -2
                ]  # denormalization requires category if they are separately normalized
                prediction = dataset.norm.revert_normalization(prediction, cat_data)

            prediction = prediction.squeeze(dim=0).detach()

        if denormalize:
            cat_data = sample[-2]
            if raw:
                #    prediction = dataset.norm.revert_normalization(prediction, cat_data)
                output_rhp = dataset.norm.revert_normalization(sample[3], cat_data)
            input_reference = dataset.norm.revert_normalization(sample[0], cat_data)
            output_reference = dataset.norm.revert_normalization(sample[-1], cat_data)
        else:
            input_reference = sample[0]
            output_reference = sample[-1]
            output_rhp = sample[3]

        if raw:
            output_reference -= output_rhp
            if self.model.nr_of_quantiles is not None:
                output_rhp = output_rhp.unsqueeze(-1).expand(
                    *output_rhp.shape, self.model.nr_of_quantiles
                )
            prediction -= output_rhp

        return prediction, input_reference, output_reference

    def test_forecast(
        self, batch_size: int = 128, calibration: bool = False
    ) -> pd.DataFrame:
        if calibration:
            assert (
                self.model.nr_of_quantiles is not None
            ), "Calibration counting requires model with quantiles"
            total_calibration = torch.empty(0)

        self.mode("test")
        dataset = self.dataset()
        if batch_size > len(dataset):
            batch_size = len(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

        results = pd.DataFrame()
        with torch.no_grad():
            for batch in loader:
                cat_data = batch[-2]
                prediction = self.model.predict(*batch[:-1])
                prediction = dataset.norm.revert_normalization(prediction, cat_data)
                output_reference = dataset.norm.revert_normalization(
                    batch[-1], cat_data
                )

                if self.model.nr_of_quantiles is not None:
                    if calibration:
                        # move quantile_dimension to front, then elementwise comparison on last two dimensions and collapse
                        batch_calibration = (
                            (output_reference < prediction.movedim(-1, 0))
                            .to(float)
                            .flatten(1)
                            .mean(-1)
                        )
                        total_calibration = torch.concat(
                            [total_calibration, batch_calibration.unsqueeze(0)]
                        )

                    if self.model_spec.symmetric_quantiles:
                        prediction = prediction[..., 0]
                    else:
                        prediction = prediction[..., prediction.shape[-1] // 2]
                batch_metrics = apply_error_metric(prediction, output_reference)

                # get mase for inhomogeneous datasets
                error_scale = dataset.get_error_scale(cat_data)
                scaled_prediction = torch.stack(
                    list(map(torch.div, prediction, error_scale))
                )
                scaled_reference = torch.stack(
                    list(map(torch.div, output_reference, error_scale))
                )
                scaled_metrics = apply_error_metric(
                    scaled_prediction, scaled_reference, ["mae", "bias"]
                )
                batch_metrics["mase"] = scaled_metrics.mae
                batch_metrics["scaled bias"] = scaled_metrics.bias

                results = pd.concat([results, batch_metrics])

            if calibration:
                final_calibration = total_calibration.mean(0)
                return results.mean().to_frame().T, final_calibration

        return results.mean().to_frame().T

    def test_rhp(self, batch_size: int = 128) -> Tensor:
        self.mode("test")
        dataset = self.dataset()
        if batch_size > len(dataset):
            batch_size = len(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

        results = pd.DataFrame()
        with torch.no_grad():
            for batch in loader:
                cat_data = batch[-2]
                prediction = dataset.norm.revert_normalization(batch[3], cat_data)
                output_reference = dataset.norm.revert_normalization(
                    batch[-1], cat_data
                )
                batch_metrics = apply_error_metric(prediction, output_reference)

                error_scale = dataset.get_error_scale(cat_data)
                scaled_prediction = torch.stack(
                    list(map(torch.div, prediction, error_scale))
                )
                scaled_reference = torch.stack(
                    list(map(torch.div, output_reference, error_scale))
                )
                scaled_metrics = apply_error_metric(
                    scaled_prediction, scaled_reference, ["mae", "bias"]
                )
                batch_metrics["mase"] = scaled_metrics.mae
                batch_metrics["scaled bias"] = scaled_metrics.bias

                results = pd.concat([results, batch_metrics])

        return results.mean().to_frame().T

    def save(self, save_obj: Union[Path, io.BytesIO]) -> None:
        """Save trained model to [save_path]"""
        # hyperparameters = self.args

        state_dict = self.model.state_dict()

        # torch.save([hyperparameters, state_dict], save_obj)
        torch.save([self.model_spec, self.dataset_spec, self.train_spec, state_dict], save_obj)

    @classmethod
    def load(cls, load_path: Union[Path, io.BytesIO]):
        """Load model saved with save_to-method"""
        model_spec, dataset_spec, train_spec, state_dict = torch.load(load_path)
        # run = cls(**hyperparameters)
        # run.model.load_state_dict(state_dict)
        model = cls(model_spec, dataset_spec, train_spec)
        model.model.load_state_dict(state_dict=state_dict)
        return model

    def reset(self) -> "ReCycleForecastModel":
        # TODO reset parameters
        raise
        return self

    def fit(self):
        return self.train_model()


#    def dropout_test_forecast(self, sample_nr: Optional[int] = 50) -> (Tensor, Tensor, Tensor):
#        self.mode('test')
#        dataset = self.dataset()
#        loader = DataLoader(dataset, batch_size=len(dataset))
#        batch = next(iter(loader))
#
#        # Set model to train mode so dropout is still applied
#        self.model.train()
#
#        predictions = []
#        with torch.no_grad():
#            for n in range(sample_nr):
#                cat_data = batch[-2]
#                single_prediction = self.model.predict(*batch[:-1])
#                prediction = dataset.norm.revert_normalization(prediction, cat_data)
#                predictions.append(single_prediction)
#
#        predictions = torch.stack(predictions, dim=0)
#
#        prediction = predictions.mean(dim=0)
#        error = predictions.std(dim=0)
#
#        input_reference = dataset.norm.revert_normalization(batch[0])
#        reference = dataset.norm.revert_normalization(batch[-1])
#
#        return prediction, error, input_reference, reference
#
#    def mc_dropout_test(self, idx: Optional[int] = None, sample_nr: Optional[int] = 50):
#        self.mode('test')
#        dataset = self.dataset()
#
#        idx = idx or randrange(0, len(dataset))
#
#        # Set model to train mode so dropout is still applied
#        self.model.train()
#
#        batch = dataset[idx]
#
#        predictions = []
#        with torch.no_grad():
#            for n in range(sample_nr):
#                cat_data = batch[-2]
#                single_prediction = self.model.predict(*batch[:-1])
#                prediction = dataset.norm.revert_normalization(prediction, cat_data)
#                predictions.append(single_prediction)
#
#        predictions = torch.stack(predictions, dim=0)
#
#        prediction = predictions.mean(dim=0)
#        error = predictions.std(dim=0)
#
#        input_reference = dataset.norm.revert_normalization(batch[0])
#        reference = dataset.norm.revert_normalization(batch[-1])
#
#        return prediction, error, input_reference, reference
