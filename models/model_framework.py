import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.error_metrics import apply_error_metric
from random import randrange

from utils.tools import EarlyStopping
from .model_dictionary import get_model_class

from typing import Optional, Tuple
from torch import Tensor
from data import ResidualDataset
from specs import ModelSpec, DatasetSpec, TrainSpec
from specs.dataset_specs import ResidualDatasetSpec

# Profiling
import cProfile, pstats, io

# Logging
import logging
logger = logging.getLogger(__name__)


class ModelFramework:
    """Wrapper class that saves and distributes all relevant features for the benchmark."""
    def __init__(
            self,
            model_spec: ModelSpec,
            dataset_spec: DatasetSpec,
            # train_spec: TrainSpec,
            # For repeated runs the already built datasets can be reused
            premade_datasets: Optional[Tuple[ResidualDataset, ResidualDataset, ResidualDataset]] = None,
):

        self.model_spec = model_spec
        self.dataset_spec = dataset_spec
        # self.train_spec = train_spec

        # Save input arguments for saving
        self.args = locals()
        del self.args['self']

        self.dataset_name = dataset_spec.data_spec.file_name
        self.model_name = model_spec.model_name

        if premade_datasets is None:
            assert type(dataset_spec) == ResidualDatasetSpec, 'ResidualDataset requires ResidualDatasetSpec'
            self.datasets = dataset_spec.create_datasets()
        else:
            logger.info('Using premade datasets')
            self.datasets = premade_datasets

        # Define run mode, 0 is train, 1 is eval 2 is testing. Used for gradient tracking and dataset selection
        self._mode = 0

        # Initialize model
        model_class = get_model_class(model_spec.model_name)
        logger.info(f'Using {model_spec.model_name} model')
        self.model = model_class(model_spec=model_spec)
        # This is unnecessary for properly coded models, but some do not manage to do this themselves
        self.model.to(model_spec.device)

    def mode(self, mode: str) -> None:
        """Sets model mode and flag for dataset selection used to select the dataset"""
        if mode in ['train', 0]:
            self.model.train()
            self._mode = 0
        elif mode in ['valid', 1]:
            self.model.eval()
            self._mode = 1
        elif mode in ['test', 2]:
            self.model.eval()
            self._mode = 2
        else:
            raise ValueError(f'Trying to set invalid mode {mode}')

    def dataset(self) -> ResidualDataset:
        return self.datasets[self._mode]

    def train_model(self, train_spec: Optional[TrainSpec]):
        """Train self.model on self.train_set for several epochs, also obtains the validation loss and returns both"""
        # if train_spec is None:
        #     train_spec = self.train_spec

        # define training method
        train_spec.clean()
        criterion = train_spec.loss
        optimizer = train_spec.optimizer(
            self.model.parameters(),
            lr=10**train_spec.learning_rate,
            **train_spec.optimizer_args
        )

        # get datasets
        train_set = self.datasets[0]
        valid_set = self.datasets[1]

        train_dataloader = DataLoader(train_set, batch_size=train_spec.batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_set, batch_size=train_spec.batch_size, shuffle=True)

        train_loss = torch.zeros(train_spec.epochs, len(train_dataloader))
        valid_loss = torch.zeros(train_spec.epochs)

        early_stopping = EarlyStopping(patience=train_spec.patience)

        if train_spec.profiling:
            # profiling
            pr = cProfile.Profile()
            pr.enable()

        for n in range(train_spec.epochs):
            if n % 10 == 0:
                logger.info(f'Starting epoch {n+1}')

            # Training phase
            train_loss[n] = self.model.train_epoch(train_dataloader, criterion, optimizer)

            # Validation phase
            valid_loss[n] = self.model.valid_epoch(valid_dataloader, criterion)
            if n % 10 == 0:
                logger.info(f'Training loss: {train_loss[n,-1].item()}')
                logger.info(f'Validation loss: {valid_loss[n]}')

            if early_stopping(valid_loss[n], self.model):
                logger.info(f'Early stopping after epoch {n+1}')
                break

        # reset to best model
        self.model.load_state_dict(early_stopping.best_state_dict)
        best_loss = early_stopping.best_loss

        if train_spec.profiling:
            pr.disable()
            s = io.StringIO()
            sortby = 'tottime'  # SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

        return train_loss.flatten(), valid_loss, best_loss

    def predict(self, idx: int or None = None, dataset_name: str = 'test', denormalize: bool = True, raw: bool = False)\
            -> (Tensor, Tensor, Tensor):
        """Used to make sample forecasts from specified set, if raw=True residuals are returned, if used"""
        self.mode(dataset_name)
        dataset = self.dataset()
        idx = idx or randrange(0, len(dataset))
        print(idx)
        sample = dataset[idx]

        self.model.eval()
        with torch.no_grad():
            prediction = self.model.predict(*sample[:-1])

            if denormalize:
                cat_data = sample[-2] # denormalization requires category if they are separately normalized
                prediction = dataset.norm.revert_normalization(prediction, cat_data)

            prediction = prediction.squeeze(dim=0).detach()

        if denormalize:
            cat_data = sample[-2]
            if raw:
            #    prediction = dataset.norm.revert_normalization(prediction, cat_data)
                output_pslp = dataset.norm.revert_normalization(sample[3], cat_data)
            input_reference = dataset.norm.revert_normalization(sample[0], cat_data)
            output_reference = dataset.norm.revert_normalization(sample[-1], cat_data)
        else:
            input_reference = sample[0]
            output_reference = sample[-1]
            output_pslp = sample[3]

        if raw:
            output_reference -= output_pslp
            if self.model.nr_of_quantiles is not None:
                output_pslp = output_pslp.unsqueeze(-1).expand(*output_pslp.shape, self.model.nr_of_quantiles)
            print(output_reference.shape, output_pslp.shape)
            prediction -= output_pslp
        return prediction, input_reference, output_reference

    def test_forecast(self, batch_size: int = 128) -> pd.DataFrame:
        self.mode('test')
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

                if self.model.nr_of_quantiles is not None:
                    prediction = prediction[..., 0]
                output_reference = dataset.norm.revert_normalization(batch[-1], cat_data)
                batch_metrics = apply_error_metric(prediction, output_reference)

                # get mase for inhomogeneous datasets
                error_scale = dataset.get_error_scale(cat_data)
                scaled_prediction = torch.stack(list(map(torch.div, prediction, error_scale)))
                scaled_reference = torch.stack(list(map(torch.div, output_reference, error_scale)))
                scaled_metrics = apply_error_metric(scaled_prediction, scaled_reference, ['mae', 'bias'])
                batch_metrics['mase'] = scaled_metrics.mae
                batch_metrics['scaled bias'] = scaled_metrics.bias

                results = pd.concat([results, batch_metrics])

        return results.mean().to_frame().T
    
    def test_pslp(self, batch_size: int = 128) -> Tensor:
        self.mode('test')
        dataset = self.dataset()
        if batch_size > len(dataset):
            batch_size = len(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

        results = pd.DataFrame()
        with torch.no_grad():
            for batch in loader:
                cat_data = batch[-2]
                prediction = dataset.norm.revert_normalization(batch[3], cat_data)
                if self.model.nr_of_quantiles is not None:
                    prediction = prediction[..., 0]
                output_reference = dataset.norm.revert_normalization(batch[-1], cat_data)
                batch_metrics = apply_error_metric(prediction, output_reference)

                error_scale = dataset.get_error_scale(cat_data)
                scaled_prediction = torch.stack(list(map(torch.div, prediction, error_scale)))
                scaled_reference = torch.stack(list(map(torch.div, output_reference, error_scale)))
                scaled_metrics = apply_error_metric(scaled_prediction, scaled_reference, ['mae', 'bias'])
                batch_metrics['mase'] = scaled_metrics.mae
                batch_metrics['scaled bias'] = scaled_metrics.bias

                results = pd.concat([results, batch_metrics])

        return results.mean().to_frame().T

    def save_to(self, save_path: str) -> None:
        """Save trained model to [save_path]"""
        hyperparameters = self.args
        state_dict = self.model.state_dict()

        torch.save([hyperparameters, state_dict], save_path)

    @classmethod
    def load_from(cls, load_path: str):
        """Load model saved with save_to-method"""
        hyperparameters, state_dict = torch.load(load_path)
        run = cls(**hyperparameters)
        run.model.load_state_dict(state_dict)
        return run

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