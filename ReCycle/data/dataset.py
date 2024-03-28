import torch
import pandas as pd

from operator import itemgetter
from torch.nn.functional import one_hot, l1_loss

# from .normalizer import Normalizer, MinMax, AbsMax
from .data_cleaner import clean_dataframe
from ..specs.dataset_specs import ResidualDatasetSpec, DatasetSpec

from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, TypeVar, Union
from pandas import DataFrame
from numpy import float32


import logging

logger = logging.getLogger(__name__)


D = TypeVar("D")


class ResidualDataset(Dataset):
    def __init__(self, data: DataFrame, dataset_spec: ResidualDatasetSpec) -> None:
        dataset_spec.check_validity()
        self.dataset_spec = dataset_spec
        data_spec = dataset_spec.data_spec

        # if reduce < 1 is given the set is reduced to the given fraction
        data = data.reset_index(drop=True)  # Quenches SettingWithCopyWarning
        logger.debug(data)
        if dataset_spec.reduce is not None:
            data = data[: int(len(data) * dataset_spec.reduce)]

        # input_data and metadata_dict column names
        self.target = data_spec.data_column_names
        self.time_column = data_spec.time_column_name
        self.meta_columns = data_spec.metadata_column_names
        self._universal_holidays = data_spec.universal_holidays

        # input_data shape
        self.historic_window = dataset_spec.historic_window
        self.forecast_window = dataset_spec.forecast_window
        self.total_window = self.historic_window + self.forecast_window
        self.features_per_step = dataset_spec.features_per_step
        self.stride = self.features_per_step

        # meta features
        self.historic_features = self.historic_window * self.features_per_step
        self.forecast_features = self.forecast_window * self.features_per_step
        self.total_window_features = self.historic_features + self.forecast_features

        # Set datatype
        logger.debug(f"{self.target=}")
        data[self.target] = data[self.target].astype(float32)

        # save type fixed dataframe
        self.dataframe = data

        # rhp specifications
        self.rhp_cycles = dataset_spec.rhp_cycles  # nr of weeks/long cycles for rhp
        self.rhp_cycle_len = (
            dataset_spec.rhp_cycle_len
        )  # nr of days/primary cycles per week/long cycle
        self.rhp_window = (
            self.rhp_cycles * self.rhp_cycle_len
        )  # nr of primary cycles in rhp

        if len(data) / self.features_per_step < self.rhp_window + self.total_window:
            raise ValueError(
                f"Not enough data to realize the given historic ({self.historic_window}), forecast ({self.forecast_window}), and rhp ({self.rhp_window}) window sizes. There seems to be only enough data for {len(data) // self.features_per_step} samples"
            )

        # raw input_data as tensor, make categories first dim
        self.unnormalized_data = torch.tensor(data[self.target].values).t()

        # initialize normalizer and get normalized input_data
        self.norm = dataset_spec.normalizer(self.unnormalized_data)
        self.normalized_data = self.norm(self.unnormalized_data)

        # reshape data into cycles (days)
        self.categories = len(self.target)
        self.normalized_data = self.normalized_data.reshape(
            self.categories, -1, self.features_per_step
        )
        self.unnormalized_data = self.unnormalized_data.reshape(
            self.categories, -1, self.features_per_step
        )

        if self.meta_columns is not None:
            self.metadata = torch.tensor(
                data[self.meta_columns].values, dtype=torch.float32
            ).t()
            self.metadata = self.metadata.reshape(
                len(self.meta_columns), -1, self.features_per_step
            )

        # Category input_data
        self.samples_per_category, self.start_indices = self._category_lengths()
        self.cumulative_samples = torch.cumsum(self.samples_per_category, dim=0)

        self.cat_offset = self.start_indices.clone()
        self.cat_offset[1:] -= self.cumulative_samples[:-1]
        self.cat_offset += self.rhp_window

        # generate metadata_dict and rhp
        self.day_metadata = self._generate_day_metadata()
        self.rhp_data = dataset_spec.rhp_dataset(
            self.dataframe,
            self.rhp_cycles,
            self.target,
            self.time_column,
            self.norm,
            self._universal_holidays,
        )

        # generate naive error estimate
        category_naive_mae = []
        for start, category in zip(self.start_indices, self.unnormalized_data):
            scale = 0
            valid_data = category[start + self.rhp_window : -1]
            for f in range(self.forecast_window):
                # Using some reshuffling the MAE for each day in each forecast window with respect to the last known
                # day can be calculated
                naive_forecast = valid_data[
                    self.historic_window - 1 : -self.forecast_window
                ]
                reference = valid_data[
                    self.historic_window + f : f + 1 - self.forecast_window
                ]
                if f + 1 == self.forecast_window:
                    reference = valid_data[self.historic_window + f :]
                loss = l1_loss(naive_forecast, reference)
                scale += loss
            category_naive_mae.append(scale / self.forecast_window)

        self.category_naive_mae = torch.tensor(category_naive_mae)

        # derive residual norm, not used anymore
        # self.residual_norm = residual_normalizer(self.normalized_data - self.rhp_data.get_local_rhp())

        # send to device
        self.device = dataset_spec.device
        if self.device is not None:
            self.to(device=self.device)

        # self check that all input_data looks like it should
        self.verify()

    def to(self, device: Optional[torch.device] = None, *args, **kwargs):
        if device is not None:
            self.device = device
            logger.info(f"Transferring to {self.device}")

        # Data tensors
        self.unnormalized_data = self.unnormalized_data.to(
            *args, device=self.device, **kwargs
        )
        self.normalized_data = self.normalized_data.to(
            *args, device=self.device, **kwargs
        )
        self.day_metadata = self.day_metadata.to(*args, device=self.device, **kwargs)

        # Misc tensors
        self.samples_per_category = self.samples_per_category.to(
            *args, device=self.device, **kwargs
        )
        self.start_indices = self.start_indices.to(*args, device=self.device, **kwargs)
        self.cumulative_samples = self.cumulative_samples.to(
            *args, device=self.device, **kwargs
        )
        self.category_naive_mae = self.category_naive_mae.to(
            *args, device=device, **kwargs
        )

        # Objects that only need to transfer their members
        self.rhp_data.to(*args, device=self.device, **kwargs)
        self.norm.to(*args, device=self.device, **kwargs)
        # self.residual_norm.to(*args, device=self.device, **kwargs)

    def __len__(self) -> int:
        length = self.cumulative_samples[-1]  # - self.rhp_window * self.categories
        logger.debug(
            f"{length=}, {self.cumulative_samples[-1]=}, {self.rhp_window=}, {self.categories=}"
        )
        return length

    def _category_lengths(self) -> Tuple[Tensor, Tensor]:
        lengths = []
        starts = []
        for column in self.target:
            cat_data = self.dataframe[column]
            first_index = cat_data.first_valid_index()
            lengths.append(len(cat_data) - first_index)
            starts.append(first_index)

        features = (
            torch.tensor(lengths) - self.total_window_features
        )  # consider window length
        cat_lengths = torch.div(
            features, self.features_per_step, rounding_mode="floor"
        )  # convert to days/cycles
        cat_lengths -= self.rhp_window  # consider rhp run up

        start_indices = torch.div(
            torch.tensor(starts), self.features_per_step, rounding_mode="floor"
        )
        logger.debug(f"{cat_lengths=}, {start_indices=}")
        return cat_lengths.clone(), start_indices.clone()

    def _generate_day_metadata(self) -> Tensor:
        # get holiday metadata_dict (only first entry per timestep)
        logger.debug(f"{self.dataframe.columns=}")
        if self._universal_holidays:
            holiday_data = torch.tensor(
                self.dataframe.loc[
                    0 :: self.features_per_step, ["holiday", "holiday_tomorrow"]
                ].values
            )
            holiday_data = holiday_data.expand(self.categories, -1, -1)
        else:
            holiday_data = [
                torch.tensor(
                    self.dataframe.loc[
                        0 :: self.features_per_step,
                        ["holiday_" + country, "holiday_" + country + "_tomorrow"],
                    ].values
                )
                for country in self.target
            ]
            holiday_data = torch.stack(holiday_data, dim=0)

        # get weekday annotation
        weekdays = torch.tensor(
            self.dataframe.loc[
                0 :: self.features_per_step, self.time_column
            ].dt.weekday.values,
            dtype=torch.int64,
        )
        weekdays = one_hot(weekdays, num_classes=7).expand(self.categories, -1, -1)

        # join tensors
        logger.debug(f"{weekdays.shape=}, {holiday_data.shape=}")
        day_metadata = torch.cat([weekdays, holiday_data], dim=-1)
        return day_metadata

    def verify(self) -> None:
        assert (
            self.unnormalized_data.shape == self.normalized_data.shape
        ), "Normalizer changes input_data layout"
        logger.info(f"Dataset length: {self.__len__()}")

    @staticmethod
    def split_data(
        full_data: DataFrame, dataset_spec: DatasetSpec
    ) -> (Tuple[DataFrame], Tuple[List]):
        logger.info("Splitting dataset")
        dataset_spec.check_validity()
        target = dataset_spec.data_spec.data_column_names

        if dataset_spec.data_spec.split_by_category:
            tests_data = train_data = valid_data = full_data
            nr_of_cats = len(target)
            train_len = int(nr_of_cats * dataset_spec.train_share)
            tests_len = int(nr_of_cats * dataset_spec.test_share)
            valid_len = nr_of_cats - (train_len + tests_len)

            perm = torch.randperm(nr_of_cats).split([train_len, valid_len, tests_len])
            train_target = list(itemgetter(*perm[0])(target))
            valid_target = list(itemgetter(*perm[1])(target))
            tests_target = list(itemgetter(*perm[2])(target))

        else:
            tests_target = train_target = valid_target = target

            # Obtain sizes for training, validation and test set
            nr_of_days = len(full_data) // dataset_spec.features_per_step
            train_len = (
                int(nr_of_days * dataset_spec.train_share)
                * dataset_spec.features_per_step
            )
            tests_len = (
                -int(nr_of_days * dataset_spec.test_share)
                * dataset_spec.features_per_step
            )
            if len(full_data) % dataset_spec.features_per_step != 0:
                logger.warning("Dataset does not contain a flat number of cycles")

            # Split residuals accordingly
            train_data = full_data[:train_len]
            valid_data = full_data[train_len:tests_len]
            tests_data = full_data[
                tests_len : nr_of_days * dataset_spec.features_per_step
            ]

        return (train_data, valid_data, tests_data), (
            train_target,
            valid_target,
            tests_target,
        )

    SET = TypeVar("SET")

    @classmethod
    def from_dataframe(
        cls: SET,
        data: Optional[Union[DataFrame, Tuple[DataFrame, DataFrame, DataFrame]]],
        dataset_spec: ResidualDatasetSpec,
    ) -> Tuple[SET, SET, SET]:
        if isinstance(data, DataFrame):
            full_data, data_column_names = clean_dataframe(
                df=data, data_spec=dataset_spec.data_spec
            )
            assert (
                dataset_spec.data_spec.data_column_names == data_column_names
            ), "Turns out it is pass by copy"
            logger.info("Dataframe prepared")
            dataframes, targets = cls.split_data(
                full_data=full_data, dataset_spec=dataset_spec
            )
        elif isinstance(data, List):
            assert (
                len(data) == 3
            ), "Must provide exactly 3 paths for train, validation and test set"
            dataframes = []
            targets = []
            for df in data:
                d, t = clean_dataframe(df=df, data_spec=dataset_spec.data_spec)
                dataframes.append(d)
                targets.append(t)
            logger.info("Dataframes prepared")
        else:
            raise ValueError("Exepcts a DataFrame or list of three dataframes")

        train_data, valid_data, tests_data = dataframes
        train_target, valid_target, tests_target = targets

        dataset_spec.data_spec.data_column_names = train_target
        train_set = cls(data=train_data, dataset_spec=dataset_spec)
        logger.info("Training input_data loaded")
        dataset_spec.data_spec.data_column_names = valid_target
        valid_set = cls(data=valid_data, dataset_spec=dataset_spec)
        logger.info("Validation input_data loaded")
        dataset_spec.data_spec.data_column_names = tests_target
        tests_set = cls(data=tests_data, dataset_spec=dataset_spec)
        logger.info("Test input_data loaded")

        return train_set, valid_set, tests_set

    @classmethod
    def from_csv(cls: SET, dataset_spec: ResidualDatasetSpec) -> Tuple[SET, SET, SET]:
        logger.info("Reading from csv")

        file = dataset_spec.data_spec.full_file_path(
            file_extension=dataset_spec.data_spec.file_extension
        )
        if isinstance(file, str):
            full_data = pd.read_csv(
                file,
                sep=dataset_spec.data_spec.sep,
                decimal=dataset_spec.data_spec.decimal,
            )
            full_data, data_column_names = clean_dataframe(
                df=full_data, data_spec=dataset_spec.data_spec
            )
            assert (
                dataset_spec.data_spec.data_column_names == data_column_names
            ), "Turns out it is pass by copy"
            logger.info("Dataframe prepared")
            dataframes, targets = cls.split_data(
                full_data=full_data, dataset_spec=dataset_spec
            )
        elif isinstance(file, List):
            assert (
                len(file) == 3
            ), "Must provide exactly 3 paths for train, validation and test set"
            dataframes = []
            targets = []
            for f in file:
                data = pd.read_csv(
                    f,
                    sep=dataset_spec.data_spec.sep,
                    decimal=dataset_spec.data_spec.decimal,
                )
                f, t = clean_dataframe(df=data, data_spec=dataset_spec.data_spec)
                dataframes.append(f)
                targets.append(t)
            logger.info("Dataframes prepared")
        else:
            raise TypeError(f"Invalid file specification: {file}")

        train_data, valid_data, tests_data = dataframes
        train_target, valid_target, tests_target = targets

        dataset_spec.data_spec.data_column_names = train_target
        train_set = cls(data=train_data, dataset_spec=dataset_spec)
        logger.info("Training input_data loaded")
        dataset_spec.data_spec.data_column_names = valid_target
        valid_set = cls(data=valid_data, dataset_spec=dataset_spec)
        logger.info("Validation input_data loaded")
        dataset_spec.data_spec.data_column_names = tests_target
        tests_set = cls(data=tests_data, dataset_spec=dataset_spec)
        logger.info("Test input_data loaded")

        return train_set, valid_set, tests_set

    def get_error_scale(self, cat_index: Tensor) -> Tensor:
        # TODO still needed?
        # if self.category_naive_mae == torch.empty(0):

        return self.category_naive_mae[cat_index]

    def __getitem__(
        self, item: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        cat_index = torch.searchsorted(
            self.cumulative_samples, item, right=True
        )  # finds index of first entry > item
        item += self.cat_offset[
            cat_index
        ]  # adjust index for category, rhp, and internal start index

        historic_slice = slice(item, item + self.historic_window)
        forecast_slice = slice(item + self.historic_window, item + self.total_window)
        logger.debug(f"{historic_slice=}, {forecast_slice=}")

        historic_data = self.normalized_data[cat_index, historic_slice].clone()
        reference = self.normalized_data[cat_index, forecast_slice].clone()

        historic_rhp, rhp_forecast = self.rhp_data.get_rhp(
            cat_index, historic_slice, forecast_slice
        )
        historic_rhp = historic_rhp.clone()
        rhp_forecast = rhp_forecast.clone()

        historic_day_metadata = self.day_metadata[cat_index, historic_slice].clone()
        forecast_day_metadata = self.day_metadata[cat_index, forecast_slice].clone()

        # TODO use/assume one feature per cycle, e.g. 3 instead of 3*24
        if self.meta_columns is not None:
            historic_metadata = self.metadata[:, historic_slice].clone()
            forecast_metadata = self.metadata[:, historic_slice].clone()

            historic_metadata = torch.permute(historic_metadata, (1, 0, 2)).resize(
                self.historic_window, len(self.meta_columns) * self.features_per_step
            )
            forecast_metadata = torch.permute(forecast_metadata, (1, 0, 2)).resize(
                self.forecast_window, len(self.meta_columns) * self.features_per_step
            )

            return (
                historic_data,
                historic_rhp,
                historic_day_metadata,
                rhp_forecast,
                forecast_day_metadata,
                cat_index,
                reference,
                historic_metadata,
                forecast_metadata,
            )
        else:
            # TODO: check if cat_index get correct device automatically
            # cat_index = torch.tensor(cat_index, device=self.device)
            return (
                historic_data,
                historic_rhp,
                historic_day_metadata,
                rhp_forecast,
                forecast_day_metadata,
                cat_index,
                reference,
            )
