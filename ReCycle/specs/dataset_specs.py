from typing import List, Optional, Type, TypeVar, Union, Tuple
import torch
from pathlib import Path

from dataclasses import dataclass
from ..data.normalizer import Normalizer, MinMax

from ..data.rhp_datasets import (
    LooseTypeRHPDataset,
    LooseTypeLastRHPDataset,
    PersistenceDataset,
)


__all__ = [
    "DataSpec",
    "DatasetSpec",
    "ResidualDatasetSpec",
]


@dataclass
class DataSpec:
    """
    Standard format to store dataset specifications. This is information considered specific to the data itself not the
    way the dataset is set up for training.

    :param Union[str, Tuple[str, str, str]] file_name: if str name of the csv-file as stored in the
        ./datasets directory. Without .csv suffix. If three str, those are interpreted as separate files for
        training, validation, and test set in that order
    :param str time_column_name: name of the column containing the time-stamp
    :param Optional[List[str]] data_column_names: name of time-series data column, if None is provided all except
        time_column names are used
    :param Optional[List[str]] metadata_column_names: names of columns with exogenous covariates or metadata_column_names
    :param Optional[str] country_code: two-letter ISO country code to determine holidays using the holidays package
    :param bool universal_holidays: only relevant if there is more than one data column. If True identical holidays
        are assumed for all countries, if False the column names must be the ISO country codes and are used to
        extract separate holidays
    :param Optional[int] downsample_rate: if not None, data is aggregate as average of non-overlapping windows of
        the specified size
    :param bool split_by_category: if True train/valid/tests set are made by selecting columns instead of dividing
        at a certain time, only use if data_column_names has sufficiently many entries
    :param bool remove_flatline: if True check the beginning of the series for consecutive 0 values and only start
        indexing once they stop
    :param str dataset_root_path: path where dataset file is stored, should generally be "./datasets/", see README
    "param str file_extension: data file extension
    """

    file_name: Union[str, Tuple[str, str, str]]
    time_column_name: str
    data_column_names: Optional[List[str]] = None
    metadata_column_names: Optional[List[str]] = None
    country_code: Optional[str] = None
    universal_holidays: bool = True
    downsample_rate: Optional[int] = None
    split_by_category: bool = False
    remove_flatline: bool = False
    root_path: str = "./datasets/"
    dataset_name: str = "custom"
    file_extension: str = ".csv"
    sep: str = ","
    decimal: str = '.'

    def full_file_path(
        self, file_extension: str = ".csv"
    ) -> Union[str, Tuple[str, str, str]]:
        # if type(self.file_name) == str:
        if isinstance(self.file_name, str):
            return self.root_path / Path(self.file_name + file_extension)
        # elif type(self.file_name) == Tuple[str, str, str]:
        elif isinstance(self.file_name, Tuple):
            assert len(self.file_name) == 3
            return tuple(
                [self.root_path + str(f) + file_extension for f in self.file_name]
            )
        else:
            raise TypeError("Invalid file_name specification")


SPEC = TypeVar("SPEC")


@dataclass
class DatasetSpec:
    """
    Standard format to store dataset specifications. This is information considered specific to the way the dataset is
    set up for training. However, the included DataSpec contains the information for the data itself

    :param int historic_window: length of the historic window
    :param int forecast_window: length of the forecast window
    :param int features_per_step: number of features of the time series, metadata_column_names features excluded
    :param DataSpec data_spec: DataSpec for data specific information
    :param Type[Normalizer] normalizer: normalizer to use for the data
    :param float train_share: fraction of the data to use for train set
    :param float test_share: fraction of the data to use for tests set
    :param Optional[float] reduce: float between 0 and 1, uses a fraction of the data for speedup during testing
    :param Optional[torch.device] device: GPU or CPU to use, if None autodetection is used
    """

    historic_window: int
    forecast_window: int
    features_per_step: int
    data_spec: DataSpec
    normalizer: Type[Normalizer] = MinMax
    train_share: float = 0.6
    test_share: float = 0.2
    reduce: Optional[float] = None
    device: Optional[torch.device] = None

    # @classmethod
    # def from_spec_name(cls: SPEC, *args, data_spec_name: str, **kwargs) -> SPEC:
    #     """Accepts the same signature as baseclass, but translates data_spec_name to a DataSpec"""
    #     # data_spec = get_data_spec(data_spec_name)
    #     data_spec = specs_dict[data_spec_name]
    #     return cls(*args, data_spec=data_spec, **kwargs)

    def check_validity(self) -> None:
        assert (
            0 <= self.train_share
        ), f"{self.train_share=} should be greater or equal 0"
        assert 0 <= self.test_share, f"{self.test_share=} should be greater or equal 0"
        assert (
            1 >= self.train_share
        ), f"{self.train_share=} should be smaller or equal 1"
        assert 1 >= self.test_share, f"{self.test_share=} should be smaller or equal 1"
        assert (self.train_share + self.test_share) <= 1, (
            f"The sum of {self.train_share=} and {self.test_share}"
            f"should be smaller than 1"
        )
        if self.reduce is not None:
            assert 0 <= self.reduce, f"{self.reduce=} should be greater or equal 0"
            assert 1 >= self.reduce, f"{self.reduce=} should be smaller or equal 1"


@dataclass
class ResidualDatasetSpec(DatasetSpec):
    residual_normalizer: Optional[Type[Normalizer]] = None
    rhp_dataset: Union[
        Type[LooseTypeRHPDataset], Type[PersistenceDataset]
    ] = LooseTypeLastRHPDataset
    rhp_cycles: int = 3
    rhp_cycle_len: int = 7
