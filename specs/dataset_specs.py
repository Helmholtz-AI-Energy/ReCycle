from dataclasses import dataclass
import torch

from typing import List, Optional, Type, TypeVar, Union, Tuple
from data.normalizer import Normalizer, MinMax, AbsMax
from data.pslp_datasets import LooseTypePSLPDataset, LooseTypeLastPSLPDataset, PersistenceDataset


__all__ = [
    'DataSpec',
    'get_data_spec',
    'DatasetSpec',
    'ResidualDatasetSpec',
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
    :param Optional[str] xlabel: label of the time axis for plotting
    :param Optional[str] ylabel: label of the value axis for plotting
    :param str root_path: path where dataset file is stored, should generally be "./datasets/", see README
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
    xlabel: Optional[str] = "Time [d]"
    ylabel: Optional[str] = "Consumption"
    root_path: str = "./datasets/"

    def full_file_path(self, file_extension: str = '.csv') -> Union[str, Tuple[str, str, str]]:
        if type(self.file_name) == str:
            return self.root_path + self.file_name + file_extension
        elif type(self.file_name) == Tuple[str, str, str]:
            return tuple([self.root_path + f + file_extension for f in self.file_name])
        else:
            raise TypeError('Invalid file_name specification')


SPEC = TypeVar('SPEC')


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
    :param float tests_share: fraction of the data to use for tests set
    :param Optional[float] reduce: float between 0 and 1, uses a fraction of the data for speedup during testing
    :param Optional[torch.device] device: GPU or CPU to use, if None autodetection is used
    """
    historic_window: int
    forecast_window: int
    features_per_step: int
    data_spec: DataSpec
    normalizer: Type[Normalizer] = MinMax
    train_share: float = 0.6
    tests_share: float = 0.2
    reduce: Optional[float] = None
    device: Optional[torch.device] = None

    @classmethod
    def from_spec_name(cls: SPEC, *args, data_spec_name: str, **kwargs) -> SPEC:
        """Accepts the same signature as baseclass, but translates data_spec_name to a DataSpec"""
        data_spec = get_data_spec(data_spec_name)
        return cls(*args, data_spec=data_spec, **kwargs)

    def check_validity(self) -> None:
        assert 0 <= self.train_share, f'{self.train_share=} should be greater or equal 0'
        assert 0 <= self.tests_share, f'{self.tests_share=} should be greater or equal 0'
        assert 1 >= self.train_share, f'{self.train_share=} should be smaller or equal 1'
        assert 1 >= self.tests_share, f'{self.tests_share=} should be smaller or equal 1'
        assert (self.train_share + self.tests_share) <= 1, (f'The sum of {self.train_share=} and {self.tests_share}'
                                                            f'should be smaller than 1')
        if self.reduce is not None:
            assert 0 <= self.reduce, f'{self.reduce=} should be greater or equal 0'
            assert 1 >= self.reduce, f'{self.reduce=} should be smaller or equal 1'

    # TODO: there should be some mechanism to use different dataset classes
    def create_datasets(self, dataset_class: Optional[Type['TimeSeriesDataset']] = None)\
            -> Tuple['TimeSeriesDataset', 'TimeSeriesDataset', 'TimeSeriesDataset']:
        self.check_validity()

        if dataset_class is None:
            from data import ResidualDataset
            dataset_class = ResidualDataset

        return dataset_class.from_csv(dataset_spec=self)


@dataclass
class ResidualDatasetSpec(DatasetSpec):
    residual_normalizer: Optional[Type[Normalizer]] = None
    pslp_dataset: Union[Type[LooseTypePSLPDataset], Type[PersistenceDataset]] = LooseTypeLastPSLPDataset
    pslp_cycles: int = 3
    pslp_cycle_len: int = 7


entsoe_de = DataSpec(file_name="entsoe_de",
                     country_code="de",
                     data_column_names=["load"],
                     time_column_name="start",
                     downsample_rate=4,
                     ylabel="Load [MW]")

entsoe_full = DataSpec(file_name="entsoe_full",
                       country_code=None,
                       universal_holidays=False,
                       data_column_names=["fr", "de", "no", "gb", "se", "ie", "it", "es", "pt", "ch", "at", "dk", "nl", "be"],
                       time_column_name="Time [s]",
                       ylabel="Load [MW]")

water = DataSpec(file_name="water",
                 country_code="de",
                 data_column_names=["Consumption"],
                 time_column_name="Date",
                 ylabel="Water Consumption [a.u.]")

uci_pt = DataSpec(file_name="uci_pt",
                  country_code="pt",
                  data_column_names=["MT_320"],
                  time_column_name="datetime",
                  split_by_category=False,
                  remove_flatline=True,
                  ylabel="Load [kW]")

informer_etth1 = DataSpec(file_name="etth1",
                          country_code="cn",
                          data_column_names=["OT"],
                          time_column_name="date",
                          ylabel='Temperature [$^\circ$C]')

informer_etth2 = DataSpec(file_name="etth2",
                          country_code="cn",
                          data_column_names=["OT"],
                          time_column_name="date",
                          ylabel='Temperature [$^\circ$C]')

minigrid = DataSpec(file_name="minigrid",
                    country_code="de",
                    data_column_names=["Load"],
                    time_column_name="date",
                    downsample_rate=None,
                    ylabel=' Consumption [kWh]')

solar = DataSpec(file_name="solar",
                 country_code="us",
                 data_column_names=["solar_mw"],
                 time_column_name="Datetime")

prices = DataSpec(file_name="prices",
                  country_code="de",
                  data_column_names=["Day-ahead Price [EUR/MWh]"],
                  time_column_name="MTU (CET/CEST)",
                  downsample_rate=4)

traffic = DataSpec(file_name="traffic_new",
                   country_code="us",
                   data_column_names=["VMT (Veh-Miles)"],
                   time_column_name="Hour")

former_traffic = DataSpec(file_name="traffic_old",
                          country_code="us",
                          data_column_names=["OT"],
                          time_column_name="date")

specs_dict = dict(entsoe_de=entsoe_de,
                  entsoe_full=entsoe_full,
                  water=water,
                  uci_pt=uci_pt,
                  etth1=informer_etth1,
                  etth2=informer_etth2,
                  minigrid=minigrid,
                  solar=solar,
                  prices=prices,
                  traffic=traffic,
                  former_traffic=former_traffic)


def get_data_spec(name: str) -> DataSpec:
    assert name in specs_dict, f'There is no know dataset specification {name}'
    spec = specs_dict[name]
    return spec
