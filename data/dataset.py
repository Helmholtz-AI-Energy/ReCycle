import torch
import pandas as pd

from operator import itemgetter
from torch.nn.functional import one_hot, l1_loss

from .normalizer import Normalizer, MinMax, AbsMax
from .data_cleaner import clean_dataframe
from .pslp_datasets import LooseTypeLastPSLPDataset
from specs.dataset_specs import ResidualDatasetSpec

from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Type, Optional, Dict, Tuple, Union, TypeVar
from pandas import DataFrame
from numpy import float32


import logging
logger = logging.getLogger(__name__)


class ResidualDataset(Dataset):
    def __init__(self, data: DataFrame, dataset_spec: ResidualDatasetSpec) -> None:
        dataset_spec.check_validity()
        data_spec = dataset_spec.data_spec

        # if reduce < 1 is given the set is reduced to the given fraction
        data = data.reset_index(drop=True)  # Quenches SettingWithCopyWarning
        logger.debug(data)
        if dataset_spec.reduce is not None:
            data = data[:int(len(data) * dataset_spec.reduce)]

        # input_data and metadata_dict column names
        self.target = data_spec.data_column_names
        self.time_column = data_spec.time_column_name
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
        logger.debug(f'{self.target=}')
        data[self.target] = data[self.target].astype(float32)

        # save type fixed dataframe
        self.dataframe = data

        # pslp specifications
        self.pslp_cycles = dataset_spec.pslp_cycles  # nr of weeks/long cycles for pslp
        self.pslp_cycle_len = dataset_spec.pslp_cycle_len  # nr of days/primary cycles per week/long cycle
        self.pslp_window = self.pslp_cycles * self.pslp_cycle_len  # nr of primary cycles in pslp

        # raw input_data as tensor, make categories first dim
        self.unnormalized_data = torch.tensor(data[self.target].values).t()

        # initialize normalizer and get normalized input_data
        self.norm = dataset_spec.normalizer(self.unnormalized_data)
        self.normalized_data = self.norm(self.unnormalized_data)

        # reshape data into cycles (days)
        self.categories = len(self.target)
        self.normalized_data = self.normalized_data.reshape(self.categories, -1, self.features_per_step)
        self.unnormalized_data = self.unnormalized_data.reshape(self.categories, -1, self.features_per_step)

        # Category input_data
        self.samples_per_category, self.start_indices = self._category_lengths()
        self.cumulative_samples = torch.cumsum(self.samples_per_category, dim=0)

        self.cat_offset = self.start_indices.clone()
        self.cat_offset[1:] -= self.cumulative_samples[:-1]
        self.cat_offset += self.pslp_window

        # generate metadata_dict and pslp
        self.metadata = self._generate_metadata()
        self.pslp_data = dataset_spec.pslp_dataset(self.dataframe, self.pslp_cycles, self.target,
                                                   self.time_column, self.norm, self._universal_holidays)

        # generate naive error estimate
        catergory_naive_mae = []
        for start, category in zip(self.start_indices, self.unnormalized_data):
            scale = 0
            valid_data = category[start+self.pslp_window:-1]
            for f in range(self.forecast_window):
                # Using some reshuffling the MAE for each day in each forecast window with respect to the last known
                # day can be calculated
                naive_forecast = valid_data[self.historic_window - 1 : -self.forecast_window]
                reference = valid_data[self.historic_window + f: f + 1 - self.forecast_window]
                if f+1 == self.forecast_window:
                    reference = valid_data[self.historic_window + f:]
                loss = l1_loss(naive_forecast, reference)
                scale += loss
            catergory_naive_mae.append(scale / self.forecast_window)

        self.catergory_naive_mae = torch.tensor(catergory_naive_mae)

        # derive residual norm, not used anymore
        # self.residual_norm = residual_normalizer(self.normalized_data - self.pslp_data.get_local_pslp())

        # send to device
        self.device = dataset_spec.device
        if self.device is not None:
            self.to(device=self.device)

        # self check that all input_data looks like it should
        self.verify()

    def to(self, device: Optional[torch.device] = None, *args, **kwargs):
        if device is not None:
            self.device = device
            logger.info(f'Transferring to {self.device}')
            print([torch.device(i) for i in range(torch.cuda.device_count())])

        # Data tensors
        self.unnormalized_data = self.unnormalized_data.to(*args, device=self.device, **kwargs)
        self.normalized_data = self.normalized_data.to(*args, device=self.device, **kwargs)
        self.metadata = self.metadata.to(*args, device=self.device, **kwargs)

        # Misc tensors
        self.samples_per_category = self.samples_per_category.to(*args, device=self.device, **kwargs)
        self.start_indices = self.start_indices.to(*args, device=self.device, **kwargs)
        self.cumulative_samples = self.cumulative_samples.to(*args, device=self.device, **kwargs)
        self.catergory_naive_mae = self.catergory_naive_mae.to(*args, device=device, **kwargs)

        # Objects that only need to transfer their members
        self.pslp_data.to(*args, device=self.device, **kwargs)
        self.norm.to(*args, device=self.device, **kwargs)
        #self.residual_norm.to(*args, device=self.device, **kwargs)

    def __len__(self) -> int:
        length = self.cumulative_samples[-1] #- self.pslp_window * self.categories
        logger.debug(f'{length=}, {self.cumulative_samples[-1]=}, {self.pslp_window=}, {self.categories=}')
        return length

    def _category_lengths(self) -> Tuple[Tensor, Tensor]:
        lengths = []
        starts = []
        for column in self.target:
            cat_data = self.dataframe[column]
            first_index = cat_data.first_valid_index()
            lengths.append(len(cat_data) - first_index)
            starts.append(first_index)

        features = torch.tensor(lengths) - self.total_window_features  # consider window length
        cat_lengths = torch.div(features, self.features_per_step, rounding_mode='floor')  # convert to days/cycles
        cat_lengths -= self.pslp_window  # consider pslp run up

        start_indices = torch.div(torch.tensor(starts), self.features_per_step, rounding_mode='floor')
        logger.debug(f'{cat_lengths=}, {start_indices=}')
        return cat_lengths.clone(), start_indices.clone()

    def _generate_metadata(self) -> Tensor:
        # get holiday metadata_dict (only first entry per timestep)
        logger.debug(f'{self.dataframe.columns=}')
        if self._universal_holidays:
            holiday_data = torch.tensor(self.dataframe.loc[0::self.features_per_step,
                                        ['holiday', 'holiday_tomorrow']].values)
            holiday_data = holiday_data.expand(self.categories, -1, -1)
        else:
            holiday_data = [torch.tensor(self.dataframe.loc[0::self.features_per_step,
                                         ['holiday_' + country, 'holiday_' + country + '_tomorrow']].values)
                            for country in self.target]
            holiday_data = torch.stack(holiday_data, dim=0)

        # get weekday annotation
        weekdays = torch.tensor(self.dataframe.loc[0::self.features_per_step, self.time_column].dt.weekday.values,
                                dtype=torch.int64)
        weekdays = one_hot(weekdays, num_classes=7).expand(self.categories, -1, -1)

        # join tensors
        logger.debug(f'{weekdays.shape=}, {holiday_data.shape=}')
        metadata = torch.cat([weekdays, holiday_data], dim=-1)
        return metadata

    def verify(self) -> None:
        assert self.unnormalized_data.shape == self.normalized_data.shape, 'Normalizer changes input_data layout'
        logger.info(f'Dataset length: {self.__len__()}')

    @staticmethod
    def prepare_dataframe(dataframe: DataFrame, metadata: Dict[str, str], target: Optional[List[str]] = None,
                          country: Optional[str] = None, downsample_rate: Optional[int] = None,
                          remove_flatline: bool = False):
        if target is None:
            logger.info('Inferring target')
            # if target is None all columns not specified as metadata_dict are considered target columns
            target = list(dataframe.columns)
            for item in metadata.values():
                logger.debug(f'{item}, {metadata}, {target}')
                target.remove(item)

        output = clean_dataframe(dataframe, target, metadata, downsample_rate, country, remove_flatline)
        return output, target

    @staticmethod
    def split_data(full_data: DataFrame, target: List[str], training_share: float, testing_share: float,
                   load_features: Optional[int] = None, by_category: bool = False) -> (Tuple[DataFrame], Tuple[List]):
        assert training_share + testing_share <= 1, 'Invalid dataset split, total larger than 1.'
        logger.info('Splitting dataset')

        if by_category:
            tests_data = train_data = valid_data = full_data
            nr_of_cats = len(target)
            train_len = int(nr_of_cats * training_share)
            tests_len = int(nr_of_cats * testing_share)
            valid_len = nr_of_cats - (train_len + tests_len)

            perm = torch.randperm(nr_of_cats).split([train_len, valid_len, tests_len])
            train_target = list(itemgetter(*perm[0])(target))
            valid_target = list(itemgetter(*perm[1])(target))
            tests_target = list(itemgetter(*perm[2])(target))

        else:
            tests_target = train_target = valid_target = target

            # Obtain sizes for training, validation and test set
            nr_of_days = len(full_data) // load_features
            train_len = int(nr_of_days * training_share) * load_features
            tests_len = - int(nr_of_days * testing_share) * load_features
            if len(full_data) % load_features != 0:
                logger.warning('Dataset does not contain a flat number of cycles')

            # Split residuals accordingly
            train_data = full_data[:train_len]
            valid_data = full_data[train_len:tests_len]
            tests_data = full_data[tests_len:nr_of_days * load_features]

        return (train_data, valid_data, tests_data), (train_target, valid_target, tests_target)

    SET = TypeVar('SET')
    @classmethod
    def from_spec(cls: SET, spec: ResidualDatasetSpec) -> Tuple[SET, SET, SET]:
        data_spec = spec.data_spec

        metadata_dict = dict(time=data_spec.time_column_name)

        return cls.from_csv(
            file=data_spec.full_file_path(file_extension='.csv'),
            historic_window=spec.historic_window,
            forecast_window=spec.forecast_window,
            features_per_step=spec.features_per_step,
            metadata_dict=metadata_dict,
            target=data_spec.data_column_names,
            normalizer=spec.normalizer,
            residual_normalizer=spec.residual_normalizer,
            pslp_dataset=spec.pslp_dataset,
            pslp_cycles=spec.pslp_cycles,
            pslp_cycle_len=spec.pslp_cycle_len,
            country=data_spec.country_code,
            downsample_rate=data_spec.downsample_rate,
            remove_flatline=data_spec.remove_flatline,
            split_by_category=data_spec.split_by_category,
            train_share=spec.train_share,
            tests_share=spec.tests_share,
            reduce=spec.reduce,
            device=spec.device
        )

    @classmethod
    def from_csv(cls: SET, file: Union[str, List[str]],
                 historic_window: int,
                 forecast_window: int,
                 features_per_step: int,
                 metadata_dict: Dict[str, str],
                 target: Optional[List[str]] = None,
                 normalizer: Type[Normalizer] = MinMax,
                 residual_normalizer: Type[Normalizer] = AbsMax,
                 pslp_dataset = LooseTypeLastPSLPDataset,
                 pslp_cycles: int = 3,
                 pslp_cycle_len: int = 7,
                 country: Optional[str] = None,
                 downsample_rate: Optional[int] = None,
                 remove_flatline: bool = False,
                 split_by_category: bool = False,
                 train_share: float = 0.6,
                 tests_share: float = 0.2,
                 reduce: Optional[float] = None,
                 device: Optional[torch.device] = None) -> Tuple[SET, SET, SET]:
        """

        :param file: dataset file or tuple of (training input_data file, validation input_data file, test input_data file)
        :param historic_window:
        :param forecast_window:
        :param features_per_step:
        :param metadata_dict:
        :param target:
        :param normalizer:
        :param residual_normalizer:
        :param pslp_window:
        :param country:
        :param downsample_rate:
        :param remove_flatline:
        :param split_by_category:
        :param train_share:
        :param tests_share:
        :param reduce:
        :return:
        """
        logger.info('Reading from csv')

        if isinstance(file, str):
            full_data = pd.read_csv(file)
            full_data, target = cls.prepare_dataframe(full_data, metadata_dict, target, country,
                                                      downsample_rate, remove_flatline)
            logger.info('Dataframe prepared')
            dataframes, targets = cls.split_data(full_data=full_data,
                                                 target=target,
                                                 training_share=train_share,
                                                 testing_share=tests_share,
                                                 load_features=features_per_step,
                                                 by_category=split_by_category)
        elif isinstance(file, List):
            assert len(file) == 3, 'Must proved exactly 3 paths for train, validation and test set'
            dataframes = []
            targets = []
            for f in file:
                data = pd.read_csv(f)
                f, t = cls.prepare_dataframe(data, metadata_dict, target, country,
                                             downsample_rate, remove_flatline)
                dataframes.append(f)
                targets.append(t)
            logger.info('Dataframes prepared')
        else:
            raise TypeError(f'Invalid file specification: {file}')

        train_data, valid_data, tests_data = dataframes
        train_target, valid_target, tests_target = targets

        data_args = dict(
            historic_window=historic_window,
            forecast_window=forecast_window,
            features_per_step=features_per_step,
            metadata=metadata_dict,
            universal_holidays=(country is not None),
            normalizer=normalizer,
            residual_normalizer=residual_normalizer,
            pslp_dataset=pslp_dataset,
            pslp_cycles=pslp_cycles,
            pslp_cycle_len=pslp_cycle_len,
            reduce=reduce,
            device=device
        )

        train_set = cls(train_data, target=train_target, **data_args)
        logger.info('Training input_data loaded')
        valid_set = cls(valid_data, target=valid_target, **data_args)
        logger.info('Validation input_data loaded')
        tests_set = cls(tests_data, target=tests_target, **data_args)
        logger.info('Test input_data loaded')

        return train_set, valid_set, tests_set

    def get_error_scale(self, cat_index: Tensor) -> Tensor:
        #if self.catergory_naive_mae == torch.empty(0):

        return self.catergory_naive_mae[cat_index]

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        cat_index = torch.searchsorted(self.cumulative_samples, item, right=True)  # finds index of first entry > item
        item += self.cat_offset[cat_index]  # adjust index for category, pslp, and internal start index

        historic_slice = slice(item, item + self.historic_window)
        forecast_slice = slice(item + self.historic_window, item + self.total_window)
        logger.debug(f'{historic_slice=}, {forecast_slice=}')

        historic_data = self.normalized_data[cat_index, historic_slice].clone()
        reference = self.normalized_data[cat_index, forecast_slice].clone()

        historic_pslp, pslp_forecast = self.pslp_data.get_pslp(cat_index, historic_slice, forecast_slice)
        historic_pslp = historic_pslp.clone()
        pslp_forecast = pslp_forecast.clone()

        historic_metadata = self.metadata[cat_index, historic_slice].clone()
        forecast_metadata = self.metadata[cat_index, forecast_slice].clone()

        # TODO: check if cat_index get correct device automatically
        #cat_index = torch.tensor(cat_index, device=self.device)

        return historic_data, historic_pslp, historic_metadata, pslp_forecast, forecast_metadata, cat_index, reference
