import torch
import pandas as pd

from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Tuple
from pandas import DataFrame

from .normalizer import Normalizer

import logging
logger = logging.getLogger(__name__)


class LooseTypePSLPDataset(Dataset):
    def __init__(self,
                 input_data: DataFrame,
                 pslp_cycles: int,
                 target: List[str],
                 time_column: str,
                 normalizer: Normalizer,
                 universal_holidays: bool) -> None:
        """This dataset type assumes preprocessing from clean_dataframe and the resulting holiday annotation"""
        self.pslp_window = pslp_cycles

        grouped_data = input_data.copy()
        grouped_data.loc[:, 'date'] = pd.to_datetime(input_data.loc[:, time_column]).dt.date
        grouped_data = grouped_data.groupby('date')

        pslp_data = []
        day_type_data = []
        group_len = grouped_data.size().values[0]
        assert (grouped_data.size().values == group_len).all(), 'Inconsistent day lengths'

        # storage for most recent pslp of each type
        last_pslp = torch.zeros(3, group_len)

        for country in target:
            reference_days = [[], [], []]
            country_pslp = []
            country_day_type = []
            data_column = country

            # get annotation column name
            if universal_holidays:
                holiday_column = 'holiday'
            else:
                holiday_column = 'holiday_' + country

            for group in grouped_data:
                date, data = group
                day_type = self.get_day_type(date, data.loc[:, holiday_column].iloc[0])
                reference_days[day_type] = self.update_ref_day(reference_days[day_type],
                                                               torch.tensor(data[data_column].values))

                if len(reference_days[day_type]) != 0:
                    last_pslp[day_type] = torch.nanmean(torch.stack(reference_days[day_type], dim=0), dim=0)
                country_pslp.append(last_pslp.clone())
                country_day_type.append(day_type)
            pslp_data.append(torch.stack(country_pslp, dim=0).clone())
            day_type_data.append(torch.tensor(country_day_type))
        self.pslp = normalizer(torch.stack(pslp_data, dim=0))
        self.day_types = torch.stack(day_type_data, dim=0)

        self.cycle_len = self.pslp.shape[-1]

    def to(self, *args, **kwargs) -> None:
        self.pslp = self.pslp.to(*args, **kwargs)
        self.day_types = self.day_types.to(*args, **kwargs)

    def update_ref_day(self, type_ref: List[Tensor], new_value: Tensor) -> List[Tensor]:
        if ref_len := len(type_ref) == self.pslp_window:
            type_ref.append(new_value)
            type_ref = type_ref[1:]
        elif ref_len in range(2, self.pslp_window):
            type_ref.append(new_value)
        elif ref_len == 1:
            type_ref.append(new_value)
        elif ref_len == 0:
            if torch.isnan(new_value).any():
                pass
            else:
                type_ref.append(new_value)
        else:
            raise RuntimeError('Something is really strange happened during processing')
        return type_ref

    @staticmethod
    def get_day_type(date: pd.Timestamp, holiday: bool) -> int:
        if holiday or date.weekday() == 6:
            day_type = 2
        elif date.weekday() < 5:
            day_type = 0
        elif date.weekday() == 5:
            day_type = 1
        else:
            raise RuntimeError('Something is really strange with the dates')
        return day_type

    def get_local_pslp(self) -> Tensor:
        day_selector_matrix = self.day_types[:, :, None, None].expand(-1, -1, -1, self.cycle_len)
        local_pslp = torch.gather(self.pslp, 2, day_selector_matrix).squeeze(-2)
        return local_pslp

    def get_pslp(self, country_idx: int, historic_slice: slice, forecast_slice: slice) -> Tuple[Tensor, Tensor]:
        pslp = self.pslp[country_idx, historic_slice]

        # get historic window (each uses most current pslp)
        historic_day_types = self.day_types[country_idx, historic_slice]
        day_selector_matrix = historic_day_types[:, None, None].expand(-1, -1, self.cycle_len)
        historic_pslp = torch.gather(pslp, 1, day_selector_matrix).squeeze(1)

        # get forecast (all use last historic pslp)
        last_pslp = pslp[-1]
        forecast_day_types = self.day_types[country_idx, forecast_slice]
        forecast_pslp = last_pslp[forecast_day_types]

        return historic_pslp, forecast_pslp


class LooseTypeLastPSLPDataset(LooseTypePSLPDataset):
    # This variant uses the last PSLP for forcast and historic.
    # So earlier days in historic will form residuals with future information,
    # but all residuals are based on the same reference
    def get_pslp(self, country_idx: int, historic_slice: slice, forecast_slice: slice) -> Tuple[Tensor, Tensor]:
        pslp = self.pslp[country_idx, historic_slice]
        last_pslp = pslp[-1]

        # get historic window
        historic_day_types = self.day_types[country_idx, historic_slice]
        historic_pslp = last_pslp[historic_day_types]

        # get forecast
        forecast_day_types = self.day_types[country_idx, forecast_slice]
        forecast_pslp = last_pslp[forecast_day_types]

        return historic_pslp, forecast_pslp


class PersistenceDataset(Dataset):
    def __init__(self,
                 input_data: DataFrame,
                 pslp_cycles: int,
                 target: List[str],
                 time_column: str,
                 normalizer: Normalizer,
                 universal_holidays: bool) -> None:
        # group to extract day_length = group_len
        grouped_data = input_data.copy()
        grouped_data.loc[:, 'date'] = pd.to_datetime(input_data.loc[:, time_column]).dt.date
        grouped_data = grouped_data.groupby('date')
        group_len = grouped_data.size().values[0]

        # normalize and reshape
        unnormalized_data = torch.tensor(input_data[target].values).t()
        normalized_data = normalizer(unnormalized_data)
        self.persistence = normalized_data.reshape(len(target), -1, group_len)

        persistence = self.persistence[:, :-1, :].clone()
        self.persistence[:, 1:, :] = persistence

    def to(self, *args, **kwargs) -> None:
        self.persistence = self.persistence.to(*args, **kwargs)

    def get_local_pslp(self) -> Tensor:
        return self.persistence

    def get_pslp(self, country_idx: int, historic_slice: slice, forecast_slice: slice) -> Tuple[Tensor, Tensor]:
        historic_persistence = self.persistence[country_idx, historic_slice]
        forecast_raw = self.persistence[country_idx, forecast_slice]
        forecast_len = len(forecast_raw)

        forecast_persistence = forecast_raw[0].expand(forecast_len, -1)

        return historic_persistence, forecast_persistence


class TendayDataset(PersistenceDataset):
    def __init__(self,
                 input_data: DataFrame,
                 pslp_cycles: int,
                 target: List[str],
                 time_column: str,
                 normalizer: Normalizer,
                 universal_holidays: bool) -> None:
        super().__init__(input_data=input_data,
                         pslp_cycles=pslp_cycles,
                         target=target,
                         time_column=time_column,
                         normalizer=normalizer,
                         universal_holidays=universal_holidays)
        pslp_cycles = 10 # hardcoded since I'm too lazy to adapt scripts to other cycles but still be adaptable later
        pslp_cycles -= 1 # first step is done by base class

        gather_tensor = self.persistence.unsqueeze(0)
        persistence = gather_tensor.clone()
        offset = persistence[:, :, 1:, :].clone()
        for offset_idx in range(pslp_cycles):
            offset = offset[:, :, :-1, :]
            persistence[:, :, offset_idx+2: , :] = offset
            gather_tensor = torch.cat([gather_tensor, persistence], dim=0)
        self.persistence=torch.mean(gather_tensor, dim=0)


def get_pslp_type(name: str):
    if name == 'persistence':
        return PersistenceDataset
    elif name == 'last_pslp':
        return LooseTypeLastPSLPDataset
    elif name == 'loose_pslp':
        return LooseTypePSLPDataset
    elif name == 'tenday':
        return TendayDataset
    else:
        raise ValueError(f'Invalid pslp type: {name}')
