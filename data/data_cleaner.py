import pandas as pd
import holidays
from datetime import timedelta

from typing import Optional, List, Dict
from pandas import DataFrame
from numpy import NaN


import logging
logger = logging.getLogger(__name__)


def clean_dataframe(df: DataFrame, target: List[str], metadata: Optional[Dict[str, str]] = None,
                    downsample_rate: Optional[int] = None, fixed_country: Optional[str] = None,
                    flatline_to_nan: bool = True) -> DataFrame:

    # Ensure increasing time sorting, discontinued as it breaks Datasets if column only contains date not time
    df[metadata['time']] = pd.to_datetime(df[metadata['time']])
    #df.sort_values(metadata['time'], ascending=True, inplace=True)

    # Remove nuisance columns
    valid = (target if metadata is None else target + list(metadata.values()))
    df = df[valid]

    if flatline_to_nan:
        starts = df[target].ne(0).idxmax()
        for column in target:
            df.loc[:starts[column] - 1, column] = NaN
        logger.info('Flatline removed')

    if downsample_rate is not None:
        new = pd.DataFrame()
        new[target] = df[target].rolling(downsample_rate).mean()[downsample_rate - 1 :: downsample_rate].reset_index(drop=True)
        logger.info('Data downsampling complete')

        if metadata is not None:
            for key in metadata:
                new[metadata[key]] = df.loc[0::downsample_rate, metadata[key]].reset_index(drop=True)

        logger.info('Metadata downsampling complete')
        df = new

    if fixed_country is not None:
        logger.info(f'Using {fixed_country} holidays')

        country = fixed_country
        local_holidays = holidays.country_holidays(country.upper())
        df['holiday'] = [n in local_holidays for n in df[metadata['time']]]
        df['holiday_tomorrow'] = [n in local_holidays for n in df[metadata['time']] + timedelta(days=1)]
    else:
        logger.info('Using individual holidays')

        for country in target:
            local_holidays = holidays.country_holidays(country.upper())
            df[metadata['time']] = pd.to_datetime(df[metadata['time']])
            df['holiday_' + country] = [n in local_holidays for n in df[metadata['time']]]
            df['holiday_' + country + '_tomorrow'] = [n in local_holidays for n in df[metadata['time']] + timedelta(days=1)]

    if not flatline_to_nan:
        logger.info('Fixing stray NaNs')
        # remove remaining NaNs by interpolating neighbours
        nan_list = df.isnull().stack()[lambda x: x].index.tolist()
        for index, column in nan_list:
            df.loc[index, column] = df.loc[index - 1:index + 2, column].mean()
    return df
