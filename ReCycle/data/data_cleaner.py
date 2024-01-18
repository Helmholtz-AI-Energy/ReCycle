import pandas as pd
import holidays
from datetime import timedelta

from typing import Optional, List, Dict, Union
from pandas import DataFrame
from numpy import NaN


import logging

logger = logging.getLogger(__name__)


def clean_dataframe(df: DataFrame, data_spec: "DataSpec") -> (DataFrame, List[str]):
    if data_spec.data_column_names is None:
        logger.info("Inferring data_column_names")
        # if data_column_names is None all columns not specified as metadata are considered data_column_names columns
        data_spec.data_column_names = list(df.columns)
        data_spec.data_column_names.remove(data_spec.time_column_name)
        for item in data_spec.metadata_column_names:
            logger.debug(f"{item}, {data_spec.data_column_names}")
            data_spec.data_column_names.remove(item)

    # shorthands for code cleanliness
    data_column_names = data_spec.data_column_names
    time_column_name = data_spec.time_column_name
    metadata_column_names = data_spec.metadata_column_names

    # Ensure increasing time sorting, discontinued as it breaks Datasets if column only contains date not time
    df[time_column_name] = pd.to_datetime(df[time_column_name])
    # df.sort_values(time_column_name, ascending=True, inplace=True)

    # Remove nuisance columns
    if type(data_spec.data_column_names) is str:
        data_column_names = [data_column_names]
    valid = (
        data_column_names
        if metadata_column_names is None
        else data_column_names + metadata_column_names
    )
    valid = [time_column_name] + valid
    df = df[valid]

    if data_spec.remove_flatline:
        starts = df[data_column_names].ne(0).idxmax()
        for column in data_column_names:
            df.loc[: starts[column] - 1, column] = NaN
        logger.info("Flatline removed")

    if data_spec.downsample_rate is not None:
        new = pd.DataFrame()
        new[data_column_names] = (
            df[data_column_names]
            .rolling(data_spec.downsample_rate)
            .mean()[data_spec.downsample_rate - 1 :: data_spec.downsample_rate]
            .reset_index(drop=True)
        )
        ld = len(new[data_column_names[0]])
        new[time_column_name] = df[time_column_name][0 :: data_spec.downsample_rate][
            :ld
        ].reset_index(drop=True)
        logger.info("Data downsampling complete")

        if metadata_column_names is not None:
            for column in metadata_column_names:
                new[column] = df.loc[
                    0 :: data_spec.downsample_rate, column
                ].reset_index(drop=True)

        logger.info("Metadata downsampling complete")
        df = new

    if data_spec.universal_holidays:
        if data_spec.country_code is None:
            logger.warning("No country specified, defaulting to Germany")
            country = "de"
        else:
            country = data_spec.country_code
        logger.info(f"Using {country} holidays")

        local_holidays = holidays.country_holidays(country.upper())
        df["holiday"] = [n in local_holidays for n in df[time_column_name]]
        df["holiday_tomorrow"] = [
            n in local_holidays for n in df[time_column_name] + timedelta(days=1)
        ]
    else:
        logger.info("Using individual holidays")

        for country in data_column_names:
            local_holidays = holidays.country_holidays(country.upper())
            df[time_column_name] = pd.to_datetime(df[time_column_name])
            df["holiday_" + country] = [
                n in local_holidays for n in df[time_column_name]
            ]
            df["holiday_" + country + "_tomorrow"] = [
                n in local_holidays for n in df[time_column_name] + timedelta(days=1)
            ]

    if not data_spec.remove_flatline:
        logger.info("Fixing stray NaNs")
        # remove remaining NaNs by interpolating neighbours
        nan_list = df.isnull().stack()[lambda x: x].index.tolist()
        for index, column in nan_list:
            df.loc[index, column] = df.loc[index - 1 : index + 2, column].mean()
    return df, data_column_names
