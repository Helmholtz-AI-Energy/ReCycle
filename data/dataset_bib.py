from dataclasses import dataclass

from typing import List, Optional


__all__ = [
    'DatasetSpec',
    'get_dataset_spec',
]


@dataclass
class DatasetSpec:
    """
    Standard format to store dataset specifications

    Args:
        file_name: name of the csv-file as stored in the ./datasets directory. Without .csv suffix

    Kwargs:
        country_code: two-letter ISO country code to determine holidays using the holidays package
    """
    file_name: str
    country_code: Optional[str]
    data_column_names: Optional[List[str]]
    time_column_names: str
    root_path: str = "./datasets/"
    downsample_rate: Optional[int] = None
    split_by_category: bool = False
    remove_flatline: bool = False
    xlabel: Optional[str] = "Time [d]"
    ylabel: Optional[str] = "Consumption"

    def __call__(self):
        return self.root_path, self.file_name, self.country_code, self.data_column_names, self.time_column_names,\
               self.downsample_rate, self.split_by_category, self.remove_flatline


entsoe_de = DatasetSpec(file_name="entsoe_de",
                        country_code="de",
                        data_column_names=["load"],
                        time_column_names="start",
                        downsample_rate=4,
                        ylabel="Load [MW]")

entsoe_full = DatasetSpec(file_name="entsoe_full",
                          country_code=None,
                          data_column_names=["fr", "de", "no", "gb", "se", "ie", "it", "es", "pt", "ch", "at", "dk", "nl", "be"],
                          time_column_names="Time [s]",
                          ylabel="Load [MW]")

water = DatasetSpec(file_name="water",
                    country_code="de",
                    data_column_names=["Consumption"],
                    time_column_names="Date",
                    ylabel="Water Consumption [a.u.]")

uci_pt = DatasetSpec(file_name="uci_pt",
                     country_code="pt",
                     data_column_names=["MT_320"],
                     time_column_names="datetime",
                     split_by_category=False,
                     remove_flatline=True,
                     ylabel="Load [kW]")

informer_etth1 = DatasetSpec(file_name="etth1",
                             country_code="cn",
                             data_column_names=["OT"],
                             time_column_names="date",
                             ylabel='Temperature [$^\circ$C]')

informer_etth2 = DatasetSpec(file_name="etth2",
                             country_code="cn",
                             data_column_names=["OT"],
                             time_column_names="date",
                             ylabel='Temperature [$^\circ$C]')

minigrid = DatasetSpec(file_name="minigrid",
                       country_code="de",
                       data_column_names=["Load"],
                       time_column_names="date",
                       downsample_rate=None,
                       ylabel=' Consumption [kWh]')

solar = DatasetSpec(file_name="solar",
                    country_code="us",
                    data_column_names=["solar_mw"],
                    time_column_names="Datetime")

prices = DatasetSpec(file_name="prices",
                     country_code="de",
                     data_column_names=["Day-ahead Price [EUR/MWh]"],
                     time_column_names="MTU (CET/CEST)",
                     downsample_rate=4)

traffic = DatasetSpec(file_name="traffic_new",
                      country_code="us",
                      data_column_names=["VMT (Veh-Miles)"],
                      time_column_names="Hour")

former_traffic = DatasetSpec(file_name="traffic_old",
                             country_code="us",
                             data_column_names=["OT"],
                             time_column_names="date")

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


def get_dataset_spec(name: str) -> DatasetSpec:
    assert name in specs_dict, f'There is no know dataset specification {name}'
    spec = specs_dict[name]
    return spec
