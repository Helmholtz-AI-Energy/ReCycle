from .models.torch_modules.oneshot_transformer import OneshotTransformer
from .models.torch_modules.mlp import MultiLayerPerceptron
from .specs.dataset_specs import DataSpec

from .specs.model_specs import MLPSpec, TransformerSpec

predefined_models_dict = dict(Transformer=OneshotTransformer, MLP=MultiLayerPerceptron)
predefined_modelspecs_dict = dict(
    Transformer=TransformerSpec,
    MLP=MLPSpec,
)

entsoe_de = DataSpec(
    file_name="de",
    country_code="de",
    data_column_names=["load"],
    time_column_name="start",
    downsample_rate=4,
    dataset_name="entsoe_de",
    # ylabel="Load [MW]",
)

entsoe_full = DataSpec(
    dataset_name="entsoe_full",
    file_name="entsoe_full",
    country_code=None,
    universal_holidays=False,
    data_column_names=[
        "fr",
        "de",
        "no",
        "gb",
        "se",
        "ie",
        "it",
        "es",
        "pt",
        "ch",
        "at",
        "dk",
        "nl",
        "be",
    ],
    time_column_name="Time [s]",
    # ylabel="Load [MW]",
)

water = DataSpec(
    dataset_name="water",
    file_name="water",
    country_code="de",
    data_column_names=["Consumption"],
    time_column_name="Date",
    # ylabel="Water Consumption [a.u.]",
)

uci_pt = DataSpec(
    dataset_name="uci_pt",
    file_name="LD2011_2014",
    country_code="pt",
    data_column_names=["MT_320"],
    time_column_name="Unnamed: 0",
    split_by_category=False,
    remove_flatline=True,
    file_extension=".txt",
    sep=";",
    decimal=","
    # ylabel="Load [kW]",
)

informer_etth1 = DataSpec(
    dataset_name="informer_etth1",
    file_name="etth1",
    country_code="cn",
    data_column_names=["OT"],
    time_column_name="date",
    # ylabel=r"Temperature [$^\circ$C]",
)

informer_etth2 = DataSpec(
    dataset_name="informer_etth2",
    file_name="etth2",
    country_code="cn",
    data_column_names=["OT"],
    time_column_name="date",
    # ylabel=r"Temperature [$^\circ$C]",
)

minigrid = DataSpec(
    dataset_name="minigrid",
    file_name="minigrid",
    country_code="de",
    data_column_names=["Load"],
    time_column_name="date",
    downsample_rate=None,
    # ylabel=" Consumption [kWh]",
)

solar = DataSpec(
    dataset_name="solar",
    file_name="solar",
    country_code="us",
    data_column_names=["solar_mw"],
    time_column_name="Datetime",
)

prices = DataSpec(
    dataset_name="prices",
    file_name="prices",
    country_code="de",
    data_column_names=["Day-ahead Price [EUR/MWh]"],
    time_column_name="MTU (CET/CEST)",
    downsample_rate=4,
)

traffic = DataSpec(
    dataset_name="traffic",
    file_name="traffic_new",
    country_code="us",
    data_column_names=["VMT (Veh-Miles)"],
    time_column_name="Hour",
)

former_traffic = DataSpec(
    dataset_name="former_traffic",
    file_name="traffic_old",
    country_code="us",
    data_column_names=["OT"],
    time_column_name="date",
)

predefined_dataspecs_dict = dict(
    entsoe_de=entsoe_de,
    entsoe_full=entsoe_full,
    water=water,
    uci_pt=uci_pt,
    etth1=informer_etth1,
    etth2=informer_etth2,
    minigrid=minigrid,
    solar=solar,
    prices=prices,
    traffic=traffic,
    former_traffic=former_traffic,
)
