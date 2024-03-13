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
    file_name="entsoe_de",
    country_code="de",
    data_column_names=["load"],
    time_column_name="start",
    downsample_rate=4,
    # ylabel="Load [MW]",
)

entsoe_full = DataSpec(
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
    file_name="water",
    country_code="de",
    data_column_names=["Consumption"],
    time_column_name="Date",
    # ylabel="Water Consumption [a.u.]",
)

uci_pt = DataSpec(
    file_name="uci_pt",
    country_code="pt",
    data_column_names=["MT_320"],
    time_column_name="datetime",
    split_by_category=False,
    remove_flatline=True,
    # ylabel="Load [kW]",
)

informer_etth1 = DataSpec(
    file_name="etth1",
    country_code="cn",
    data_column_names=["OT"],
    time_column_name="date",
    # ylabel=r"Temperature [$^\circ$C]",
)

informer_etth2 = DataSpec(
    file_name="etth2",
    country_code="cn",
    data_column_names=["OT"],
    time_column_name="date",
    # ylabel=r"Temperature [$^\circ$C]",
)

minigrid = DataSpec(
    file_name="minigrid",
    country_code="de",
    data_column_names=["Load"],
    time_column_name="date",
    downsample_rate=None,
    # ylabel=" Consumption [kWh]",
)

solar = DataSpec(
    file_name="solar",
    country_code="us",
    data_column_names=["solar_mw"],
    time_column_name="Datetime",
)

prices = DataSpec(
    file_name="prices",
    country_code="de",
    data_column_names=["Day-ahead Price [EUR/MWh]"],
    time_column_name="MTU (CET/CEST)",
    downsample_rate=4,
)

traffic = DataSpec(
    file_name="traffic_new",
    country_code="us",
    data_column_names=["VMT (Veh-Miles)"],
    time_column_name="Hour",
)

former_traffic = DataSpec(
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
