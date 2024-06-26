from pathlib import Path
import io

import torch
import pandas as pd

import ReCycle
from ReCycle import run
from ReCycle.specs import configure_run
from ReCycle.models.model_framework import ReCycleForecastModel
# from ReCycle.specs import DatasetSpec


data_root = Path("/".join(ReCycle.__path__[0].split("/")[:-1]) + "/data/")
data_filename = "anonymized_dataset"

default_params = {
    "model_name": "Transformer",
    "historic_window": 7,
    "forecast_window": 7,
    "features_per_step": 24,
    "dataset_root_path": data_root,
    "dataset_file_path": data_filename,
    "dataset_time_col": "Date",
    "dataset_data_cols": ["Consumption"],
    # "dataset_country_code": "de",
    "epochs": 20,
    "batch_size": 1,
    "d_model": 32,
    "train_share": 0.33,
    "test_share": 0.32,
    "rhp_cycles": 2,
}


transformer_params = {
    "nheads": 1,
    "d_feedforward": 32,
    "d_hidden": 32,
    "meta_token": True,
}

mlp_params = {}


def test_custom_dataset():
    data_spec = ReCycle.specs.dataset_specs.DataSpec(
        file_name=default_params["dataset_file_path"],
        time_column_name=default_params["dataset_time_col"],
        data_column_names=default_params["dataset_data_cols"],
        root_path=data_root,
    )
    dataset_spec = ReCycle.specs.dataset_specs.ResidualDatasetSpec(
        historic_window=7,
        forecast_window=default_params["forecast_window"],
        features_per_step=default_params["features_per_step"],
        data_spec=data_spec,
        train_share=default_params["train_share"],
        test_share=default_params["test_share"],
        rhp_cycles=default_params["rhp_cycles"],
    )

    datasets = ReCycle.data.dataset.ResidualDataset.from_csv(dataset_spec)

    for dataset in datasets:
        print(len(dataset.dataframe) / 24)
        print(dataset.total_window, dataset.rhp_window)
        print(dataset.cumulative_samples)
        print(len(dataset.dataframe) / 24 - dataset.total_window - dataset.rhp_window)
        # print(len(dataset))
        assert len(dataset) > 0


def test_entsoe_de_dataset():
    data_spec = ReCycle.globals.entsoe_de
    data_spec.root_path = "~/Data/ENTSO-E_WesterEuropePowerConsumption/"
    dataset_spec = ReCycle.specs.dataset_specs.ResidualDatasetSpec(
        historic_window=default_params["historic_window"],
        forecast_window=default_params["forecast_window"],
        features_per_step=default_params["features_per_step"],
        data_spec=data_spec,
    )
    datasets = ReCycle.data.dataset.ResidualDataset.from_csv(dataset_spec)

    assert datasets[0].categories == 1

    for dataset in datasets:
        print(len(dataset))
        print(len(dataset.dataframe) / 24)
        print(len(dataset.dataframe) / 24 - len(dataset))
        print(
            len(dataset.dataframe)
            - (len(dataset) + dataset.total_window + dataset.rhp_window) * 24
        )
        assert (
            len(dataset.dataframe)
            == (len(dataset) + dataset.total_window + dataset.rhp_window) * 24
        )


def test_training(tmp_path):
    params = {}
    params.update(default_params)
    params["model_args"] = transformer_params
    params["action"] = "train"
    params["save_checkpoint_path"] = tmp_path

    model_spec, dataset_spec, train_spec, action_spec = configure_run(**params)

    run.run_action(
        model_spec=model_spec,
        dataset_spec=dataset_spec,
        train_spec=train_spec,
        action_spec=action_spec,
    )
    del params


def test_testing(tmp_path):
    params = {}
    params.update(default_params)
    params["model_args"] = transformer_params
    params["action"] = "train"
    params["save_checkpoint_path"] = tmp_path

    model_spec, dataset_spec, train_spec, action_spec = configure_run(**params)
    run.run_action(
        model_spec=model_spec,
        dataset_spec=dataset_spec,
        train_spec=train_spec,
        action_spec=action_spec,
    )

    params["action"] = "test"
    params["load_checkpoint"] = tmp_path
    model_spec, dataset_spec, train_spec, action_spec = configure_run(**params)
    run.run_action(
        model_spec=model_spec,
        dataset_spec=dataset_spec,
        train_spec=train_spec,
        action_spec=action_spec,
    )


def test_entsoe_de(tmp_path):
    params = {}
    params.update(default_params)
    params["dataset_root_path"] = "~/Data/ENTSO-E_WesterEuropePowerConsumption/"
    params["dataset_name"] = "entsoe_de"
    params["model_args"] = transformer_params
    params["action"] = "train"
    params["save_checkpoint_path"] = tmp_path
    params["epochs"] = 2

    model_spec, dataset_spec, train_spec, action_spec = configure_run(**params)
    run.run_action(
        model_spec=model_spec,
        dataset_spec=dataset_spec,
        train_spec=train_spec,
        action_spec=action_spec,
    )

    params["action"] = "test"
    params["load_checkpoint"] = tmp_path
    model_spec, dataset_spec, train_spec, action_spec = configure_run(**params)
    run.run_action(
        model_spec=model_spec,
        dataset_spec=dataset_spec,
        train_spec=train_spec,
        action_spec=action_spec,
    )


# def test_uci_pt():
#     print("the uci_pt dataset might require some cropping if this test fails")
#     with tempfile.TemporaryDirectory() as save_checkpoint_path:
#         params = {}
#         params.update(default_params)
#         params["dataset_root_path"] = "~/Data/electricityloaddiagrams20112014/"
#         params["dataset_name"] = "uci_pt"
#         params["model_args"] = transformer_params
#         params["action"] = "train"
#         params["save_checkpoint_path"] = save_checkpoint_path
#
#         model_spec, dataset_spec, train_spec, action_spec = configure_run(**params)
#         run.run_action(
#             model_spec=model_spec,
#             dataset_spec=dataset_spec,
#             train_spec=train_spec,
#             action_spec=action_spec,
#         )
#
#         params["action"] = "test"
#         params["load_checkpoint"] = save_checkpoint_path + "/"
#         model_spec, dataset_spec, train_spec, action_spec = configure_run(**params)
#         run.run_action(
#             model_spec=model_spec,
#             dataset_spec=dataset_spec,
#             train_spec=train_spec,
#             action_spec=action_spec,
#         )


def test_mlp(tmp_path):
    params = {}
    params.update(default_params)
    params["model_name"] = "MLP"
    params["model_args"] = mlp_params
    params["action"] = "train"
    params["save_checkpoint_path"] = tmp_path

    model_spec, dataset_spec, train_spec, action_spec = configure_run(**params)

    run.run_action(
        model_spec=model_spec,
        dataset_spec=dataset_spec,
        train_spec=train_spec,
        action_spec=action_spec,
    )


def test_dataset_from_dataframe():
    data_spec = ReCycle.specs.dataset_specs.DataSpec(
        file_name=default_params["dataset_file_path"],
        time_column_name=default_params["dataset_time_col"],
        data_column_names=default_params["dataset_data_cols"],
        root_path=data_root,
    )
    dataset_spec = ReCycle.specs.dataset_specs.ResidualDatasetSpec(
        historic_window=7,
        forecast_window=default_params["forecast_window"],
        features_per_step=default_params["features_per_step"],
        data_spec=data_spec,
        train_share=default_params["train_share"],
        test_share=default_params["test_share"],
        rhp_cycles=default_params["rhp_cycles"],
    )

    csv_datasets = ReCycle.data.dataset.ResidualDataset.from_csv(dataset_spec)

    data = pd.read_csv(
        data_spec.full_file_path(file_extension=data_spec.file_extension),
        sep=data_spec.sep,
        decimal=data_spec.decimal,
    )
    df_datasets = ReCycle.data.dataset.ResidualDataset.from_dataframe(
        data, dataset_spec
    )

    for df_ds, csv_ds in zip(df_datasets, csv_datasets):
        for sample1, sample2 in zip(df_ds, csv_ds):
            input1 = sample1[0]
            input2 = sample2[0]
            assert torch.all(torch.eq(input1, input2))


def test_inference(tmp_path):
    params = {}
    params.update(default_params)
    params["model_args"] = transformer_params
    params["action"] = "train"
    params["save_checkpoint_path"] = tmp_path

    model_spec, dataset_spec, train_spec, action_spec = configure_run(**params)
    run.run_action(
        model_spec=model_spec,
        dataset_spec=dataset_spec,
        train_spec=train_spec,
        action_spec=action_spec,
    )

    params["action"] = "infer"
    params["load_checkpoint"] = tmp_path
    params["input_path"] = data_root / Path(data_filename + ".csv")
    params["output_path"] = tmp_path / Path("out.csv")
    model_spec, dataset_spec, train_spec, action_spec = configure_run(**params)
    run.run_action(
        model_spec=model_spec,
        dataset_spec=dataset_spec,
        train_spec=train_spec,
        action_spec=action_spec,
    )


def test_metadata_mlp(tmp_path):
    params = {}
    params.update(default_params)
    params["model_name"] = "MLP"
    params["model_args"] = mlp_params
    params["action"] = "train"
    params["save_checkpoint_path"] = tmp_path
    params["dataset_metadata_cols"] = ["Metadata_1", "Metadata_2", "Metadata_3"]

    model_spec, dataset_spec, train_spec, action_spec = configure_run(**params)

    run.run_action(
        model_spec=model_spec,
        dataset_spec=dataset_spec,
        train_spec=train_spec,
        action_spec=action_spec,
    )


def test_metadata_transformer(tmp_path):
    params = {}
    params.update(default_params)
    params["model_args"] = transformer_params
    params["action"] = "train"
    params["save_checkpoint_path"] = tmp_path
    params["dataset_metadata_cols"] = ["Metadata_1", "Metadata_2", "Metadata_3"]

    model_spec, dataset_spec, train_spec, action_spec = configure_run(**params)

    run.run_action(
        model_spec=model_spec,
        dataset_spec=dataset_spec,
        train_spec=train_spec,
        action_spec=action_spec,
    )


def test_no_data_col_names():
    pass


# def test_checkpointing(tmp_path):
#     raise


def test_serialization(tmp_path):
    """Test serializtion of the model after training without writing it to a file. And training and inference by passing a dataframe directly."""
    # configure run
    params = {
        "model_name": "Transformer",
        "historic_window": 7,
        "forecast_window": 7,
        "features_per_step": 24,
        "epochs": 20,
        "batch_size": 1,
        "d_model": 32,
        "train_share": 0.33,
        "test_share": 0.32,
        "rhp_cycles": 2,
        "model_args": transformer_params,
        "action": "train",
        "save_checkpoint_path": tmp_path,
        "dataset_time_col": "Date",
        "dataset_data_cols": ["Consumption"],
    }

    model_spec, dataset_spec, train_spec, action_spec = configure_run(**params)

    forecaster = ReCycleForecastModel(model_spec, dataset_spec, train_spec)

    # load data
    train_df = pd.read_csv(data_root / Path(data_filename + '.csv'))
    valid_df = pd.read_csv(data_root / Path(data_filename + '.csv'))
    test_df = pd.read_csv(data_root / Path(data_filename + '.csv'))
    input_df = pd.read_csv(data_root / Path(data_filename + '.csv'))

    # train
    forecaster.train_model(train_df=train_df, valid_df=valid_df)
    model_as_bytes = io.BytesIO()
    forecaster.save(model_as_bytes)
    # test
    forecaster.test_forecast(test_df=test_df)
    # load model
    del forecaster
    forecaster = ReCycleForecastModel.load(model_as_bytes)
    # predict
    pred = forecaster.predict(input_df)
    print(pred)


# def test_quantiles_interface():
#     pass
#
#
# def test_metadata_interface():
#     pass
