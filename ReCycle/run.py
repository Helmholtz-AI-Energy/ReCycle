import os
import io

# from random import randrange
from typing import Optional, Union, Tuple
from pathlib import Path
# import dataclasses

import pandas as pd

from .models.model_framework import ReCycleForecastModel
from .data.data_cleaner import clean_dataframe
from .data.dataset import ResidualDataset

# TODO output visualization to files in the log dir
# from .visualisation import (
#     plot_losses,
#     plot_sample,
#     plot_quantiles,
#     plot_calibration,
# )

from .specs import ModelSpec, DatasetSpec, TrainSpec, ActionSpec

# Logging
import logging

logger = logging.getLogger(__name__)


def run_action(
    model_spec: ModelSpec,
    dataset_spec: DatasetSpec,
    train_spec: TrainSpec,
    action_spec: ActionSpec,
) -> Optional[Union[io.BytesIO, None, pd.DataFrame, Tuple[float, Tuple[ResidualDataset]]]]:
    # TODO update return type hints
    action_spec.check_validity()

    model_spec.check_validity()
    dataset_spec.check_validity()
    checkpoint_file_name = Path("_".join([model_spec.model_name, dataset_spec.data_spec.dataset_name, dataset_spec.data_spec.file_name]) + ".pt")

    if action_spec.load_path is not None:
        load_file = action_spec.load_path / checkpoint_file_name

        forecaster = ReCycleForecastModel.load(load_file)
        if forecaster.model_spec != model_spec:
            logger.warn(f"loaded model spec not identical with specified one: {forecaster.model_spec} and {model_spec}")
        # NOTE the comparisons evaluate to false when they should not, maybe because of float precision?
        # assert model_spec == forecaster.model_spec
        # assert dataset_spec == forecaster.dataset_spec
        old_train_spec = forecaster.train_spec
        forecaster.train_spec = train_spec

        # # TODO log difference between old and new specs
        logger.info(
            f"Loading from {load_file}:\n {forecaster.model_spec=}\n {forecaster.dataset_spec=}\n {old_train_spec=}\n replacing with {train_spec=}"
        )

    else:
        forecaster = ReCycleForecastModel(model_spec, dataset_spec, train_spec)

    if action_spec.action == "train":
        train_loss, valid_loss, best_loss = forecaster.train_model()

        if isinstance(action_spec.save_path, Path):
            os.makedirs(action_spec.save_path, exist_ok=True)
            save_obj = action_spec.save_path / checkpoint_file_name
        elif isinstance(action_spec.save_path, io.BytesIO):
            save_obj = action_spec.save_path
        else:
            raise ValueError("Unexpected type of object to save checkpoint to.")

        forecaster.save(save_obj)

        logger.info("Model saved")
        return save_obj

    elif action_spec.action == "test":
        logger.info("Evaluating on test set")
        forecaster.mode("test")
        if model_spec.quantiles is not None:
            print(f"Quantiles: {train_spec.loss.get_quantile_values()}")
        test_batchsize = (
            train_spec.batch_size
            if train_spec.batch_size < len(forecaster.dataset())
            else len(forecaster.dataset())
        )
        print("Network prediction:")
        if model_spec.quantiles is None:
            result_summary = forecaster.test_forecast(batch_size=test_batchsize)
        else:
            result_summary, calibration = forecaster.test_forecast(
                batch_size=test_batchsize, calibration=True
            )
            print(calibration)

        print(result_summary)

        # Persistence for reference
        print("Load profiling:")
        rhp_summary = forecaster.test_rhp(batch_size=test_batchsize)
        print(rhp_summary)
        return result_summary, rhp_summary

    elif action_spec.action == "infer":
        # load input data from file
        input_df = pd.read_csv(action_spec.input_path)

        # input_spec = dataclasses.replace(dataset_spec.data_spec)
        input_df, data_column_names = clean_dataframe(
            df=input_df, data_spec=dataset_spec.data_spec
        )

        pred, input_reference, output_reference = forecaster.predict(
            input_df, action_spec.output_path
        )
        pred_df = pd.DataFrame(pred.numpy())
        pred_df.to_csv(action_spec.output_path)
        # TODO add time stamps to pred_df

        return pred_df

    elif action_spec.action == "hpo":
        raise
        train_loss, valid_loss, best_loss = forecaster.train_model()

        return best_loss, forecaster.datasets

    # if action_spec.plot_prediction:
    #     xlabel = dataset_spec.data_spec.xlabel
    #     ylabel = dataset_spec.data_spec.ylabel
    #     # res_label = "Residual " + ylabel

    #     if model_spec.quantiles is not None:
    #         # plot quantiles
    #         idx = randrange(len(forecaster.datasets[2]))
    #         print(f"Sample nr: {idx}")

    #         prediction, input_data, reference = forecaster.predict(
    #             dataset_name="test", idx=idx
    #         )
    #         plot_quantiles(prediction, reference)

    #     else:
    #         logger.info("plotting predictions")
    #         for n in range(4):
    #             idx = randrange(len(forecaster.datasets[2]))
    #             print(f"Sample nr: {idx}")

    #             prediction, input_data, reference = forecaster.predict(
    #                 dataset_name="test", idx=idx
    #             )
    #             plot_sample(
    #                 historic_data=input_data,
    #                 forecast_data=prediction,
    #                 forecast_reference=reference,
    #                 xlabel=xlabel,
    #                 ylabel=ylabel,
    #             )
    #             # time_resolved_error(prediction, reference)

    #             # prediction, input_data, reference = forecaster.predict(dataset_name='test', idx=idx, raw=True)
    #             # plot_prediction(prediction, input_data, reference, plot_input=False, xlabel=xlabel, ylabel=res_label,
    #             #                 is_residual_plot=True)
