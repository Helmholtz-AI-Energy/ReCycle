import os

# from random import randrange
from typing import Optional, Tuple
from pathlib import Path

import torch
import pandas as pd

from .models.model_framework import ReCycleForecastModel
from .data.data_cleaner import clean_dataframe

# TODO output visualization to files in the log dir
# from .visualisation import (
#     plot_losses,
#     plot_sample,
#     plot_quantiles,
#     plot_calibration,
# )

from .specs import ModelSpec, DatasetSpec, TrainSpec, ActionSpec
from .data.dataset import ResidualDataset

# Logging
import logging

logger = logging.getLogger(__name__)


def run_action(
    model_spec: ModelSpec,
    dataset_spec: DatasetSpec,
    train_spec: TrainSpec,
    action_spec: ActionSpec,
) -> Optional[Tuple[float, Tuple[ResidualDataset, ResidualDataset, ResidualDataset]]]:
    action_spec.check_validity()

    model_spec.check_validity()
    dataset_spec.check_validity()
    if action_spec.load_path is not None:
        load_path = action_spec.load_path or action_spec.save_path
        load_file = (
            load_path
            + "_".join([model_spec.model_name, dataset_spec.data_spec.file_name])
            + ".pt"
        )

        # TODO this should not use torch
        model_spec, dataset_spec, old_train_spec, state_dict = torch.load(load_file)
        # TODO log difference between old and new train spec
        logger.info(
            f"Loading from {load_file}:\n {model_spec=}\n {dataset_spec=}\n {old_train_spec=}"
        )

        # TODO don't call this a run, we have a generic forecast model wrapper around a torch or other model
        run = ReCycleForecastModel(model_spec, dataset_spec, train_spec)
        run.model.load_state_dict(state_dict=state_dict)
    else:
        run = ReCycleForecastModel(model_spec, dataset_spec, train_spec)

    if action_spec.action == "train":
        train_loss, valid_loss, best_loss = run.train_model(train_spec)

        # TODO generate model file names, such that models are not accidentally overwritten
        # set upt save location
        os.makedirs(action_spec.save_path, exist_ok=True)
        save_file = action_spec.save_path / Path(
            "_".join([model_spec.model_name, dataset_spec.data_spec.file_name]) + ".pt"
        )

        # get model state dictionary and save
        state_dict = run.model.state_dict()
        torch.save([model_spec, dataset_spec, train_spec, state_dict], f=save_file)
        logger.info("Model saved")

        # if action_spec.plot_loss:
        #     logger.info("Plotting loss")
        #     plot_losses(train_loss, valid_loss)

    elif action_spec.action == "test":
        logger.info("Evaluating on test set")
        run.mode("test")
        if model_spec.quantiles is not None:
            print(f"Quantiles: {train_spec.loss.get_quantile_values()}")
        test_batchsize = (
            train_spec.batch_size
            if train_spec.batch_size < len(run.dataset())
            else len(run.dataset())
        )
        print("Network prediction:")
        if model_spec.quantiles is None:
            result_summary = run.test_forecast(batch_size=test_batchsize)
        else:
            result_summary, calibration = run.test_forecast(
                batch_size=test_batchsize, calibration=True
            )
            print(calibration)
            # quantiles = train_spec.loss.get_quantile_values()
            # plot_calibration(calibration, quantiles)

        print(result_summary)

        # Persistence for reference
        print("Load profiling:")
        rhp_summary = run.test_rhp(batch_size=test_batchsize)
        print(rhp_summary)

    elif action_spec.action == "infer":
        # load input data from file
        input_df = pd.read_csv(action_spec.input_path)
        input_df, data_column_names = clean_dataframe(
            df=input_df, data_spec=dataset_spec.data_spec
        )

        pred, input_reference, output_reference = run.predict(
            input_df, action_spec.output_path
        )
        pred_df = pd.DataFrame(pred.numpy())
        pred_df.to_csv(action_spec.output_path)

        return (
            pred,
            # pred_df,
        )

    elif action_spec.action == "hpo":
        raise
        train_loss, valid_loss, best_loss = run.train_model(train_spec)

        return best_loss, run.datasets

    # if action_spec.plot_prediction:
    #     xlabel = dataset_spec.data_spec.xlabel
    #     ylabel = dataset_spec.data_spec.ylabel
    #     # res_label = "Residual " + ylabel

    #     if model_spec.quantiles is not None:
    #         # plot quantiles
    #         idx = randrange(len(run.datasets[2]))
    #         print(f"Sample nr: {idx}")

    #         prediction, input_data, reference = run.predict(
    #             dataset_name="test", idx=idx
    #         )
    #         plot_quantiles(prediction, reference)

    #     else:
    #         logger.info("plotting predictions")
    #         for n in range(4):
    #             idx = randrange(len(run.datasets[2]))
    #             print(f"Sample nr: {idx}")

    #             prediction, input_data, reference = run.predict(
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

    #             # prediction, input_data, reference = run.predict(dataset_name='test', idx=idx, raw=True)
    #             # plot_prediction(prediction, input_data, reference, plot_input=False, xlabel=xlabel, ylabel=res_label,
    #             #                 is_residual_plot=True)
