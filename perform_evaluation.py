import os
from random import randrange

import torch

from models import ModelFramework
from utils.visualisation import (
    plot_losses,
    plot_sample,
    plot_quantiles,
    plot_calibration,
)

from typing import Optional, Tuple
from specs import ModelSpec, DatasetSpec, TrainSpec, ActionSpec
from data import ResidualDataset

# Logging
import logging

logger = logging.getLogger(__name__)


def perform_evaluation(
    model_spec: ModelSpec,
    dataset_spec: DatasetSpec,
    train_spec: TrainSpec,
    action_spec: ActionSpec,
    # For repeated runs the already built datasets can be reused
    premade_datasets: Optional[
        Tuple[ResidualDataset, ResidualDataset, ResidualDataset]
    ] = None,
) -> Optional[Tuple[float, Tuple[ResidualDataset, ResidualDataset, ResidualDataset]]]:
    action_spec.check_validity()

    if action_spec.load:
        load_path = action_spec.load_path or action_spec.save_path
        load_file = (
            load_path
            + "_".join([model_spec.model_name, dataset_spec.data_spec.file_name])
            + ".pt"
        )

        model_spec, dataset_spec, old_train_spec, state_dict = torch.load(load_file)
        logger.info(
            f"Loading from {load_file}:\n {model_spec=}\n {dataset_spec=}\n {old_train_spec=}"
        )

        model_spec.check_validity()
        dataset_spec.check_validity()

        run = ModelFramework(
            model_spec, dataset_spec, premade_datasets=premade_datasets
        )
        run.model.load_state_dict(state_dict=state_dict)
    else:
        model_spec.check_validity()
        dataset_spec.check_validity()

        run = ModelFramework(
            model_spec, dataset_spec, premade_datasets=premade_datasets
        )

    if action_spec.train:
        train_loss, valid_loss, best_loss = run.train_model(train_spec)

        if action_spec.hyper_optimization_interrupt:
            return best_loss, run.datasets

        if action_spec.save:
            # set upt save location
            os.makedirs(action_spec.save_path, exist_ok=True)
            save_file = (
                action_spec.save_path
                + "_".join([model_spec.model_name, dataset_spec.data_spec.file_name])
                + ".pt"
            )

            # get model state dictionary and save
            state_dict = run.model.state_dict()
            torch.save([model_spec, dataset_spec, train_spec, state_dict], f=save_file)
            logger.info("Model saved")

        if action_spec.plot_loss:
            logger.info("Plotting loss")
            plot_losses(train_loss, valid_loss)

    if action_spec.test:
        logger.info("Evaluating test set")
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
            quantiles = train_spec.loss.get_quantile_values()
            plot_calibration(calibration, quantiles)

        print(result_summary)

        # Persistence for reference
        print("Load profiling:")
        rhp_summary = run.test_rhp(batch_size=test_batchsize)
        print(rhp_summary)

    if action_spec.plot_prediction:
        xlabel = dataset_spec.data_spec.xlabel
        ylabel = dataset_spec.data_spec.ylabel
        res_label = "Residual " + ylabel

        if model_spec.quantiles is not None:
            # plot quantiles
            idx = randrange(len(run.datasets[2]))
            print(f"Sample nr: {idx}")

            prediction, input_data, reference = run.predict(
                dataset_name="test", idx=idx
            )
            plot_quantiles(prediction, reference)

        else:
            logger.info("plotting predictions")
            for n in range(4):
                idx = randrange(len(run.datasets[2]))
                print(f"Sample nr: {idx}")

                prediction, input_data, reference = run.predict(
                    dataset_name="test", idx=idx
                )
                plot_sample(
                    historic_data=input_data,
                    forecast_data=prediction,
                    forecast_reference=reference,
                    xlabel=xlabel,
                    ylabel=ylabel,
                )
                # time_resolved_error(prediction, reference)

                # prediction, input_data, reference = run.predict(dataset_name='test', idx=idx, raw=True)
                # plot_prediction(prediction, input_data, reference, plot_input=False, xlabel=xlabel, ylabel=res_label,
                #                 is_residual_plot=True)
