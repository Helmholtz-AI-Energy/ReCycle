from random import randrange

from models import ModelFramework
from utils.visualisation import plot_losses, plot_sample

from specs import ModelSpec, DatasetSpec, TrainSpec, ActionSpec

# Logging
import logging
logger = logging.getLogger(__name__)


def perform_evaluation(
        model_spec: ModelSpec,
        dataset_spec: DatasetSpec,
        train_spec: TrainSpec,
        action_spec: ActionSpec
) -> None:
    if action_spec.load:
        logger.error('Loading not implemented')
        run = None
    else:
        run = ModelFramework(model_spec, dataset_spec)

    if action_spec.train:
        train_loss, valid_loss, best_loss = run.train_model(train_spec)
        if action_spec.plot_loss:
            logger.info('Plotting loss')
            plot_losses(train_loss, valid_loss)

    if action_spec.save:
        logger.warning('Saving not implemented')
        pass

    if action_spec.test:
        logger.info('Evaluating test set')
        run.mode('test')
        test_batchsize = train_spec.batch_size if train_spec.batch_size < len(run.dataset()) else len(run.dataset())
        print('Network prediction:')
        result_summary = run.test_forecast(batch_size=test_batchsize)
        print(result_summary)

        # Persistence for reference
        print('Load profiling:')
        pslp_summary = run.test_pslp(batch_size=test_batchsize)
        print(pslp_summary)

    if action_spec.plot_prediction:
        xlabel = dataset_spec.data_spec.xlabel
        ylabel = dataset_spec.data_spec.ylabel
        res_label = "Residual " + ylabel

        if (model_spec.quantiles is not None) or (model_spec.custom_quantiles is not None):
            # plot quantiles
            pass
        else:
            logger.info('plotting predictions')
            for n in range(4):
                idx = randrange(len(run.datasets[2]))
                print(f'Sample nr: {idx}')

                prediction, input_data, reference = run.predict(dataset_name='test', idx=idx)
                plot_sample(
                    historic_data=input_data,
                    forecast_data=prediction,
                    forecast_reference=reference,
                    xlabel=xlabel,
                    ylabel=ylabel)
                # time_resolved_error(prediction, reference)

                # prediction, input_data, reference = run.predict(dataset_name='test', idx=idx, raw=True)
                # plot_prediction(prediction, input_data, reference, plot_input=False, xlabel=xlabel, ylabel=res_label,
                #                 is_residual_plot=True)

