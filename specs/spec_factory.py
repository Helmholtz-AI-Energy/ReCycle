import torch

from .model_specs import get_model_spec, ModelSpec
from .dataset_specs import get_data_spec, DataSpec, DatasetSpec, ResidualDatasetSpec
from .train_specs import TrainSpec
from .action_specs import ActionSpec
from utils.quantile_loss import get_quantile_loss

from typing import Optional, Type, Union, Tuple, List
from data.embeddings import FullEmbedding
from data.normalizer import Normalizer, MinMax
from data.pslp_datasets import LooseTypePSLPDataset, LooseTypeLastPSLPDataset, PersistenceDataset
from torch import Tensor
from torch.nn.modules.loss import _Loss, L1Loss
from torch.optim import Optimizer, Adam

import logging
logger = logging.getLogger(__name__)


__all__ = {
    'spec_factory'
}


def spec_factory(
        model_name: str,

        historic_window: int,
        forecast_window: int,
        features_per_step: int,

        meta_features: Optional[int] = None,
        d_model: int = None,
        embedding: Optional[Union[str, FullEmbedding]] = None,
        embedding_args: Optional[dict] = None,
        residual_input: bool = True,
        residual_forecast: bool = True,
        custom_quantiles: Optional[Union[List[int], Tensor]] = None,
        quantiles: Optional[int] = None,
        assume_symmetric_quantiles: bool = False,
        model_args: Optional[dict] = None,

        normalizer: Type[Normalizer] = MinMax,
        train_share: float = 0.6,
        tests_share: float = 0.2,
        reduce: Optional[float] = None,

        residual_normalizer: Optional[Type[Normalizer]] = None,
        pslp_dataset: Union[Type[LooseTypePSLPDataset], Type[PersistenceDataset]] = LooseTypeLastPSLPDataset,
        pslp_cycles: int = 3,
        pslp_cycle_len: int = 7,

        dataset_name: Optional[str] = None,
        file_name: Optional[Union[str, Tuple[str, str, str]]] = None,
        time_column_name: Optional[str] = None,
        data_column_names: Optional[List[str]] = None,
        metadata_column_names: Optional[List[str]] = None,
        country_code: Optional[str] = None,
        universal_holidays: bool = True,
        downsample_rate: Optional[int] = None,
        split_by_category: bool = False,
        remove_flatline: bool = False,
        xlabel: Optional[str] = "Time [d]",
        ylabel: Optional[str] = "Consumption",
        root_path: str = "./datasets/",

        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 200,
        patience: Optional[int] = 20,
        loss: _Loss = L1Loss(),
        optimizer: Type[Optimizer] = Adam,
        optimizer_args: Optional[dict] = None,
        profiling: bool = False,

        device: torch.device = None,

        train: bool = True,
        plot_loss: bool = True,

        save: bool = True,
        save_path: str = './saved_models/',
        load: bool = False,
        load_path: Optional[str] = None,

        test: bool = True,
        plot_prediction: bool = True,

        **kwargs
) -> (ModelSpec, DatasetSpec, TrainSpec, ActionSpec):
    if model_args is None:
        model_args = kwargs
    else:
        for argument in kwargs:
            logger.warning(f'Unused argument {argument} = {kwargs[argument]}')

    # set up ModelSpec
    model_args = model_args or {}

    # autodetect device
    device = device or _get_device()
    logger.info(f'Using {device}')

    model_spec_class = get_model_spec(model_name)
    if type(embedding) is str:
        embedding_args = embedding_args or {}
        model_spec = model_spec_class.from_embedding_name(
            model_name=model_name,

            historic_window=historic_window,
            forecast_window=forecast_window,
            features_per_step=features_per_step,

            meta_features=meta_features,
            d_model=d_model,
            embedding=embedding,
            embedding_args=embedding_args,
            residual_input=residual_input,
            residual_forecast=residual_forecast,
            custom_quantiles=custom_quantiles,
            quantiles=quantiles,
            assume_symmetric_quantiles=assume_symmetric_quantiles,

            device=device,
            **model_args,
        )
    else:
        model_spec = model_spec_class(
            model_name=model_name,
            historic_window=historic_window,
            forecast_window=forecast_window,
            features_per_step=features_per_step,

            meta_features=meta_features,
            d_model=d_model,
            embedding=embedding,
            residual_input=residual_input,
            residual_forecast=residual_forecast,
            custom_quantiles=custom_quantiles,
            quantiles=quantiles,
            assume_symmetric_quantiles=assume_symmetric_quantiles,

            device=device,
            **model_args,
        )

    # set up DatasetSpec
    if dataset_name is not None:
        data_spec = get_data_spec(dataset_name)
    else:
        assert file_name is not None and time_column_name is not None, ('Either dataset_name or file_name and'
                                                                        'time_column_name must be specified')
        data_spec = DataSpec(
            file_name=file_name,
            time_column_name=time_column_name,
            data_column_names=data_column_names,
            metadata_column_names=metadata_column_names,
            country_code=country_code,
            universal_holidays=universal_holidays,
            downsample_rate=downsample_rate,
            split_by_category=split_by_category,
            remove_flatline=remove_flatline,
            xlabel=xlabel,
            ylabel=ylabel,
            root_path=root_path,
        )

    dataset_spec = ResidualDatasetSpec(
        historic_window=historic_window,
        forecast_window=forecast_window,
        features_per_step=features_per_step,
        data_spec=data_spec,
        normalizer=normalizer,
        train_share=train_share,
        tests_share=tests_share,
        reduce=reduce,
        device=device,
        residual_normalizer=residual_normalizer,
        pslp_dataset=pslp_dataset,
        pslp_cycles=pslp_cycles,
        pslp_cycle_len=pslp_cycle_len,
    )

    # Prepare TrainSpec arguments
    if (custom_quantiles is not None) or (quantiles is not None):
        # quantiles will always overwrite loss to quantile loss
        loss = get_quantile_loss(
            custom_quantiles=custom_quantiles,
            quantile_nr=quantiles,
            symmetric_quantiles=assume_symmetric_quantiles
        )

    patience = patience or epochs
    optimizer_args = optimizer_args or {} #{'weight_decay': 0.1}

    # set up TrainSpec
    train_spec = TrainSpec(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        loss=loss,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        profiling=profiling,
    )

    # set up ActionSpec
    load_path = load_path or save_path
    action_spec = ActionSpec(
        train=train,
        plot_loss=plot_loss,
        save=save,
        save_path=save_path,
        load=load,
        load_path=load_path,
        test=test,
        plot_prediction=plot_prediction
    )

    return model_spec, dataset_spec, train_spec, action_spec


def _get_device():
    """Deduces the device models and data should be saved and run on"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
