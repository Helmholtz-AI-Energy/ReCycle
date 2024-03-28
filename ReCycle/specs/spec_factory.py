from typing import Optional, Type, Union, Tuple, List
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss, L1Loss
from torch.optim import Optimizer, Adam

from .model_specs import ModelSpec
from .dataset_specs import DataSpec, DatasetSpec, ResidualDatasetSpec
from .train_specs import TrainSpec
from .action_specs import ActionSpec
from ..globals import predefined_dataspecs_dict, predefined_modelspecs_dict

from ..utils.quantile_loss import get_quantile_loss

from ..data.embeddings import FullEmbedding
from ..data.normalizer import Normalizer, MinMax
from ..data.rhp_datasets import (
    LooseTypeRHPDataset,
    LooseTypeLastRHPDataset,
    PersistenceDataset,
)
import logging

logger = logging.getLogger(__name__)


def configure_run(
    action: str,
    model_name: str,
    historic_window: int,
    forecast_window: int,
    features_per_step: int,
    meta_features: Optional[int] = 9,
    d_model: int = None,
    embedding: Optional[Union[str, FullEmbedding]] = "default",
    dropout: float = 0.0,
    residual_input: bool = True,
    residual_forecast: bool = True,
    custom_quantiles: Optional[Union[List[int], Tensor]] = None,
    quantiles: Optional[int] = None,
    symmetric_quantiles: bool = False,
    invert_quantiles: bool = False,
    normalizer: Type[Normalizer] = MinMax,
    train_share: float = 0.6,
    test_share: float = 0.2,
    reduce: Optional[float] = None,
    residual_normalizer: Optional[Type[Normalizer]] = None,
    rhp_dataset: Union[
        Type[LooseTypeRHPDataset], Type[PersistenceDataset]
    ] = LooseTypeLastRHPDataset,
    rhp_cycles: int = 3,
    rhp_cycle_len: int = 7,
    dataset_name: Optional[str] = None,
    dataset_file_path: Optional[Union[str, Tuple[str, str, str]]] = None,
    dataset_time_col: Optional[str] = None,
    dataset_data_cols: Optional[List[str]] = None,
    dataset_metadata_cols: Optional[List[str]] = None,
    dataset_root_path: str = "./datasets/",
    dataset_country_code: Optional[str] = None,
    universal_holidays: bool = True,
    downsample_rate: Optional[int] = None,
    split_by_category: bool = False,
    remove_flatline: bool = False,
    log_learning_rate: float = -3,
    batch_size: int = 32,
    epochs: int = 200,
    patience: Optional[int] = 20,
    loss: _Loss = L1Loss(),
    optimizer: Type[Optimizer] = Adam,
    optimizer_args: Optional[dict] = None,
    profiling: bool = False,
    save_checkpoint_path: str = "./saved_models/",
    load_checkpoint: Optional[str] = None,
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    # model parameters
    model_args={},
) -> (ModelSpec, DatasetSpec, TrainSpec, ActionSpec):
    # set up ModelSpec

    # autodetect device
    device = _get_device()
    logger.info(f"Using {device}")

    model_spec_class = predefined_modelspecs_dict[model_name]
    if custom_quantiles is not None:
        quantiles = len(custom_quantiles)
    if isinstance(embedding, str):
        meta_cols = 0
        if dataset_metadata_cols is not None:
            meta_cols = len(dataset_metadata_cols)
        model_spec = model_spec_class.from_embedding_name(
            model_name=model_name,
            historic_window=historic_window,
            forecast_window=forecast_window,
            features_per_step=features_per_step,
            meta_features=meta_features,
            meta_cols=meta_cols,
            d_model=d_model,
            embedding=embedding,
            dropout=dropout,
            residual_input=residual_input,
            residual_forecast=residual_forecast,
            custom_quantiles=custom_quantiles,
            quantiles=quantiles,
            symmetric_quantiles=symmetric_quantiles,
            device=device,
            **model_args,
        )
    elif isinstance(embedding, FullEmbedding):
        model_spec = model_spec_class(
            model_name=model_name,
            historic_window=historic_window,
            forecast_window=forecast_window,
            features_per_step=features_per_step,
            meta_features=meta_features,
            d_model=d_model,
            embedding=embedding,
            dropout=dropout,
            residual_input=residual_input,
            residual_forecast=residual_forecast,
            custom_quantiles=custom_quantiles,
            quantiles=quantiles,
            symmetric_quantiles=symmetric_quantiles,
            device=device,
            **model_args,
        )
    else:
        raise ValueError()

    # set up DatasetSpec
    if dataset_name is not None:
        data_spec = predefined_dataspecs_dict[dataset_name]
        data_spec.root_path = dataset_root_path
        # TODO update the other stuff if necessary
    else:
        assert (
            dataset_file_path is not None and dataset_time_col is not None
        ), "Either dataset_name or file_name and time_column_name must be specified"
        data_spec = DataSpec(
            file_name=dataset_file_path,
            time_column_name=dataset_time_col,
            data_column_names=dataset_data_cols,
            metadata_column_names=dataset_metadata_cols,
            country_code=dataset_country_code,
            universal_holidays=universal_holidays,
            downsample_rate=downsample_rate,
            split_by_category=split_by_category,
            remove_flatline=remove_flatline,
            root_path=dataset_root_path,
        )

    dataset_spec = ResidualDatasetSpec(
        historic_window=historic_window,
        forecast_window=forecast_window,
        features_per_step=features_per_step,
        data_spec=data_spec,
        normalizer=normalizer,
        train_share=train_share,
        test_share=test_share,
        reduce=reduce,
        device=device,
        residual_normalizer=residual_normalizer,
        rhp_dataset=rhp_dataset,
        rhp_cycles=rhp_cycles,
        rhp_cycle_len=rhp_cycle_len,
    )

    # Prepare TrainSpec arguments
    if (custom_quantiles is not None) or (quantiles is not None):
        # quantiles will always overwrite loss to quantile loss
        loss = get_quantile_loss(
            custom_quantiles=custom_quantiles,
            quantile_nr=quantiles,
            symmetric_quantiles=symmetric_quantiles,
            invert_quantiles=invert_quantiles,
        )

    patience = patience or epochs
    optimizer_args = optimizer_args or {}  # {'weight_decay': 0.1}

    # set up TrainSpec
    train_spec = TrainSpec(
        log_learning_rate=log_learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        loss=loss,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        profiling=profiling,
    )

    # set up ActionSpec
    action_spec = ActionSpec(
        action=action,
        save_path=save_checkpoint_path,
        load_path=load_checkpoint,
        input_path=input_path,
        output_path=output_path,
    )

    return model_spec, dataset_spec, train_spec, action_spec


def _get_device():
    """Deduces the device models and data should be saved and run on"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
