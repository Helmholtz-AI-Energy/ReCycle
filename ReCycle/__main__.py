# import argparse
import logging

import click
from configparser import ConfigParser
from pathlib import Path

from . import run
from .specs import configure_run


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
# quantiles = None
# symmetric_quantiles = False
# invert = False

# with symmetric quantiles only half the quantiles and the median need to be predicted
# if symmetric_quantiles:
#     quantiles = quantiles // 2 + 1
# TODO make sure number of quantiles is set correctly
# TODO enable load specs from config file

# # TODO: enable parsing from string for: normalizer, residual_normalizer, rhp_dataset, loss and optimizer

# if parsed["custom_quantiles"]:
#     parsed["custom_quantiles"] = [0.8, 0.9]
# else:
#     parsed["custom_quantiles"] = None


def configure(ctx, param, value):
    if value is not None:
        path = Path(value)
        if not path.exists():
            raise ValueError(f"{value} not a file")
        cfg = ConfigParser()
        cfg.read(path)
        try:
            options = dict(cfg["options"])
        except KeyError:
            options = {}
        ctx.default_map = options


# TODO if hpo, should multiple values be allowed for all model parameters?
# TODO check that naming of "token", "timestep" and so on is consistent
# TODO replace argparse with click and implement config file
@click.command()
# config file
@click.option(
    "--config",
    "-c",
    "config_file",
    type=click.Path(dir_okay=False),
    default=None,
    callback=configure,
    is_eager=True,
    expose_value=False,
    help="",
    show_default=True,
)
# basic setup
@click.argument(
    "action", type=click.Choice(("train", "test", "infer", "hpo"), case_sensitive=False)
)
# TODO fix this in action spec, still defaults train is true
# dataset options
@click.option(
    "--dataset_name",
    "dataset_name",
    type=str,
    default=None,
    show_default=True,
    help="Name of existing predefined dataset",
)
# TODO make choice
@click.option(
    "--dataset_file_path",
    "dataset_file_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        writable=False,
        path_type=Path,
    ),
)
# TODO this could be separate files for traing, validation, and test
@click.option(
    "--dataset_time_column",
    "dataset_time_col",
    type=str,
    default=None,
    show_default=True,
    help="Index of column in datsaset file or dataframe containing time stamps. If None, the first column is used.",
)
@click.option(
    "--dataset_data_columns",
    "dataset_data_cols",
    type=str,
    default=None,
    show_default=True,
    help="Index of columns in dataset file or dataframe containing time series data values.",
)
# TODO allow multiple values
@click.option(
    "--dataset_metadata_columns",
    "dataset_metadata_cols",
    type=str,
    default=None,
    show_default=True,
    help="index of columns in the dataset file or dataframe containing metadata",
)
# TODO allow multiple values
@click.option(
    "--dataset_train_share",
    "train_share",
    type=float,
    default=0.6,
    show_default=True,
    help="Fraction of dataset used for training",
)
@click.option(
    "--dataset_test_share",
    "test_share",
    type=float,
    default=0.2,
    show_default=True,
    help="Fraction of dataset used for validation",
)
@click.option(
    "--dataset_reduce",
    "reduce",
    type=float,
    default=None,
    show_default=True,
    help="Reduce dataset to given fraction. Applied before train/test split",
)
# TODO more dataset options: country?, downsample rate, remove flatline?, split by category?
# recent history profile
@click.option(
    "--rhp_cycles",
    "rhp_cycles",
    type=int,
    default=3,
    show_default=True,
    help="Number of past cycles to use for recent history profile",
)
@click.option(
    "--rhp_cycle_len",
    "rhp_cycle_len",
    type=int,
    default=7,
    show_default=True,
    help="Number of tokens in the sequence that make up one cycle of the recent history",
)
# training parameters
@click.option(
    "--model_name",
    "model_name",
    type=str,
    default="Transformer",
    show_default=True,
    help="Model to perform action with",
)
@click.option(
    "--model_historic_window",
    "historic_window",
    type=int,
    default=21,
    show_default=True,
    help="Number of input tokens the model uses",
)
@click.option(
    "--model_forecast_window",
    "forecast_window",
    type=int,
    default=7,
    show_default=True,
    help="Number of output tokens",
)
@click.option(
    "--model_primary_cycle_len",
    "features_per_step",
    type=int,
    default=24,
    show_default=True,
    help="Number of time steps to be collated into a single sequence token. For a monovariate timeseries this is the number of input features per token",
)
@click.option(
    "--model_meta_features",
    "meta_features",
    type=int,
    default=9,
    show_default=True,
    help="Number of metadata features.",
)
# TODO infer this from the len of metadata column names?
@click.option(
    "--model_d",
    "d_model",
    type=int,
    default=32,
    show_default=True,
    help="Latent space dimension of the model (per head for Transformer)",
)
# TODO update/make naming more generic for other models
@click.option(
    "--embedding", "embedding", type=str, default="default", show_default=True, help=""
)
# TODO fix naming, go to choice, fix help message
@click.option(
    "--model_residual_input",
    "residual_input",
    type=bool,
    default=True,
    show_default=True,
    help="If true, model uses residual inputs",
)
# TODO why is residual input/forecast a model parameter? would expect this to be handled on the dataset side
@click.option(
    "--model_residual_forecast",
    "residual_forecast",
    type=bool,
    default=True,
    show_default=True,
    help="If true, the model produces residual forecasts",
)
# @click.option(
#     "--model_quantiles",
#     "-q",
#     "quantiles",
#     multiple=True,
#     default=[None],
#     help="Number of quantiles, or lits of quantiles, each entry given with -q quantile1 -q quantile2 and so on. If giving a number of quantiles, mind the '--symmetric_quantiles' flag.",
# )
@click.option("--quantiles", "quantiles", default=None, type=int)
# TODO fix quantiles
@click.option(
    "--model_symmetric_quantiles",
    "symmetric_quantiles",
    type=bool,
    default=True,
    show_default=True,
    help="If true, and quantiles are not set explicitely, generates the quantiles symmetrically rather than traditionally.",
)
# TODO what does traditionally mean?
@click.option(
    "--model_invert_quantiles",
    "invert_quantiles",
    type=bool,
    default=False,
    show_default=True,
    help="Makes symmetric quantiles be lower instead of upper",
)
# TODO wot?
@click.option(
    "--model_log_learning_rate",
    "log_learning_rate",
    type=float,
    default=-3.0,
    show_default=True,
    help="log of the model learning rate",
)
@click.option(
    "--model_batch_size",
    "batch_size",
    type=int,
    default=32,
    show_default=True,
    help="batch size during model training",
)
@click.option(
    "--model_epochs",
    "epochs",
    type=int,
    default=200,
    show_default=True,
    help="number of training epochs",
)
# TODO should these go into a training spec, instead of the model?
@click.option(
    "--model_patience",
    "patience",
    type=int,
    default=200,
    show_default=True,
    help="patience for early stopping during training",
)
# TODO profiling?
# TODO loss and optimizer options
@click.option(
    "--model_load_checkpoint",
    "load_checkpoint",
    type=click.Path(),
    default=None,
    show_default=True,
    help="Model checkpoint file to load",  # TODO
)
# TODO check. this should probably always be a file, or the most recent checkpoint from the dir
@click.option(
    "--model_save_checkpoint",
    "save_checkpoint_path",
    type=click.Path(),
    default=Path("./saved_models/"),
    show_default=True,
    help="",  # TODO
)
# TODO check. this should be a path and checkpoint filenames should be generated from epoch/iteration and monitor metric value?
@click.option(
    "--model_Transformer_num_heads",
    "model_Transformer_nheads",
    type=int,
    default=1,
    show_default=True,
    help="Number of transformer heads",
)
@click.option(
    "--model_Transformer_d_feedforward",
    "model_Transformer_dim_feedforward",
    type=int,
    default=32,
    show_default=True,
    help="dimension of final transformer layer",
)
# TODO consistent and better naming, dim_feedfoward should probably be d_out, this could also be a generic model parameter, rather than transformer specific
# TODO infer this from output window and primary cycle len
@click.option(
    "--model_Transformer_d_hidden",
    "model_Transformer_d_hidden",
    type=int,
    default=32,
    show_default=True,
    help="Dimension of the hidden latent space in the feedforward layer of an attention block",
)
@click.option(
    "--model_Transformer_meta_token",
    "model_Transformer_meta_token",
    type=bool,
    default=True,
    show_default=True,
    help="Use metadata as transformer decoder input",
)
# TODO divide the config into sections, at least one per spec object or so
def main(**kwargs):
    # NOTE build specs objects from parameters
    model_name = kwargs["model_name"]
    model_params = {
        "_".join(k.split("_")[2:]): kwargs[k]
        for k in kwargs
        if (len(k.split("_")) >= 2 and k.split("_")[1]) == model_name
    }
    for key in model_params:
        del kwargs[f"model_{model_name}_{key}"]
    kwargs["model_args"] = model_params
    model_spec, dataset_spec, train_spec, action_spec = configure_run(**kwargs)

    # TODO build dataset from data and dataset specs
    # TODO build model from model specs
    # TODO run action with model on data from action specs
    # TODO train_spec is only needed when actually training
    run.run_action(
        model_spec=model_spec,
        dataset_spec=dataset_spec,
        train_spec=train_spec,
        action_spec=action_spec,
    )


main()
