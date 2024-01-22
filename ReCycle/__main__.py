import argparse

from .specs import spec_factory
from .perform_evaluation import perform_evaluation

# Logging
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")


parser = argparse.ArgumentParser(description="Load forecasting evaluation framework")

default_model = "Transformer"
default_dataset = "entsoe_de"

custom_quantiles = False
quantiles = None
assume_symmetric_quantiles = False
invert = False

# with symmetric quantiles only half the quantiles and the median need to be predicted
if assume_symmetric_quantiles:
    quantiles = quantiles // 2 + 1

train_mode_override = True
if train_mode_override:
    train = True
    load = False
    save = True
else:
    train = False
    load = True
    save = False

parser.add_argument(
    "-m", "--model_name", default=default_model, help="specifies model to use"
)

# General arguments
parser.add_argument(
    "--historic_window", default=21, type=int, help="number of input time steps"
)
parser.add_argument(
    "--forecast_window", default=7, type=int, help="number of forecast time steps"
)
parser.add_argument(
    "--features_per_step",
    default=24,
    type=int,
    help="number of features per input time step",
)

# Model arguments
parser.add_argument(
    "--meta_features",
    default=9,
    type=int,
    help="number of metadata_column_names features",
)
parser.add_argument(
    "--d_model", default=32, type=int, help="number of features after embedding"
)
parser.add_argument(
    "--embedding", default="default", help="embedding to use, check embeddings.py"
)
parser.add_argument(
    "--residual_input",
    default=True,
    type=bool,
    help="whether to use residual inputs",
)
parser.add_argument(
    "--residual_forecast",
    default=True,
    type=bool,
    help="whether to make residual forecasts",
)
parser.add_argument(
    "--custom_quantiles",
    default=custom_quantiles,
    type=bool,
    help="whether to use hardcoded quantiles",
)
parser.add_argument(
    "--quantiles",
    default=quantiles,
    type=int,
    help="number of quantiles for autogeneration",
)
parser.add_argument(
    "--assume_symmetric_quantiles",
    action="store_true",
    default=assume_symmetric_quantiles,
    help="whether quantiles are build symmetrically from the center or traditionally",
)
parser.add_argument(
    "--invert",
    action="store_true",
    default=invert,
    help="makes symmetric quantiles be lower quantiles instead of upper",
)
# Transformer specific arguments
parser.add_argument("--nheads", default=1, type=int, help="number of transformer heads")
parser.add_argument(
    "--dim_feedforward",
    default=32,
    type=int,
    help="dimension of final linear layer in transformer",
)
parser.add_argument(
    "--d_hidden", default=32, type=int, help="dimension of encoder linear layer"
)
# parser.add_argument('--meta_token', default=True, type=bool, help='whether to use non-zero start_token')
# TODO: add malformer, dropout

# dataset setup arguments
# parser.add_argument('--normalizer', default='min_max', type=str,
#                     help='data normalization to use, check normalizers.py')
parser.add_argument(
    "--train_share",
    default=0.6,
    type=float,
    help="fraction of dataset used for training",
)
parser.add_argument(
    "--tests_share",
    default=0.2,
    type=float,
    help="fraction of dataset used for testing",
)
parser.add_argument(
    "--reduce",
    default=None,
    type=float,
    help="reduce dataset size to specified fraction for testing",
)

# parser.add_argument('--residual_normalizer', default='none',
#                     help='normalization to use for residuals, check normalizers.py')
# parser.add_argument('--rhp_dataset', default='last_rhp', type=str,
#                     help='class of rhp to use, cf rhp_datasets.py')
parser.add_argument(
    "--rhp_cycles",
    default=3,
    type=int,
    help="number of past instances to use for rhp",
)
parser.add_argument(
    "--rhp_cycle_len",
    default=7,
    type=int,
    help="number of steps making occurrence of each rhp instance certain",
)

# Data specifications
parser.add_argument(
    "--dataset_name",
    default=default_dataset,
    help="fills in following specifications for known residuals, use None otherwise",
)
# These are for custom datasets probably easier to create a new spec in specs.dataset_specs though
parser.add_argument(
    "--file_name",
    default=None,
    type=str,
    help="name of the data file (should be csv)",
)
parser.add_argument(
    "--time_column_name",
    default=None,
    type=str,
    help="name of time data column in csv file",
)
parser.add_argument(
    "--data_column_names",
    default=None,
    type=str,
    help="name of data column in csv file",
)
# TODO: add country, downsample rate, remove flatline, split by category and others

# Training specifications
parser.add_argument(
    "--learning_rate", default=-3.0, type=float, help="log of learning rate"
)
parser.add_argument(
    "--batch_size", default=32, type=int, help="batch size during training"
)
parser.add_argument("--epochs", default=200, type=int, help="number of epochs to train")
parser.add_argument(
    "--patience", default=20, type=int, help="patience for early stopping"
)
parser.add_argument(
    "--profiling",
    action="store_true",
    default=False,
    help="enable training profiling",
)
# TODO: add loss and optimizer specification specification

# Training actions
parser.add_argument(
    "--train", action="store_true", default=train, help="train the model"
)
parser.add_argument(
    "--plot_loss",
    action="store_true",
    default=True,
    help="plot loss after training",
)
# Save and load
parser.add_argument("--save", action="store_true", default=save, help="save model")
parser.add_argument(
    "--save_path", default="./saved_models/", help="path to store model if saving"
)
parser.add_argument(
    "--load", action="store_true", default=load, help="load model from file"
)
parser.add_argument(
    "--load_path",
    default=None,
    help="path to load model from, if None same as save_path",
)

# Processing steps
parser.add_argument(
    "--test",
    action="store_true",
    default=True,
    help="evaluate error metrics on test set",
)
parser.add_argument(
    "--plot_prediction",
    action="store_true",
    default=False,
    help="plot some example predictions",
)

# TODO: enable parsing from string for: normalizer, residual_normalizer, rhp_dataset, loss and optimizer

parsed = vars(parser.parse_args())

if parsed["custom_quantiles"]:
    parsed["custom_quantiles"] = [0.8, 0.9]
else:
    parsed["custom_quantiles"] = None

model_spec, dataset_spec, train_spec, action_spec = spec_factory(**parsed)

perform_evaluation(
    model_spec=model_spec,
    dataset_spec=dataset_spec,
    train_spec=train_spec,
    action_spec=action_spec,
)
