import torch
import random
from mpi4py import MPI
import argparse

from specs import spec_factory
from perform_evaluation import perform_evaluation

from propulate import Islands
from propulate.propagators import SelectBest, SelectWorst
from propulate.utils import get_default_propagator

from typing import Callable, Dict


class ObjectiveFunction:
    def __init__(self, func: Callable, **kwargs) -> None:
        self.function = func
        self.kwargs = kwargs
        self.device, self.rank = self.get_device()

    @staticmethod
    def get_device(): # -> torch.device:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        device = torch.device(f'cuda:{rank%4}')
        return device, rank

    def __call__(self, hypers: Dict) -> float:
        print(f'{self.device=}, {self.rank=}')
        return self.function(**self.kwargs, **hypers, device=self.device)


def main_objective(**kwargs) -> float:
    datasets = None
    n_loops = 3
    total_loss = 0.

    specs = spec_factory(hyper_optimization_interrupt=True, **kwargs)

    for _ in range(n_loops):
        loss, datasets = perform_evaluation(*specs, premade_datasets=datasets)
        total_loss += loss

    return total_loss / n_loops


def dummy(params, device=None):
    return sum([params[x] ** 2 for x in params])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='oneshot_transformer', type=str, help='model to use')
    parser.add_argument('--dataset_name', default='uci_pt', type=str, help='dataset to use')
    parser.add_argument('--mode', default='evolve', type=str, help='chose evolve, extract or continue')
    parsed = parser.parse_args()

    model_name = parsed.model_name
    dataset_name = parsed.dataset_name
    mode = parsed.mode
    checkpoint = dataset_name + "_" + model_name + ".p"

    # model parameters
    model_args = dict(
        historic_window=21,
        forecast_window=7,
        features_per_step=24,
        dropout=0.,
        residual_input=True,
        residual_forecast=True,
        quantiles=None,
    )

    # set propulate parameters
    NUM_GENERATIONS = 100  # Set number of generations.
    POP_SIZE = 20  # Set size of breeding population.
    num_migrants = 1

    if mode == 'evolve':
        # standard optimisation mode
        load_checkpoint = ""
    elif mode == 'extract':
        # extract the best individual from checkpoint
        load_checkpoint = checkpoint
        NUM_GENERATIONS = 1
    elif mode == 'continue':
        # standard optimisation continuing from checkpoint
        load_checkpoint = checkpoint
    else:
        raise ValueError(f'Invalid mode: {mode}')

    # choose rhp_type
    if dataset_name in ['prices']:
        rhp_type = 'persistence'
    elif dataset_name in ['solar']:
        rhp_type = 'tenday'
    else:
        rhp_type = 'last_rhp'

    # Add meta features
    if dataset_name in ['solar']:
        meta_features = None
    else:
        meta_features = 9
    
    function = ObjectiveFunction(main_objective,
                                 **model_args,
                                 model_name=model_name,
                                 dataset_name=dataset_name,
                                 rhp_type=rhp_type,
                                 meta_features=meta_features,
                                 num_encoder_layers=1,
                                 num_decoder_layers=1
                                 )

    if dataset_name in ["entsoe_de", "water"]:
        # Entso-e de stats
        limits = dict(
            d_model=[4, 100],
            batch_size=[1, 256],
            learning_rate=[-5., -2.],
            nhead=[1, 8],
            #num_encoder_layers=[1, 5],
            #num_decoder_layers=[1, 5],
            dim_feedforward=[4, 1024],
            d_hidden=[4, 1024],
        )
    elif dataset_name in ["entsoe_full", "uci_pt", "solar", "etth1", "etth2", "prices"]:
        # Entsoe full ant uci pt stats
        limits = dict(
            d_model=[4, 1000],
            batch_size=[1, 256],
            learning_rate=[-5., -2.],
            nhead=[1, 8],
            #num_encoder_layers=[1, 5],
            #num_decoder_layers=[1, 5],
            dim_feedforward=[4, 1024],
            d_hidden=[4, 1024],
        )
    elif dataset_name in ["minigrid"]:
        # minigrid stats
        limits = dict(
            d_model=[4, 100],
            batch_size=[1, 27],
            learning_rate=[-5., -2.],
            nhead=[1, 8],
            #num_encoder_layers=[1, 5],
            #num_decoder_layers=[1, 5],
            dim_feedforward=[4, 1024],
            d_hidden=[4, 1024],
        )
    else:
        raise TypeError(f'Invalid dataset specification: {dataset_name}')

    rng = random.Random(MPI.COMM_WORLD.rank)
    propagator = get_default_propagator(POP_SIZE, limits, 0.7, 0.4, 0.1, rng=rng)
    islands = Islands(
        function,
        propagator,
        rng,
        generations=NUM_GENERATIONS,
        num_isles=12,
        load_checkpoint=load_checkpoint,  # pop_cpt.p",
        save_checkpoint=checkpoint,
        migration_probability=0.9,
        emigration_propagator=SelectBest,
        immigration_propagator=SelectWorst,
        pollination=True,
    )
    islands.evolve(top_n=1, logging_interval=1, DEBUG=2)

