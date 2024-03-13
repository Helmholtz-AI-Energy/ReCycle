from ..specs.dataset_specs import DatasetSpec
from .dataset import ResidualDataset


def build_datasets(dataset_spec: DatasetSpec):
    dataset_spec.check_validity()

    return ResidualDataset.from_csv(dataset_spec=dataset_spec)
