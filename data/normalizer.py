from abc import ABC, abstractmethod

import torch
from torch import Tensor

from typing import Optional, Callable, Iterable, Type


import logging

logger = logging.getLogger(__name__)


def map_to_tensor(function: Callable, obj: Iterable) -> Tensor:
    return torch.tensor(list(map(function, obj)))


class Normalizer(ABC):
    def __init__(
        self, data: Tensor, use_categories: Optional[bool] = None, eps: float = 1e-15
    ) -> None:
        # unless category use is specified input_data with more than 1 dim is treated with categories
        logger.debug(f"{use_categories=}, {data.shape=}")
        self._use_categories = use_categories or (data.shape[0] != 1)
        self._eps = eps
        self.set_normalization(data)

    @abstractmethod
    def to(self, *args, **kwargs) -> None:
        """Applies torch.Tensor.to(*args, **kwargs) to all normalization constants"""
        raise NotImplementedError

    @abstractmethod
    def set_normalization(self, data: Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, data: Tensor, categories: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def revert_normalization(
        self, data: Tensor, categories: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError

    def __call__(self, data: Tensor, categories: Optional[Tensor] = None) -> Tensor:
        return self.normalize(data, categories)


class MinMax(Normalizer):
    _min: Tensor
    _max: Tensor

    def __init__(
        self, data: Tensor, use_categories: Optional[bool] = None, eps: float = 1e-15
    ) -> None:
        super(MinMax, self).__init__(data=data, use_categories=use_categories, eps=eps)

    def to(self, *args, **kwargs) -> None:
        self._min = self._min.to(*args, **kwargs)
        self._max = self._max.to(*args, **kwargs)

    def set_normalization(self, data: Tensor) -> None:
        # if self._use_categories:
        #    self._min = map_to_tensor(torch.min, data)
        #    self._max = map_to_tensor(torch.max, data)
        # else:
        #    self._min = data.min()
        #    self._max = data.max()

        start = 1 if self._use_categories else 0
        dims = [*range(start, data.dim())]
        self._min = data.nan_to_num(nan=torch.inf).amin(dim=dims)
        self._max = data.nan_to_num(nan=-torch.inf).amax(dim=dims)

    def normalize(self, data: Tensor, categories: Optional[Tensor] = None) -> Tensor:
        if self._use_categories:
            if categories is None:
                categories = torch.arange(data.shape[0])
            else:
                assert (
                    data.shape[0] == categories.shape[0]
                ), "invalid category specification"

            data_min = self._min[categories]
            data_max = self._max[categories]

            output_data = (
                (data.transpose(0, -1) - data_min) / (data_max - data_min + self._eps)
            ).transpose(0, -1)
        else:
            output_data = (data - self._min) / (self._max - self._min + self._eps)

        return output_data

    def revert_normalization(
        self, data: Tensor, categories: Optional[Tensor] = None
    ) -> Tensor:
        if self._use_categories:
            assert categories is not None, "category specification missing"

            data_min = self._min[categories]
            data_max = self._max[categories]

            output_data = (
                data.transpose(0, -1) * (data_max - data_min) + data_min
            ).transpose(0, -1)
        else:
            output_data = data * (self._max - self._min) + self._min

        return output_data


class ZeroMean(Normalizer):
    _mean: Tensor
    _sigma: Tensor

    def __init__(
        self, data: Tensor, use_categories: Optional[bool] = None, eps: float = 1e-15
    ) -> None:
        super(ZeroMean, self).__init__(
            data=data, use_categories=use_categories, eps=eps
        )

    def to(self, *args, **kwargs) -> None:
        self._mean = self._mean.to(*args, **kwargs)
        self._sigma = self._sigma.to(*args, **kwargs)

    def set_normalization(self, data: Tensor) -> None:
        if self._use_categories:
            self._mean = map_to_tensor(torch.mean, data)
            self._sigma = map_to_tensor(torch.std, data)
        else:
            self._mean = data.mean()
            self._sigma = data.std()

    def normalize(self, data: Tensor, categories: Optional[Tensor] = None) -> Tensor:
        if self._use_categories:
            if categories is None:
                categories = torch.arange(data.shape[0])
            else:
                assert (
                    data.shape[0] == categories.shape[0]
                ), "invalid category specification"

            data_mean = self._mean[categories]
            data_std = self._sigma[categories]

            output_data = (
                (data.transpose(0, -1) - data_mean) / (data_std + self._eps)
            ).transpose(0, -1)
        else:
            output_data = (data - self._mean) / (self._sigma + self._eps)

        return output_data

    def revert_normalization(
        self, data: Tensor, categories: Optional[Tensor] = None
    ) -> Tensor:
        if self._use_categories:
            assert categories is not None, "category specification missing"

            data_mean = self._mean[categories]
            data_std = self._sigma[categories]

            output_data = (data.transpose(0, -1) * data_std + data_mean).transpose(
                0, -1
            )
        else:
            output_data = data * self._sigma + self._mean

        return output_data


class AbsMax(Normalizer):
    _max: Tensor

    def __init__(
        self, data: Tensor, use_categories: Optional[bool] = None, eps: float = 1e-15
    ) -> None:
        super(AbsMax, self).__init__(data=data, use_categories=use_categories, eps=eps)

    def to(self, *args, **kwargs) -> None:
        self._max = self._max.to(*args, **kwargs)

    def set_normalization(self, data: Tensor) -> None:
        if self._use_categories:
            self._max = map_to_tensor(torch.max, torch.abs(data))
        else:
            self._max = torch.abs(data).max()

    def normalize(self, data: Tensor, categories: Optional[Tensor] = None) -> Tensor:
        if self._use_categories:
            if categories is None:
                categories = torch.arange(data.shape[0])
            else:
                assert (
                    data.shape[0] == categories.shape[0]
                ), "invalid category specification"

            data_max = self._max[categories]

            output_data = (data.transpose(0, -1) / data_max).transpose(0, -1)
        else:
            output_data = data / self._max

        return output_data

    def revert_normalization(
        self, data: Tensor, categories: Optional[Tensor] = None
    ) -> Tensor:
        if self._use_categories:
            assert categories is not None, "category specification missing"

            data_max = self._max[categories]

            output_data = (data.transpose(0, -1) * data_max).transpose(0, -1)
        else:
            output_data = data * self._max

        return output_data


class NoNormalizer(Normalizer):
    def __init__(
        self, data: Tensor, use_categories: Optional[bool] = None, eps: float = 1e-15
    ) -> None:
        super(NoNormalizer, self).__init__(
            data=data, use_categories=use_categories, eps=eps
        )

    def to(self, *args, **kwargs) -> None:
        pass

    def set_normalization(self, data: Tensor) -> None:
        pass

    def normalize(self, data: Tensor, categories: Optional[Tensor] = None) -> Tensor:
        return data

    def revert_normalization(
        self, data: Tensor, categories: Optional[Tensor] = None
    ) -> Tensor:
        return data


def select_normalizer(name: str) -> Type[Normalizer]:
    if name == "min_max":
        return MinMax
    elif name == "abs_max":
        return AbsMax
    elif name == "zero_mean":
        return ZeroMean
    elif name == "none":
        return NoNormalizer
    else:
        raise TypeError
