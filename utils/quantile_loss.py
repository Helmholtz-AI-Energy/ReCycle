import torch

from torch.nn.modules.loss import _Loss

from typing import Optional, List, Union
from torch import Tensor


class QuantileLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self,
                 quantiles: Union[List[float], Tensor],
                 reduction: str = 'mean') -> None:

        super().__init__(reduction=reduction)

        self.quantile_nr = len(quantiles)
        # Only quantiles - 0.5 is needed so do it right away
        self._quantile_values = torch.tensor(quantiles) - 0.5

    def get_quantile_values(self) -> Tensor:
        return self._quantile_values() + 0.5

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape[:-1] == target.shape, f'Input shape {input.shape[:-1]} does not match target shape {target.shape}'
        assert input.shape[-1] == self.quantile_nr, f'Input quantiles {input.shape[-1]} do not match specified quantiles {self.quantile_nr}'

        difference = input - target.unsqueeze(-1).expand(*target.shape, self.quantile_nr)

        # Calculate q * |x| for x > 0; (1 - q) |x| for x < 0
        # where q = self.quantile_values, x = difference
        # write as (0.5 + sign(x)(q - 0.5)) * |x| for vectorization
        # This is where the 0.5 that would be added is subtracted again add we drop both
        # So calculate 0.5 + sign(x)(q - 0.5), but since |x| = sign(x)*x and sign(x)**2 = 1
        # becomes 0.5 * sign(x) * x + (q - 0.5) * x
        point_loss = 0.5 * torch.abs(difference) + torch.mul(self._quantile_values.expand(*target.shape, -1), difference)

        if self.reduction == 'mean':
            output = torch.mean(point_loss)
        elif self.reduction == 'sum':
            output = torch.sum(point_loss)
        else:
            raise ValueError('Invalid reduction type')

        return output


class SymmetricQuantileLoss(QuantileLoss):
    def __init__(self,
                 quantile_nr: int = 2,
                 custom_quantiles: Optional[Union[List[int], Tensor]] = None,
                 reduction: str = 'mean') -> None:
        """
        Defines a symmetric variant of the quantile loss

        :param int quantile_nr: 1 yields the median, 2 the median and the center 50% quantile, 3 median and the
            center 33% and 66%, ...
        :param Optional[Union[List[int], Tensor]] custom_quantiles: Specifies a list of quantile values. Note that this
            specifies the values between the boundaries so 0.5 yields the 75% quantile which can be symmetrized
        :param str reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed.
        """
        assert quantile_nr > 0, f'Non-zero, positive number of quantiles required'

        if custom_quantiles is not None:
            quantile_values = torch.tensor(custom_quantiles)
        else:
            quantile_values = torch.arange(0, 1, 1 / quantile_nr)

        quantile_values /= 2
        quantile_values += 0.5

        super().__init__(quantiles=quantile_values, reduction=reduction)

    def get_quantile_values(self) -> float:
        """This version returns the value of the symmetrized, enclosed interval so median is 0%, 25%-75% is 50%, ..."""
        return self.get_quantile_values() * 2


class EquidistantQuantileLoss(QuantileLoss):
    def __init__(self, quantile_nr: int = 9, reduction: str = 'mean') -> None:
        """
        Evenly spaces quantile_nr quantiles between 0 and 100%. Note if quantile_nr is even this will skip the median
        :param quantile_nr: 1 yields the median (50% quantile), 2 the 33% and 66% quantile,
            3 yields 25%, 50%, and 75% quantile, ...
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed.
        """
        assert quantile_nr > 0, f'Non-zero, positive number of quantiles required'
        stride = 1. / (quantile_nr + 1)
        quantile_values = torch.arange(stride, 1., stride)[:quantile_nr]

        super().__init__(quantiles=quantile_values, reduction=reduction)


def get_quantile_loss(
        custom_quantiles: Optional[Union[List[float], Tensor]] = None,
        quantile_nr: Optional[int] = None,
        symmetric_quantiles: bool = False,
        reduction: str = 'mean'
) -> QuantileLoss:

    if custom_quantiles is not None:
        if symmetric_quantiles:
            return SymmetricQuantileLoss(custom_quantiles=custom_quantiles, reduction=reduction)
        else:
            return QuantileLoss(quantiles=custom_quantiles, reduction=reduction)
    else:
        assert quantile_nr is not None, f'Either quantile_nr of custom_quantiles must be provided'
        if symmetric_quantiles:
            return SymmetricQuantileLoss(quantile_nr=quantile_nr, reduction=reduction)
        else:
            return EquidistantQuantileLoss(quantile_nr=quantile_nr, reduction=reduction)
