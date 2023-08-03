import torch
import pandas as pd
from torch import nn

from torch import Tensor

eps = 1e-8

# All error metrics here assume two dimensions: (sequence length, feature number)
# or three dimensions: (batch size, sequence length, feature number)
# Note: Add new metric to dictionary and the end of file


class ErrorMetric(nn.Module):
    """Base class for all error metrics"""
    @staticmethod
    def decorate_output(value: float) -> str:
        """Each metric should specify how to print the result correctly"""
        raise NotImplementedError


class MAPELoss(ErrorMetric):
    @staticmethod
    def forward(prediction: Tensor, reference: Tensor, time_resolved: bool = False) -> Tensor:
        """Calculates the mean average percentage error of [prediction] with respect to [reference]"""
        normalized = torch.abs(1 - torch.div(prediction, reference + eps))

        if time_resolved:
            dims = tuple(range(normalized.dim()))[:-2]  # last two are assumed to be time dimensions and maintained
            result = 100 * torch.mean(normalized, dim=dims).flatten()
        else:
            result = 100 * torch.mean(normalized)

        return result

    @staticmethod
    def decorate_output(value: float) -> str:
        return f'MAPE: {value}%'


class MAELoss(ErrorMetric):
    @staticmethod
    def forward(prediction: Tensor, reference: Tensor, time_resolved: bool = False) -> Tensor:
        """Calculates the mean absolute error of [prediction] with respect to [reference]"""
        absolute = torch.abs(reference - prediction)

        if time_resolved:
            dims = tuple(range(absolute.dim()))[:-2]  # last two are assumed to be time dimensions and maintained
            result = torch.mean(absolute, dim=dims).flatten()
        else:
            result = torch.mean(absolute)
        return result

    @staticmethod
    def decorate_output(value: float) -> str:
        return f'MAE: {value}'


class Bias(ErrorMetric):
    @staticmethod
    def forward(prediction: Tensor, reference: Tensor, time_resolved: bool = False) -> Tensor:
        """Calculates the bias (average difference) of [prediction] with respect to [reference]"""
        deviation = reference - prediction

        if time_resolved:
            dims = tuple(range(deviation.dim()))[:-2]  # last two are assumed to be time dimensions and maintained
            result = torch.mean(deviation, dim=dims).flatten()
        else:
            result = torch.mean(deviation)
        return result

    @staticmethod
    def decorate_output(value: float) -> str:
        return f'Bias: {value}'


class RPELoss(ErrorMetric):
    @staticmethod
    def forward(prediction: Tensor, reference: Tensor) -> Tensor:
        """Calculates the relative peak error of [prediction] with respect to [reference]"""
        # Find actual peaks and their positions
        reference_peak, position = torch.max(reference.flatten(start_dim=-2), dim=-1)
        # Find corresponding predictions
        prediction_peak = torch.gather(prediction.flatten(start_dim=-2), dim=0, index=position.unsqueeze(dim=-1))
        peak_ratio = torch.div(prediction_peak, reference_peak)
        result = torch.mean(100 * (peak_ratio - 1))
        return result

    @staticmethod
    def decorate_output(value: float) -> str:
        return f'RPE: {value}%'


class NSELoss(ErrorMetric):
    @staticmethod
    def forward(prediction: Tensor, reference: Tensor) -> Tensor:
        """Calculates the Nash-Sutcliffe efficiency of [prediction] with respect to [reference]"""
        # Calculate mean square deviation of reference
        reference_mean = torch.mean(reference, dim=(-1, -2), keepdim=True)
        reference_deviation = (reference - reference_mean)**2
        normalization = torch.sum(reference_deviation, dim=(-1, -2))

        # Calculate squared deviation of prediction from reference
        squared_deviation = torch.sum((reference - prediction)**2, dim=(-1, -2))
        result = torch.mean(1 - torch.div(squared_deviation, normalization))
        return result

    @staticmethod
    def decorate_output(value: float) -> str:
        return f'NSE: {value}'


class MSELoss(ErrorMetric):
    @staticmethod
    def forward(prediction: Tensor, reference: Tensor, time_resolved: bool = False) -> Tensor:
        """Calculates the mean square error of [prediction] with respect to [reference]"""
        if time_resolved:
            squared = (prediction - reference)**2
            dims = tuple(range(squared.dim()))[:-2]  # last two are assumed to be time dimensions and maintained
            result = torch.mean(squared, dim=dims).flatten()
        else:
            result = nn.MSELoss()(prediction, reference)
        return result

    @staticmethod
    def decorate_output(value: float) -> str:
        return f'MSE: {value}'


class PAPELoss(ErrorMetric):
    @staticmethod
    def forward(prediction: Tensor, reference: Tensor) -> Tensor:
        """Calculates the peak absolute percentage error of [prediction] with respect to [reference]"""
        normalized = torch.abs(1 - torch.div(prediction, reference + eps))
        result = torch.mean(100 * torch.max(normalized.flatten(start_dim=-2), dim=-1)[0])
        return result

    @staticmethod
    def decorate_output(value: float) -> str:
        return f'PAPE: {value}%'


error_metric_dict = dict(
    mape=MAPELoss,
    mae=MAELoss,
    mse=MSELoss,
    bias=Bias,
    #rpe=RPELoss,
    #nse=NSELoss,
    #pape=PAPELoss
)

time_resolvable_metrics = dict(
    mape=MAPELoss,
    mae=MAELoss,
    mse=MSELoss,
    bias=Bias
)


def apply_error_metric(prediction: Tensor, reference: Tensor, metric: str or list = 'all') -> pd.DataFrame:
    if metric == 'all':
        metric = error_metric_dict.keys()
    elif type(metric) is str:
        metric = [metric]
    
    prediction = prediction.cpu()
    reference = reference.cpu()

    result_summary = {}
    for key in metric:
        error_metric = error_metric_dict[key]()
        result = error_metric(prediction, reference)
        result_summary[key] = [result.item()]
        #print(error_metric.decorate_output(round_to_significant(result.item())))

    return pd.DataFrame(result_summary)
