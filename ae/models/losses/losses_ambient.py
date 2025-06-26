import torch
from torch import nn as nn, Tensor

from ae.models import AmbientDriftNetwork, AmbientDiffusionNetwork
from ae.models.losses.losses_autoencoder import vector_mse, matrix_mse


class AmbientDriftLoss(nn.Module):
    """
        Compute the mean square error between vectors
    """

    def __init__(self, *args, **kwargs):
        """
            Compute the mean square error between two matrices under any matrix-norm.

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    def forward(self, model: AmbientDriftNetwork, input_data: Tensor, target: Tensor) -> Tensor:
        """
            Compute the mean square error between two matrices under any matrix-norm.

        :param model: AmbientDriftNetwork
        :param input_data: tensor of shape (n, a, b)
        :param target: tensor of shape (n, a, b)
        :return: tensor of shape (1, ).
        """
        model_output = model.forward(input_data)
        return vector_mse(model_output, target)


class AmbientCovarianceLoss(nn.Module):
    """
        Compute the mean square error between two matrices under any matrix-norm.
    """

    def __init__(self, *args, **kwargs):
        """
            Compute the mean square error between two matrices under any matrix-norm.

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    def forward(self, model: AmbientDiffusionNetwork, input_data: Tensor, target: Tensor) -> Tensor:
        """
            Compute the mean square error between two matrices under any matrix-norm.

        :param model: AmbientDiffusionNetwork
        :param input_data: tensor of shape (n, a, b)
        :param target: tensor of shape (n, a, b)
        :return: tensor of shape (1, ).
        """
        model_output = model.forward(input_data)
        model_output = torch.bmm(model_output, model_output.mT)
        return matrix_mse(model_output, target)
