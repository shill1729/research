import torch
import torch.nn as nn
from torch import Tensor
from typing import Union

from ae.models.sdes_latent import ambient_quadratic_variation_drift

def vector_mse(a, b):
    square_error = torch.linalg.vector_norm(a - b, ord=2, dim=1) ** 2
    mse = torch.mean(square_error)
    return mse

def matrix_mse(a, b, norm="fro"):
    square_error = torch.linalg.matrix_norm(a - b, ord=norm) ** 2
    mse = torch.mean(square_error)
    return mse

class TangentSpaceAnglesLoss(nn.Module):
    """ Equivalent to minimizing the frobenius error of P-P_hat, we can make the angle between subspaces zero.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(observed_frame, decoder_jacobian, metric_tensor):
        _, _, d = decoder_jacobian.size()
        # Compute the eigenvalue decomposition (EVD) of the metric tensor
        evals, evecs = torch.linalg.eigh(metric_tensor)
        # Compute the inverse square root of the eigenvalues, using the reciprocal square-root function .rsqrt()
        inv_sqrt_evals = torch.diag_embed(evals.rsqrt())
        # Compute the square root matrix via EVD
        gframe = torch.bmm(evecs, torch.bmm(inv_sqrt_evals, evecs.mT))
        model_frame = torch.bmm(decoder_jacobian, gframe)
        a = torch.bmm(observed_frame.mT, model_frame)
        sigma_sq_sum = torch.linalg.matrix_norm(a, ord="fro") ** 2
        loss = 2 * torch.mean(d - sigma_sq_sum)
        return loss


class MatrixMSELoss(nn.Module):
    """
        Compute the mean square error between two matrices under any matrix-norm.
    """

    def __init__(self, norm="fro", *args, **kwargs):
        """
            Compute the mean square error between two matrices under any matrix-norm.

        :param norm: the matrix norm to use: "fro", "nuc", -2, 2, inf, -inf, etc
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.norm = norm

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
            Compute the mean square error between two matrices under any matrix-norm.
        :param output: tensor of shape (n, a, b)
        :param target: tensor of shape (n, a, b)
        :return: tensor of shape (1, ).
        """
        return matrix_mse(output, target)


class ReconstructionLoss(nn.Module):
    """
        Compute the reconstruction loss between an auto-encoder and a given point cloud.
    """

    def __init__(self, *args, **kwargs):
        """
            Compute the reconstruction loss between an auto-encoder and a given point cloud.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, reconstructed_points: Tensor, points: Tensor) -> Tensor:
        """

        :param reconstructed_points: tensor of shape (n, D)
        :param points: tensor of shape (n, D)
        :return:
        """
        reconstruction_error = self.reconstruction_loss(reconstructed_points, points)
        return reconstruction_error


class ContractivePenalty(nn.Module):
    """
        Ensure an auto-encoder is contractive by bounding the Frobenius norm of its encoder's Jacobian.
    """

    def __init__(self, norm="fro", *args, **kwargs):
        """
            Ensure an auto-encoder is contractive by bounding the Frobenius norm of its encoder's Jacobian.
        :param norm: the matrix norm to use: "fro", "nuc", -2, 2, inf, -inf, etc
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.norm = norm

    def forward(self, encoder_jacobian: Tensor) -> Tensor:
        """
            Ensure an auto-encoder is contractive by bounding the Frobenius norm of its encoder's Jacobian.

        :param encoder_jacobian: tensor of shape (n, d, D)
        :return: tensor of shape (1, ).
        """
        encoder_jacobian_norm = torch.linalg.matrix_norm(encoder_jacobian, ord=self.norm)
        contraction_penalty = torch.mean(encoder_jacobian_norm ** 2)
        return contraction_penalty


class TangentBundleRegularization(nn.Module):
    """
        A regularization term to train the autoencoder's orthogonal projection to approximate an observed orthogonal
        projection
    """

    def __init__(self, norm:Union[str, int]="fro", *args, **kwargs):
        """
            A regularization term to train the autoencoder's orthogonal projection to approximate an observed
            orthogonal projection
        :param norm:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.matrix_mse = MatrixMSELoss(norm)

    def forward(self, decoder_jacobian: Tensor, metric_tensor: Tensor, true_projection: Tensor):
        """
            A regularization term to train the autoencoder's orthogonal projection to approximate an observed
            orthogonal projection

        :param decoder_jacobian: tensor of shape (n, D, d)
        :param metric_tensor: tensor of shape (n, d, d)
        :param true_projection: tensor of shape (n, D, D), the observed orthogonal projection onto the tangent space
        :return:
        """
        # Consider using EVD to compute inverses instead.
        inverse_metric_tensor = torch.linalg.inv(metric_tensor)
        model_projection = torch.bmm(decoder_jacobian, torch.bmm(inverse_metric_tensor, decoder_jacobian.mT))
        return self.matrix_mse(model_projection, true_projection)

class TangentBundleApproxPenalty(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(decoder_jacobian: Tensor, encoder_jacobian: Tensor, true_projection: Tensor):
        """
        Compute d - Tr(D_pi P D_phi) where:
        - D_pi is the encoder jacobian (d x D)
        - D_phi is the decoder jacobian (D x d)
        - P is the true projection matrix (D x D)
        - d is the latent dimensionality

        Args:
            decoder_jacobian: Tensor of shape (..., D, d) - D_phi
            encoder_jacobian: Tensor of shape (..., d, D) - D_pi
            true_projection: Tensor of shape (..., D, D) - P

        Returns:
            Tensor: d - Tr(D_pi P D_phi)
        """

        # Compute the matrix product D_pi P D_phi
        # encoder_jacobian @ true_projection @ decoder_jacobian
        # Shape: (..., d, D) @ (..., D, D) @ (..., D, d)
        # Result: (..., d, d)

        temp = torch.bmm(encoder_jacobian, true_projection)  # (..., d, D)
        product = torch.bmm(temp, decoder_jacobian)  # (..., d, d)

        # Compute the trace of the resulting matrix
        # For batched matrices, we sum over the diagonal elements
        trace = torch.diagonal(product, dim1=-2, dim2=-1).sum(dim=-1)  # (...,)

        # Get the dimensionality d (latent dimension)
        d = encoder_jacobian.shape[-2]  # d

        # Return d - Tr(D_pi P D_phi)
        return torch.mean(d - trace)

class NormalDriftPenalty(nn.Module):

    def __init__(self, *args, **kwargs):
        """
            Train the autoencoder to have quadratic variation term's normal component match the normal component
            of the observed drift. N(mu-0.5 q_model) = 0, where N is observed normal projection, and mu is data.

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(encoder_jacobian, decoder_hessian, ambient_cov, ambient_drift, observed_frame):
        """

        :param encoder_jacobian: tensor of shape (n, d, D)
        :param decoder_hessian: tensor of shape (n, D, d)
        :param ambient_cov: tensor of shape (n, D, D)
        :param ambient_drift: tensor of shape (n, D)
        :param observed_frame: tensor of shape (n, D, d)
        :return: tensor of shape (1, ).
        """
        # Ito's lemma from D -> d gives a proxy to the local covariance using the Jacobian of the encoder
        bbt_proxy = torch.bmm(torch.bmm(encoder_jacobian, ambient_cov), encoder_jacobian.mT)
        # The QV correction from d -> D with the proxy-local cov: q^i = < bb^T, nabla^2 phi^i >_F
        qv = ambient_quadratic_variation_drift(bbt_proxy, decoder_hessian)

        # The ambient drift mu = Dphi a + 0.5 q should satisfy v := mu-0.5 q has N v = 0 since v = Dphi a is tangent
        # to the manifold and N is the normal projection onto the manifold.
        tangent_drift = ambient_drift - 0.5 * qv

        # Compute v-Pv and minimize this norm. Use P=HH^T to avoid DxD products--Is this really necessary?
        frame_transpose_times_tangent_vector = torch.bmm(observed_frame.mT, tangent_drift.unsqueeze(2))
        tangent_projected = torch.bmm(observed_frame, frame_transpose_times_tangent_vector).squeeze(2)
        normal_projected_tangent_drift = tangent_drift - tangent_projected
        return torch.mean(torch.linalg.vector_norm(normal_projected_tangent_drift, ord=2, dim=1))


class DiffeomorphicRegularization(nn.Module):
    """
        A naive method to ensure diffeomorphism conditions for an auto-encoder pair
    """

    def __init__(self, norm="fro", *args, **kwargs):
        """
            A naive method to ensure diffeomorphism conditions for an auto-encoder pair
        :param norm:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.matrix_mse = MatrixMSELoss(norm)

    def forward(self, decoder_jacobian: Tensor, encoder_jacobian: Tensor):
        """
            A naive method to ensure diffeomorphism conditions for an auto-encoder pair

        :param decoder_jacobian: tensor of shape (n, D, d)
        :param encoder_jacobian: tensor of shape (n, d, D)
        :return: tensor of shape (1, ).
        """
        # For large D >> d, it is more memory efficient to store d x d, therefore we compute Dpi Dphi
        model_product = torch.bmm(encoder_jacobian, decoder_jacobian)
        # Subtract identity matrix in-place without expanding across batches
        n, d, _ = model_product.size()
        # Create a diagonal matrix in-place
        diag_indices = torch.arange(d)
        model_product[:, diag_indices, diag_indices] -= 1.0
        # Compute the matrix MSE between (Dpi * Dphi) and I without explicitly creating I
        return self.matrix_mse(model_product, torch.zeros_like(model_product))