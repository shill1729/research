from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from ae.models.autoencoder import AutoEncoder
from ae.models.local_neural_sdes import ambient_quadratic_variation_drift, AutoEncoderDiffusion


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
        sigma_sq_sum = torch.linalg.matrix_norm(a, ord="fro")**2
        loss = 2*torch.mean(d-sigma_sq_sum)
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

    def forward(self, input_data: Tensor, target: Tensor) -> Tensor:
        """
            Compute the mean square error between two matrices under any matrix-norm.
        :param input_data: tensor of shape (n, a, b)
        :param target: tensor of shape (n, a, b)
        :return: tensor of shape (1, ).
        """
        square_error = torch.linalg.matrix_norm(input_data - target, ord=self.norm) ** 2
        mse = torch.mean(square_error)
        return mse


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

    def __init__(self, norm="fro", *args, **kwargs):
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


class TangentDriftAlignment(nn.Module):

    def __init__(self, *args, **kwargs):
        """
            A regularization term to align the higher-order geometry of an autoencoder with an observed
        ambient drift. Specifically, the ambient drift minus the 2nd-order ito-correction term should be
        tangent to the manifold. Hence the orthogonal projection onto the normal bundle should be zero. We minimize
        this norm to make it zero for an observed orthgonal projection. The model inputs go into the
        ito-correction term. A proxy to the local-covariance is used by encoding the ambient covariance via
        Ito's lemma when mapping from D -> d. The ito-correction term also uses the Hessian of the decoder,
        so this is a second-order regularization.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.reconstruction_loss = nn.MSELoss()

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
        return torch.mean(torch.linalg.vector_norm(normal_projected_tangent_drift, ord=2))


class TangentDriftAlignment2(nn.Module):
    """
        A regularization term to align the higher-order geometry of an autoencoder with an observed
        ambient drift. Specifically, the ambient drift minus the 2nd-order ito-correction term should be
        tangent to the manifold. Hence the orthogonal projection onto the normal bundle should be zero. We minimize
        this norm to make it zero for an observed orthgonal projection. The model inputs go into the
        ito-correction term. A proxy to the local-covariance is used by encoding the ambient covariance via
        Ito's lemma when mapping from D -> d. The ito-correction term also uses the Hessian of the decoder,
        so this is a second-order regularization.
    """

    def __init__(self, *args, **kwargs):
        """
            A regularization term to align the higher-order geometry of an autoencoder with an observed
        ambient drift. Specifically, the ambient drift minus the 2nd-order ito-correction term should be
        tangent to the manifold. Hence the orthogonal projection onto the normal bundle should be zero. We minimize
        this norm to make it zero for an observed orthgonal projection. The model inputs go into the
        ito-correction term. A proxy to the local-covariance is used by encoding the ambient covariance via
        Ito's lemma when mapping from D -> d. The ito-correction term also uses the Hessian of the decoder,
        so this is a second-order regularization.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.reconstruction_loss = nn.MSELoss()

    @staticmethod
    def forward(encoder_jacobian, decoder_hessian, ambient_cov, ambient_drift, observed_normal_proj):
        """
            A regularization term to align the higher-order geometry of an autoencoder with an observed
        ambient drift. Specifically, the ambient drift minus the 2nd-order ito-correction term should be
        tangent to the manifold. Hence the orthogonal projection onto the normal bundle should be zero. We minimize
        this norm to make it zero for an observed orthgonal projection. The model inputs go into the
        ito-correction term. A proxy to the local-covariance is used by encoding the ambient covariance via
        Ito's lemma when mapping from D -> d. The ito-correction term also uses the Hessian of the decoder,
        so this is a second-order regularization.

        :param encoder_jacobian: tensor of shape (n, d, D)
        :param decoder_hessian: tensor of shape (n, D, d)
        :param ambient_cov: tensor of shape (n, D, D)
        :param ambient_drift: tensor of shape (n, D)
        :param observed_normal_proj: tensor of shape (n, D, D)
        :return: tensor of shape (1, ).
        """
        # Ito's lemma from D -> d gives a proxy to the local covariance using the Jacobian of the encoder
        bbt_proxy = torch.bmm(torch.bmm(encoder_jacobian, ambient_cov), encoder_jacobian.mT)
        # The QV correction from d -> D with the proxy-local cov: q^i = < bb^T, nabla^2 phi^i >_F
        qv = ambient_quadratic_variation_drift(bbt_proxy, decoder_hessian)
        # The ambient drift mu = Dphi a + 0.5 q should satisfy v := mu-0.5 q has P v = 0 since v = Dphi a is tangent
        # to the manifold
        tangent_drift = ambient_drift - 0.5 * qv
        # Compute N . (mu-0.5 q)
        normal_projected_tangent_drift = torch.bmm(observed_normal_proj, tangent_drift.unsqueeze(2)).squeeze(2)
        # Return the mean of the norm squared
        loss = torch.mean(torch.linalg.vector_norm(normal_projected_tangent_drift, ord=2))
        return loss


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


# To add a new loss regularization, simply add a weight to the weight class, then modify TotalLoss
@dataclass
class LossWeights:
    reconstruction: float = 1.
    encoder_contraction_weight: float = 0.0
    decoder_contraction_weight: float = 0.0
    tangent_space_error_weight: float = 0.0
    tangent_angle_weight: float = 0.0
    tangent_drift_weight: float = 0.0
    diffeomorphism_reg: float = 0.0


class TotalLoss(nn.Module):
    def __init__(self, weights: LossWeights, norm="fro", device="cpu", *args, **kwargs):
        """
        Compute the total loss for the autoencoder, as a weighted sum of the individual losses.

        :param weights: LossWeights dataclass instance
        """
        super().__init__(*args, **kwargs)
        self.weights = weights
        self.reconstruction_loss = ReconstructionLoss()
        self.contractive_reg = ContractivePenalty(norm="fro")
        self.diffeomorphism_reg = DiffeomorphicRegularization(norm)
        self.tangent_bundle_reg = TangentBundleRegularization(norm="fro")
        self.drift_alignment_reg = TangentDriftAlignment2()
        self.tangent_angles_reg = TangentSpaceAnglesLoss()
        self.device = device

    def forward(self, autoencoder: AutoEncoder, x, targets):
        """
        Compute the weighted total loss.

        :param autoencoder: The autoencoder model
        :param x: Input data to the autoencoder
        :param targets: Target data for reconstruction and regularization: (P, H, Sigma, mu)
        """
        true_projection, observed_frame, ambient_cov, ambient_drift = targets
        n, D, _ = true_projection.size()
        # Orthogonal projection onto the normal bundle
        true_normal_proj = torch.eye(D, device=self.device).expand(n, D, D) - true_projection
        # Encoded local coordinates
        z = autoencoder.encoder(x)
        # Reconstructed ambient coordinates
        reconstructed_points = autoencoder.decoder(z)

        # Initialize total loss.
        total_loss = 0
        # We always compute the reconstruction loss
        reconstruction_loss = self.reconstruction_loss(reconstructed_points, x)
        total_loss += self.weights.reconstruction * reconstruction_loss

        # Check which objects we need
        need_decoder_jacobian = (self.weights.tangent_space_error_weight > 0 or
                                 self.weights.diffeomorphism_reg > 0 or
                                 self.weights.decoder_contraction_weight > 0 or
                                 self.weights.tangent_angle_weight > 0
                                 )
        need_encoder_jacobian = (self.weights.diffeomorphism_reg > 0 or
                                 self.weights.encoder_contraction_weight > 0 or
                                 self.weights.tangent_drift_weight > 0
                                 )
        need_decoder_hessian = self.weights.tangent_drift_weight > 0
        need_metric_tensor = (self.weights.tangent_space_error_weight > 0 or
                              self.weights.tangent_angle_weight > 0
                              )

        decoder_jacobian = None
        encoder_jacobian = None
        decoder_hessian = None
        metric_tensor = None
        latent_distances = None
        if need_decoder_jacobian:
            decoder_jacobian = autoencoder.decoder_jacobian(z)
        if need_encoder_jacobian:
            encoder_jacobian = autoencoder.encoder_jacobian(x)
        if need_decoder_hessian:
            decoder_hessian = autoencoder.decoder_hessian(z)
        if need_metric_tensor and need_decoder_jacobian:
            metric_tensor = torch.bmm(decoder_jacobian.mT, decoder_jacobian)
        # Contractive regularization
        if self.weights.encoder_contraction_weight > 0:
            contractive_loss = self.contractive_reg(encoder_jacobian)
            total_loss += self.weights.encoder_contraction_weight * contractive_loss
        if self.weights.decoder_contraction_weight > 0:
            contractive_loss = self.contractive_reg(decoder_jacobian)
            total_loss += self.weights.decoder_contraction_weight * contractive_loss
        # Tangent Bundle regularization
        if self.weights.tangent_space_error_weight > 0:
            tangent_bundle_loss = self.tangent_bundle_reg.forward(decoder_jacobian, metric_tensor, true_projection)
            total_loss += self.weights.tangent_space_error_weight * tangent_bundle_loss
        # Drift alignment regularization
        if self.weights.tangent_drift_weight > 0:
            drift_alignment_loss = self.drift_alignment_reg.forward(encoder_jacobian,
                                                                    decoder_hessian,
                                                                    ambient_cov,
                                                                    ambient_drift,
                                                                    true_normal_proj)
            total_loss += self.weights.tangent_drift_weight * drift_alignment_loss
        # Diffeomorphism regularization
        if self.weights.diffeomorphism_reg > 0:
            diffeomorphism_loss1 = self.diffeomorphism_reg(decoder_jacobian, encoder_jacobian)
            total_loss += self.weights.diffeomorphism_reg * diffeomorphism_loss1
        if self.weights.tangent_angle_weight > 0:
            total_loss += self.weights.tangent_angle_weight * self.tangent_angles_reg.forward(observed_frame,
                                                                                              decoder_jacobian,
                                                                                              metric_tensor)
        return total_loss


class LocalDiffusionLoss(nn.Module):
    """
        A custom loss function for training an AutoEncoderDiffusion model. This loss combines several components:
        - Mean squared error (MSE) between the predicted covariance and the target covariance.
        - A tangent drift loss that penalizes deviations in the normal projection of the drift vector.

        The total loss is a weighted sum of these components, allowing for control over the influence of the tangent drift loss.

        Parameters:
        -----------
        tangent_drift_weight : float, optional (default=0.0)
            Weight applied to the tangent drift loss component.
        norm : str, optional (default="fro")
            Norm type to be used in the MSE loss calculations. The norm is applied when computing the matrix MSE.

        Attributes:
        -----------
        tangent_drift_weight : float
            Weight applied to the tangent drift loss component.
        cov_mse : MatrixMSELoss
            MSE loss instance for comparing model and target covariances in the ambient space.
        tangent_drift_loss : function
            Function used to calculate the tangent drift loss component.
    """

    def __init__(self, norm="fro"):
        """

        :param tangent_drift_weight: weight for the tangent drift alignment penalty
        :param norm: matrix norm for the covariance error
        """
        super().__init__()
        self.cov_mse = MatrixMSELoss(norm=norm)

    def forward(self, ae_diffusion: AutoEncoderDiffusion, x: Tensor, encoded_cov: Tensor):
        """
        Computes the total loss for the AutoEncoderDiffusion model.

        Parameters:
        -----------
        ae_diffusion : AutoEncoderDiffusion
            The AutoEncoderDiffusion model whose output is being evaluated.
        x : Tensor
            Input tensor to the autoencoder.
        encoded_cov : tuple of Tensors
            the encoded covariance computed using the encoder's jacobian and the ambient observed covariance.

        Returns:
        --------
        total_loss : Tensor
            The computed total loss combining covariance MSE, local covariance MSE, and weighted tangent drift loss.
        """
        bbt = ae_diffusion.compute_local_covariance(x)
        local_cov_mse = self.cov_mse(bbt, encoded_cov)
        total_loss = local_cov_mse
        return total_loss


class LocalDriftLoss(nn.Module):
    """
    A custom loss function for training an AutoEncoderDrift model. This loss function computes:
    - Mean squared error (MSE) between the model-predicted latent drift and the observed latent drift, encoded

    Parameters:
    -----------
    None. Inherits from nn.Module.

    Methods:
    --------
    forward(drift_model: AutoEncoderDrift, x: Tensor, targets: Tuple[Tensor, Tensor]) -> Tensor
        Computes the loss given the input data, the drift model, and the target drift and covariance.
    """

    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(ae_diffusion: AutoEncoderDiffusion, x: Tensor,
                observed_ambient_drift: Tensor) -> Tensor:
        """
        Computes the loss for the AutoEncoderDrift model.

        Parameters:
        -----------
        drift_model : AutoEncoderDrift
            The AutoEncoderDrift model whose output is being evaluated.
        x : Tensor
            Input tensor to the autoencoder.
        targets : tuple of Tensors
            A tuple containing the observed ambient drift vector and the target encoded covariance matrix
            (mu, encoded_cov).

        Returns:
        --------
        loss : Tensor
            The computed loss combining ambient drift error and latent drift error.
        """
        z, _, _, q = ae_diffusion.compute_sde_manifold_tensors(x)
        model_latent_drift = ae_diffusion.latent_sde.drift_net.forward(z)
        dpi = ae_diffusion.autoencoder.encoder_jacobian(x)
        tangent_drift_vector = observed_ambient_drift - 0.5 * q
        observed_latent_drift = torch.bmm(dpi, tangent_drift_vector.unsqueeze(2)).squeeze(2)
        latent_sq_error = torch.linalg.vector_norm(model_latent_drift - observed_latent_drift, ord=2, dim=1) ** 2
        latent_drift_mse = torch.mean(latent_sq_error)
        return latent_drift_mse
