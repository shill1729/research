import torch
from torch import nn as nn, Tensor

from ae.models import AutoEncoderDiffusion
from ae.models.sdes_latent import ambient_quadratic_variation_drift
from ae.models.losses.losses_autoencoder import MatrixMSELoss


class LocalCovarianceLoss(nn.Module):
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

    def __init__(self, norm, ambient=False):
        """

        :param tangent_drift_weight: weight for the tangent drift alignment penalty
        :param norm: matrix norm for the covariance error
        """
        super().__init__()
        self.cov_mse = MatrixMSELoss(norm="fro")
        self.ambient = ambient


    def forward(self, ae_diffusion: AutoEncoderDiffusion, x: Tensor, targets: Tensor):
        """
        Computes the total loss for the AutoEncoderDiffusion model.

        Parameters:
        -----------
        ae_diffusion : AutoEncoderDiffusion
            The AutoEncoderDiffusion model whose output is being evaluated.
        x : Tensor
            Input tensor to the autoencoder.
        targets : tuple of Tensors
            the encoded covariance computed using the encoder's jacobian and the ambient observed covariance.

        Returns:
        --------
        total_loss : Tensor
            The computed total loss combining covariance MSE, local covariance MSE, and weighted tangent drift loss.
        """
        # TODO comput ambient loss
        cov, encoded_cov, dphi = targets
        bbt = ae_diffusion.compute_local_covariance(x)
        if self.ambient:
            ambient_model_cov = torch.bmm(dphi, torch.bmm(bbt, dphi.mT))
            return self.cov_mse(ambient_model_cov, cov)
        else:
        # TODO for computing latent loss
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

    def __init__(self, ambient=False,*args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.ambient = ambient

    def forward(self, ae_diffusion: AutoEncoderDiffusion,
                x: Tensor,
                targets: Tensor) -> Tensor:
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
            (mu, encoded_cov, decoder_hessian).

        Returns:
        --------
        loss : Tensor
            The computed loss combining ambient drift error and latent drift error.
        """
        # TODO: toggle between ambient or latent losses
        # This below commented out is the ambient loss
        z = ae_diffusion.autoencoder.encoder(x)
        mu, encoded_cov, decoder_jacobian, decoder_hessian = targets
        if self.ambient:
            b = ae_diffusion.latent_sde.drift_net(z)
            q_hat = ambient_quadratic_variation_drift(encoded_cov, decoder_hessian)
            tangent_drift = torch.bmm(decoder_jacobian, b.unsqueeze(-1)).squeeze(-1)
            mu_hat = tangent_drift + 0.5 * q_hat
            sq_error = torch.linalg.vector_norm(mu_hat-mu, ord=2, dim=1)**2
            mse = torch.mean(sq_error)
            return mse
        else:
        # TODO: the below code computes the drift loss in the latent space by targeting
        #  Dpi(mu-0.5 q)
            # This code uses the encoded covariance from the AE
            # Option 2: use the Decoder for encoding the covariance
            # dphi = ae_diffusion.autoencoder.decoder_jacobian(z)
            # g = torch.bmm(dphi.mT, dphi)
            # ginv = torch.linalg.inv(g)
            # dpi_min = torch.bmm(ginv, dphi.mT)
            # encoded_cov = torch.bmm(dpi_min, torch.bmm(cov, dpi_min.mT))

            dpi = ae_diffusion.autoencoder.encoder_jacobian(x)
            hessian_decoder = ae_diffusion.autoencoder.decoder_hessian(z)
            q = ambient_quadratic_variation_drift(encoded_cov, hessian_decoder)
            # Model latent drift
            model_latent_drift = ae_diffusion.latent_sde.drift_net.forward(z)
            # Target latent drift, based on pre-trained AE
            tangent_drift_vector = mu - 0.5 * q
            observed_latent_drift = torch.bmm(dpi, tangent_drift_vector.unsqueeze(2)).squeeze(2)
            # Latent drift L^2 error
            latent_sq_error = torch.linalg.vector_norm(model_latent_drift - observed_latent_drift, ord=2, dim=1) ** 2
            latent_drift_mse = torch.mean(latent_sq_error)
            return latent_drift_mse
