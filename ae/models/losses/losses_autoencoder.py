from dataclasses import dataclass

from ae.models.autoencoder import AutoEncoder
from ae.models.losses.penalties import *


# To add a new loss regularization, simply add a weight to the weight class, then modify TotalLoss
@dataclass
class LossWeights:
    reconstruction: float = 1.
    encoder_contraction_weight: float = 0.0
    decoder_contraction_weight: float = 0.0
    tangent_space_error_weight: float = 0.0
    tangent_angle_weight: float = 0.0
    tangent_approx_weight: float = 0.0
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
        self.drift_alignment_reg = NormalDriftPenalty()
        self.tangent_angles_reg = TangentSpaceAnglesLoss()
        self.tangent_approx_reg = TangentBundleApproxPenalty()
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
        # The re-encoded points: not used, unless you want to penalize recycliing
        # z_hat_hat = autoencoder.encoder(reconstructed_points)

        # Initialize total loss.
        total_loss = 0
        # We always compute the reconstruction loss
        reconstruction_loss = self.reconstruction_loss(reconstructed_points, x)
        total_loss += self.weights.reconstruction * reconstruction_loss

        # Check which objects we need
        need_decoder_jacobian = (self.weights.tangent_space_error_weight > 0 or
                                 self.weights.diffeomorphism_reg > 0 or
                                 self.weights.decoder_contraction_weight > 0 or
                                 self.weights.tangent_angle_weight > 0 or
                                 self.weights.tangent_drift_weight > 0 or
                                 self.weights.tangent_approx_weight > 0
                                 )
        need_encoder_jacobian = (self.weights.diffeomorphism_reg > 0 or
                                 self.weights.encoder_contraction_weight > 0 or
                                 self.weights.tangent_drift_weight > 0 or
                                 self.weights.tangent_approx_weight > 0
                                 )
        need_decoder_hessian = self.weights.tangent_drift_weight > 0
        need_metric_tensor = (self.weights.tangent_space_error_weight > 0 or
                              self.weights.tangent_angle_weight > 0 or
                              self.weights.tangent_drift_weight > 0
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
            # Using the penrose min norm inverse
            # ginv = torch.linalg.inv(metric_tensor)
            # encoder_min = torch.bmm(ginv, decoder_jacobian.mT)
            # Old implementation:
            # drift_alignment_loss = self.drift_alignment_reg.forward(encoder_jacobian=encoder_jacobian,
            #                                                         decoder_hessian=decoder_hessian,
            #                                                         ambient_cov=ambient_cov,
            #                                                         ambient_drift=ambient_drift,
            #                                                         observed_normal_proj=true_normal_proj,
            #                                                         decoder_jacobian=decoder_jacobian)
            drift_alignment_loss = self.drift_alignment_reg.forward(encoder_jacobian=encoder_jacobian,
                                                                    decoder_hessian=decoder_hessian,
                                                                    ambient_cov=ambient_cov,
                                                                    ambient_drift=ambient_drift,
                                                                    observed_frame=observed_frame)
            total_loss += self.weights.tangent_drift_weight * drift_alignment_loss
        # Diffeomorphism regularization
        if self.weights.diffeomorphism_reg > 0:
            diffeomorphism_loss1 = self.diffeomorphism_reg(decoder_jacobian, encoder_jacobian)
            total_loss += self.weights.diffeomorphism_reg * diffeomorphism_loss1
        if self.weights.tangent_angle_weight > 0:
            tangent_angle_loss = self.tangent_angles_reg.forward(observed_frame, decoder_jacobian, metric_tensor)
            total_loss += self.weights.tangent_angle_weight * tangent_angle_loss
        if self.weights.tangent_approx_weight > 0:
            approx_tangent_penalty = self.tangent_approx_reg(decoder_jacobian, encoder_jacobian, true_projection)
            total_loss += self.weights.tangent_approx_weight * approx_tangent_penalty
        return total_loss
