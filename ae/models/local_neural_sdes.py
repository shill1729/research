import torch.nn as nn
from typing import List, Optional
import torch
import numpy as np
from torch import Tensor

from ae.models.ffnn import FeedForwardNeuralNet
from ae.models.autoencoder import AutoEncoder
from ae.sdes import SDE


# The Second order term coming from Ito's applied to vector valued functions (by applying Ito's component wise)
def ambient_quadratic_variation_drift(latent_covariance: Tensor, decoder_hessian: Tensor) -> Tensor:
    qv = torch.einsum("njk,nrkj->nr", latent_covariance, decoder_hessian)
    return qv


class LatentNeuralSDE(nn.Module):
    def __init__(self,
                 intrinsic_dim: int,
                 h1: List[int],
                 h2: List[int],
                 drift_act: nn.Module,
                 diffusion_act: nn.Module,
                 encoder_act: Optional[nn.Module] = None,
                 *args,
                 **kwargs):
        """

        :param intrinsic_dim:
        :param h1:
        :param h2:
        :param drift_act:
        :param diffusion_act:
        :param encoder_act:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.intrinsic_dim = intrinsic_dim
        neurons_mu = [intrinsic_dim] + h1 + [intrinsic_dim]
        neurons_sigma = [intrinsic_dim] + h2 + [intrinsic_dim ** 2]
        activations_mu = [drift_act for _ in range(len(neurons_mu) - 2)] + [encoder_act]
        activations_sigma = [diffusion_act for _ in range(len(neurons_sigma) - 2)] + [encoder_act]
        self.drift_net = FeedForwardNeuralNet(neurons_mu, activations_mu, normalize=True)
        self.diffusion_net = FeedForwardNeuralNet(neurons_sigma, activations_sigma, normalize=True)

    def diffusion(self, z: Tensor) -> Tensor:
        """

        :param z:
        :return:
        """
        return self.diffusion_net(z).view((z.size(0), self.intrinsic_dim, self.intrinsic_dim))

    def latent_drift_fit(self, t: float, z: np.ndarray) -> np.ndarray:
        """ For numpy EM SDE solvers"""
        with torch.no_grad():
            return self.drift_net(torch.tensor(z, dtype=torch.float32)).detach().numpy()

    def latent_diffusion_fit(self, t: float, z: np.ndarray) -> np.ndarray:
        """ For numpy EM SDE solvers"""
        d = z.shape[0]
        with torch.no_grad():
            return self.diffusion_net(torch.tensor(z, dtype=torch.float32)).view((d, d)).detach().numpy()

    def sample_paths(self, z0: np.ndarray, tn: float, ntime: int, npaths: int) -> np.ndarray:
        """

        :param z0:
        :param tn:
        :param ntime:
        :param npaths:
        :return:
        """
        # Initialize SDE object
        latent_sde = SDE(self.latent_drift_fit, self.latent_diffusion_fit)
        # Generate sample ensemble
        latent_ensemble = latent_sde.sample_ensemble(z0, tn, ntime, npaths, noise_dim=self.intrinsic_dim)
        return latent_ensemble


class AutoEncoderDiffusion(nn.Module):
    def __init__(self,
                 latent_sde: LatentNeuralSDE,
                 ae: AutoEncoder,
                 *args,
                 **kwargs):
        """

        :param extrinsic_dim:
        :param intrinsic_dim:
        :param hidden_dims: [h1, h2, h3] where each h is a list of ints for hidden dimensions of AE, and drift,
        diffusion
        :param activations: [encoder_act, decoder_act, drift_act, diffusion_act]
        :param sde_final_act: optional final activation for drift and diffusion
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.observed_projection = None
        self.observed_ambient_drift = None

        self.latent_sde = latent_sde
        self.autoencoder = ae
        self.intrinsic_dim = ae.intrinsic_dim
        self.extrinsic_dim = ae.extrinsic_dim

    def lift_sample_paths(self, latent_ensemble) -> np.ndarray:
        # Lift the latent paths to the ambient space
        lifted_ensemble = np.array([self.autoencoder.decoder(torch.tensor(path, dtype=torch.float32)).detach().numpy()
                                    for path in latent_ensemble])
        return lifted_ensemble

    def compute_sde_manifold_tensors(self, x: Tensor):
        z = self.autoencoder.encoder.forward(x)
        dphi = self.autoencoder.decoder_jacobian(z)
        latent_diffusion = self.latent_sde.diffusion(z)
        latent_covariance = torch.bmm(latent_diffusion, latent_diffusion.mT)
        hessian = self.autoencoder.decoder_hessian(z)
        q = ambient_quadratic_variation_drift(latent_covariance, hessian)
        return z, dphi, latent_diffusion, q

    def compute_local_covariance(self, x: Tensor):
        z = self.autoencoder.encoder.forward(x)
        local_diffusion = self.latent_sde.diffusion(z)
        local_covariance = torch.bmm(local_diffusion, local_diffusion.mT)
        return local_covariance

    def compute_local_drift(self, x: Tensor):
        z = self.autoencoder.encoder.forward(x)
        local_drift = self.latent_sde.drift_net(z)
        return local_drift

    def compute_ambient_drift(self, x: Tensor) -> Tensor:
        z, dphi, b, q = self.compute_sde_manifold_tensors(x)
        local_drift = self.latent_sde.drift_net.forward(z)
        tangent_drift = torch.bmm(dphi, local_drift.unsqueeze(2)).squeeze()
        mu_model = tangent_drift + 0.5 * q
        return mu_model

    def compute_ambient_covariance(self, x: Tensor) -> Tensor:
        z, dphi, b, q = self.compute_sde_manifold_tensors(x)
        bbt = torch.bmm(b, b.mT)
        cov_model = torch.bmm(dphi, torch.bmm(bbt, dphi.mT))
        return cov_model



