import torch.nn as nn
from typing import List, Optional
import torch
import numpy as np
from torch import Tensor

from ae.models.ffnn import FeedForwardNeuralNet
from ae.models.autoencoder import AutoEncoder
from ae.sdes import SDE, SDEtorch


# The Second order term coming from Ito's applied to vector valued functions (by applying Ito's component wise)
def ambient_quadratic_variation_drift(latent_covariance: Tensor, decoder_hessian: Tensor) -> Tensor:
    qv = torch.einsum("njk,nrkj->nr", latent_covariance, decoder_hessian)
    return qv


class LatentNeuralSDE(nn.Module):
    def __init__(self,
                 intrinsic_dim: int,
                 drift_layers: List[int],
                 diff_layers: List[int],
                 drift_act: nn.Module,
                 diffusion_act: nn.Module,
                 encoder_act: Optional[nn.Module] = None,
                 spectral_normalize=False,
                 weight_normalize=False,
                 fro_normalize=False,
                 fro_max_norm=1.,
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
        neurons_mu = [intrinsic_dim] + drift_layers + [intrinsic_dim]
        neurons_sigma = [intrinsic_dim] + diff_layers + [intrinsic_dim ** 2]
        activations_mu = [drift_act for _ in range(len(neurons_mu) - 2)] + [encoder_act]
        activations_sigma = [diffusion_act for _ in range(len(neurons_sigma) - 2)] + [encoder_act]
        self.drift_net = FeedForwardNeuralNet(neurons_mu,
                                              activations_mu,
                                              spectral_normalize=spectral_normalize,
                                              weight_normalize=weight_normalize,
                                              fro_normalize=fro_normalize,
                                              fro_max_norm=fro_max_norm)
        self.diffusion_net = FeedForwardNeuralNet(neurons_sigma,
                                                  activations_sigma,
                                                  spectral_normalize=spectral_normalize,
                                                  weight_normalize=weight_normalize,
                                                  fro_normalize=fro_normalize,
                                                  fro_max_norm=fro_max_norm)

    def diffusion(self, z: Tensor) -> Tensor:
        """

        :param z:
        :return:
        """
        return self.diffusion_net(z).view((z.size(0), self.intrinsic_dim, self.intrinsic_dim))

    def latent_drift_numpy(self, t: float, z: np.ndarray) -> np.ndarray:
        """ For numpy EM SDE solvers"""
        with torch.no_grad():
            return self.drift_net(torch.tensor(z, dtype=torch.float32)).cpu().detach().numpy()

    def latent_drift_torch(self, t: float, z: torch.Tensor) -> torch.Tensor:
        """ For numpy EM SDE solvers"""
        with torch.no_grad():
            return self.drift_net(z)

    def latent_diffusion_numpy(self, t: float, z: np.ndarray) -> np.ndarray:
        """ For numpy EM SDE solvers"""
        d = z.shape[0]
        with torch.no_grad():
            return self.diffusion_net(torch.tensor(z, dtype=torch.float32)).view((d, d)).cpu().detach().numpy()

    def latent_diffusion_torch(self, t: float, z: torch.Tensor) -> torch.Tensor:
        """ For numpy EM SDE solvers"""
        d = z.shape[0]
        with torch.no_grad():
            return self.diffusion_net(z).view((d, d))

    def sample_paths(self, z0: Tensor, tn: float, ntime: int, npaths: int) -> Tensor:
        """

        :param z0:
        :param tn:
        :param ntime:
        :param npaths:
        :return:
        """
        # Initialize SDE object
        # latent_sde = SDE(self.latent_drift_fit, self.latent_diffusion_fit)
        latent_sde = SDEtorch(self.latent_drift_torch, self.latent_diffusion_torch)
        # Generate sample ensemble
        latent_ensemble = latent_sde.sample_ensemble(z0, tn, ntime, npaths, noise_dim=self.intrinsic_dim,
                                                     device=z0.device)
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

    def lift_sample_paths(self, latent_ensemble) -> torch.Tensor:
        # Lift the latent paths to the ambient space
        # lifted_ensemble = np.array([self.autoencoder.decoder(torch.tensor(path, dtype=torch.float32)).detach().numpy()
        #                             for path in latent_ensemble])
        lifted_ensemble = torch.stack([self.autoencoder.decoder(path)
                                    for path in latent_ensemble])
        return lifted_ensemble

    def compute_sde_manifold_tensors(self, x: Tensor, cov=None):
        z = self.autoencoder.encoder.forward(x)
        dphi = self.autoencoder.decoder_jacobian(z)
        latent_diffusion = self.latent_sde.diffusion(z)
        if cov is None:
            latent_covariance = torch.bmm(latent_diffusion, latent_diffusion.mT)
        else:
            dpi = self.autoencoder.encoder_jacobian(x)
            latent_covariance = torch.bmm(dpi, torch.bmm(cov, dpi.mT))
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

    def compute_ambient_drift(self, x: Tensor, cov=None) -> Tensor:
        z, dphi, b, q = self.compute_sde_manifold_tensors(x, cov)
        local_drift = self.latent_sde.drift_net.forward(z)
        tangent_drift = torch.bmm(dphi, local_drift.unsqueeze(2)).squeeze()
        mu_model = tangent_drift + 0.5 * q
        return mu_model

    def compute_ambient_covariance(self, x: Tensor, cov=None) -> Tensor:
        z, dphi, b, q = self.compute_sde_manifold_tensors(x, cov)
        bbt = torch.bmm(b, b.mT)
        cov_model = torch.bmm(dphi, torch.bmm(bbt, dphi.mT))
        return cov_model

    def compute_ambient_diffusion(self, x: Tensor, cov=None) -> Tensor:
        z, dphi, b, q = self.compute_sde_manifold_tensors(x, cov)
        diffusion = torch.bmm(dphi, b)
        return diffusion

    def ambient_diffusion_wrapper(self, t, x: Tensor):
        return self.compute_ambient_diffusion(x.view((1, 2))).view((2, ))

    def ambient_drift_wrapper(self, t, x: Tensor):
        return self.compute_ambient_drift(x.view((1, 2))).view((2, ))

    def direct_ambient_sample_paths(self, x0: Tensor, tn: float, ntime: int, npaths: int) -> Tensor:
        """

        :param z0:
        :param tn:
        :param ntime:
        :param npaths:
        :return:
        """
        # Initialize SDE object
        # latent_sde = SDE(self.latent_drift_fit, self.latent_diffusion_fit)
        ambient_sde = SDEtorch(self.ambient_drift_wrapper, self.ambient_diffusion_wrapper)
        # Generate sample ensemble
        ambient_ensemble = ambient_sde.sample_ensemble(x0, tn, ntime, npaths, noise_dim=self.extrinsic_dim,
                                                     device=x0.device)
        return ambient_ensemble




