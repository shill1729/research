"""
    An implementation of an auto-encoder using PyTorch. It is implemented as a class with various methods
    for computing objects from differential geometry (e.g. orthogonal projections).
"""
from typing import List, Optional
from torch import Tensor

import torch.nn as nn
import numpy as np
import torch

from ae.models.ffnn import FeedForwardNeuralNet

def decoded_covariance(local_cov: Tensor, decoder_jacobian: Tensor) -> Tensor:
    """

    :param local_cov:
    :param decoder_jacobian:
    :return:
    """
    return torch.bmm(torch.bmm(decoder_jacobian, local_cov), decoder_jacobian.mT)

class AutoEncoder(nn.Module):
    def __init__(self,
                 extrinsic_dim: int,
                 intrinsic_dim: int,
                 hidden_dims: List[int],
                 encoder_act: nn.Module,
                 decoder_act: nn.Module,
                 final_act: Optional[nn.Module] = None,
                 spectral_normalize: bool = False,
                 weight_normalize: bool = False,
                 fro_normalize: bool = False,
                 fro_max_norm: float = 5.,
                 tie_weights=False,
                 *args,
                 **kwargs):
        """
        An Auto-encoder using FeedForwardNeuralNet for encoding and decoding.

        Many methods are provided for differential geometry computations.

        :param extrinsic_dim: the observed extrinsic high dimension
        :param intrinsic_dim: the latent intrinsic dimension
        :param hidden_dims: list of hidden dimensions for the encoder and decoder
        :param encode_act: the encoder activation function
        :param decode_act: the decoder activation function
        :param final_act: the final activation layer for the encoder
        :param args: args to pass to nn.Module
        :param kwargs: kwargs to pass to nn.Module
        """
        super().__init__(*args, **kwargs)
        self.intrinsic_dim = intrinsic_dim
        self.extrinsic_dim = extrinsic_dim
        # Encoder and decoder architecture:
        encoder_neurons = [extrinsic_dim] + hidden_dims + [intrinsic_dim]
        # The decoder's layer structure is the reverse of the encoder
        decoder_neurons = encoder_neurons[::-1]
        if final_act is None:
            encoder_acts = [encoder_act] * (len(hidden_dims) + 1)
        else:
            encoder_acts = [encoder_act] * len(hidden_dims) + [final_act]
        # The decoder has no final activation, so it can target anything in the ambient space
        decoder_acts = [decoder_act] * len(hidden_dims) + [None]
        # TODO pass the normalization options
        self.encoder = FeedForwardNeuralNet(encoder_neurons, encoder_acts, spectral_normalize=spectral_normalize)
        self.decoder = FeedForwardNeuralNet(decoder_neurons, decoder_acts, spectral_normalize=spectral_normalize)

        # Tie the weights of the decoder to be the transpose of the encoder, in reverse order
        self.weights_tied = tie_weights
        if self.weights_tied:
            self.decoder.tie_weights(self.encoder)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the autoencoder.

        :param x: the observed point cloud of shape (batch_size, extrinsic_dim)
        :return: the reconstructed point cloud x_hat of shape
                 (batch_size, extrinsic_dim)
        """
        z = self.encoder.forward(x)
        x_hat = self.decoder.forward(z)
        return x_hat

    def encoder_jacobian(self, x: Tensor) -> Tensor:
        """
        Compute the Jacobian of the encoder at the given input.

        :param x: Input tensor of shape (batch_size, extrinsic_dim)
        :return: Jacobian matrix of shape (batch_size, intrinsic_dim, extrinsic_dim)
        """
        return self.encoder.jacobian_network(x)

    def decoder_jacobian(self, z: Tensor) -> Tensor:
        """
        Compute the Jacobian of the decoder at the given latent representation.

        :param z: Latent tensor of shape (batch_size, intrinsic_dim)
        :return: Jacobian matrix of shape (batch_size, extrinsic_dim, intrinsic_dim)
        """
        return self.decoder.jacobian_network(z)

    def decoder_hessian(self, z: Tensor) -> Tensor:
        """
        Compute the Hessian of the decoder at the given latent representation.

        :param z: Latent tensor of shape (batch_size, intrinsic_dim)
        :return: Hessian matrix of shape (batch_size, extrinsic_dim, intrinsic_dim, intrinsic_dim)
        """
        return self.decoder.hessian_network(z)

    def encoder_hessian(self, x: Tensor) -> Tensor:
        """
        Compute the Hessian of the encoder at the given input.

        :param x: Input tensor of shape (batch_size, extrinsic_dim)
        :return: Hessian matrix of shape (batch_size, intrinsic_dim, extrinsic_dim, extrinsic_dim)
        """
        return self.encoder.hessian_network(x)

    def neural_orthogonal_projection(self, z: Tensor) -> Tensor:
        """
            Compute the orthogonal projection onto the tangent space of the decoder at z.

            :param z: Latent tensor of shape (batch_size, intrinsic_dim)
            :return: Orthogonal projection matrix of shape
                     (batch_size, extrinsic_dim, extrinsic_dim)
        """
        dphi = self.decoder_jacobian(z)
        g = torch.bmm(dphi.mT, dphi)
        g_inv = torch.linalg.inv(g)
        P = decoded_covariance(g_inv, dphi)
        return P

    def neural_metric_tensor(self, z: Tensor) -> Tensor:
        """
            Compute the Riemannian metric tensor induced by the decoder at z.

            :param z: Latent tensor of shape (batch_size, intrinsic_dim)
            :return: Metric tensor of shape (batch_size, intrinsic_dim, intrinsic_dim)
        """
        dphi = self.decoder_jacobian(z)
        g = torch.bmm(dphi.mT, dphi)
        return g

    def compute_orthonormal_frame(self, x):
        z = self.encoder(x)
        dphi = self.decoder_jacobian(z)
        g = torch.bmm(dphi.mT, dphi)
        ginv = torch.linalg.inv(g)
        evals, evecs = torch.linalg.eigh(ginv)
        # Square root matrix via EVD:
        gframe = torch.bmm(evecs, torch.bmm(torch.diag_embed(torch.sqrt(evals)), evecs.mT))
        Hmodel = torch.bmm(dphi, gframe)
        return Hmodel

    def neural_volume_measure(self, z: Tensor) -> Tensor:
        """
            Compute the volume measure induced by the Riemannian metric tensor.

            :param z: Latent tensor of shape (batch_size, intrinsic_dim)
            :return: Volume measure tensor of shape (batch_size,)
        """
        g = self.metric_tensor(z)
        return torch.sqrt(torch.linalg.det(g))

    def brownian_drift_1(self, z: Tensor) -> Tensor:
        """
        Compute the drift vector field of Brownian motion on the manifold:
            0.5 * ∇_g · g^{-1}
        where the divergence is taken row-wise with respect to the Riemannian metric
        induced by the decoder at z.

        :param z: Latent tensor of shape (batch_size, intrinsic_dim)
        :return: Drift vector field of shape (batch_size, intrinsic_dim)
        """
        z = z.requires_grad_(True)
        batch_size, d = z.shape

        # Compute metric and its inverse
        g = self.neural_metric_tensor(z)  # (B, d, d)
        g_inv = torch.linalg.inv(g)       # (B, d, d)

        # Compute log(sqrt(det g)) = 0.5 * log det g
        log_sqrt_det_g = 0.5 * torch.logdet(g)  # (B,)
        grad_log_vol = torch.autograd.grad(
            outputs=log_sqrt_det_g.sum(), inputs=z, create_graph=True
        )[0]  # (B, d)

        # Compute divergence of each row of g^{-1}
        divergence_rows = []
        for i in range(d):
            row = g_inv[:, i, :]  # (B, d)
            grads = []
            for j in range(d):
                grad_row_j = torch.autograd.grad(
                    outputs=row[:, j].sum(), inputs=z, create_graph=True
                )[0][:, j]  # (B,)
                grads.append(grad_row_j)
            div_row_i = torch.stack(grads, dim=1).sum(dim=1)  # (B,)
            divergence_rows.append(div_row_i)

        divergence = torch.stack(divergence_rows, dim=1)  # (B, d)

        # Add the term from the volume form correction
        result = 0.5 * (divergence + grad_log_vol)  # (B, d)
        return result

    def brownian_drift_2(self, z: Tensor) -> Tensor:
        """
        Compute the intrinsic Brownian‐motion drift on the manifold via
        b^k = 0.5 * g^{ij} Gamma^k_{ij},
        with Gamma^k_{ij} = 0.5 * g^{kℓ}(∂_i g_{jℓ} + ∂_j g_{iℓ} - ∂_ℓ g_{ij}).

        More efficient than row‐wise divergence + volume correction.
        """
        z = z.requires_grad_(True)
        B, d = z.shape

        # 1) metric and inverse
        g = self.neural_metric_tensor(z)        # (B, d, d)
        g_inv = torch.linalg.inv(g)             # (B, d, d)

        drift = torch.zeros(B, d, device=z.device, dtype=z.dtype)
        for b in range(B):
            # 2) compute ∂_m g_{ij}(z[b]) via one Jacobian call
            def metric_fn(y: Tensor) -> Tensor:
                # returns (d,d) for single sample
                return self.neural_metric_tensor(y.unsqueeze(0))[0]
            # metric_grad[m, i, j] = ∂_m g_{ij}
            metric_grad = torch.autograd.functional.jacobian(
                metric_fn, z[b], create_graph=True
            )  # (d, d, d)

            # 4) contract to get b^k = 0.5 * g_inv^{ij} Γ^k_{ij}
            # note: Γ^k_{ij} = 0.5 * g_inv[b, k, ℓ] * (metric_grad[i,j,ℓ] + metric_grad[j,i,ℓ] - metric_grad[ℓ,i,j])
            gamma = 0.5 * torch.einsum(
                'kℓ,ijℓ->ijk',
                g_inv[b],
                (metric_grad.permute(2, 1, 0)  # ∂_i g_{jℓ}
                 + metric_grad.permute(1, 2, 0)  # ∂_j g_{iℓ}
                 - metric_grad)                  # - ∂_ℓ g_{ij}
            )  # (i,j,k)

            drift[b] = 0.5 * torch.einsum('ij,ijk->k', g_inv[b], gamma)

        return drift



    def lift_sample_paths(self, latent_ensemble: np.ndarray) -> np.ndarray:
        """
        Lift the latent paths to the ambient space using the decoder.

        :param latent_ensemble: An array of latent paths of shape
                                (num_samples, path_length, intrinsic_dim)
        :return: Lifted ensemble in the ambient space of shape
                 (num_samples, path_length, extrinsic_dim)
        """
        lifted_ensemble = np.array([self.decoder(torch.tensor(path, dtype=torch.float32)).detach().numpy()
                                    for path in latent_ensemble])
        return lifted_ensemble

    def plot_surface(self, a: float, b: float, grid_size: int, ax=None, title=None, dim=3, device="cpu") -> None:
        """
        Plot the surface produced by the neural-network chart.

        :param title:
        :param a: the lb of the encoder range box [a,b]^d
        :param b: the ub of the encoder range box [a,b]^d
        :param grid_size: grid size for the mesh of the encoder range
        :param ax: plot axis object
        :param dim: dimension to plot in, default 3
        :return:
        """
        if dim == 3:
            ux = np.linspace(a, b, grid_size)
            vy = np.linspace(a, b, grid_size)
            u, v = np.meshgrid(ux, vy, indexing="ij")
            x1 = np.zeros((grid_size, grid_size))
            x2 = np.zeros((grid_size, grid_size))
            x3 = np.zeros((grid_size, grid_size))
            for i in range(grid_size):
                for j in range(grid_size):
                    x0 = np.column_stack([u[i, j], v[i, j]])
                    x0 = torch.tensor(x0, dtype=torch.float32, device=device)
                    xx = self.decoder(x0).cpu().detach().numpy()
                    x1[i, j] = xx[0, 0]
                    x2[i, j] = xx[0, 1]
                    x3[i, j] = xx[0, 2]
            if ax is not None:
                ax.plot_surface(x1, x2, x3, alpha=0.5, cmap="magma")
                if title is not None:
                    ax.set_title(title)
                else:
                    ax.set_title("NN manifold")
            else:
                raise ValueError("'ax' cannot be None")
        elif dim == 2:
            u = np.linspace(a, b, grid_size)
            x1 = np.zeros((grid_size, grid_size))
            x2 = np.zeros((grid_size, grid_size))
            for i in range(grid_size):
                x0 = np.column_stack([u[i]])
                x0 = torch.tensor(x0, dtype=torch.float32, device=device)
                xx = self.decoder(x0).detach().numpy()
                x1[i] = xx[0, 0]
                x2[i] = xx[0, 1]
            if ax is not None:
                ax.plot(x1, x2, alpha=0.9)
                if title is not None:
                    ax.set_title(title)
                else:
                    ax.set_title("NN manifold")
        return None




