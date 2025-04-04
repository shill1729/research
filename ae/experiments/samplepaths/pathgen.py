"""
    Sample path generation of the ground truth, ambient model and SDE-AEs.
"""
import torch
import numpy as np

from ae.toydata.datagen import ToyData
from ae.experiments.training.training import Trainer
from ae.models.local_neural_sdes import AutoEncoderDiffusion
from ae.sdes import SDE
from ae.utils import random_rotation_matrix, pad

class SamplePathGenerator:
    def __init__(self, toydata: ToyData, trainer: Trainer):
        """

        :param toydata:
        :param trainer:
        """
        self.toydata = toydata
        self.trainer = trainer
        self.ambient_sde = SDE(self.trainer.ambient_drift.drift_numpy, self.trainer.ambient_diffusion.diffusion_numpy)
        self.ground_truth_paths = None
        self.vanilla_ambient_paths = None
        self.ae_paths = None
        self.x0 = None
        self.z0_dict = None


    @staticmethod
    def __generate_paths_ae(z0, model: AutoEncoderDiffusion, tn, npaths, ntime):
        """

        :param z0:
        :param model:
        :param tn:
        :param npaths:
        :param ntime:
        :return: tuple (ambient_paths, latent_paths), of shape (npaths, ntime+1, D) and (npaths, ntime+1, d)
        """
        latent_paths = model.latent_sde.sample_paths(z0, tn, ntime, npaths)
        ambient_paths = model.lift_sample_paths(latent_paths)
        return ambient_paths, latent_paths

    @staticmethod
    def __get_z0(model: AutoEncoderDiffusion, x0_torch, name):
        """

        :param model:
        :param x0_torch:
        :param name:
        :return:
        """
        z0_tensor = model.autoencoder.encoder(x0_torch)
        x0_hat = model.autoencoder.decoder(z0_tensor).detach().numpy().squeeze(0)

        x0_numpy = x0_torch.squeeze(0).detach().numpy()
        z0_numpy = z0_tensor.detach().numpy().squeeze(0)
        print("\n " + str(name))
        print("l1 Recon Error for x0 = " + str(np.linalg.vector_norm(x0_hat - x0_numpy, ord=1)))
        print("l2 Recon Error for x0 = " + str(np.linalg.vector_norm(x0_hat - x0_numpy, ord=2)))
        return z0_numpy

    def get_best_point(self, num_samples=1000, seed=None, embed=False):
        """
        Generate random points and find the single point that minimizes
        the maximum reconstruction error over all models.

        :param models: Dictionary of autoencoder models {name: model}
        :param point_cloud: Object with a generate(num_samples, seed) method to create points
        :param num_samples: Number of points to sample
        :param seed: Random seed for reproducibility
        :return: The best point x0 (numpy array) minimizing max recon error across models
        """
        # Make sure we are generating in the original interior
        self.toydata.set_point_cloud()
        # x0_samples = self.toydata.point_cloud.generate(num_samples, seed=seed)[0]  # numpy (num_samples, D)
        d = self.toydata.point_cloud.dimension
        x0_samples = self.toydata.generate_data(num_samples, d, None, "cpu", embed)["x"]
        best_x0 = None
        best_max_recon_error = float("inf")

        for i in range(num_samples):
            x0 = x0_samples[i]  # numpy (D,)
            x0_torch = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)  # torch (1, D)

            max_recon_error = float("-inf")  # Track max error for this point

            for name, model in self.trainer.models.items():
                # print(x0_torch.size())
                z0_tensor = model.autoencoder.encoder(x0_torch)
                x0_hat = model.autoencoder.decoder(z0_tensor).detach().numpy().squeeze(0)

                x0_numpy = x0_torch.squeeze(0).detach().numpy()

                # Compute L2 reconstruction error
                recon_error = np.linalg.norm(x0_hat - x0_numpy, ord=2)
                max_recon_error = max(max_recon_error, recon_error)  # Take max over all models

            # Keep track of the point with the smallest max reconstruction error
            if max_recon_error < best_max_recon_error:
                best_max_recon_error = max_recon_error
                best_x0 = x0
        return best_x0

    def get_bd_point(self):
        """

        :return:
        """
        # self.toydata.set_point_cloud()
        # TODO: does not work for embedded data
        if self.toydata.large_dim is not None:
            raise NotImplemented("Sampling from the boundary for embedded data is not implemented yet.")
        a = self.toydata.surface.bounds()[0][0] - 0.01
        b = self.toydata.surface.bounds()[0][1] + 0.01
        bd_x0 = self.toydata.point_cloud.np_phi(a, b).reshape((1, 3)).squeeze(0)
        return bd_x0

    def generate_paths(self, tn, ntime, npaths, seed=None, embed=False, large_dim=3):
        """
        Generate ensemble paths for ground truth, ambient model, and autoencoder models

        Args:
            tn (float): Maximum time horizon
            ntime (int): Number of time steps per path
            npaths (int): Number of sample paths per model
            seed (int, optional): Random seed for reproducibility

        Returns:
            tuple: (ambient_gt, vanilla_ambient_paths, ambient_ae_paths, local_gt, local_ae_paths).
            ambient_ae_paths and local_ae_paths are dictionaries of key/items = name/ensemble
        """
        # TODO: right now the initial point 'x0' is generated internally. Do we want the option to pass it?
        # x0 = self.toydata.point_cloud.generate(1, seed=seed)[0]  # numpy (D,)
        # x0 = self.get_best_point(embed=embed).detach().numpy()
        x0 = self.get_bd_point()
        print("Point with smallest reconstruction error")
        print(x0)
        # x0 = self.toydata.point_cloud.np_phi(*[0.8, 0.8]).squeeze(1)
        # print("Point near the boundary")
        # print(x0)
        x0_torch = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)  # torch (1,D)
        z0_dict = {name: self.__get_z0(model, x0_torch, name) for name, model in self.trainer.models.items()}
        ambient_gt, local_gt = self.toydata.ground_truth_ensemble(x0, tn, ntime, npaths)
        # Embed ground-truth ambient paths if testing in D >> 3
        if embed:
            embedded_gt = np.zeros((npaths, ntime+1, large_dim))
            for i in range(npaths):
                Q = random_rotation_matrix(D=large_dim, seed=self.toydata.embedding_seed)
                embedded_gt[i] = pad(ambient_gt[i], large_dim) @ Q.T
        vanilla_ambient_paths = self.ambient_sde.sample_ensemble(x0, tn, ntime, npaths, noise_dim=large_dim)
        ae_paths = {name: self.__generate_paths_ae(z0_dict[name], model, tn, npaths, ntime) for name, model in self.trainer.models.items()}
        ambient_ae_paths = {name: ae_paths[name][0] for name in self.trainer.models.keys()}
        local_ae_paths = {name: ae_paths[name][1] for name in self.trainer.models.keys()}
        if embed:
            return embedded_gt, vanilla_ambient_paths, ambient_ae_paths, local_gt, local_ae_paths
        else:
            return ambient_gt, vanilla_ambient_paths, ambient_ae_paths, local_gt, local_ae_paths
