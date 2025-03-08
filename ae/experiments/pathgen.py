"""
    Sample path generation of the ground truth, ambient model and SDE-AEs.
"""
import torch
import numpy as np

from ae.experiments.datagen import ToyData
from ae.experiments.training import Trainer
from ae.models.local_neural_sdes import AutoEncoderDiffusion
from ae.sdes.sdes import SDE


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
        ambient_paths = np.zeros((npaths, ntime + 1, 3))
        for j in range(npaths):
            ambient_paths[j, :, :] = model.autoencoder.decoder(
                torch.tensor(latent_paths[j, :, :], dtype=torch.float32)).detach().numpy()
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

    def generate_paths(self, tn, ntime, npaths, seed=None):
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
        x0 = self.toydata.point_cloud.generate(1, seed=seed)[0]  # numpy (D,)
        ambient_gt, local_gt = self.toydata.ground_truth_ensemble(x0, tn, ntime, npaths)
        vanilla_ambient_paths = self.ambient_sde.sample_ensemble(x0, tn, ntime, npaths, noise_dim=3)
        x0_torch = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)  # torch (1,D)
        z0_dict = {name: self.__get_z0(model, x0_torch, name) for name, model in self.trainer.models.items()}
        ae_paths = {name: self.__generate_paths_ae(z0_dict[name], model, tn, npaths, ntime) for name, model in self.trainer.models.items()}
        ambient_ae_paths = {name: ae_paths[name][0] for name in self.trainer.models.keys()}
        local_ae_paths = {name: ae_paths[name][1] for name in self.trainer.models.keys()}
        return ambient_gt, vanilla_ambient_paths, ambient_ae_paths, local_gt, local_ae_paths
