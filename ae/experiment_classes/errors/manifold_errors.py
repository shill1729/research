import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ae.toydata.datagen import ToyData
from ae.models.sdes_latent import AutoEncoderDiffusion
from ae.models.losses.losses_autoencoder import TotalLoss, LossWeights
from ae.models.losses.losses_ambient import AmbientDriftLoss, AmbientCovarianceLoss
from ae.models.sdes_ambient import AmbientDriftNetwork, AmbientDiffusionNetwork
from ae.utils.performance_analysis import compute_test_losses
from ae.experiment_classes.training.helpers import save_plot
from ae.experiment_classes.training.training import Trainer
from ae.utils.plot_functions import plot_interior_boundary_highlight, plot_interior_boundary_recon, plot_interior_boundary_latent

class GeometryError:
    def __init__(self, toydata: ToyData, trainer: Trainer,
                 eps_max=1., device="cpu", show=False, embed=False):
        self.toydata = toydata
        self.ae_dict = trainer.models
        self.ambient_drift_loss = AmbientDriftLoss()
        self.ambient_diffusion_loss = AmbientCovarianceLoss()
        self.eps_max = eps_max
        self.device = device
        self.ae_loss = TotalLoss(LossWeights())
        self.ambient_drift = trainer.ambient_drift
        self.ambient_diffusion = trainer.ambient_diffusion
        self.trainer = trainer
        self.show = show
        self.embed = embed

    def generate_test_data(self, epsilon=1., n=1000, test_seed=None):
        self.toydata.set_point_cloud(epsilon, True)
        return self.toydata.generate_data(n, 2, test_seed, self.device, embed=self.embed)

    def compute_ambient_diff_and_drift_loss(self,
                                            drift_model: AmbientDriftNetwork,
                                            diffusion_model: AmbientDiffusionNetwork,
                                            x_subset, cov_subset, mu_subset):
        # Comment this and use it if you want to use local.
        # z_subset = self.trainer.models["vanilla"].autoencoder.encoder(x_subset)
        # 3/11/2025: it doesnt make sense to train mu(z) and sigma(z) ambiently.
        # The vanilla AE has its own mu(z) and sigma(z) network that gets trained locally in R^d and
        # mapped up to ambient dimension.
        # The vanilla ambient model must have mu(x) and sigma(x) in R^D (R^{D x D}) for sample paths
        # unless you do some encoding and decoding which I guess we could test too.
        mu_loss = self.ambient_drift_loss.forward(drift_model, x_subset, mu_subset)
        sigma_loss = self.ambient_diffusion_loss.forward(diffusion_model, x_subset, cov_subset)
        return torch.sqrt(mu_loss).item(), torch.sqrt(sigma_loss).item()

    def compute_our_ambient_diff_and_drift_loss(self, aedf: AutoEncoderDiffusion, x_subset, cov_subset, mu_subset):
        # Compute model diffusion loss
        mu_model = aedf.compute_ambient_drift(x_subset, cov_subset)
        cov_model = aedf.compute_ambient_covariance(x_subset)
        mu_loss = torch.sqrt(torch.mean(torch.linalg.vector_norm(mu_model - mu_subset, ord=2, dim=1) ** 2))
        cov_loss = torch.sqrt(torch.mean(torch.linalg.matrix_norm(cov_model - cov_subset, ord="fro") ** 2))
        return mu_loss.item(), cov_loss.item()

    # -------------------------------------
    # Helper: determine which test samples are in the interior (training domain)
    # -------------------------------------
    def is_interior_local(self, local_coords: torch.Tensor, bounds_list):
        interior_mask = torch.ones(local_coords.shape[0], dtype=torch.bool, device=local_coords.device)
        # interior_mask = torch.ones(local_coords.shape[0], dtype=torch.bool, device=self.device)
        for i, (low, high) in enumerate(bounds_list):
            interior_mask = interior_mask & (local_coords[:, i] >= low) & (local_coords[:, i] <= high)
        interior_mask = torch.tensor(interior_mask.detach().numpy(), dtype=torch.bool)
        return interior_mask.detach()

    def subset_error(self, mask,
                     model: AutoEncoderDiffusion,
                     x_test, mu_test, cov_test, p_test, h_test):
        if torch.any(mask):
            x_bnd = x_test[mask]
            p_bnd = p_test[mask]
            frame_bnd = h_test[mask]
            cov_bnd = cov_test[mask]
            mu_bnd = mu_test[mask]
            losses_subset = compute_test_losses(model,
                                                self.ae_loss,
                                                x_bnd, p_bnd, frame_bnd, cov_bnd, mu_bnd,
                                                device=self.device)
            drift_loss, diffusion_loss = self.compute_our_ambient_diff_and_drift_loss(model, x_bnd, cov_bnd, mu_bnd)
            ambient_drift_loss, ambient_diff_loss = self.compute_ambient_diff_and_drift_loss(self.ambient_drift,
                                                                                             self.ambient_diffusion,
                                                                                             x_bnd, cov_bnd, mu_bnd)
            return losses_subset, drift_loss, diffusion_loss, ambient_drift_loss, ambient_diff_loss

    def compute_and_plot_errors(self, eps_grid_size, num_test, test_seed, local_space=False):
        test_dict = self.generate_test_data(self.eps_max, num_test, test_seed)
        x_test_full = test_dict["x"]
        mu_test_full = test_dict["mu"]
        cov_test_full = test_dict["cov"]
        p_test_full = test_dict["p"]
        h_test_full = test_dict["orthonormal_frame"]
        local_x_test_full = test_dict["local_x"]

        epsilons = np.linspace(0.01, self.eps_max, eps_grid_size)

        loss_keys = [
            "reconstruction loss", "encoder contractive loss", "decoder contractive loss",
            "tangent bundle loss", "tangent angle loss", "tangent drift alignment loss",
            "diffeomorphism loss"
        ]

        model_loss_storage = {name: {lk: [] for lk in loss_keys} for name in self.ae_dict.keys()}
        model_drift_losses = {name: [] for name in self.ae_dict.keys()}
        model_diffusion_losses = {name: [] for name in self.ae_dict.keys()}
        ambient_drift_losses = []
        ambient_diffusion_losses = []

        # Print interior losses
        interior_mask = self.is_interior_local(local_x_test_full, self.toydata.surface.bounds())
        results = {}
        for name, model in self.ae_dict.items():
            losses_subset, drift_loss, diffusion_loss, ambient_drift_loss, ambient_diff_loss = \
                self.subset_error(interior_mask, model, x_test_full, mu_test_full, cov_test_full, p_test_full,
                                  h_test_full)

            combined_losses = {**losses_subset,
                               'drift_loss': drift_loss,
                               'diffusion_loss': diffusion_loss,
                               'ambient_drift_loss': ambient_drift_loss,
                               'ambient_diff_loss': ambient_diff_loss}

            results[name] = combined_losses
        interior_test_loss_df = pd.DataFrame(results)
        print("Interior test loss:")
        print(interior_test_loss_df)

        for eps in epsilons:
            current_bounds = [(b[0] - eps, b[1] + eps) for b in self.toydata.surface.bounds()]
            current_mask = self.is_interior_local(local_x_test_full, current_bounds)

            x_test = x_test_full[current_mask]
            mu_test = mu_test_full[current_mask]
            cov_test = cov_test_full[current_mask]
            p_test = p_test_full[current_mask]
            h_test = h_test_full[current_mask]
            local_x_test = local_x_test_full[current_mask]
            interior_mask = self.is_interior_local(local_x_test, self.toydata.surface.bounds())
            boundary_mask = ~interior_mask

            for name, model in self.ae_dict.items():
                losses = self.subset_error(boundary_mask, model, x_test, mu_test, cov_test, p_test, h_test)
                if losses is not None:
                    losses_subset, drift_loss, diffusion_loss, ambient_drift_loss, ambient_diff_loss = losses

                    for key in loss_keys:
                        model_loss_storage[name][key].append(losses_subset.get(key, np.nan))

                    model_drift_losses[name].append(drift_loss)
                    model_diffusion_losses[name].append(diffusion_loss)

                    if name == list(self.ae_dict.keys())[0]:
                        ambient_drift_losses.append(ambient_drift_loss)
                        ambient_diffusion_losses.append(ambient_diff_loss)
                else:
                    for key in loss_keys:
                        model_loss_storage[name][key].append(np.nan)
                    model_drift_losses[name].append(np.nan)
                    model_diffusion_losses[name].append(np.nan)
                    if name == list(self.ae_dict.keys())[0]:
                        ambient_drift_losses.append(np.nan)
                        ambient_diffusion_losses.append(np.nan)

        # Regular loss plots
        for key in loss_keys:
            fig = plt.figure()
            for name in self.ae_dict.keys():
                plt.plot(epsilons, model_loss_storage[name][key], label=name)
            plt.xlabel('Epsilon')
            plt.ylabel(key)
            plt.title(f'{key} vs Epsilon')
            plt.legend()
            if self.show:
                plt.show()
            save_plot(fig, self.trainer.exp_dir, plot_name=key+"_bd_errors")

        # Drift Loss Plot
        fig = plt.figure()
        for name in self.ae_dict.keys():
            plt.plot(epsilons, model_drift_losses[name], label=f"{name} drift loss")
        # plt.plot(epsilons, ambient_drift_losses, linestyle="--", label="Ambient drift loss", color="black")
        plt.xlabel("Epsilon")
        plt.ylabel("Drift Loss")
        plt.title("Drift Loss vs Epsilon")
        plt.legend()
        if self.show:
            plt.show()
        save_plot(fig, self.trainer.exp_dir, plot_name="drift_errors")

        # Diffusion Loss Plot
        fig = plt.figure()
        for name in self.ae_dict.keys():
            plt.plot(epsilons, model_diffusion_losses[name], label=f"{name} diffusion loss")
        # plt.plot(epsilons, ambient_diffusion_losses, linestyle="--", label="Ambient diffusion loss", color="black")
        plt.xlabel("Epsilon")
        plt.ylabel("Diffusion Loss")
        plt.title("Diffusion Loss vs Epsilon")
        plt.legend()
        if self.show:
            plt.show()
        save_plot(fig, self.trainer.exp_dir, plot_name="diffusion_bd_errors")
        plt.close(fig)

    def plot_int_bd_surface(self, epsilon=0.5):
        for name, model in self.trainer.models.items():
            fig = plot_interior_boundary_highlight(epsilon, self.toydata, model, name, self.device)
            save_plot(fig, self.trainer.exp_dir, plot_name=name+"_interior_boundary_highlight_gt_pts")
            fig2 = plot_interior_boundary_recon(epsilon, self.toydata, model, name, self.device)
            save_plot(fig2, self.trainer.exp_dir, plot_name=name+"_interior_boundary_highlight_recon_pts")
            fig3 = plot_interior_boundary_latent(epsilon, self.toydata, model, name, self.device)
            save_plot(fig3, self.trainer.exp_dir, plot_name=name + "_interior_boundary_highlight_latent_pts")
