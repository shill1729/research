"""
Run this module then update the path in load_models.py and then run feynmankac_stats.py
"""
import torch
import torch.nn as nn
import sympy as sp
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- IMPORTS FROM CODE BASE ---
from ae.symbolic.diffgeo import RiemannianManifold
from ae.toydata.pointclouds import PointCloud
from ae.toydata.local_dynamics import *
from ae.toydata.surfaces import *

from ae.models.autoencoder import AutoEncoder
from ae.models.local_neural_sdes import LatentNeuralSDE, AutoEncoderDiffusion
from ae.models.losses import LossWeights, TotalLoss, LocalDiffusionLoss, LocalDriftLoss
from ae.models.losses import AmbientDriftLoss, AmbientDiffusionLoss
from ae.models.ambient_sdes import AmbientDriftNetwork, AmbientDiffusionNetwork

from ae.utils import process_data
from ae.models.fitting import ThreeStageFit, fit_model
from ae.utils.performance_analysis import compute_test_losses

from ae.sdes import SDE
from ae.experiments.helpers import *
# TODO: only plot the boundary errors. Print the interior errors.
# ---------------------------
# SET HYPERPARAMETERS & SEEDS
# ---------------------------
device = torch.device("cpu")
train_seed = None
test_seed = None
norm = "fro"
# torch.manual_seed(train_seed)
penalty = "reconstruction loss"
# Point cloud parameters
num_points = 30
num_test = 20000
batch_size = 20
eps_max = 1.
eps_grid_size = 20

# The intrinsic and extrinsic dimensions.
extrinsic_dim, intrinsic_dim = 3, 2
hidden_dims = [32]
diffusion_layers = [16]
drift_layers = [16]
lr = 0.001
weight_decay = 0.
epochs_ae = 12000
epochs_diffusion = 9000
epochs_drift = 9000
print_freq = 500
# Diffeo weight for accumulative orders
diffeo_weight_12 = 0.08  # this is the separate diffeo_weight for just the First order and second order
# First order weight: 0.08 was good
tangent_angle_weight = 0.08
# Second order weights accumulative
tangent_angle_weight2 = 0.08  # the first order weight for the second order model, if accumulating penalties
tangent_drift_weight = 0.001
# diffeo weight alone
# diffeo_weight = 0.2  # I think making this higher helps contract but worsens the second order
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
drift_act = nn.Tanh()
diffusion_act = nn.Tanh()
npaths = 2
ntime = 1000
tn = 1
# TODO: toggle these for annealing the second order weight
# anneal_weights = { "tangent_drift_weight": lambda epoch: tangent_drift_weight * (epoch / epochs_ae) if epoch >
# np.round(epochs_ae/3) else 0.}
anneal_weights = None

anneal_tag = "annealed_2nd" if anneal_weights is not None else "not_annealed"
# -------------------------------------
# CHOOSE THE SURFACE AND GET THE BOUNDS
# -------------------------------------
surface = Paraboloid(5., 5.)
bounds = surface.bounds()  # native bounds in the local coordinates
# For testing, we will enlarge these bounds by ε.

# Initialize the manifold and dynamics.
manifold = RiemannianManifold(surface.local_coords(), surface.equation())
dynamics = LangevinHarmonicOscillator()
local_drift = dynamics.drift(manifold)
local_diffusion = dynamics.diffusion(manifold)

cloud_train = PointCloud(manifold, bounds, local_drift, local_diffusion, compute_orthogonal_proj=True)
if __name__ == "__main__":
    params = {
        "num_points": num_points,
        "num_test": num_test,
        "batch_size": batch_size,
        "eps_max": eps_max,
        "eps_grid_size": eps_grid_size,
        "extrinsic_dim": extrinsic_dim,
        "intrinsic_dim": intrinsic_dim,
        "hidden_dims": hidden_dims,
        "diffusion_layers": diffusion_layers,
        "drift_layers": drift_layers,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs_ae": epochs_ae,
        "epochs_diffusion": epochs_diffusion,
        "epochs_drift": epochs_drift,
        "print_freq": print_freq,
        "tangent_angle_weight": tangent_angle_weight,
        "tangent_drift_weight": tangent_drift_weight,
        "diffeo_weight": diffeo_weight_12,
        "npaths": npaths,
        "ntime": ntime,
        "tn": tn,
    }
    # Automatically save to the right folder:
    base_dir = "trained_models/"+surface.__class__.__name__+"/"+dynamics.__class__.__name__
    exp_dir = setup_experiment_dir(params, base_dir, anneal_tag)
    print(f"Saving results to {exp_dir}")
    # -------------------------------------
    # GENERATE TRAINING DATA (POINT CLOUD)
    # -------------------------------------

    # Generate: points, weights, drifts, covariances, and local coordinates.
    x, _, mu, cov, local_x = cloud_train.generate(num_points, seed=train_seed)
    # Process data: process_data returns projection matrices (p) and an orthonormal frame.
    x, mu, cov, p, _, orthonormal_frame = process_data(x, mu, cov, d=intrinsic_dim, return_frame=True, device=device)
    # (Keep local_x separately for interior vs. boundary checks)

    # -------------------------------------
    # SET UP MODELS WITH DIFFERENT LOSSES
    # -------------------------------------
    # Vanilla: no extra penalty.
    weights_vanilla = LossWeights()
    # First order: activate only tangent angle loss.
    weights_first = LossWeights()
    weights_first.diffeomorphism_reg = diffeo_weight_12
    weights_first.tangent_angle_weight = tangent_angle_weight
    # Second order: activate only up to tangent drift alignment loss.
    weights_second = LossWeights()
    weights_second.diffeomorphism_reg = diffeo_weight_12
    weights_second.tangent_angle_weight = tangent_angle_weight2
    weights_second.tangent_drift_weight = tangent_drift_weight
    # Diffeomorphic model:
    # weights_diffeo = LossWeights()
    # weights_diffeo.diffeomorphism_reg = diffeo_weight

    # Instantiate the AutoEncoder + Neural SDE models.
    # Vanilla model:
    # TODO: pass the device call
    ae_vanilla = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
    latent_sde_vanilla = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers,
                                         drift_act, diffusion_act, encoder_act)
    ae_diffusion_vanilla = AutoEncoderDiffusion(latent_sde_vanilla, ae_vanilla)

    # First order model:
    ae_first = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
    latent_sde_first = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers,
                                       drift_act, diffusion_act, encoder_act)
    ae_diffusion_first = AutoEncoderDiffusion(latent_sde_first, ae_first)

    # Second order model:
    ae_second = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
    latent_sde_second = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers,
                                        drift_act, diffusion_act, encoder_act)
    ae_diffusion_second = AutoEncoderDiffusion(latent_sde_second, ae_second)

    # #Diffeomorphic model:
    # ae_diffeo = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
    # latent_sde_diffeo = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers,
    #                                     drift_act, diffusion_act, encoder_act)
    # ae_diffusion_diffeo = AutoEncoderDiffusion(latent_sde_diffeo, ae_diffeo)

    # Ambient networks for dynamics
    # TODO: be able to switch between R^d\to R^D and R^D to R^D to compare to our model
    ambient_drift = AmbientDriftNetwork(extrinsic_dim, extrinsic_dim, drift_layers, drift_act)
    ambient_diffusion = AmbientDiffusionNetwork(extrinsic_dim, extrinsic_dim, diffusion_layers, diffusion_act)

    # -------------------------------------
    # TRAIN EACH MODEL
    # -------------------------------------
    fit3 = ThreeStageFit(lr, epochs_ae, epochs_diffusion, epochs_drift, weight_decay, batch_size, print_freq)


    def fit_wrapper(ae_diffusion: AutoEncoderDiffusion, weights, anneal_weights):
        trained_model = fit3.three_stage_fit(ae_diffusion, weights, x, mu, cov, p, orthonormal_frame, anneal_weights,
                                             norm, device)
        return trained_model


    print("Training vanilla model (no extra penalties)...")
    ae_diffusion_vanilla = fit_wrapper(ae_diffusion_vanilla, weights_vanilla, None)
    print("\nTraining first order model (tangent angle loss)...")
    ae_diffusion_first = fit_wrapper(ae_diffusion_first, weights_first, None)
    print("\nTraining second order model (tangent drift alignment loss)...")
    ae_diffusion_second = fit_wrapper(ae_diffusion_second, weights_second, anneal_weights=anneal_weights)
    print("Training ambient drift network")
    z = ae_diffusion_vanilla.autoencoder.encoder(x).detach()
    ambient_drift_loss = AmbientDriftLoss()
    fit_model(ambient_drift, ambient_drift_loss, x, mu, lr, epochs_drift, print_freq, weight_decay, batch_size)
    ambient_diffusion_loss = AmbientDiffusionLoss()
    fit_model(ambient_diffusion, ambient_diffusion_loss, x, cov, lr, epochs_drift, print_freq, weight_decay, batch_size)

    # -------------------------------------
    # Prepare a TotalLoss instance for testing (used for reconstruction loss)
    # -------------------------------------
    ae_loss = TotalLoss(weights_vanilla, norm)  # (loss modules inside TotalLoss)

    # -------------------------------------
    # Instantiate diffusion and drift loss modules (for testing)
    # -------------------------------------
    diffusion_loss_obj = LocalDiffusionLoss(norm)
    drift_loss_obj = LocalDriftLoss()

    # -------------------------------------
    # Helper: determine which test samples are in the interior (training domain)
    # -------------------------------------
    def is_interior_local(local_coords: torch.Tensor, bounds_list):
        interior_mask = torch.ones(local_coords.shape[0], dtype=torch.bool, device=local_coords.device)
        for i, (low, high) in enumerate(bounds_list):
            interior_mask = interior_mask & (local_coords[:, i] >= low) & (local_coords[:, i] <= high)
        return interior_mask.cpu().numpy()

    # -------------------------------------
    # Helpers to compute diffusion and drift losses on a subset
    # -------------------------------------
    def compute_diff_and_drift(model: AutoEncoderDiffusion, x_subset, cov_subset, mu_subset):
        if x_subset.shape[0] == 0:
            return np.nan, np.nan
        dpi = model.autoencoder.encoder.jacobian_network(x_subset).detach()
        # Compute the encoded covariance: dpi * cov * dpi^T
        encoded_cov = torch.bmm(torch.bmm(dpi, cov_subset), dpi.transpose(1, 2))
        diff_loss = diffusion_loss_obj.forward(ae_diffusion=model, x=x_subset, encoded_cov=encoded_cov)
        drift_loss_val = drift_loss_obj.forward(ae_diffusion=model, x=x_subset, observed_ambient_drift=mu_subset)
        return diff_loss.item(), drift_loss_val.item()

    def compute_ambient_diff_and_drift(aedf: AutoEncoderDiffusion,
                                       drift_model: AmbientDriftNetwork,
                                       diffusion_model: AmbientDiffusionNetwork,
                                       x_subset, cov_subset, mu_subset):
        # TODO see above todo regaring using R^d to R^D or R^D to R^D for the comparison ambient NNs
        # z_subset = aedf.autoencoder.encoder(x_subset)
        mu_loss = ambient_drift_loss.forward(drift_model, x_subset, mu_subset).item()
        sigma_loss = ambient_diffusion_loss.forward(diffusion_model, x_subset, cov_subset).item()
        return mu_loss, sigma_loss

    def compute_our_ambient_diff_and_drift(aedf: AutoEncoderDiffusion, x_subset, cov_subset, mu_subset):
        # Compute model diffusion loss
        mu_model = aedf.compute_ambient_drift(x_subset)
        cov_model = aedf.compute_ambient_covariance(x_subset)
        mu_loss = torch.mean(torch.linalg.vector_norm(mu_model-mu_subset, ord=2, dim=1)**2)
        cov_loss = torch.mean(torch.linalg.matrix_norm(cov_model-cov_subset, ord="fro")**2)
        return mu_loss.item(), cov_loss.item()


    def compute_and_plot_errors(local_space=False):
        # -------------------------
        # Generate one large test set for the largest region
        # -------------------------
        max_bounds = [(b[0] - eps_max, b[1] + eps_max) for b in bounds]
        cloud_test = PointCloud(manifold, max_bounds, local_drift, local_diffusion,
                                compute_orthogonal_proj=True)
        # Generate a large test set (adjust num_test if needed)
        x_test_full, _, mu_test_full, cov_test_full, local_x_test_full = cloud_test.generate(num_test, seed=test_seed)
        x_test_full, mu_test_full, cov_test_full, p_test_full, _, orthonormal_frame_test_full = process_data(
            x_test_full, mu_test_full, cov_test_full, d=intrinsic_dim, return_frame=True, device=device)
        local_x_test_full = torch.tensor(local_x_test_full, dtype=torch.float32, device=device)

        # -------------------------
        # Pre-define helper functions for subsetting
        # -------------------------
        def is_interior_local(local_coords: torch.Tensor, bounds_list):
            interior_mask = torch.ones(local_coords.shape[0], dtype=torch.bool, device=local_coords.device)
            for i, (low, high) in enumerate(bounds_list):
                interior_mask = interior_mask & (local_coords[:, i] >= low) & (local_coords[:, i] <= high)
            return interior_mask.cpu().numpy()

        def is_in_current_bounds(local_coords: torch.Tensor, current_bounds):
            mask = torch.ones(local_coords.shape[0], dtype=torch.bool, device=local_coords.device)
            for i, (low, high) in enumerate(current_bounds):
                mask = mask & (local_coords[:, i] >= low) & (local_coords[:, i] <= high)
            return mask.cpu().numpy()

        # Helper functions to compute losses (unchanged)
        def subset_reconstruction_losses(model):
            if np.any(interior_mask):
                x_int = x_test[interior_mask]
                p_int = p_test[interior_mask]
                frame_int = orthonormal_frame_test[interior_mask]
                cov_int = cov_test[interior_mask]
                mu_int = mu_test[interior_mask]
                losses_int = compute_test_losses(model, ae_loss, x_int, p_int, frame_int, cov_int, mu_int,
                                                 device=device)
                rec_int = losses_int[penalty]
            else:
                rec_int = np.nan

            if np.any(boundary_mask):
                x_bnd = x_test[boundary_mask]
                p_bnd = p_test[boundary_mask]
                frame_bnd = orthonormal_frame_test[boundary_mask]
                cov_bnd = cov_test[boundary_mask]
                mu_bnd = mu_test[boundary_mask]
                losses_bnd = compute_test_losses(model, ae_loss, x_bnd, p_bnd, frame_bnd, cov_bnd, mu_bnd,
                                                 device=device)
                rec_bnd = losses_bnd[penalty]
            else:
                rec_bnd = np.nan

            return rec_int, rec_bnd

        def subset_diffusion_and_drift_losses(model, local=True):
            if np.any(interior_mask):
                x_int = x_test[interior_mask]
                cov_int = cov_test[interior_mask]
                mu_int = mu_test[interior_mask]
                if local:
                    diff_int, drift_int = compute_diff_and_drift(model, x_int, cov_int, mu_int)
                else:
                    diff_int, drift_int = compute_our_ambient_diff_and_drift(model, x_int, cov_int, mu_int)
            else:
                diff_int, drift_int = np.nan, np.nan

            if np.any(boundary_mask):
                x_bnd = x_test[boundary_mask]
                cov_bnd = cov_test[boundary_mask]
                mu_bnd = mu_test[boundary_mask]
                if local:
                    diff_bnd, drift_bnd = compute_diff_and_drift(model, x_bnd, cov_bnd, mu_bnd)
                else:
                    diff_bnd, drift_bnd = compute_our_ambient_diff_and_drift(model, x_bnd, cov_bnd, mu_bnd)
            else:
                diff_bnd, drift_bnd = np.nan, np.nan

            return diff_int, diff_bnd, drift_int, drift_bnd

        def subset_ambient_diffusion_and_drift_losses(model, drift_m, diff_m):
            if np.any(interior_mask):
                x_int = x_test[interior_mask]
                cov_int = cov_test[interior_mask]
                mu_int = mu_test[interior_mask]
                diff_int, drift_int = compute_ambient_diff_and_drift(model, drift_m, diff_m, x_int, cov_int, mu_int)
            else:
                diff_int, drift_int = np.nan, np.nan

            if np.any(boundary_mask):
                x_bnd = x_test[boundary_mask]
                cov_bnd = cov_test[boundary_mask]
                mu_bnd = mu_test[boundary_mask]
                diff_bnd, drift_bnd = compute_ambient_diff_and_drift(model, drift_m, diff_m, x_bnd, cov_bnd, mu_bnd)
            else:
                diff_bnd, drift_bnd = np.nan, np.nan

            return diff_int, diff_bnd, drift_int, drift_bnd

        # -------------------------
        # Loop over different epsilon values
        # -------------------------
        epsilons = np.linspace(0.05, eps_max, eps_grid_size)
        # Initialize lists for errors (Reconstruction, Diffusion, Drift)
        errors_vanilla_interior, errors_vanilla_boundary = [], []
        errors_first_interior, errors_first_boundary = [], []
        errors_second_interior, errors_second_boundary = [], []
        # errors_diffeo_interior, errors_diffeo_boundary = [], []
        # Diffusion errors
        errors_vanilla_diff_interior, errors_vanilla_diff_boundary = [], []
        errors_first_diff_interior, errors_first_diff_boundary = [], []
        errors_second_diff_interior, errors_second_diff_boundary = [], []
        # errors_diffeo_diff_interior, errors_diffeo_diff_boundary = [], []
        errors_ambient_diffusion_interior, errors_ambient_diffusion_boundary = [], []
        # Drift errors
        errors_vanilla_drift_interior, errors_vanilla_drift_boundary = [], []
        errors_first_drift_interior, errors_first_drift_boundary = [], []
        errors_second_drift_interior, errors_second_drift_boundary = [], []
        # errors_diffeo_drift_interior, errors_diffeo_drift_boundary = [], []
        errors_ambient_drift_interior, errors_ambient_drift_boundary = [], []

        for eps in epsilons:
            # Define current test region for this epsilon
            current_bounds = [(b[0] - eps, b[1] + eps) for b in bounds]
            # Subset the full test set to those points that lie in current_bounds
            current_mask = is_in_current_bounds(local_x_test_full, current_bounds)
            # Subset test data accordingly
            x_test = x_test_full[current_mask]
            mu_test = mu_test_full[current_mask]
            cov_test = cov_test_full[current_mask]
            p_test = p_test_full[current_mask]
            orthonormal_frame_test = orthonormal_frame_test_full[current_mask]
            local_x_test = local_x_test_full[current_mask]

            # Now determine interior vs. boundary using the training bounds (i.e. [a, b]^d)
            interior_mask = is_interior_local(local_x_test, bounds)
            boundary_mask = ~interior_mask

            # Compute reconstruction losses for each model on this subset
            rec_vanilla_int, rec_vanilla_bnd = subset_reconstruction_losses(ae_diffusion_vanilla)
            rec_first_int, rec_first_bnd = subset_reconstruction_losses(ae_diffusion_first)
            rec_second_int, rec_second_bnd = subset_reconstruction_losses(ae_diffusion_second)
            # rec_diffeo_int, rec_diffeo_bnd = subset_reconstruction_losses(ae_diffusion_diffeo)

            # Compute diffusion and drift losses
            diff_vanilla_int, diff_vanilla_bnd, drift_vanilla_int, drift_vanilla_bnd = subset_diffusion_and_drift_losses(
                ae_diffusion_vanilla, local_space)
            diff_first_int, diff_first_bnd, drift_first_int, drift_first_bnd = subset_diffusion_and_drift_losses(
                ae_diffusion_first, local_space)
            diff_second_int, diff_second_bnd, drift_second_int, drift_second_bnd = subset_diffusion_and_drift_losses(
                ae_diffusion_second, local_space)
            # diff_diffeo_int, diff_diffeo_bnd, drift_diffeo_int, drift_diffeo_bnd = subset_diffusion_and_drift_losses(
            #     ae_diffusion_diffeo, local_space)

            # Ambient losses for vanilla model
            diffusion_ambient_int, diffusion_ambient_bnd, drift_ambient_int, drift_ambient_bnd = subset_ambient_diffusion_and_drift_losses(
                ae_diffusion_vanilla, ambient_drift, ambient_diffusion)

            # Vanilla auto encoder reconstruction loss
            errors_vanilla_interior.append(rec_vanilla_int)
            errors_vanilla_boundary.append(rec_vanilla_bnd)
            # First order (diffeo+tangent space) recon loss
            errors_first_interior.append(rec_first_int)
            errors_first_boundary.append(rec_first_bnd)
            # Second order (diffeo+tangent space+ito curvature) recon loss
            errors_second_interior.append(rec_second_int)
            errors_second_boundary.append(rec_second_bnd)
            # # Diffeo only AE recon loss
            # errors_diffeo_interior.append(rec_diffeo_int)
            # errors_diffeo_boundary.append(rec_diffeo_bnd)
            # Covariance losses for each AE: 0th order
            errors_vanilla_diff_interior.append(diff_vanilla_int)
            errors_vanilla_diff_boundary.append(diff_vanilla_bnd)
            # Covariance loss under the 1st Order AE
            errors_first_diff_interior.append(diff_first_int)
            errors_first_diff_boundary.append(diff_first_bnd)
            # Covariance loss under the 2nd order AE
            errors_second_diff_interior.append(diff_second_int)
            errors_second_diff_boundary.append(diff_second_bnd)
            # # Covariance loss under the diffeo penalized AE
            # errors_diffeo_diff_interior.append(diff_diffeo_int)
            # errors_diffeo_diff_boundary.append(diff_diffeo_bnd)
            # Covariance loss for an ambient NN for model-covariance
            errors_ambient_diffusion_interior.append(diffusion_ambient_int)
            errors_ambient_diffusion_boundary.append(diffusion_ambient_bnd)

            # Drift loss for the vanilla AE
            errors_vanilla_drift_interior.append(drift_vanilla_int)
            errors_vanilla_drift_boundary.append(drift_vanilla_bnd)
            # Drift loss for the First order AE
            errors_first_drift_interior.append(drift_first_int)
            errors_first_drift_boundary.append(drift_first_bnd)
            # Drift loss for the 2nd order AE
            errors_second_drift_interior.append(drift_second_int)
            errors_second_drift_boundary.append(drift_second_bnd)
            # # Drift loss for the diffeo penalized AE
            # errors_diffeo_drift_interior.append(drift_diffeo_int)
            # errors_diffeo_drift_boundary.append(drift_diffeo_bnd)
            # Drift loss for the ambient NN for drift model
            errors_ambient_drift_interior.append(drift_ambient_int)
            errors_ambient_drift_boundary.append(drift_ambient_bnd)

            print(f"Epsilon {eps:.2f}:")
            print(f"  Vanilla - Interior Rec. Loss: {rec_vanilla_int:.4f}, Boundary Rec. Loss: {rec_vanilla_bnd:.4f}")
            print(f"  First   - Interior Rec. Loss: {rec_first_int:.4f}, Boundary Rec. Loss: {rec_first_bnd:.4f}")
            print(f"  Second  - Interior Rec. Loss: {rec_second_int:.4f}, Boundary Rec. Loss: {rec_second_bnd:.4f}")
            # print(f"  Diffeo  - Interior Rec. Loss: {rec_diffeo_int:.4f}, Boundary Rec. Loss: {rec_diffeo_bnd:.4f}\n")

        # -------------------------
        # Plot the error curves as before (using your plotting code)
        # -------------------------
        test_set_multiple_range = np.arange(1, eps_grid_size + 1) * num_test  # adjust if desired
        space = "Local" if local_space else "Ambient"
        fig = plt.figure(figsize=(8, 8))
        # Row 1: Reconstruction Loss
        plt.subplot(3, 2, 1)
        plt.plot(test_set_multiple_range, errors_vanilla_interior, marker='o', label='Vanilla')
        plt.plot(test_set_multiple_range, errors_first_interior, marker='o', label='First Order')
        plt.plot(test_set_multiple_range, errors_second_interior, marker='o', label='Second Order')
        # plt.plot(test_set_multiple_range, errors_diffeo_interior, marker='o', label='Diffeo')
        plt.xlabel('Test Set Subset Size')
        plt.ylabel('Interior Rec. Loss')
        plt.title(penalty+ 'Loss - Interior')
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(epsilons, errors_vanilla_boundary, marker='o', label='Vanilla')
        plt.plot(epsilons, errors_first_boundary, marker='o', label='First Order')
        plt.plot(epsilons, errors_second_boundary, marker='o', label='Second Order')
        # plt.plot(epsilons, errors_diffeo_boundary, marker='o', label='Diffeo')
        plt.xlabel('Epsilon')
        plt.ylabel('Boundary Rec. Loss')
        plt.title(penalty+' Loss - Boundary')
        plt.legend()

        # Row 2: Diffusion Loss
        plt.subplot(3, 2, 3)
        plt.plot(test_set_multiple_range, errors_vanilla_diff_interior, marker='o', label='Vanilla')
        plt.plot(test_set_multiple_range, errors_first_diff_interior, marker='o', label='First Order')
        plt.plot(test_set_multiple_range, errors_second_diff_interior, marker='o', label='Second Order')
        # plt.plot(test_set_multiple_range, errors_diffeo_diff_interior, marker='o', label='Diffeo')
        plt.plot(test_set_multiple_range, errors_ambient_diffusion_interior, marker='o', label='Ambient model')
        plt.xlabel('Test Set Subset Size')
        plt.ylabel('Interior Cov. Loss')
        plt.title(space + ' Covariance Loss - Interior')
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(epsilons, errors_vanilla_diff_boundary, marker='o', label='Vanilla')
        plt.plot(epsilons, errors_first_diff_boundary, marker='o', label='First Order')
        plt.plot(epsilons, errors_second_diff_boundary, marker='o', label='Second Order')
        # plt.plot(epsilons, errors_diffeo_diff_boundary, marker='o', label='Diffeo')
        plt.plot(epsilons, errors_ambient_diffusion_boundary, marker='o', label='Ambient')
        plt.xlabel('Epsilon')
        plt.ylabel('Boundary Cov. Loss')
        plt.title(space + ' Covariance Loss - Boundary')
        plt.legend()

        # Row 3: Drift Loss
        plt.subplot(3, 2, 5)
        plt.plot(test_set_multiple_range, errors_vanilla_drift_interior, marker='o', label='Vanilla')
        plt.plot(test_set_multiple_range, errors_first_drift_interior, marker='o', label='First Order')
        plt.plot(test_set_multiple_range, errors_second_drift_interior, marker='o', label='Second Order')
        # plt.plot(test_set_multiple_range, errors_diffeo_drift_interior, marker='o', label='Diffeo')
        plt.plot(test_set_multiple_range, errors_ambient_drift_interior, marker='o', label='Ambient')
        plt.xlabel('Test Set Subset Size')
        plt.ylabel('Interior Drift Loss')
        plt.title(space + ' Drift Loss - Interior')
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(epsilons, errors_vanilla_drift_boundary, marker='o', label='Vanilla')
        plt.plot(epsilons, errors_first_drift_boundary, marker='o', label='First Order')
        plt.plot(epsilons, errors_second_drift_boundary, marker='o', label='Second Order')
        # plt.plot(epsilons, errors_diffeo_drift_boundary, marker='o', label='Diffeo')
        plt.plot(epsilons, errors_ambient_drift_boundary, marker='o', label='Ambient')
        plt.xlabel('Epsilon')
        plt.ylabel('Boundary Drift Loss')
        plt.title(space + ' Drift Loss - Boundary')
        plt.legend()

        plt.tight_layout()
        plt.show()
        save_plot(fig, exp_dir, plot_name="int_vs_bd_errors")


        # Plot SDEs
        z0_true = x[0, :2].detach()
        x0 = x[0, :]

        z0 = ae_diffusion_first.autoencoder.encoder.forward(x0).detach()
        true_latent_paths = cloud_train.latent_sde.sample_ensemble(z0_true, tn, ntime, npaths)
        model_latent_paths = ae_diffusion_first.latent_sde.sample_paths(z0, tn, ntime, npaths)
        basic_ambient_paths = SDE(ambient_drift.drift_numpy, ambient_diffusion.diffusion_numpy).sample_ensemble(x0.detach(), tn, ntime, npaths, noise_dim=3)
        true_ambient_paths = np.zeros((npaths, ntime + 1, extrinsic_dim))
        model_ambient_paths = np.zeros((npaths, ntime + 1, extrinsic_dim))
        for j in range(npaths):
            model_ambient_paths[j, :, :] = ae_diffusion_first.autoencoder.decoder(
                torch.tensor(model_latent_paths[j, :, :], dtype=torch.float32)).detach().numpy()
            for i in range(ntime + 1):
                true_ambient_paths[j, i, :] = np.squeeze(cloud_train.np_phi(*true_latent_paths[j, i, :]))

        x_test_full = x_test_full.detach()
        fig = plt.figure()
        ax = plt.subplot(111, projection="3d")
        ax.scatter(x_test_full[:, 0], x_test_full[:, 1], x_test_full[:, 2], alpha=0.01)
        for i in range(npaths):
            ax.plot3D(true_ambient_paths[i, :, 0], true_ambient_paths[i, :, 1], true_ambient_paths[i, :, 2], c="black",
                      alpha=0.8)
            ax.plot3D(model_ambient_paths[i, :, 0], model_ambient_paths[i, :, 1], model_ambient_paths[i, :, 2],
                      c="blue",
                      alpha=0.8)
            ax.plot3D(basic_ambient_paths[i, :, 0], basic_ambient_paths[i, :, 1], basic_ambient_paths[i, :, 2],
                      c="red",
                      alpha=0.8)
        ae_diffusion_second.autoencoder.plot_surface(-1, 1, grid_size=30, ax=ax, title="Reconstruction")
        # Create legend handles
        legend_elements = [
            Line2D([0], [0], color="black", lw=2, label="True Path"),
            Line2D([0], [0], color="blue", lw=2, label="2nd-order AE-SDE Path"),
            Line2D([0], [0], color="red", lw=2, label="Ambient SDE model Path")
        ]

        # Add legend
        ax.legend(handles=legend_elements, loc="best")
        plt.show()
        save_plot(fig, exp_dir, plot_name="model_sample_paths_surface")

    compute_and_plot_errors(False)

    torch.save(ae_diffusion_vanilla.state_dict(), os.path.join(exp_dir, "ae_diffusion_vanilla.pth"))
    torch.save(ae_diffusion_first.state_dict(), os.path.join(exp_dir, "ae_diffusion_first.pth"))
    torch.save(ae_diffusion_second.state_dict(), os.path.join(exp_dir, "ae_diffusion_second.pth"))
    # torch.save(ae_diffusion_diffeo.state_dict(), os.path.join(exp_dir, "ae_diffusion_diffeo.pth"))

    torch.save(ambient_drift.state_dict(), os.path.join(exp_dir, "ambient_drift.pth"))
    torch.save(ambient_diffusion.state_dict(), os.path.join(exp_dir, "ambient_diffusion.pth"))

    print("Models successfully saved.")
