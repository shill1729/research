"""
    For each surface and dynamics pair, we train every combination of penalized autoencoder
    We compute the relevant losses on interpolation test points and extrapolation test sets.

    Run table_processor.py after this using the surface name to convert to latex with highlighted best entries.
"""
import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
from scipy.stats import ttest_ind
from datetime import datetime

from ae.symbolic.diffgeo import RiemannianManifold
from ae.toydata.pointclouds import PointCloud
from ae.models.autoencoder import AutoEncoder
from ae.models.local_neural_sdes import LatentNeuralSDE, AutoEncoderDiffusion
from ae.models.losses import TotalLoss, LossWeights, LocalDriftLoss, LocalDiffusionLoss
from ae.utils import select_device, set_grad_tracking, process_data
from ae.toydata.surfaces import *
from ae.toydata.local_dynamics import *
from ae.performance_analysis import compute_test_losses
from ae.models.fitting import fit_model, ThreeStageFit

n_trials = 1
# Point cloud sample parameters
n_train = 10
n_test = 30
training_seed = 17
eps = 0.05

# Model architecture:
extrinsic_dim = 3
intrinsic_dim = 2
# Same amount of neurons for every network
hidden_layers = [2]
drift_layers = [2]
diffusion_layers = [2]
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
drift_act = nn.Tanh()
diffusion_act = nn.Tanh()
norm = "fro"
# Penalty coefficients for non-vanilla AEs:
tangent_angle_weight = 0.
tangent_drift_weight = 0.
diffeo_weight = 0.
# Training parameters:
lr = 0.001
epochs_ae = 2
epochs_diffusion = 2
epochs_drift = 2
print_freq = 1
weight_decay = 0.
batch_size = int(n_train / 2)

# Define the surfaces and dynamics
# surfaces = [Paraboloid(), HyperbolicParaboloid(), ProductSurface(), RationalSurface(), Sphere(), Torus()]
# dynamics = [BrownianMotion(), RiemannianBrownianMotion(), LangevinDoubleWell(), ArbitraryMotion()]
surfaces = [ProductSurface(2)]
dynamics = [RiemannianBrownianMotion()]

diffusion_loss = LocalDiffusionLoss(norm="fro")
drift_loss = LocalDriftLoss()


def evaluate(bounds,
             local_drift,
             local_diffusion,
             models,
             loss_functions,
             diffusion_loss,
             drift_loss,
             manifold: RiemannianManifold,
             ae_names):
    # Evaluate on test data.
    # Declare the point cloud object and generate the data.
    print("Generating test point cloud...")
    point_cloud = PointCloud(manifold, bounds, local_drift, local_diffusion, compute_orthogonal_proj=True)
    x_test, _, mu_test, cov_test, _ = point_cloud.generate(n=n_test, seed=training_seed)
    print("Pre-processing the ambient covariance into the orthogonal projection and orthonormal frame...")
    # Use process_data to convert to torch and compute the orthogonal projection and orthonormal frame
    x_test, mu_test, cov_test, P_test, N_test, H_test = process_data(x_test, mu_test, cov_test, d=intrinsic_dim,
                                                                     return_frame=True)
    list_of_test_losses_dicts = []

    for model, loss in zip(models, loss_functions):
        test_ae_losses = compute_test_losses(model, loss, x_test, P_test, H_test, cov_test, mu_test)
        # Compute diffusion losses for testing set:
        dpi_test = model.autoencoder.encoder.jacobian_network(x_test).detach()
        encoded_cov_test = torch.bmm(torch.bmm(dpi_test, cov_test), dpi_test.mT)
        diffusion_loss_test = diffusion_loss.forward(ae_diffusion=model,
                                                     x=x_test,
                                                     encoded_cov=encoded_cov_test
                                                     )
        drift_loss_test = drift_loss.forward(ae_diffusion=model,
                                             x=x_test,
                                             observed_ambient_drift=mu_test
                                             )

        test_ae_losses["diffusion loss"] = diffusion_loss_test.detach().numpy()
        test_ae_losses["drift loss"] = drift_loss_test.detach().numpy()

        list_of_test_losses_dicts.append(test_ae_losses)

    # Create DataFrame from all results
    df = pd.DataFrame(list_of_test_losses_dicts)
    df = df.transpose()
    print(df)
    # Hard coded indices of the losses from compute_test losses transposed. These should include
    # reconstruction loss, tangent angle, tangent drift, diffeomorphism, and diffusion and drift
    # TODO replace the index hardcoding with something more robust to changes...
    df = df.iloc[[0, 4, 5, 6, 7, 8], :]
    df.columns = ae_names

    print("\nComparison Table:")
    print(df.to_latex())
    return df


def average_losses(results_list):
    """
    Compute the average loss values for 'interp' and 'extrap' across a list of results.

    Args:
        results_list (list): A list where each element is a dictionary containing 'interp' and 'extrap'.
                             Each of these keys maps to a DataFrame of losses.

    Returns:
        dict: A dictionary with averaged loss values for 'interp' and 'extrap'.
    """
    # Initialize dictionaries to store accumulated sums and counts for averaging
    aggregated = {"interp": None, "extrap": None}

    for result in results_list:
        for key in ["interp", "extrap"]:
            # Extract the DataFrame for the current key
            current_df = result[key]

            # Accumulate sums of DataFrames
            if aggregated[key] is None:
                aggregated[key] = current_df.copy()
            else:
                aggregated[key] += current_df

    # Compute averages
    averaged_results = {
        key: aggregated[key] / len(results_list)
        for key in aggregated
    }

    return averaged_results


def save_results(results, filename):
    """Save results as a CSV and LaTeX file."""
    results.to_csv(f"{filename}.csv")
    with open(f"{filename}.tex", "w") as f:
        f.write(results.to_latex())


for surface in surfaces:
    surface_name = surface.__class__.__name__
    print(surface)

    # Create directory for the surface
    surface_dir = os.path.join("experiment_results", surface_name)
    os.makedirs(surface_dir, exist_ok=True)

    # Generate manifold--this is a class with sympy methods essentially
    manifold = RiemannianManifold(surface.local_coords(), surface.equation())

    for dyn in dynamics:
        dyn_name = dyn.__class__.__name__
        # Create subdirectories for interpolation and extrapolation
        interp_dir = os.path.join(surface_dir, "interp")
        extrap_dir = os.path.join(surface_dir, "extrap")
        os.makedirs(interp_dir, exist_ok=True)
        os.makedirs(extrap_dir, exist_ok=True)

        print(dyn)
        print(dyn.drift(manifold))
        print(dyn.diffusion(manifold))
        results = []
        for n in range(n_trials):
            # Generate point cloud:
            cloud = PointCloud(manifold, surface.bounds(),
                               dyn.drift(manifold), dyn.diffusion(manifold),
                               compute_orthogonal_proj=True)
            x, _, mu, cov, local_x = cloud.generate(n_train, seed=None)
            x, mu, cov, P, _, H = process_data(x, mu, cov, d=2, return_frame=True)

            # Define the vanilla AE model
            # 0th-order model
            vanilla_weights = LossWeights()
            vanilla_weights.diffeomorphism_reg1 = diffeo_weight
            ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_layers, encoder_act, decoder_act)
            ae_loss = TotalLoss(vanilla_weights, norm)
            vanilla_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers, drift_act, diffusion_act,
                                          encoder_act)
            vanilla_ae_diffusion = AutoEncoderDiffusion(vanilla_sde, ae)

            # Define the tangent-regularized AE Model
            # 1st-order model
            tbae_weights = LossWeights()
            tbae_weights.tangent_angle_weight = tangent_angle_weight
            tbae_weights.diffeomorphism_reg = diffeo_weight
            tbae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_layers, encoder_act, decoder_act)
            tbae_loss = TotalLoss(tbae_weights, norm)
            tbae_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers, drift_act, diffusion_act,
                                       encoder_act)
            tbae_diffusion = AutoEncoderDiffusion(tbae_sde, tbae)

            # Define the tangent-drift aligned AE model
            # Second order
            ito_ae_weights = LossWeights()
            ito_ae_weights.tangent_angle_weight = tangent_angle_weight
            ito_ae_weights.tangent_drift_weight = tangent_drift_weight
            ito_ae_weights.diffeomorphism_reg = diffeo_weight
            ito_ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_layers, encoder_act, decoder_act)
            ito_ae_loss = TotalLoss(ito_ae_weights, norm)
            itoae_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers, drift_act, diffusion_act,
                                        encoder_act)
            ito_ae_diffusion = AutoEncoderDiffusion(itoae_sde, ito_ae)

            # Edit this if you want to just run a few models.
            models = [vanilla_ae_diffusion, tbae_diffusion, ito_ae_diffusion]
            loss_functions = [ae_loss, tbae_loss, ito_ae_loss]
            weight_list = [vanilla_weights, tbae_weights, ito_ae_weights]
            ae_names = ["0th Order", "1st Order", "2nd Order"]
            drift_models = []
            diffusion_models = []

            i = 0
            for model, weights in zip(models, weight_list):
                print("\nTrial = " + str(n))
                fit3 = ThreeStageFit(lr, epochs_ae, epochs_diffusion, epochs_drift, weight_decay, batch_size,
                                     print_freq)
                model = fit3.three_stage_fit(model, weights, x, mu, cov, P, H)
                # Done training
                # diffusion_models.append(model.latent_sde)
                # drift_models.append(model_drift)
                i += 1

            # Evaluate interpolation and extrapolation
            interp_df = evaluate(surface.bounds(),
                                 dyn.drift(manifold),
                                 dyn.diffusion(manifold),
                                 models,
                                 loss_functions,
                                 diffusion_loss,
                                 drift_loss,
                                 manifold,
                                 ae_names)
            extrap_df = evaluate(
                [(b[0] - eps, b[1] + eps) for b in surface.bounds()],
                dyn.drift(manifold),
                dyn.diffusion(manifold),
                models,
                loss_functions,
                diffusion_loss,
                drift_loss,
                manifold,
                ae_names
            )
            results.append({"interp": interp_df, "extrap": extrap_df})

        averaged_results = average_losses(results)
        print("Averaged Interpolation Losses:")
        print(averaged_results["interp"])

        print("\nAveraged Extrapolation Losses:")
        print(averaged_results["extrap"])

        # Save results in organized folders
        interp_filepath = os.path.join(interp_dir, f"{dyn_name}.csv")
        extrap_filepath = os.path.join(extrap_dir, f"{dyn_name}.csv")

        print(f"Saving averaged interpolation results to {interp_filepath}...")
        averaged_results["interp"].to_csv(interp_filepath)

        print(f"Saving averaged extrapolation results to {extrap_filepath}...")
        averaged_results["extrap"].to_csv(extrap_filepath)
