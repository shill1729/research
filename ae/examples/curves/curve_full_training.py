"""
    This module fits 4 autoencoder-SDEs and an Ambient SDE model.
    The 4 AE-SDEs go through all three-stages of training: AE, diffusion, drift.
"""
# TODO: this module can now train surfaces and its companion module 'curve_full_assessment.py' can load surfaces
#  so we should rename and change the directory maybe.
# Standard library imports
import torch.nn as nn
import copy
import torch
import os
import json

# Custom library imports
from ae.toydata.curves import *
from ae.toydata.surfaces import *
from ae.toydata.local_dynamics import *
from ae.toydata import RiemannianManifold, PointCloud
from ae.utils import process_data
from ae.models import AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion, fit_model, ThreeStageFit
from ae.models import LossWeights, AmbientDriftNetwork, AmbientDiffusionNetwork
from ae.models.losses.losses_ambient import AmbientDriftLoss, AmbientCovarianceLoss
# TODO: embed into higher dimension
# TODO: we are currently using the 2-norm for the Tangent Bundle error!
# TODO: Check contractive errors

# Model configuration parameters
# Set 'compare_mse' to True when you want to compare 1st and 2nd order models being trained
# with Latent MSEs vs Ambient MSEs
compare_mse = False # TODO: currently tests tangent penalty approx vs true tangent penalty
# NOTE: These are only used when 'compare_mse=False'
use_ambient_cov_mse = False
use_ambient_drift_mse = False
# NOTE: Toggle this to make all the autoencoders and all the SDEs use the same initial weights.
use_same_initial_weights = True
train_seed = None
n_train = 50
batch_size = int(n_train * 0.8)

# Architecture parameters
intrinsic_dim = 1
extrinsic_dim = 2
hidden_dims = [16]
diff_layers = [2]
drift_layers = [2]


# Training parameters
lr = 0.001
weight_decay = 0.
epochs_ae = 9000
epochs_diffusion = 1000
epochs_drift = 1000
print_freq = 1000

# Penalty weights
#: 1., 0.001, 0.001/0.002 worked well for paraboloid on many dynamics
diffeo_weight = 0.01
first_order_weight = 0.01
second_order_weight = 0.01

# Activation functions
encoder_act = nn.Tanh()
final_act = nn.Tanh() # for the encoder
decoder_act = nn.Tanh()
drift_act = nn.Tanh()
diffusion_act = nn.Tanh()

if __name__ == "__main__":
    # Pick the manifold and dynamics
    curve = Parabola()
    dynamics = RiemannianBrownianMotion()
    manifold = RiemannianManifold(curve.local_coords(), curve.equation())
    local_drift = dynamics.drift(manifold)
    local_diffusion = dynamics.diffusion(manifold)

    # Generate and preprocess data
    point_cloud = PointCloud(manifold, curve.bounds(), local_drift, local_diffusion, True)
    print("Extrinsic drift")
    print(point_cloud.extrinsic_drift)
    x, _, mu, cov, local_x = point_cloud.generate(n=n_train, seed=train_seed)
    x, mu, cov, p, n, h = process_data(x, mu, cov, d=intrinsic_dim)

    # Define AE loss configurations
    # IMPORTANT:
    # When 'compare_mse=True', we overwrite vanilla and diffeo to be First and Second order using Latent MSE
    mse_choice = {"vanilla": False, "diffeo": False, "first_order": True, "second_order": True}
    if compare_mse:
        # If we are comparing latent vs ambient MSE, use only first and second order
        print("Comparing latent MSE vs ambient MSE")
        weight_configs = {
            # First order overwrite
            "vanilla": LossWeights(diffeomorphism_reg=diffeo_weight,
                                   tangent_angle_weight=first_order_weight,
                                   tangent_drift_weight=0.),
            # Second order overwrite. 'curve_full_assessment.py' handles the relabeling.
            "diffeo": LossWeights(diffeomorphism_reg=diffeo_weight,
                                  tangent_angle_weight=first_order_weight,
                                  tangent_drift_weight=second_order_weight),
            "first_order": LossWeights(diffeomorphism_reg=diffeo_weight,
                                       tangent_approx_weight=first_order_weight,
                                       tangent_drift_weight=0.),
            "second_order": LossWeights(diffeomorphism_reg=diffeo_weight,
                                        tangent_approx_weight=first_order_weight,
                                        tangent_drift_weight=second_order_weight)
        }
    else:
        # Otherwise we are doing ablation test for different penalties
        weight_configs = {
            "vanilla": LossWeights(diffeomorphism_reg=0.0,
                                   tangent_space_error_weight=0.0,
                                   tangent_drift_weight=0.0),
            "diffeo": LossWeights(diffeomorphism_reg=diffeo_weight,
                                  tangent_space_error_weight=0.0,
                                  tangent_drift_weight=0.0),
            "first_order": LossWeights(diffeomorphism_reg=diffeo_weight,
                                       tangent_space_error_weight=first_order_weight,
                                       tangent_drift_weight=0.0),
            "second_order": LossWeights(diffeomorphism_reg=diffeo_weight,
                                        tangent_space_error_weight=first_order_weight,
                                        tangent_drift_weight=second_order_weight)
        }

    # Fit each AE-SDE variant and save results
    fit3 = ThreeStageFit(lr, epochs_ae, epochs_diffusion, epochs_drift, weight_decay, batch_size, print_freq)
    base_dir = "saved_models"
    os.makedirs(base_dir, exist_ok=True)

    # Build the reference models
    # NOTE: These are only used for 'use_same_initial_weights=True'
    ae_ref = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act, final_act=final_act)
    latent_sde_ref = LatentNeuralSDE(intrinsic_dim, drift_layers, diff_layers, drift_act, diffusion_act,
                                     encoder_act=final_act)
    aedf_ref = AutoEncoderDiffusion(latent_sde_ref, ae_ref)

    for name, weights in weight_configs.items():
        print(f"Training AE-SDE model with {name} weights")

        ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
        latent_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diff_layers, drift_act, diffusion_act)
        aedf = AutoEncoderDiffusion(latent_sde, ae)
        if use_same_initial_weights:
            aedf.load_state_dict(copy.deepcopy(aedf_ref.state_dict()))
        # Train all three stages
        if not compare_mse:
            print("All models trained with same MSE choice")
            fit3.three_stage_fit(aedf, weights, x, mu, cov, p, h,
                                 ambient_cov_mse=use_ambient_cov_mse,
                                 ambient_drift_mse=use_ambient_drift_mse)
        else:
            print("First and Second order models are being trained in Latent and Ambient")
            # If we are comparing MSE then pick the choice for each model name.
            fit3.three_stage_fit(aedf, weights, x, mu, cov, p, h,
                                 ambient_cov_mse=mse_choice[name],
                                 ambient_drift_mse=mse_choice[name])

        subdir = os.path.join(base_dir, name)
        os.makedirs(subdir, exist_ok=True)
        torch.save(aedf.state_dict(), os.path.join(subdir, "aedf.pth"))

    config = {
        "train_seed": train_seed,
        "n_train": n_train,
        "batch_size": batch_size,
        "intrinsic_dim": intrinsic_dim,
        "extrinsic_dim": extrinsic_dim,
        "hidden_dims": hidden_dims,
        "drift_layers": drift_layers,
        "diff_layers": diff_layers,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs_ae": epochs_ae,
        "epochs_diffusion": epochs_diffusion,
        "epochs_drift": epochs_drift,
        "print_freq": print_freq,
        "diffeo_weight": diffeo_weight,
        "first_order_weight": first_order_weight,
        "second_order_weight": second_order_weight,
        "manifold": curve.__class__.__name__,
        "dynamics": dynamics.__class__.__name__
    }

    with open(os.path.join(base_dir, "config.json"), "w") as f:
        json.dump(config, f)

    # Train and save ambient models once
    # print("Training ambient diffusion model")
    # ambient_diff_model = AmbientDiffusionNetwork(extrinsic_dim, extrinsic_dim, diff_layers, diffusion_act)
    # fit_model(ambient_diff_model, AmbientCovarianceLoss(), x, cov, lr, epochs_diffusion, print_freq, weight_decay, batch_size)

    # print("Training ambient drift model")
    # ambient_drift_model = AmbientDriftNetwork(extrinsic_dim, extrinsic_dim, drift_layers, drift_act)
    # fit_model(ambient_drift_model, AmbientDriftLoss(), x, mu, lr, epochs_drift, print_freq, weight_decay, batch_size)

    # torch.save(ambient_diff_model.state_dict(), os.path.join(base_dir, "ambient_diff_model.pth"))
    # torch.save(ambient_drift_model.state_dict(), os.path.join(base_dir, "ambient_drift_model.pth"))

    print("All models and configurations saved.")
