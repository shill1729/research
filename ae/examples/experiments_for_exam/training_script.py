"""
    Looping over all curves.

    This module fits 4 autoencoder-SDEs and an Ambient SDE model.
    The 4 AE-SDEs go through all three-stages of training: AE, diffusion, drift.
"""
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

# Model configuration parameters
# Set 'compare_mse' to True when you want to compare 1st and 2nd order models being trained
# with Latent MSEs vs Ambient MSEs
compare_mse = False
# NOTE: These are only used when 'compare_mse=False'
use_ambient_cov_mse = False
use_ambient_drift_mse = False
# NOTE: Toggle this to make all the autoencoders and all the SDEs use the same initial weights.
use_same_initial_weights = True
# Do you want to tie the autoencoder weights?
tie_weights = True
train_seed = None
n_train = 50
batch_size = int(n_train * 0.5)

# Architecture parameters
intrinsic_dim = 2
extrinsic_dim = 3
hidden_dims = [16, 16]
diff_layers = [16, 16]
drift_layers = [16, 16]


# Training parameters
lr = 0.0005
weight_decay = 0.
epochs_ae = 9000
epochs_diffusion = 9000
epochs_drift = 9000
print_freq = 1000

# Penalty weights
#: 1., 0.001, 0.001/0.002 worked well for paraboloid on many dynamics
diffeo_weight = 0.05
first_order_weight = 0.001
second_order_weight = 0.001

# Activation functions
encoder_act = nn.Tanh()
final_act = nn.Tanh() # for the encoder
decoder_act = nn.Tanh()
drift_act = nn.Tanh()
diffusion_act = nn.Tanh()

# List of curves to train on
# curve_list = [
#     # Parabola(a=1.0),
#     Cubic(a=1.05),
#     # SineCurve(amplitude=1.0, frequency=1, phase=0.0),
#     FastSineCurve(),
#     RationalCurve(a=1, b=1),
#     BellCurve(a=1, b=1.0),
#     ArctangentCurve(),
#     TanhCurve(),
#     SigmoidCurve(),
#     GaussianModulatedSineCurve()
# ]

curve_list = [
    # Paraboloid()
    ProductSurface(),
    # RationalSurface(),
    # DeepParaboloid()

]


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
    print("Ablation testing: vanilla, diffeo, diffeo+1st, diffeo+1st+2nd")
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


if __name__ == "__main__":

    # Pick dynamics here:
    dynamics = ArbitraryMotion2()

    for curve in curve_list:
        print(f"\n=== Training on curve: {curve.__class__.__name__} ===")

        manifold = RiemannianManifold(curve.local_coords(), curve.equation())
        local_drift = dynamics.drift(manifold)
        local_diffusion = dynamics.diffusion(manifold)

        point_cloud = PointCloud(manifold, curve.bounds(), local_drift, local_diffusion, True)
        x, _, mu, cov, local_x = point_cloud.generate(n=n_train, seed=train_seed)
        x, mu, cov, p, n, h = process_data(x, mu, cov, d=intrinsic_dim)

        # Initialize the reference model once per curve if using same weights
        ae_ref = AutoEncoder(extrinsic_dim,
                             intrinsic_dim,
                             hidden_dims,
                             encoder_act,
                             decoder_act,
                             final_act=final_act,
                             tie_weights=tie_weights)
        latent_sde_ref = LatentNeuralSDE(intrinsic_dim,
                                         drift_layers,
                                         diff_layers,
                                         drift_act,
                                         diffusion_act,
                                         encoder_act=final_act)
        aedf_ref = AutoEncoderDiffusion(latent_sde_ref, ae_ref)

        fit3 = ThreeStageFit(lr, epochs_ae, epochs_diffusion, epochs_drift,
                             weight_decay, batch_size, print_freq)

        base_dir = os.path.join("saved_models", curve.__class__.__name__)
        os.makedirs(base_dir, exist_ok=True)

        for name, weights in weight_configs.items():
            print(f"Training AE-SDE model with {name} weights")
            ae = AutoEncoder(extrinsic_dim,
                             intrinsic_dim,
                             hidden_dims,
                             encoder_act,
                             decoder_act,
                             final_act=final_act,
                             tie_weights=tie_weights)
            latent_sde = LatentNeuralSDE(intrinsic_dim,
                                         drift_layers,
                                         diff_layers,
                                         drift_act,
                                         diffusion_act,
                                         encoder_act=final_act)
            aedf = AutoEncoderDiffusion(latent_sde, ae)

            if use_same_initial_weights:
                aedf.load_state_dict(copy.deepcopy(aedf_ref.state_dict()))
            # Fit the 3-stage model
            fit3.three_stage_fit(aedf, weights, x, mu, cov, p, h,
                                 ambient_cov_mse=use_ambient_cov_mse,
                                 ambient_drift_mse=use_ambient_drift_mse)

            model_dir = os.path.join(base_dir, name)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(aedf.state_dict(), os.path.join(model_dir, "aedf.pth"))

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

    print("All curves processed.")
