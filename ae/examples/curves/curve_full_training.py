"""
    This module fits 4 autoencoder-SDEs and an Ambient SDE model.
    The 4 AE-SDEs go through all three-stages of training: AE, diffusion, drift.
"""

# Standard library imports
import torch.nn as nn
import torch
import os
import json

# Custom library imports
from ae.toydata.curves import *
from ae.toydata.local_dynamics import *
from ae.toydata import RiemannianManifold, PointCloud
from ae.utils import process_data
from ae.models import AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion, fit_model, ThreeStageFit
from ae.models import LossWeights, AmbientDriftNetwork, AmbientDiffusionNetwork
from ae.models.losses.losses_ambient import AmbientDriftLoss, AmbientCovarianceLoss

# Model configuration parameters
use_ambient_losses = True
train_seed = None
n_train = 30
batch_size = int(n_train / 2)

# Architecture parameters
intrinsic_dim = 1
extrinsic_dim = 2
hidden_dims = [32, 32, 32]
drift_layers = [32, 32, 32]
diff_layers = [32, 32, 32]

# Training parameters
lr = 0.001
weight_decay = 0.
epochs_ae = 9000
epochs_diffusion = 9000
epochs_drift = 9000
print_freq = 1000

# Penalty weights
diffeo_weight = 0.01
first_order_weight = 0.01
second_order_weight = 0.025

# Activation functions
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
drift_act = nn.Tanh()
diffusion_act = nn.Tanh()

# Pick the manifold and dynamics
curve = SineCurve()
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
weight_configs = {
    "vanilla":     LossWeights(diffeomorphism_reg=0.0,
                               tangent_angle_weight=0.0,
                               tangent_drift_weight=0.0),
    "diffeo":      LossWeights(diffeomorphism_reg=diffeo_weight,
                               tangent_angle_weight=0.0,
                               tangent_drift_weight=0.0),
    "first_order": LossWeights(diffeomorphism_reg=diffeo_weight,
                               tangent_angle_weight=first_order_weight,
                               tangent_drift_weight=0.0),
    "second_order":LossWeights(diffeomorphism_reg=diffeo_weight,
                               tangent_angle_weight=first_order_weight,
                               tangent_drift_weight=second_order_weight),
}

# Fit each AE-SDE variant and save results
fit3 = ThreeStageFit(lr, epochs_ae, epochs_diffusion, epochs_drift, weight_decay, batch_size, print_freq)
base_dir = "saved_models"
os.makedirs(base_dir, exist_ok=True)

for name, weights in weight_configs.items():
    print(f"Training AE-SDE model with {name} weights")

    ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
    latent_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diff_layers, drift_act, diffusion_act)
    aedf = AutoEncoderDiffusion(latent_sde, ae)

    fit3.three_stage_fit(aedf, weights, x, mu, cov, p, h, ambient=use_ambient_losses)

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
print("Training ambient diffusion model")
ambient_diff_model = AmbientDiffusionNetwork(extrinsic_dim, extrinsic_dim, diff_layers, diffusion_act)
fit_model(ambient_diff_model, AmbientCovarianceLoss(), x, cov, lr, epochs_diffusion, print_freq, weight_decay, batch_size)

print("Training ambient drift model")
ambient_drift_model = AmbientDriftNetwork(extrinsic_dim, extrinsic_dim, drift_layers, drift_act)
fit_model(ambient_drift_model, AmbientDriftLoss(), x, mu, lr, epochs_drift, print_freq, weight_decay, batch_size)

torch.save(ambient_diff_model.state_dict(), os.path.join(base_dir, "ambient_diff_model.pth"))
torch.save(ambient_drift_model.state_dict(), os.path.join(base_dir, "ambient_drift_model.pth"))

print("All models and configurations saved.")
