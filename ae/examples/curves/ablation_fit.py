import torch.nn as nn
import torch
import os
import json

from ae.toydata.curves import *
from ae.toydata.local_dynamics import *
from ae.toydata import RiemannianManifold, PointCloud
from ae.utils import process_data
from ae.models import AutoEncoder, fit_model
from ae.models import LossWeights, TotalLoss


# Model configuration parameters
train_seed = None  # Set fixed seeds for reproducibility
test_seed = None
n_train = 30
batch_size = int(n_train/2)

# Architecture parameters
intrinsic_dim = 1
extrinsic_dim = 2
hidden_dims = [16, 16, 16]

# Training parameters
lr = 0.001
weight_decay = 0.
# Training epochs
epochs_ae = 5000
print_freq = 1000
# Penalty weights
diffeo_weight = 0.01
first_order_weight = 0.01
second_order_weight = 0.01

# Activation functions
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()

# ============================================================================
# Generate data and train models
# ============================================================================

# Pick the manifold and dynamics
curve = SineCurve()
dynamics = RiemannianBrownianMotion()
manifold = RiemannianManifold(curve.local_coords(), curve.equation())
local_drift = dynamics.drift(manifold)
local_diffusion = dynamics.diffusion(manifold)

# Generate point cloud and process the data
point_cloud = PointCloud(manifold, curve.bounds(), local_drift, local_diffusion, True)
print("Extrinsic drift")
print(point_cloud.extrinsic_drift)
x, _, mu, cov, local_x = point_cloud.generate(n=n_train, seed=train_seed)
x, mu, cov, p, n, h = process_data(x, mu, cov, d=intrinsic_dim)

# Define weight configurations for the four AE variants
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

# Store trained models if needed
trained_models = {}

# Iterate over each configuration and train the AE
for name, weights in weight_configs.items():
    print(f"Training autoencoder with {name} weights...")

    ae = AutoEncoder(
        extrinsic_dim=extrinsic_dim,
        intrinsic_dim=intrinsic_dim,
        hidden_dims=hidden_dims,
        encoder_act=encoder_act,
        decoder_act=decoder_act
    )

    ae_loss = TotalLoss(weights)

    fit_model(
        model=ae,
        loss=ae_loss,
        input_data=x,
        targets=(p, h, cov, mu),
        lr=lr,
        epochs=epochs_ae,
        print_freq=print_freq,
        weight_decay=weight_decay,
        batch_size=batch_size
    )

    trained_models[name] = ae

# Gather together all hyperparameters and files into a config:
config = {
    "train_seed": train_seed,
    "test_seed": test_seed,
    "n_train": n_train,
    "batch_size": batch_size,
    "intrinsic_dim": intrinsic_dim,
    "extrinsic_dim": extrinsic_dim,
    "hidden_dims": hidden_dims,
    "lr": lr,
    "weight_decay": weight_decay,
    "epochs_ae": epochs_ae,
    "print_freq": print_freq,
    "diffeo_weight": diffeo_weight,
    "first_order_weight": first_order_weight,
    "second_order_weight": second_order_weight,
    "manifold": curve.__class__.__name__,
    "dynamics": dynamics.__class__.__name__
}

# After training completes, save the models and important variables.
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump(config, f)

for name, _ in weight_configs.items():
    torch.save(trained_models[name].state_dict(), os.path.join(save_dir, "ae_"+str(name)+".pth"))

# Optionally, save any learned AE representations or encodings if needed
print("Models and configuration saved successfully.")
