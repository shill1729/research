import torch.nn as nn

from ae.models import AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion, ThreeStageFit
from ae.models import LossWeights
from ae.toydata import RiemannianManifold, PointCloud
from ae.toydata.local_dynamics import *
from ae.toydata.surfaces import *
from ae.toydata.curves import *
from ae.utils import process_data

import torch
import json
import os

# Model configuration parameters
train_seed = None  # Set fixed seeds for reproducibility
test_seed = None
n_train = 30
batch_size = 20
model_grid = 90 # resolution for true vs model curves
num_grid = 30  # grid resolution per axis

# Architecture parameters
intrinsic_dim = 1
extrinsic_dim = 2
hidden_dims = [32]
drift_layers = [32]
diff_layers = [32]

# Training parameters
lr = 0.001
weight_decay = 0.
# Training epochs
epochs_ae = 9000
epochs_diffusion = 9000
epochs_drift = 9000
print_freq = 1000
# Penalty weights
diffeo_weight = 0.5
first_order_weight = 0.
second_order_weight = 0.



# Activation functions
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
drift_act = nn.Tanh()
diffusion_act = nn.Tanh()

# ============================================================================
# Generate data and train models
# ============================================================================

# Pick the manifold and dynamics
curve = Parabola()
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

# Declare an instance of an AE
ae = AutoEncoder(extrinsic_dim=extrinsic_dim,
                 intrinsic_dim=intrinsic_dim,
                 hidden_dims=hidden_dims,
                 encoder_act=encoder_act,
                 decoder_act=decoder_act)

# Declare an instance of the local coordinate SDE networks
latent_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diff_layers, drift_act, diffusion_act)

# Initialize the AE-SDE object
aedf = AutoEncoderDiffusion(latent_sde, ae)
weights = LossWeights(tangent_angle_weight=first_order_weight,
                      tangent_drift_weight=second_order_weight,
                      diffeomorphism_reg=diffeo_weight)

# Train the model using three-stage fitting
fit3 = ThreeStageFit(lr, epochs_ae, epochs_diffusion, epochs_drift, weight_decay, batch_size, print_freq)
fit3.three_stage_fit(aedf, weights, x, mu, cov, p, h)


# === Define save paths ===
save_dir = "saved_models/"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "aedf_model.pt")
config_path = os.path.join(save_dir, "config.json")

# === Save model state ===
torch.save({
    'ae_state_dict': aedf.autoencoder.state_dict(),
    'sde_state_dict': aedf.latent_sde.state_dict()
}, model_path)

# === Save config and metadata ===
config = {
    "intrinsic_dim": intrinsic_dim,
    "extrinsic_dim": extrinsic_dim,
    "hidden_dims": hidden_dims,
    "drift_layers": drift_layers,
    "diff_layers": diff_layers,
    "activation_functions": {
        "encoder": encoder_act.__class__.__name__,
        "decoder": decoder_act.__class__.__name__,
        "drift": drift_act.__class__.__name__,
        "diffusion": diffusion_act.__class__.__name__
    },
    "lr": lr,
    "weight_decay": weight_decay,
    "epochs": {
        "ae": epochs_ae,
        "diffusion": epochs_diffusion,
        "drift": epochs_drift
    },
    "batch_size": batch_size,
    "print_freq": print_freq,
    "loss_weights": {
        "tangent_angle_weight": first_order_weight,
        "tangent_drift_weight": second_order_weight,
        "diffeomorphism_reg": diffeo_weight
    },
    "manifold_class": curve.__class__.__name__,
    "dynamics_class": dynamics.__class__.__name__,
    "point_cloud_bounds": curve.bounds(),
    "training_data_shape": x.shape
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)
