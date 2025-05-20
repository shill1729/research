import torch.nn as nn
import torch
import os
import json

from ae.toydata.curves import *
from ae.toydata.local_dynamics import *
from ae.toydata import RiemannianManifold, PointCloud
from ae.utils import process_data
from ae.models import AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion, fit_model, ThreeStageFit
from ae.models import LossWeights, AmbientDriftNetwork, AmbientDiffusionNetwork
from ae.models.losses import AmbientDriftLoss, AmbientDiffusionLoss

# Model configuration parameters
train_seed = None  # Set fixed seeds for reproducibility
test_seed = None
n_train = 50
batch_size = 25

# Architecture parameters
intrinsic_dim = 1
extrinsic_dim = 2
hidden_dims = [16]
drift_layers = [16]
diff_layers = [16]

# Training parameters
lr = 0.001
weight_decay = 0.
# Training epochs
epochs_ae = 15000
epochs_diffusion = 15000
epochs_drift = 15000
print_freq = 1000
# Penalty weights
diffeo_weight = 0.25
first_order_weight = 0.01
second_order_weight = 0.01

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

# Instantiate and train the Euclidean ambient SDE model
ambient_drift_model = AmbientDriftNetwork(extrinsic_dim, extrinsic_dim, drift_layers, drift_act)
ambient_diff_model = AmbientDiffusionNetwork(extrinsic_dim, extrinsic_dim, diff_layers, diffusion_act)
ambient_drift_loss = AmbientDriftLoss()
ambient_diff_loss = AmbientDiffusionLoss()

print("Training ambient diffusion model")
fit_model(ambient_diff_model, ambient_diff_loss, x, cov, lr, epochs_diffusion, print_freq, weight_decay, batch_size)
print("\nTraining ambient drift model")  # Fixed typo: was "diffusion" instead of "drift"
fit_model(ambient_drift_model, ambient_drift_loss, x, mu, lr, epochs_drift, print_freq, weight_decay, batch_size)


# Gather together all hyperparameters and files into a config:
config = {
    "train_seed": train_seed,
    "test_seed": test_seed,
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

# After training completes, save the models and important variables.
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump(config, f)

torch.save(aedf.state_dict(), os.path.join(save_dir, "aedf.pth"))
torch.save(ambient_drift_model.state_dict(), os.path.join(save_dir, "ambient_drift_model.pth"))
torch.save(ambient_diff_model.state_dict(), os.path.join(save_dir, "ambient_diff_model.pth"))

# Optionally, save any learned AE representations or encodings if needed
print("Models and configuration saved successfully.")