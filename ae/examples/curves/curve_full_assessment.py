import os
import json
import importlib
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ae.models import AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion
from ae.models import AmbientDriftNetwork, AmbientDiffusionNetwork
from ae.models.sdes_latent import ambient_quadratic_variation_drift
from ae.toydata import RiemannianManifold, PointCloud
from ae.utils import process_data
from ae.experiment_classes.errors.ae_errors import compute_all_losses_for_model

# Number of test points in new sample
n_test = 60000
num_grid = 50
eps_max = 0.5
eps_grid_size = 20

test_seed = None

# Load configuration
save_dir = "saved_models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(save_dir, "config.json"), "r") as f:
    config = json.load(f)

intrinsic_dim = config["intrinsic_dim"]
extrinsic_dim = config["extrinsic_dim"]
hidden_dims = config["hidden_dims"]
drift_layers = config["drift_layers"]
diff_layers = config["diff_layers"]
manifold_name = config["manifold"]
dynamics_name = config["dynamics"]

print("Loading "+str(manifold_name)+" and "+str(dynamics_name))
# Dynamically import curve and dynamics
curves_mod = importlib.import_module("ae.toydata.curves")
ManifoldClass = getattr(curves_mod, manifold_name)
curve = ManifoldClass()

dynamics_mod = importlib.import_module("ae.toydata.local_dynamics")
DynamicsClass = getattr(dynamics_mod, dynamics_name)
dynamics = DynamicsClass()

manifold = RiemannianManifold(curve.local_coords(), curve.equation())
local_drift = dynamics.drift(manifold)
local_diffusion = dynamics.diffusion(manifold)
point_cloud = PointCloud(manifold, curve.bounds(), local_drift, local_diffusion, True)
x, _, mu, cov, local_x = point_cloud.generate(n=n_test, seed=test_seed)
x, mu, cov, p, n, h = process_data(x, mu, cov, d=intrinsic_dim)

# Load ambient models
ambient_drift_model = AmbientDriftNetwork(extrinsic_dim, extrinsic_dim, drift_layers, torch.nn.Tanh())
ambient_drift_model.load_state_dict(torch.load(os.path.join(save_dir, "ambient_drift_model.pth"), map_location=device))
ambient_drift_model.eval()

ambient_diff_model = AmbientDiffusionNetwork(extrinsic_dim, extrinsic_dim, diff_layers, torch.nn.Tanh())
ambient_diff_model.load_state_dict(torch.load(os.path.join(save_dir, "ambient_diff_model.pth"), map_location=device))
ambient_diff_model.eval()

# Load each AE-SDE variant
model_names = ["vanilla", "diffeo", "first_order", "second_order"]
aedf_models = {}

for name in model_names:
    subdir = os.path.join(save_dir, name)
    ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, torch.nn.Tanh(), torch.nn.Tanh())
    latent_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diff_layers, torch.nn.Tanh(), torch.nn.Tanh())
    aedf = AutoEncoderDiffusion(latent_sde, ae)
    aedf.load_state_dict(torch.load(os.path.join(subdir, "aedf.pth"), map_location=device))
    aedf.eval()
    aedf_models[name] = aedf

#======================================================================================================================
# Compute interpolation errors
#======================================================================================================================
losses = {}
for name, model in aedf_models.items():
    loss_for_this_model = compute_all_losses_for_model(model, x, mu, cov, p, intrinsic_dim)
    losses[name] = loss_for_this_model
# Convert to data frame
interp_losses = pd.DataFrame.from_dict(losses, orient="columns")
pd.set_option('display.max_columns', None)
print("Interpolation losses (new test data sampled from training region)")
print(interp_losses.round(6))

#======================================================================================================================
# Compute extrapolation errors
#======================================================================================================================
# Setup for extrapolation shells using a single large sample
eps_grid = torch.linspace(0.01, eps_max, steps=eps_grid_size)
bounds = curve.bounds()
base_lower = torch.tensor([b[0] for b in bounds])
base_upper = torch.tensor([b[1] for b in bounds])

# Generate from the largest shell
shell_lower = base_lower - eps_max
shell_upper = base_upper + eps_max
new_bounds = [(l.item(), u.item()) for l, u in zip(shell_lower, shell_upper)]

point_cloud_shell = PointCloud(manifold, new_bounds, local_drift, local_diffusion, True)
x_all, _, mu_all, cov_all, local_x_all = point_cloud_shell.generate(n=n_test, seed=test_seed)
local_x_all = torch.tensor(local_x_all, dtype=x.dtype, device=x.device)
x_all, mu_all, cov_all, p_all, _, _ = process_data(x_all, mu_all, cov_all, d=intrinsic_dim)

# Evaluate extrapolation losses on increasing shells
shell_results = []
for eps in eps_grid:
    lower = base_lower - eps
    upper = base_upper + eps
    mask = torch.any((local_x_all < lower) | (local_x_all > upper), dim=1)
    mask &= torch.all((local_x_all >= shell_lower) & (local_x_all <= shell_upper), dim=1)

    if mask.sum() == 0:
        continue

    x_shell = x_all[mask]
    mu_shell = mu_all[mask]
    cov_shell = cov_all[mask]
    p_shell = p_all[mask]

    shell_loss = {"epsilon": eps.item(), "n_points": mask.sum().item()}
    for name, model in aedf_models.items():
        loss_vals = compute_all_losses_for_model(model, x_shell, mu_shell, cov_shell, p_shell, intrinsic_dim)
        for k, v in loss_vals.items():
            shell_loss[f"{name}/{k}"] = v
    shell_results.append(shell_loss)

shell_df = pd.DataFrame(shell_results)

pd.set_option('display.max_columns', None)
print("\nExtrapolation losses on shells:")
print(shell_df.iloc[:, 0:2].round(6))

groups = {
    "Reconstruction": ["Reconstruction"],
    "Tangent geometry": ["Tangent penalty", "Ito penalty", "Diffeomorphism Error"],
    "Encoder/Decoder Conditioning": [
        "Min Smallest Decoder SV", "Max Largest Decoder SV",
        "Min Smallest Encoder SV", "Max Largest Encoder SV"
    ],
    "Metric Geometry": ["Moore-Penrose error"],
    "Ambient Consistency": ["Ambient Cov Errors", "Ambient Drift Errors"]
}

for group_name, keys in groups.items():
    # Special layout for Encoder/Decoder group
    if group_name == "Encoder/Decoder Conditioning":
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(len(keys), 1, figsize=(8, 8), sharex=True)
        if len(keys) == 1:
            axes = [axes]

    for ax, key in zip(axes, keys):
        for model in aedf_models:
            col = f"{model}/{key}"
            if col in shell_df.columns:
                ax.plot(shell_df["epsilon"], shell_df[col], label=model, marker='o')
        ax.set_ylabel(key)
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Epsilon")
    fig.suptitle(f"{group_name} vs Epsilon")
    plt.tight_layout()
    plt.show()

#======================================================================================================================
# Manifold reconstruction plot
#======================================================================================================================
phi_u, u_grid = point_cloud_shell.get_curve(num_grid=num_grid)
phi_u_tensor = torch.tensor(phi_u, dtype=torch.float32, device=device)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(phi_u[:, 0], phi_u[:, 1], label="Ground Truth", color="black", linewidth=2)

for name, model in aedf_models.items():
    with torch.no_grad():
        encoded = model.autoencoder.encoder(phi_u_tensor)
        decoded = model.autoencoder.decoder(encoded).cpu().numpy()
    ax.plot(decoded[:, 0], decoded[:, 1], label=name, linestyle="--")

ax.set_title("Manifold Reconstruction via AutoEncoder")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

#======================================================================================================================
# Drift vector field reconstruction plot
#======================================================================================================================
# compute ground‐truth drift at each u
mu_gt = np.zeros((num_grid, extrinsic_dim))
for i, u in enumerate(u_grid):
    mu_gt[i] = point_cloud_shell.np_extrinsic_drift(u)[:, 0]

# compute each model's ambient drift
mu_models = {}
for name in ["vanilla", "first_order", "second_order"]:
    arr = aedf_models[name].compute_ambient_drift(phi_u_tensor).detach()
    mu_models[name] = arr
# More aggressive decimation - show fewer arrows for cleaner visualization
idx = np.arange(0, num_grid, 4)  # Show every 4th arrow instead of every 2nd
# Overlay comparison (show difference from ground truth)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

model_names = ["Vanilla", "First Order", "Second Order"]
model_keys = ["vanilla", "first_order", "second_order"]

for ax, name, key in zip(axes, model_names, model_keys):
    # Plot manifold
    ax.plot(phi_u[:, 0], phi_u[:, 1], 'k-', linewidth=2, alpha=0.4, label='Manifold')

    # Ground truth in gray
    ax.quiver(
        phi_u[idx, 0], phi_u[idx, 1],
        mu_gt[idx, 0], mu_gt[idx, 1],
        angles='xy', scale_units='xy', scale=1.0, pivot='mid',
        width=0.002, color='gray', alpha=0.5, label='Ground Truth'
    )

    # Model prediction in color
    ax.quiver(
        phi_u[idx, 0], phi_u[idx, 1],
        mu_models[key][idx, 0], mu_models[key][idx, 1],
        angles='xy', scale_units='xy', scale=1.0, pivot='mid',
        width=0.003, color='red', alpha=0.8, label=name
    )

    ax.set_title(f"{name} vs Ground Truth")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend()

plt.tight_layout()
plt.show()
