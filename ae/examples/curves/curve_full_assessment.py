# TODO: this module can now load surfaces and its companion module 'curve_full_training.py' can train surfaces
#  so we should rename and change the directory maybe.
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
from ae.examples.curves.curve_full_training import compare_mse
from ae.examples.curves.curve_full_training import encoder_act, decoder_act, final_act, drift_act, diffusion_act

# NOTE:
# Choose a base directory for saved models: either 'save_models' or a specific name
# save_dir = "paraboloid_rotation_drift"
# save_dir = "paraboloid_harmonic_drift"
# save_dir = "product_rbm"
save_dir = "saved_models"

# Number of test points in new sample
n_test = 20000
num_grid = 100
eps_min = -0.1
eps_max = 0.5
eps_grid_size = 20

test_seed = None

# If training was done to compare latent vs ambient MSE or not.
if compare_mse:
    display_name_map = {
        "vanilla": "1st-latent",
        "diffeo": "2nd-latent",
        "first_order": "1st-ambient",
        "second_order": "2nd-ambient"
    }
else:
    display_name_map = {
        "vanilla": "vanilla",
        "diffeo": "diffeo",
        "first_order": "first_order",
        "second_order": "second_order"
    }


# Load configuration
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
if intrinsic_dim == 2:
    geom_mod = importlib.import_module("ae.toydata.surfaces")
elif intrinsic_dim == 1:
    geom_mod = importlib.import_module("ae.toydata.curves")
else:
    raise NotImplementedError("Only curves and surfaces are implemented currently.")
ManifoldClass = getattr(geom_mod, manifold_name)
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
# ambient_drift_model = AmbientDriftNetwork(extrinsic_dim, extrinsic_dim, drift_layers, torch.nn.Tanh())
# ambient_drift_model.load_state_dict(torch.load(os.path.join(save_dir, "ambient_drift_model.pth"), map_location=device))
# ambient_drift_model.eval()
#
# ambient_diff_model = AmbientDiffusionNetwork(extrinsic_dim, extrinsic_dim, diff_layers, torch.nn.Tanh())
# ambient_diff_model.load_state_dict(torch.load(os.path.join(save_dir, "ambient_diff_model.pth"), map_location=device))
# ambient_diff_model.eval()

# Load each AE-SDE variant
model_names = ["vanilla", "diffeo", "first_order", "second_order"]
aedf_models = {}

for name in model_names:
    subdir = os.path.join(save_dir, name)
    ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act, final_act=final_act)
    # TODO: decide on what is more performant: using the final activation layer of the encoder or not
    latent_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diff_layers, drift_act, diffusion_act, encoder_act=None)
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
interp_losses.columns = [display_name_map.get(col, col) for col in interp_losses.columns]

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
print(base_lower)
print(base_upper)

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
    #  TODO: review this and make sure it is correct then delete the conditional for eps > 0.
    # if eps >= 0:
    outside_training_mask = torch.any((local_x_all < base_lower - eps_min) | (local_x_all > base_upper + eps_min), dim=1)
    within_eps_max = torch.all((local_x_all >= lower) & (local_x_all <= upper), dim=1)
    mask = outside_training_mask & within_eps_max
    # else:
    #     inside_train = torch.all((local_x_all >= base_lower) &
    #                              (local_x_all <= base_upper), dim=1)
    #     inside_inner = torch.all((local_x_all >= lower) &
    #                              (local_x_all <= upper), dim=1)
    #     mask = inside_train & (~inside_inner)  # between inner and outer faces

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

# TODO: refactor this into a plotting module
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
                ax.plot(shell_df["epsilon"]+eps_min, shell_df[col], label=display_name_map[model], marker='o')
        ax.axvline(0., color='k', linestyle='--', linewidth=1)
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
# TODO: refactor this into a plotting module
if intrinsic_dim == 1:
    phi_u, u_grid = point_cloud_shell.get_curve(num_grid=num_grid)
    phi_u_tensor = torch.tensor(phi_u, dtype=torch.float32, device=device)
elif intrinsic_dim == 2:
    phi_u, u_grid = point_cloud_shell.get_surface(num_grid=num_grid)
    phi_u_tensor = torch.tensor(phi_u.reshape(-1, phi_u.shape[-1]), dtype=torch.float32, device=device)
    print(phi_u_tensor.size())
else:
    raise NotImplementedError("Only 1D and 2D manifolds are currently supported.")

if intrinsic_dim == 1:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(phi_u[:, 0], phi_u[:, 1], label="Ground Truth", color="black", linewidth=2)
    for name, model in aedf_models.items():
        with torch.no_grad():
            encoded = model.autoencoder.encoder(phi_u_tensor)
            decoded = model.autoencoder.decoder(encoded).cpu().numpy()
        ax.plot(decoded[:, 0], decoded[:, 1], label=display_name_map[name], linestyle="--")
    ax.set_title("Manifold Reconstruction via AutoEncoder")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
elif intrinsic_dim == 2:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        phi_u[:, :, 0], phi_u[:, :, 1], phi_u[:, :, 2], color='gray', alpha=0.3, rstride=1, cstride=1, linewidth=0
    )
    for name, model in aedf_models.items():
        with torch.no_grad():
            encoded = model.autoencoder.encoder(phi_u_tensor.view(-1, phi_u_tensor.shape[-1]))
            decoded = model.autoencoder.decoder(encoded).cpu().numpy().reshape(phi_u.shape)
        ax.plot_surface(
            decoded[:, :, 0], decoded[:, :, 1], decoded[:, :, 2],
            label=display_name_map[name], alpha=0.5, rstride=1, cstride=1
        )
    ax.set_title("Surface Reconstruction via AutoEncoder")
    plt.tight_layout()
    plt.show()


#======================================================================================================================
# Drift vector field reconstruction plot
#======================================================================================================================
model_names = ["Vanilla", "First Order", "Second Order"]
model_keys = ["vanilla", "first_order", "second_order"]
# TODO refactor this into a plotting module
if intrinsic_dim == 1:
    # compute ground‐truth drift at each u
    mu_gt = np.zeros((num_grid, extrinsic_dim))
    for i, u in enumerate(u_grid):
        mu_gt[i] = point_cloud_shell.np_extrinsic_drift(u)[:, 0]

    # compute each model's ambient drift
    mu_models = {}
    for name in model_keys:
        arr = aedf_models[name].compute_ambient_drift(phi_u_tensor).detach()
        mu_models[name] = arr
    # More aggressive decimation - show fewer arrows for cleaner visualization
    idx = np.arange(0, num_grid, 3)  # Show every 4th arrow instead of every 2nd
    # Overlay comparison (show difference from ground truth)
    fig, axes = plt.subplots(1, 3, figsize=(8, 8))
    for ax, name, key in zip(axes, model_names, model_keys):
        # Plot manifold
        ax.plot(phi_u[:, 0], phi_u[:, 1], 'k-', linewidth=2, alpha=0.4, label='Manifold')

        # Ground truth in gray
        ax.quiver(
            phi_u[idx, 0], phi_u[idx, 1],
            mu_gt[idx, 0], mu_gt[idx, 1],
            angles='xy', scale_units='xy', pivot='mid',
            width=0.002, color='black', alpha=0.6, label='Ground Truth'
        )

        # Model prediction in color
        ax.quiver(
            phi_u[idx, 0], phi_u[idx, 1],
            mu_models[key][idx, 0], mu_models[key][idx, 1],
            angles='xy', scale_units='xy', pivot='mid',
            width=0.003, color='red', alpha=0.4, label=display_name_map[key]
        )

        ax.set_title(f"{display_name_map[key]} vs Ground Truth")
        ax.grid(True, alpha=0.3)
        # ax.set_aspect('equal')
        ax.legend()

    plt.tight_layout()
    plt.show()
elif intrinsic_dim == 2:
    # Reshape phi_u back to grid for plotting
    U1, U2 = u_grid  # This is assumed to be (U1, U2) from get_surface
    phi_u_grid = phi_u.reshape(num_grid, num_grid, extrinsic_dim)

    # Compute ground truth drift field on the grid
    mu_gt = np.zeros((num_grid, num_grid, extrinsic_dim))
    for i in range(num_grid):
        for j in range(num_grid):
            u = U1[i, j]
            v = U2[i, j]
            mu_gt[i, j, :] = point_cloud_shell.np_extrinsic_drift(u, v)[:, 0]

    # Compute model drifts
    mu_models = {}
    for name in model_keys:
        mu_pred = aedf_models[name].compute_ambient_drift(phi_u_tensor).detach().numpy()
        mu_models[name] = mu_pred.reshape(num_grid, num_grid, extrinsic_dim)

    # Downsample for cleaner vector field plots
    skip = 5  # every 5th grid point
    X = phi_u_grid[::skip, ::skip, 0]
    Y = phi_u_grid[::skip, ::skip, 1]
    Z = phi_u_grid[::skip, ::skip, 2]

    U_gt = mu_gt[::skip, ::skip, 0]
    V_gt = mu_gt[::skip, ::skip, 1]
    W_gt = mu_gt[::skip, ::skip, 2]

    for key, name in zip(model_keys, model_names):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Surface plot
        ax.plot_surface(
            phi_u_grid[:, :, 0], phi_u_grid[:, :, 1], phi_u_grid[:, :, 2],
            color='lightgray', alpha=0.3, rstride=1, cstride=1, linewidth=0
        )

        # Ground truth quiver
        ax.quiver(
            X, Y, Z, U_gt, V_gt, W_gt,
            length=0.1, normalize=True, color='black', alpha=0.6, label='Ground Truth'
        )

        # Model drift quiver
        U_m = mu_models[key][::skip, ::skip, 0]
        V_m = mu_models[key][::skip, ::skip, 1]
        W_m = mu_models[key][::skip, ::skip, 2]
        ax.quiver(
            X, Y, Z, U_m, V_m, W_m,
            length=0.1, normalize=True, color='red', alpha=0.5, label=display_name_map[key]
        )

        ax.set_title(f"{display_name_map[key]} Drift Field vs Ground Truth")
        plt.tight_layout()
        plt.show()


#======================================================================================================================
# Covariance eigenvalue scatter plots
#======================================================================================================================
# TODO refactor this into a different module and import it.
def compute_eigenvalue_spectrum(cov_matrices):
    """Compute ordered eigenvalues for a batch of symmetric PSD matrices."""
    # cov_matrices: (N, D, D)
    evals = torch.linalg.eigvalsh(cov_matrices)  # (N, D), sorted ascending
    return evals.detach().numpy()

# Evaluate ground truth covariance at grid points
if intrinsic_dim == 1:
    cov_gt = np.zeros((num_grid, extrinsic_dim, extrinsic_dim))
    for i, u in enumerate(u_grid):
        cov_gt[i] = point_cloud_shell.np_extrinsic_covariance(u)
    cov_gt = torch.tensor(cov_gt, dtype=torch.float32, device=device)
elif intrinsic_dim == 2:
    cov_gt = np.zeros((num_grid, num_grid, extrinsic_dim, extrinsic_dim))
    for i in range(num_grid):
        for j in range(num_grid):
            u = U1[i, j]
            v = U2[i, j]
            cov_gt[i, j] = point_cloud_shell.np_extrinsic_covariance(u, v)
    cov_gt = torch.tensor(cov_gt, dtype=torch.float32, device=device)
    cov_gt = cov_gt.view(-1, extrinsic_dim, extrinsic_dim)
else:
    raise NotImplementedError("Only 1D and 2D manifolds are currently supported.")

# Stack the input points
x_eval = phi_u_tensor  # shape (N, D), consistent with drift evaluations

# Generate scatter plots for each model
for key in model_keys:
    model = aedf_models[key]

    cov_hat = model.compute_ambient_covariance(x_eval)  # shape (N, D, D)

    evals_gt = compute_eigenvalue_spectrum(cov_gt)      # shape (N, D)
    evals_pred = compute_eigenvalue_spectrum(cov_hat)   # shape (N, D)

    d = evals_gt.shape[1]
    fig, axes = plt.subplots(1, d, figsize=(4 * d, 4), sharex=True, sharey=True)
    if d == 1:
        axes = [axes]

    for i in range(d):
        axes[i].scatter(evals_gt[:, i], evals_pred[:, i], alpha=0.5, s=10)
        axes[i].plot([evals_gt[:, i].min(), evals_gt[:, i].max()],
                     [evals_gt[:, i].min(), evals_gt[:, i].max()],
                     'k--', linewidth=1)
        axes[i].set_title(f"Eigenvalue {i+1}")
        axes[i].set_xlabel("Ground Truth")
        axes[i].set_ylabel("Model Prediction")
        axes[i].grid(True)

    fig.suptitle(f"Spectral Scatter Plot: {display_name_map[key]}")
    plt.tight_layout()
    plt.show()
