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
from ae.utils import process_data, embed_data_with_qr_matrix
from ae.experiment_classes.errors.ae_errors import compute_all_losses_for_model
from ae.experiments.train_and_save import compare_mse, embedding_dim, embedding_matrix, embed_data
from ae.experiments.train_and_save import encoder_act, decoder_act, final_act, drift_act, diffusion_act

# NOTE:
# Choose a base directory for saved models: either 'save_models' or a specific name
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
observed_dim = config["observed_dim"]
hidden_dims = config["hidden_dims"]
drift_layers = config["drift_layers"]
diff_layers = config["diff_layers"]
manifold_name = config["manifold"]
dynamics_name = config["dynamics"]
print("Intrinsic dimension is = "+str(intrinsic_dim))
print("Extrinsic dimension is = "+str(extrinsic_dim))
print("Observed dimension is = "+str(observed_dim))
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
if embed_data:
    x, mu, cov, p, n, h = embed_data_with_qr_matrix(x, mu, cov, p, n, h, embedding_matrix)
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
    ae = AutoEncoder(observed_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act, final_act=final_act)
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
x_all, mu_all, cov_all, p_all, n_all, h_all = process_data(x_all, mu_all, cov_all, d=intrinsic_dim)
if embed_data:
    x_all, mu_all, cov_all, p_all, n_all, h_all = embed_data_with_qr_matrix(x_all, mu_all,
                                                                            cov_all, p_all, n_all, h_all, embedding_matrix)
# Evaluate extrapolation losses on increasing shells
shell_results = []
for eps in eps_grid:
    lower = base_lower - eps
    upper = base_upper + eps
    # TODO: something is off: often we can see interpolation loss for 2nd being smallest yet the extrap
    #  starts off greater. This makes no sense because the beginning of the plot is in the training region!
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
    # "Encoder/Decoder Conditioning": [
    #     "Min Smallest Decoder SV", "Max Largest Decoder SV",
    #     "Min Smallest Encoder SV", "Max Largest Encoder SV"
    # ],
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
# TODO: this fails for embedded data in high dimension because we are returning extrinsic dimension GT surfaces/curves
# TODO: plot mean-curvature for curves in a side by side plot of the manifold
if intrinsic_dim == 1:
    phi_u, u_grid = point_cloud_shell.get_curve(num_grid=num_grid)
    phi_u_tensor = torch.tensor(phi_u, dtype=torch.float32, device=device)
    # print(phi_u_tensor.size())
elif intrinsic_dim == 2:
    phi_u, u_grid = point_cloud_shell.get_surface(num_grid=num_grid)
    phi_u_tensor = torch.tensor(phi_u.reshape(-1, phi_u.shape[-1]), dtype=torch.float32, device=device)
    # print(phi_u_tensor.size())
    # print(phi_u.shape)
else:
    raise NotImplementedError("Only 1D and 2D manifolds are currently supported.")
if embed_data:
    phi_u_tensor = phi_u_tensor @ embedding_matrix.T
    phi_u = phi_u @ embedding_matrix.T.cpu().numpy()
if intrinsic_dim == 1:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(phi_u_tensor[:, 0], phi_u_tensor[:, 1], label="Ground Truth", color="black", linewidth=2)
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
    # phi_u_tensor_reshaped = phi_u_tensor.cpu().numpy().reshape(phi_u.shape)
    ax.plot_surface(
        phi_u[:, :, 0],
        phi_u[:, :, 1],
        phi_u[:, :, 2],
        color='gray', alpha=0.3, rstride=1, cstride=1, linewidth=0
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
    # TODO: this Seems to be redundant; phi_u is already of that shape!
    phi_u_grid = phi_u.reshape(num_grid, num_grid, observed_dim)

    # Compute ground truth drift field on the grid
    mu_gt = np.zeros((num_grid, num_grid, extrinsic_dim))
    for i in range(num_grid):
        for j in range(num_grid):
            u = U1[i, j]
            v = U2[i, j]
            mu_gt[i, j, :] = point_cloud_shell.np_extrinsic_drift(u, v)[:, 0]
    if embed_data:
        mu_gt = mu_gt @ embedding_matrix.T.cpu().numpy()

    # Compute model drifts
    mu_models = {}
    for name in model_keys:
        mu_pred = aedf_models[name].compute_ambient_drift(phi_u_tensor).detach().numpy()
        mu_models[name] = mu_pred.reshape(num_grid, num_grid, observed_dim)

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


# #======================================================================================================================
# # Covariance eigenvalue scatter plots
# #======================================================================================================================
# # TODO refactor this into a different module and import it.
# def compute_eigenvalue_spectrum(cov_matrices):
#     """Compute ordered eigenvalues for a batch of symmetric PSD matrices."""
#     # cov_matrices: (N, D, D)
#     evals = torch.linalg.eigvalsh(cov_matrices)  # (N, D), sorted ascending
#     return evals.detach().numpy()
#
# # Evaluate ground truth covariance at grid points
# if intrinsic_dim == 1:
#     cov_gt = np.zeros((num_grid, extrinsic_dim, extrinsic_dim))
#     for i, u in enumerate(u_grid):
#         cov_gt[i] = point_cloud_shell.np_extrinsic_covariance(u)
#     cov_gt = torch.tensor(cov_gt, dtype=torch.float32, device=device)
# elif intrinsic_dim == 2:
#     cov_gt = np.zeros((num_grid, num_grid, extrinsic_dim, extrinsic_dim))
#     for i in range(num_grid):
#         for j in range(num_grid):
#             u = U1[i, j]
#             v = U2[i, j]
#             cov_gt[i, j] = point_cloud_shell.np_extrinsic_covariance(u, v)
#     cov_gt = torch.tensor(cov_gt, dtype=torch.float32, device=device)
#     cov_gt = cov_gt.view(-1, extrinsic_dim, extrinsic_dim)
# else:
#     raise NotImplementedError("Only 1D and 2D manifolds are currently supported.")
#
# # Stack the input points
# x_eval = phi_u_tensor  # shape (N, D), consistent with drift evaluations
#
# # Generate scatter plots for each model
# for key in model_keys:
#     model = aedf_models[key]
#
#     cov_hat = model.compute_ambient_covariance(x_eval)  # shape (N, D, D)
#
#     evals_gt = compute_eigenvalue_spectrum(cov_gt)      # shape (N, D)
#     evals_pred = compute_eigenvalue_spectrum(cov_hat)   # shape (N, D)
#
#     d = evals_gt.shape[1]
#     fig, axes = plt.subplots(1, d, figsize=(4 * d, 4), sharex=True, sharey=True)
#     if d == 1:
#         axes = [axes]
#
#     for i in range(d):
#         axes[i].scatter(evals_gt[:, i], evals_pred[:, i], alpha=0.5, s=10)
#         axes[i].plot([evals_gt[:, i].min(), evals_gt[:, i].max()],
#                      [evals_gt[:, i].min(), evals_gt[:, i].max()],
#                      'k--', linewidth=1)
#         axes[i].set_title(f"Eigenvalue {i+1}")
#         axes[i].set_xlabel("Ground Truth")
#         axes[i].set_ylabel("Model Prediction")
#         axes[i].grid(True)
#
#     fig.suptitle(f"Spectral Scatter Plot: {display_name_map[key]}")
#     plt.tight_layout()
#     plt.show()

# TODO: for one dimensional curves, plot the mean curvature.
# ======================================================================================================================
# Extrinsic curvature / mean curvature via decoder Jacobian & Hessian
#   - For d=1 (curves): plots κ(u) vs u for each AE-SDE variant and a κ_pred vs κ_gt scatter.
#   - For d=2 (surfaces): computes |H| on a grid (compact heatmap).
# ======================================================================================================================

def _safe_pinv_sym(mat, rcond=1e-10):
    # mat: (N,d,d) symmetric PSD. Use pinv for stability near singular points.
    return torch.linalg.pinv(mat, rcond=rcond)

def _project_normal(J):
    """
    Given J (N,D,d), return normal projector P_N (N,D,D) using P_T = J (J^T J)^-1 J^T.
    """
    N, D, d = J.shape
    g = torch.matmul(J.transpose(1, 2), J)                      # (N,d,d)
    g_inv = _safe_pinv_sym(g)                                   # (N,d,d)
    P_T = torch.matmul(J, torch.matmul(g_inv, J.transpose(1, 2)))  # (N,D,D)
    I = torch.eye(D, device=J.device, dtype=J.dtype).unsqueeze(0).expand(N, D, D)
    P_N = I - P_T
    return P_N, g, g_inv

def mean_curvature_from_decoder(ae, z, expect_d=None):
    """
    Mean curvature vector magnitude from decoder Jacobian/Hessian at z.
    Works for d in {1,2}. Returns |H| (N,) and H_vec (N,D).
    Requires:
      ae.decoder_jacobian(z): (N,D,d)
      ae.decoder_hessian(z): (N,D,d,d)
    """
    # Obtain J and H
    try:
        J = ae.decoder_jacobian(z)              # (N,D,d)
        H = ae.decoder_hessian(z)               # (N,D,d,d)
    except AttributeError:
        raise RuntimeError("AutoEncoder must provide decoder_jacobian and decoder_hessian.")

    N, D, d = J.shape
    if expect_d is not None and d != expect_d:
        raise ValueError(f"Expected intrinsic dim {expect_d} but decoder Jacobian has d={d}.")

    P_N, g, g_inv = _project_normal(J)  # (N,D,D), (N,d,d), (N,d,d)

    # Accumulate (1/d) * sum_{i,j} g^{ij} * P_N H_{ij}
    H_vec = torch.zeros((N, D), device=J.device, dtype=J.dtype)
    for i in range(d):
        for j in range(d):
            H_ij = H[:, :, i, j]                    # (N,D)
            II_ij = torch.bmm(P_N, H_ij.unsqueeze(-1)).squeeze(-1)  # (N,D)
            coeff = g_inv[:, i, j].unsqueeze(-1)    # (N,1)
            H_vec += coeff * II_ij
    H_vec = H_vec / float(d)                         # (N,D)
    H_mag = torch.linalg.norm(H_vec, dim=1)          # (N,)
    return H_mag, H_vec

def curve_curvature_ground_truth(phi_u, u_grid, eps=1e-12):
    """
    κ_gt for a parametric curve φ(u) ∈ R^D sampled on u_grid.
    Parameterization-invariant: κ = ||(I - tt^T) φ''|| / ||φ'||^2, with t = φ'/||φ'||.
    """
    phi_u = np.asarray(phi_u)           # (N,D)
    u_grid = np.asarray(u_grid).reshape(-1)  # (N,)
    N, D = phi_u.shape

    # first and second derivative w.r.t. u (non-uniform grid OK)
    dphi_du = np.gradient(phi_u, u_grid, axis=0, edge_order=2)   # (N,D)
    d2phi_du2 = np.gradient(dphi_du, u_grid, axis=0, edge_order=2)  # (N,D)

    speed = np.linalg.norm(dphi_du, axis=1, keepdims=True) + eps
    t = dphi_du / speed                                           # (N,D)
    # normal component of acceleration:
    t_dot_a = np.sum(d2phi_du2 * t, axis=1, keepdims=True)        # (N,1)
    a_normal = d2phi_du2 - t_dot_a * t                            # (N,D)
    kappa = np.linalg.norm(a_normal, axis=1) / (np.squeeze(speed,1)**2 + eps)  # (N,)
    return kappa

# ----- CURVES: κ(u) line plot + scatter -------------------------------------------------------------------------------
if intrinsic_dim == 1:
    # Ground-truth curvature from the analytical grid you already produced: phi_u, u_grid
    kappa_gt = curve_curvature_ground_truth(phi_u, u_grid)

    # For each model: map ground-truth points x=φ(u) -> z via encoder, compute |H| from decoder (which equals κ)
    kappa_models = {}
    # Now compute using the formula (need gradients inside, so no torch.no_grad around the curvature call)
    for key in aedf_models:
        z = aedf_models[key].autoencoder.encoder(phi_u_tensor)   # (N,1)
        k_hat, _ = mean_curvature_from_decoder(aedf_models[key].autoencoder, z, expect_d=1)
        kappa_models[key] = k_hat.detach().cpu().numpy()

    # Plot κ(u): GT vs models
    plt.figure(figsize=(9, 5))
    plt.plot(u_grid, kappa_gt, label="Ground Truth", linewidth=2, color="black")
    for key in aedf_models:
        plt.plot(u_grid, kappa_models[key], linestyle="--", label=display_name_map.get(key, key))
    plt.xlabel("u")
    plt.ylabel("Curvature κ")
    plt.title("Mean Curvature along Parameter u")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Scatter κ_pred vs κ_gt for each model
    fig, axes = plt.subplots(1, min(len(aedf_models), 4), figsize=(12, 3.5), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for ax, key in zip(axes, aedf_models.keys()):
        y = kappa_models[key]
        ax.scatter(kappa_gt, y, s=10, alpha=0.5)
        lo, hi = float(np.min(kappa_gt)), float(np.max(kappa_gt))
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        ax.set_title(display_name_map.get(key, key))
        ax.set_xlabel("κ (GT)")
        ax.set_ylabel("κ (Model)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ----- SURFACES (optional): |H|(u,v) heatmap --------------------------------------------------------------------------
elif intrinsic_dim == 2:
    # Build grid-based tensors for J and H via decoder and compare |H| magnitudes on the grid
    # Prepare (u,v) grid latent codes by encoding GT points
    # with torch.no_grad():
    #     z_grid = aedf_models["vanilla"].autoencoder.encoder(phi_u_tensor)  # (N,2), using any model's encoder to get z
    # Compute |H| for each model
    H_maps = {}
    for key in aedf_models:
        z = aedf_models[key].autoencoder.encoder(phi_u_tensor)  # (N,2)
        H_mag, _ = mean_curvature_from_decoder(aedf_models[key].autoencoder, z, expect_d=2)
        H_maps[key] = H_mag.detach().cpu().numpy().reshape(num_grid, num_grid)

    # Simple heatmaps: GT (finite differences) vs each model.
    # Ground-truth |H| from finite differences using the same projector formula on (u,v)
    # Build numeric J and H on grid:
    # TODO: this seems redundant for exactly the same reason as above. --phi_u is already in this shape
    Phi = phi_u.reshape(num_grid, num_grid, observed_dim)  # (Nu,Nv,D)
    U1, U2 = u_grid  # from earlier
    # Finite differences along grid:
    def _grad2_grid(F, U1, U2):
        # F: (Nu, Nv, D)
        Fu = np.gradient(F, U1[:,0], axis=0, edge_order=2)                  # ∂u φ
        Fv = np.gradient(F, U2[0,:], axis=1, edge_order=2)                  # ∂v φ
        Fuu = np.gradient(Fu, U1[:,0], axis=0, edge_order=2)                # ∂uu φ
        Fvv = np.gradient(Fv, U2[0,:], axis=1, edge_order=2)                # ∂vv φ
        Fuv = np.gradient(Fu, U2[0,:], axis=1, edge_order=2)                # ∂uv φ
        return Fu, Fv, Fuu, Fuv, Fvv

    Fu, Fv, Fuu, Fuv, Fvv = _grad2_grid(Phi, U1, U2)  # each (Nu,Nv,D)
    # Flatten and convert to torch to reuse projector formula
    Fu_t = torch.tensor(Fu.reshape(-1, observed_dim), dtype=torch.float32, device=device)
    Fv_t = torch.tensor(Fv.reshape(-1, observed_dim), dtype=torch.float32, device=device)
    J_gt = torch.stack([Fu_t, Fv_t], dim=2)  # (N,D,2)
    P_N_gt, g_gt, ginv_gt = _project_normal(J_gt)

    Fuu_t = torch.tensor(Fuu.reshape(-1, observed_dim), dtype=torch.float32, device=device)
    Fvv_t = torch.tensor(Fvv.reshape(-1, observed_dim), dtype=torch.float32, device=device)
    Fuv_t = torch.tensor(Fuv.reshape(-1, observed_dim), dtype=torch.float32, device=device)

    H_gt_vec = torch.zeros((J_gt.shape[0], observed_dim), device=device)
    H_terms = { (0,0): Fuu_t, (0,1): Fuv_t, (1,0): Fuv_t, (1,1): Fvv_t }
    for i in range(2):
        for j in range(2):
            Hij = torch.bmm(P_N_gt, H_terms[(i,j)].unsqueeze(-1)).squeeze(-1)  # (N,D)
            coeff = ginv_gt[:, i, j].unsqueeze(-1)
            H_gt_vec += coeff * Hij
    H_gt_vec = H_gt_vec / 2.0
    H_gt_mag = torch.linalg.norm(H_gt_vec, dim=1).detach().cpu().numpy().reshape(num_grid, num_grid)

    # Plot GT and one or two models to keep it compact
    # keys_to_show = list(aedf_models.keys())[:3]  # pick up to 3
    print(aedf_models.keys())
    keys_to_show = ["vanilla", "first_order", "second_order"]
    cols = 1 + len(keys_to_show)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]
    im = axes[0].imshow(H_gt_mag, origin="lower", extent=[U2.min(), U2.max(), U1.min(), U1.max()], aspect='auto')
    axes[0].set_title("|H| (GT)")
    axes[0].set_xlabel("v"); axes[0].set_ylabel("u")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    for ax, key in zip(axes[1:], keys_to_show):
        im2 = ax.imshow(H_maps[key], origin="lower", extent=[U2.min(), U2.max(), U1.min(), U1.max()], aspect='auto')
        ax.set_title(f"|H| ({display_name_map.get(key,key)})")
        ax.set_xlabel("v"); ax.set_ylabel("u")
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("Surface Mean Curvature Magnitude")
    plt.tight_layout()
    plt.show()



