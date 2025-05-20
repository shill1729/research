import os
import json
import importlib
import torch
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import numpy as np
import seaborn as sns  # Add this to your imports at the top if not already present

from ae.toydata import RiemannianManifold, PointCloud
from ae.sdes import SDEtorch, SDE
from ae.models import AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion
from ae.models import AmbientDriftNetwork, AmbientDiffusionNetwork
from ae.utils import process_data
# TODO:
#  CODE for
#  1. Revisit ablation tests and prove 2nd-order beats 1st-order
#  2. Write up new script
#
#    theoretical part:
#   improvement of generalization gap for autoencoder-UAT's.


n_test = 10000
# Parameters for sample paths
tn = 1.
ntime = 500
npaths = 500
plot_sample_paths = False

# Resolution for comparing true curve to model curve
model_grid = 100
# Grid for the box [a,b]^2 for L^2 error of coefficients in ambient psace
num_grid = 40
# Epsilon for boundary extension
epsilon = 0.05




# =======================
# 1. Load Configuration
# =======================
save_dir = "saved_models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(save_dir, "config.json"), "r") as f:
    config = json.load(f)


train_seed = config["train_seed"]
test_seed = config["test_seed"]
n_train = config["n_train"]
batch_size = config["batch_size"]

intrinsic_dim = config["intrinsic_dim"]
extrinsic_dim = config["extrinsic_dim"]
hidden_dims = config["hidden_dims"]
drift_layers = config["drift_layers"]
diff_layers = config["diff_layers"]

lr = config["lr"]
weight_decay = config["weight_decay"]
epochs_ae = config["epochs_ae"]
epochs_diffusion = config["epochs_diffusion"]
epochs_drift = config["epochs_drift"]
print_freq = config["print_freq"]

diffeo_weight = config["diffeo_weight"]
first_order_weight = config["first_order_weight"]
second_order_weight = config["second_order_weight"]

manifold_name = config["manifold"]
dynamics_name = config["dynamics"]

# =======================
# 2. Dynamically Load Manifold and Dynamics
# =======================
curves_mod = importlib.import_module("ae.toydata.curves")
ManifoldClass = getattr(curves_mod, manifold_name)
curve = ManifoldClass()
print(curve.bounds())

dynamics_mod = importlib.import_module("ae.toydata.local_dynamics")
DynamicsClass = getattr(dynamics_mod, dynamics_name)
dynamics = DynamicsClass()

# Instantiate a manifold object
manifold = RiemannianManifold(curve.local_coords(), curve.equation())
local_drift = dynamics.drift(manifold)
local_diffusion = dynamics.diffusion(manifold)
# Pass it to a point cloud object for test data.
# TODO: should we make two point clouds for testing interpolation and extrapolation error? We need to change
#  curve.bounds() to be adjusted by epsilon from the config file
point_cloud = PointCloud(manifold, curve.bounds(), local_drift, local_diffusion, True)


print(f"Loaded manifold: {manifold_name}, dynamics: {dynamics_name}.")

# =======================
# 3. Load Neural Network Models Correctly
# =======================
# Reconstruct architectures using saved hyperparameters
ae = AutoEncoder(
    extrinsic_dim=extrinsic_dim,
    intrinsic_dim=intrinsic_dim,
    hidden_dims=hidden_dims,
    encoder_act=torch.nn.Tanh(),
    decoder_act=torch.nn.Tanh()
)

latent_sde = LatentNeuralSDE(
    intrinsic_dim,
    drift_layers,
    diff_layers,
    torch.nn.Tanh(),
    torch.nn.Tanh()
)

# Load full AutoEncoderDiffusion model state
aedf = AutoEncoderDiffusion(latent_sde, ae)
aedf.load_state_dict(torch.load(os.path.join(save_dir, "aedf.pth"), map_location=device))

# Load Ambient models
ambient_drift_model = AmbientDriftNetwork(
    extrinsic_dim, extrinsic_dim, drift_layers, torch.nn.Tanh()
)
ambient_drift_model.load_state_dict(torch.load(os.path.join(save_dir, "ambient_drift_model.pth"), map_location=device))

ambient_diff_model = AmbientDiffusionNetwork(
    extrinsic_dim, extrinsic_dim, diff_layers, torch.nn.Tanh()
)
ambient_diff_model.load_state_dict(torch.load(os.path.join(save_dir, "ambient_diff_model.pth"), map_location=device))

# Go into evaluation mode
ambient_drift_model.eval()
ambient_drift_model.eval()
aedf.eval()
# =======================
# 4. Build SDEtorch object
# =======================
ambient_sde = SDEtorch(ambient_drift_model.drift_torch, ambient_diff_model.diffusion_torch)


print("All models loaded successfully. Ready for evaluation.")
x, _, mu, cov, local_x = point_cloud.generate(n=n_test, seed=train_seed)
x, mu, cov, p, n, h = process_data(x, mu, cov, d=intrinsic_dim)
#====
# Point cloud reconstruction
#====
x_encoded = aedf.autoencoder.encoder.forward(x)
x_recon = aedf.autoencoder.decoder.forward(x_encoded)
# Generate points along the manifold for visualization
local_x_rng = np.linspace(curve.bounds()[0][0], curve.bounds()[0][1], model_grid)
x_rng = np.zeros((model_grid, extrinsic_dim))
for i in range(model_grid):
    x_rng[i, :] = point_cloud.np_phi(local_x_rng[i]).squeeze()
x_rng = torch.tensor(x_rng, dtype=torch.float32)
local_x_rng_tensor = torch.tensor(local_x_rng, dtype=torch.float32)
# TODO: note the decoder u -> phi(u) stretches the x coordinate! from u to phi^1(u), of course!
# We can either generate test data in a larger true latent space and encode/decoder it
x_rng_encoded = aedf.autoencoder.encoder.forward(x_rng)
x_rng_decoded_amb = aedf.autoencoder.decoder.forward(x_rng_encoded)
# OR just pass the enlarged local true data and see where it is decoded to
local_x_rng_decoded = aedf.autoencoder.decoder.forward(local_x_rng_tensor.unsqueeze(-1))


# ============================================================================
# Sample Path Generation and Analysis
# ============================================================================
# Initialize for path generation
x0 = x[0, :]  # Starting point
d = intrinsic_dim  # Intrinsic dimension
time_grid = np.linspace(0, tn, ntime + 1)  # Time grid for path evolution

# 1. Generate Ground Truth paths
z0_true = local_x[0, :]  # Use the true latent coordinates for the starting point
gt_ambient_paths = np.zeros((npaths, ntime + 1, extrinsic_dim))
gt_local_paths = point_cloud.latent_sde.sample_ensemble(z0_true, tn, ntime, npaths)

for j in range(npaths):
    for i in range(ntime + 1):
        gt_ambient_paths[j, i, :] = np.squeeze(point_cloud.np_phi(gt_local_paths[j, i, :]))
gt_ambient_paths = torch.from_numpy(gt_ambient_paths)

# 2. Generate AE-SDE Model paths: AE+NN for coefficients
z0 = x_encoded[0, :].detach() # Encoded starting point
aesde_local_paths = aedf.latent_sde.sample_paths(z0, tn, ntime, npaths)
aesde_ambient_paths = aedf.lift_sample_paths(aesde_local_paths).detach()

# 3. Generate Euclidean Ambient SDE model paths: just NN for coefficients
nnsde_ambient_paths = ambient_sde.sample_ensemble(x0, tn, ntime, npaths)


# ============================================================================
# Trajectory Visualization
# ============================================================================
if plot_sample_paths:
    # Plot sample paths from all three models
    fig = plt.figure(figsize=(12, 8))
    for j in range(min(10, npaths)):  # Plot only a subset for clarity
        plt.plot(gt_ambient_paths[j, :, 0], gt_ambient_paths[j, :, 1], c="black", alpha=0.5)
        plt.plot(aesde_ambient_paths[j, :, 0], aesde_ambient_paths[j, :, 1], c="red", alpha=0.5)
        # plt.plot(model_direct_ambient_paths[j, :, 0].detach(), model_direct_ambient_paths[j, :, 1].detach(), c="purple",
        #          alpha=0.5)
        plt.plot(nnsde_ambient_paths[j, :, 0].detach(), nnsde_ambient_paths[j, :, 1].detach(), c="blue", alpha=0.5)


    # Add the manifold for reference
    plt.plot(x_rng[:, 0], x_rng[:, 1], c="green", linewidth=2, alpha=0.1)

    plt.title("Sample Path Trajectories Comparison")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(["Ground Truth", "AE-SDE", "Euclidean-SDE", "Manifold"])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('curve_plots/trajectory_comparison.png')
    plt.show()



#=============================================================================
# Comparison of drift vector fields in ambient space
#=============================================================================
# Define the domain [a,b]^2.
a, b = curve.bounds()[0][0], curve.bounds()[0][1]  # you may change these values as needed

# Generate a uniform grid of points in [a, b]^2.
x_vals = np.linspace(a, b, num_grid)
y_vals = np.linspace(a, b, num_grid)
X, Y = np.meshgrid(x_vals, y_vals)
# Each point is a 2D coordinate (x, y)
points = np.column_stack([X.ravel(), Y.ravel()])
# True curve:
true_curve = np.array([point_cloud.np_phi(x_vals[i]) for i in range(num_grid)]).squeeze()
model_curve = aedf.autoencoder.decoder(torch.tensor(x_vals.reshape((num_grid, 1)), dtype=torch.float32)).detach()
# -------------------------------------------------------------------
# Compute the vector fields.
# -------------------------------------------------------------------

# 1. True Ambient Drift:
# Note: point_cloud.np_extrinsic_drift is not vectorized and expects only the x-coordinate.
# We compute it for each grid point separately.
true_drift = np.array([point_cloud.np_extrinsic_drift(pt[0]) for pt in points])
# each call returns a 2D vector.

# 2. NN Ambient Drift: vectorized, so we call it directly on the (n,2) array.
drift_nn = ambient_drift_model(torch.tensor(points, dtype=torch.float32)).detach().numpy()

# 3. AE Ambient Drift: also vectorized.
drift_ae = aedf.compute_ambient_drift(torch.tensor(points, dtype=torch.float32)).detach().numpy()

print("Shape of drifts")
print(true_drift.shape)
print(drift_nn.shape)
print(drift_ae.shape)

# -------------------------------------------------------------------
# Plotting the vector fields side by side.
# -------------------------------------------------------------------
print("xvals and curve shapes")
print(x_vals.shape)
print(true_curve.shape)
print(model_curve.shape)
# Create a figure with three subplots side by side.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# True Ambient Drift Quiver Plot
axes[0].quiver(X, Y,
               true_drift[:, 0].reshape(X.shape),
               true_drift[:, 1].reshape(Y.shape),
               angles='xy', scale_units='xy', scale=1)
axes[0].plot(true_curve[:, 0], true_curve[:, 1], c="blue")
axes[0].plot(model_curve[:, 0], model_curve[:, 1], c="red")
axes[0].set_title("True Ambient Drift")
axes[0].set_xlim(a, b)
axes[0].set_ylim(a, b)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_aspect('equal')

# NN Ambient Drift Quiver Plot
axes[1].quiver(X, Y,
               drift_nn[:, 0].reshape(X.shape),
               drift_nn[:, 1].reshape(Y.shape),
               angles='xy', scale_units='xy', scale=1)
axes[1].plot(true_curve[:, 0], true_curve[:, 1], c="blue")
axes[1].plot(model_curve[:, 0], model_curve[:, 1], c="red")
axes[1].set_title("NN Ambient Drift")
axes[1].set_xlim(a, b)
axes[1].set_ylim(a, b)
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_aspect('equal')

# AE Ambient Drift Quiver Plot
axes[2].quiver(X, Y,
               drift_ae[:, 0].reshape(X.shape),
               drift_ae[:, 1].reshape(Y.shape),
               angles='xy', scale_units='xy', scale=1)
axes[2].plot(true_curve[:, 0], true_curve[:, 1], c="blue")
axes[2].plot(model_curve[:, 0], model_curve[:, 1], c="red")
axes[2].set_title("AE Ambient Drift")
axes[2].set_xlim(a, b)
axes[2].set_ylim(a, b)
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].set_aspect('equal')

plt.tight_layout()
plt.savefig('curve_plots/drift_fields.png')
plt.show()


# -------------------------------------------------------------------
# Compute L^2 Errors over the grid.
# -------------------------------------------------------------------
# Compute the squared error at each grid point
squared_error_nn = np.sum((true_drift.squeeze() - drift_nn) ** 2, axis=1)
squared_error_ae = np.sum((true_drift.squeeze() - drift_ae) ** 2, axis=1)

# Compute the area element of each grid cell
dx = (b - a) / (num_grid - 1)
dy = (b - a) / (num_grid - 1)
area_element = dx * dy

# Numerical integration (sum over all grid points)
l2_error_nn = np.sqrt(np.sum(squared_error_nn) * area_element)
l2_error_ae = np.sqrt(np.sum(squared_error_ae) * area_element)

print(f"L^2(R^2) Error of NN Drift Approximation: {l2_error_nn:.6f}")
print(f"L^2(R^2) Error of AE Drift Approximation: {l2_error_ae:.6f}")


# L^2 error on the manifold:
# -------------------------------------------------------------------
# Compute L^2 Errors Over the Manifold Using Sampled Data and MC
# -------------------------------------------------------------------

# Evaluate true and model drifts at the sampled points.
true_drift_samples = np.array([point_cloud.np_extrinsic_drift(u) for u in local_x.squeeze()])
drift_nn_samples = ambient_drift_model(x).detach().numpy()
drift_ae_samples = aedf.compute_ambient_drift(x).detach().numpy()

# Compute squared errors.
squared_error_nn_samples = np.sum((true_drift_samples.squeeze() - drift_nn_samples) ** 2, axis=1)
squared_error_ae_samples = np.sum((true_drift_samples.squeeze() - drift_ae_samples) ** 2, axis=1)

# Compute volume element (Jacobian determinant) at each sampled point.
# volume_elements = np.array([point_cloud.np_volume_measure(u) for u in local_x.squeeze()])

# Monte Carlo estimate of the integral.
l2_error_nn_manifold = np.sqrt(np.mean(squared_error_nn_samples))
l2_error_ae_manifold = np.sqrt(np.mean(squared_error_ae_samples))

print(f"L^2(M) Error of NN Drift (MC Estimate): {l2_error_nn_manifold:.6f}")
print(f"L^2(M) Error of AE Drift (MC Estimate): {l2_error_ae_manifold:.6f}")


# -------------------------------------------------------------------
# Compute L^2(M) Error of the Covariance Field
# -------------------------------------------------------------------

# 1. Generate test samples on the manifold
x_test, _, _, _, local_u = point_cloud.generate(n=10000, seed=test_seed)
# 2. True extrinsic covariance at each sampled intrinsic coordinate
true_cov = np.array([point_cloud.np_extrinsic_covariance(u)
                     for u in local_u.squeeze()])

# 3. Predicted covariance from the Euclidean‐NN model
ambient_diff_model.eval()
with torch.no_grad():
    A_nn = ambient_diff_model(torch.tensor(x_test, dtype=torch.float32)).numpy()
cov_nn = np.matmul(A_nn, A_nn.transpose(0, 2, 1))

# 4. Predicted covariance from the AE‐SDE model
cov_ae = aedf.compute_ambient_covariance(torch.tensor(x_test, dtype=torch.float32)).detach().numpy()

# 5. Frobenius‐norm squared errors and RMS
sqerr_nn = np.sum((true_cov - cov_nn)**2, axis=(1, 2))
sqerr_ae = np.sum((true_cov - cov_ae)**2, axis=(1, 2))
l2cov_nn = np.sqrt(np.mean(sqerr_nn))
l2cov_ae = np.sqrt(np.mean(sqerr_ae))

print(f"L^2(M) Error of NN Covariance (RMS Frobenius): {l2cov_nn:.6f}")
print(f"L^2(M) Error of AE Covariance (RMS Frobenius): {l2cov_ae:.6f}")




# ============================================================================
# Feynman-Kac Analysis with Multiple Test Functions
# ============================================================================
def chart_error_vectorized(paths):
    """
    Vectorized version of chart_error for ensemble paths

    Args:
        paths (torch.Tensor): Ensemble paths of shape (n_ensemble, n_time, n_dim)

    Returns:
        numpy.ndarray: Error values of shape (n_ensemble, n_time)
    """
    # Extract individual coordinates for vectorized computation
    x_coords = paths[:, :, 0]  # Shape: (n_ensemble, n_time)
    y_coords = paths[:, :, 1]  # Shape: (n_ensemble, n_time)

    n_ensemble, n_time = x_coords.shape
    errors = np.zeros((n_ensemble, n_time))

    # Compute errors for each point in the ensemble
    for i in range(n_ensemble):
        for t in range(n_time):
            expected_z = point_cloud.np_phi(x_coords[i, t].detach())[intrinsic_dim].squeeze()
            errors[i, t] = np.abs(expected_z - y_coords[i, t].detach().numpy())
    errors = torch.tensor(errors, dtype=torch.float32, device=paths.device)
    return errors

# Define a set of test functions for Feynman-Kac analysis
def test_functions(x):
    """
    Compute multiple test functions for Feynman-Kac analysis.
    Returns a dictionary of function values.
    """
    results = {
        'manifold_constr': chart_error_vectorized(x),
        'x': x[:, :, 0],
        'y': x[:, :, 1],
        'sin(y)-sin(x^2)': torch.sin(x[..., 1])-torch.sin(x[..., 0]**2),
        'exp(y)-exp(x^2)': torch.exp(x[..., 1]) - torch.exp(x[..., 0] ** 2),
        'cos(y)-cos(x^p)': torch.cos(x[..., 1]) - torch.cos(torch.abs(x[..., 0]) ** 1.9),
        '(y over x) minus x': (x[..., 1]/x[...,0]-x[...,0])**2
    }
    return results


# Compute test function values for all three models
gt_fk = test_functions(gt_ambient_paths.detach())
aesde_fk = test_functions(aesde_ambient_paths.detach())
# aesde_direct_fk = test_functions(model_direct_ambient_paths.detach())
eucl_fk = test_functions(nnsde_ambient_paths.detach())
# Compute mean values
gt_means = {k: torch.mean(v, dim=0) for k, v in gt_fk.items()}
aesde_means = {k: torch.mean(v, dim=0) for k, v in aesde_fk.items()}
eucl_means = {k: torch.mean(v, dim=0) for k, v in eucl_fk.items()}

# Compute errors
aesde_errors = {k: np.abs(gt_means[k].detach().numpy() - aesde_means[k].detach().numpy()) for k in gt_means}
eucl_errors = {k: np.abs(gt_means[k].detach().numpy() - eucl_means[k].detach().numpy()) for k in gt_means}


# Relative errors of FK statistics:
# Relative errors with small epsilon for stability
rel_stability = 10**-9
aesde_rel_errors = {
    k: np.abs(gt_means[k].detach().numpy() - aesde_means[k].detach().numpy()) /
       (np.abs(gt_means[k].detach().numpy()) + rel_stability)
    for k in gt_means
}
eucl_rel_errors = {
    k: np.abs(gt_means[k].detach().numpy() - eucl_means[k].detach().numpy()) /
       (np.abs(gt_means[k].detach().numpy()) + rel_stability)
    for k in gt_means
}


for func_name in gt_means.keys():
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Plot absolute error
    axes[0].plot(time_grid, eucl_errors[func_name], c="blue", label="Euclidean SDE")
    axes[0].plot(time_grid, aesde_errors[func_name], c="red", label="AE-SDE")
    axes[0].set_title(f"FK Absolute Error for {func_name}")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Absolute Error")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Plot relative error
    axes[1].plot(time_grid, eucl_rel_errors[func_name], c="blue", label="Euclidean SDE")
    axes[1].plot(time_grid, aesde_rel_errors[func_name], c="red", label="AE-SDE")
    axes[1].set_title(f"FK Relative Error for {func_name}")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Relative Error")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'curve_plots/fk_abs_rel_error_{func_name}.png')
    plt.show()


# Distributional comparisons

# --- Extract end points in intrinsic coordinate ---
gt_end_u = gt_ambient_paths[:, -1, 0].numpy()  # True SDE, final x-coordinate (u)
nn_end_u = nnsde_ambient_paths[:, -1, 0].detach().numpy()  # NN-SDE, final x-coordinate (u)
ae_end_u = aesde_ambient_paths[:, -1, 0].detach().numpy()  # AE-SDE, final x-coordinate (u)

# --- Plot histograms ---
bins = np.linspace(curve.bounds()[0][0]-epsilon, curve.bounds()[0][1]+epsilon, 50)
plt.hist(gt_end_u, bins, alpha=0.5, label='True', density=True)
plt.hist(nn_end_u, bins, alpha=0.5, label='NN-SDE', density=True)
plt.hist(ae_end_u, bins, alpha=0.5, label='AE-SDE', density=True)
plt.legend()
plt.title('Endpoint $x$-Distributions')
plt.xlabel('$x$')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- 1-Wasserstein Distances ---
W_nn = wasserstein_distance(gt_end_u, nn_end_u)
W_ae = wasserstein_distance(gt_end_u, ae_end_u)
print("\nx-coordinate distribution comparison")
print(f"Wasserstein-1 True vs NN-SDE: {W_nn:.6f}")
print(f"Wasserstein-1 True vs AE-SDE: {W_ae:.6f}")

# --- MMD with RBF Kernel ---
def mmd_rbf(X, Y, sigma=0.05):
    X = X[:, None]
    Y = Y[:, None]
    XX = np.exp(-((X - X.T) ** 2) / (2 * sigma ** 2)).mean()
    YY = np.exp(-((Y - Y.T) ** 2) / (2 * sigma ** 2)).mean()
    XY = np.exp(-((X - Y.T) ** 2) / (2 * sigma ** 2)).mean()
    return XX + YY - 2 * XY

M_nn = mmd_rbf(gt_end_u, nn_end_u)
M_ae = mmd_rbf(gt_end_u, ae_end_u)
print(f"MMD-RBF True vs NN-SDE: {M_nn:.6e}")
print(f"MMD-RBF True vs AE-SDE: {M_ae:.6e}")



# --- Plot Histograms and KDEs Together ---
plt.figure(figsize=(8, 5))
sns.histplot(gt_end_u, bins=bins, color='black', label='True', kde=True, stat='density', alpha=0.3)
sns.histplot(nn_end_u, bins=bins, color='blue', label='NN-SDE', kde=True, stat='density', alpha=0.3)
sns.histplot(ae_end_u, bins=bins, color='red', label='AE-SDE', kde=True, stat='density', alpha=0.3)

plt.legend()
plt.title('Endpoint $x$-Distributions with KDE')
plt.xlabel('$x$')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#==========================================================
# Distributional comparisons for y coordinate
#==========================================================


# --- Extract end points in extrinsic coordinate ---
gt_end_u = gt_ambient_paths[:, -1, 1].numpy()  # True SDE, final x-coordinate (u)
nn_end_u = nnsde_ambient_paths[:, -1, 1].detach().numpy()  # NN-SDE, final x-coordinate (u)
ae_end_u = aesde_ambient_paths[:, -1, 1].detach().numpy()  # AE-SDE, final x-coordinate (u)

# --- Plot histograms ---
combined_data = np.concatenate([gt_end_u, nn_end_u, ae_end_u])
bins = np.linspace(np.min(combined_data), np.max(combined_data), 90)

plt.hist(gt_end_u, bins, alpha=0.5, label='True', density=True)
plt.hist(nn_end_u, bins, alpha=0.5, label='NN-SDE', density=True)
plt.hist(ae_end_u, bins, alpha=0.5, label='AE-SDE', density=True)
plt.legend()
plt.title('Endpoint $y$-Distributions')
plt.xlabel('$y$')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\ny-coordinate distribution comparison")
# --- 1-Wasserstein Distances ---
W_nn = wasserstein_distance(gt_end_u, nn_end_u)
W_ae = wasserstein_distance(gt_end_u, ae_end_u)
print(f"Wasserstein-1 True vs NN-SDE: {W_nn:.6f}")
print(f"Wasserstein-1 True vs AE-SDE: {W_ae:.6f}")

# --- MMD with RBF Kernel ---
def mmd_rbf(X, Y, sigma=0.05):
    X = X[:, None]
    Y = Y[:, None]
    XX = np.exp(-((X - X.T) ** 2) / (2 * sigma ** 2)).mean()
    YY = np.exp(-((Y - Y.T) ** 2) / (2 * sigma ** 2)).mean()
    XY = np.exp(-((X - Y.T) ** 2) / (2 * sigma ** 2)).mean()
    return XX + YY - 2 * XY

M_nn = mmd_rbf(gt_end_u, nn_end_u)
M_ae = mmd_rbf(gt_end_u, ae_end_u)
print(f"MMD-RBF True vs NN-SDE: {M_nn:.6e}")
print(f"MMD-RBF True vs AE-SDE: {M_ae:.6e}")

# --- Plot Histograms and KDEs Together ---
plt.figure(figsize=(8, 5))
sns.histplot(gt_end_u, bins=bins, color='black', label='True', kde=True, stat='density', alpha=0.3)
sns.histplot(nn_end_u, bins=bins, color='blue', label='NN-SDE', kde=True, stat='density', alpha=0.3)
sns.histplot(ae_end_u, bins=bins, color='red', label='AE-SDE', kde=True, stat='density', alpha=0.3)

plt.legend()
plt.title('Endpoint $y$-Distributions with KDE')
plt.xlabel('$y$')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
