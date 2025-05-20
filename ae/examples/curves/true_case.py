import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from ae.sdes import SDE, SDEtorch

from ae.toydata.curves import *
from ae.toydata.surfaces import *
from ae.toydata.local_dynamics import *
from ae.toydata import RiemannianManifold, PointCloud, ProductSurface
from ae.utils import process_data
from ae.models import AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion, fit_model, ThreeStageFit
from ae.models import LossWeights, AmbientDriftNetwork, AmbientDiffusionNetwork
from ae.models.losses import AmbientDriftLoss, AmbientDiffusionLoss
# Model configuration parameters
train_seed = None  # Set fixed seeds for reproducibility
test_seed = None
n_train = 30
n_test = 2000
batch_size = 20
model_grid = 90 # resolution for true vs model curves
num_grid = 30  # grid resolution per axis

# Architecture parameters
intrinsic_dim = 1
extrinsic_dim = 2
epsilon = 0.1
hidden_dims = [16]
drift_layers = [16]
diff_layers = [16]

# Training parameters
lr = 0.001
weight_decay = 0.
# Training epochs
epochs_ae = 9000
epochs_diffusion = 9000
epochs_drift = 9000
print_freq = 1000
# Penalty weights
diffeo_weight = 0.2
first_order_weight = 0.02
second_order_weight = 0.02

# Sample path input
tn = 1.
ntime = 1000
npaths = 500


# Activation functions
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
drift_act = nn.Tanh()
diffusion_act = nn.Tanh()

# ============================================================================
# Generate data and train models
# ============================================================================

# Pick the manifold and dynamics
curve = BellCurve()
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

# Instantiate and train the Euclidean ambient SDE model
ambient_drift_model = AmbientDriftNetwork(extrinsic_dim, extrinsic_dim, drift_layers, drift_act)
ambient_diff_model = AmbientDiffusionNetwork(extrinsic_dim, extrinsic_dim, diff_layers, diffusion_act)
ambient_drift_loss = AmbientDriftLoss()
ambient_diff_loss = AmbientDiffusionLoss()

# Local SDE models directly in intrinsic coordinates
local_drift_model = AmbientDriftNetwork(intrinsic_dim, intrinsic_dim, drift_layers, drift_act)
local_diff_model = AmbientDiffusionNetwork(intrinsic_dim, intrinsic_dim, diff_layers, diffusion_act)

# Prepare local targets using your formulas from the notes:
z = local_x
b_targets = np.zeros((n_train, intrinsic_dim))
a_targets = np.zeros((n_train, intrinsic_dim, intrinsic_dim))
for i in range(n_train):
    b_targets[i] = point_cloud.np_local_drift(*z[i, :].T).squeeze()
    a_targets[i] = point_cloud.np_local_covariance(*z[i, :].T)  # Same here; from Σ = σσ^T
z = torch.tensor(local_x, dtype=torch.float32)  # Local coordinates already from point_cloud.generate()
print(b_targets.shape)
print(a_targets.shape)
b_targets = torch.tensor(b_targets, dtype=torch.float32)
a_targets = torch.tensor(a_targets, dtype=torch.float32)



# Losses
local_drift_loss = AmbientDriftLoss()
local_diff_loss = AmbientDiffusionLoss()

print("Training ambient diffusion model")
fit_model(ambient_diff_model, ambient_diff_loss, x, cov, lr, epochs_diffusion, print_freq, weight_decay, batch_size)
print("\nTraining ambient drift model")  # Fixed typo: was "diffusion" instead of "drift"
fit_model(ambient_drift_model, ambient_drift_loss, x, mu, lr, epochs_drift, print_freq, weight_decay, batch_size)

# Training
print("Training intrinsic diffusion model")
fit_model(local_diff_model, local_diff_loss, z, a_targets, lr, epochs_diffusion, print_freq, weight_decay, batch_size)

print("\nTraining intrinsic drift model")
fit_model(local_drift_model, local_drift_loss, z, b_targets, lr, epochs_drift, print_freq, weight_decay, batch_size)



# Optionally project by passing the orthogonal projection or implicit function
ambient_sde = SDEtorch(ambient_drift_model.drift_torch, ambient_diff_model.diffusion_torch)

# ============================================================================
# Model performance evaluation
# ============================================================================


# Generate test data with expanded bounds
train_bounds = curve.bounds()[0]
large_bounds = [(train_bounds[0] - epsilon, train_bounds[1] + epsilon)]
print("Training bounds:")
print(train_bounds)
print("Large bounds")
print(large_bounds)
point_cloud = PointCloud(manifold, large_bounds, local_drift, local_diffusion, True)
x_test, _, mu_test, cov_test, local_x_test = point_cloud.generate(n=n_test, seed=test_seed)
x_test, mu_test, cov_test, p_test, n_test, h_test = process_data(x_test, mu_test, cov_test, d=intrinsic_dim)



# Generate points along the manifold for visualization
local_x_rng = np.linspace(large_bounds[0][0], large_bounds[0][1], model_grid)
x_rng = np.zeros((model_grid, extrinsic_dim))
for i in range(model_grid):
    # True parameterization is np_phi
    x_rng[i, :] = point_cloud.np_phi(local_x_rng[i]).squeeze()
x_rng = torch.tensor(x_rng, dtype=torch.float32)
local_x_rng_tensor = torch.tensor(local_x_rng, dtype=torch.float32)
# TODO: note the decoder u -> phi(u) stretches the x coordinate! from u to phi^1(u), of course!
# We can either generate test data in a larger true latent space and encode/decoder it


# ============================================================================
# Sample Path Generation and Analysis
# ============================================================================

# Initialize for path generation
x0 = x[0, :]  # Starting point
d = intrinsic_dim  # Intrinsic dimension
time_grid = np.linspace(0, tn, ntime + 1)  # Time grid for path evolution

# 1. Generate Ground Truth paths
z0_true = local_x[0, :]  # Use the true latent coordinates for the starting point
ambient_paths = np.zeros((npaths, ntime + 1, extrinsic_dim))
latent_paths = point_cloud.latent_sde.sample_ensemble(z0_true, tn, ntime, npaths)

for j in range(npaths):
    for i in range(ntime + 1):
        # Lift the local coordinate paths to ambient space with the true parameterization
        ambient_paths[j, i, :] = np.squeeze(point_cloud.np_phi(latent_paths[j, i, :]))
ambient_paths = torch.from_numpy(ambient_paths)

# 2. Generate intrinsic SDE Model paths
local_sde = SDE(local_drift_model.drift_numpy, local_diff_model.diffusion_numpy)

# Sample local paths
z0_local = local_x[0, :]  # Starting point in local coordinates
model_local_paths = local_sde.sample_ensemble(z0_local, tn, ntime, npaths)

# Map to ambient space using true parameterization
model_ambient_paths = np.zeros((npaths, ntime + 1, extrinsic_dim))
for j in range(npaths):
    for i in range(ntime + 1):
        model_ambient_paths[j, i, :] = np.squeeze(point_cloud.np_phi(model_local_paths[j, i, :]))
model_ambient_paths = torch.from_numpy(model_ambient_paths)


# 3. Generate Euclidean Ambient SDE model paths
euclidean_model_paths = ambient_sde.sample_ensemble(x0, tn, ntime, npaths)


# 4. Generate AE-SDE ambient paths DIRECTLY
# model_direct_ambient_paths = aedf.direct_ambient_sample_paths(x0, tn, ntime, npaths)

# ============================================================================
# Trajectory Visualization
# ============================================================================

# Plot sample paths from all three models
fig = plt.figure(figsize=(12, 8))
for j in range(min(10, npaths)):  # Plot only a subset for clarity
    plt.plot(ambient_paths[j, :, 0], ambient_paths[j, :, 1], c="black", alpha=0.5)
    plt.plot(model_ambient_paths[j, :, 0], model_ambient_paths[j, :, 1], c="red", alpha=0.5)
    # plt.plot(model_direct_ambient_paths[j, :, 0].detach(), model_direct_ambient_paths[j, :, 1].detach(), c="purple",
    #          alpha=0.5)
    plt.plot(euclidean_model_paths[j, :, 0].detach(), euclidean_model_paths[j, :, 1].detach(), c="blue", alpha=0.5)


# Add the manifold for reference
# plt.plot(x_rng_decoded[:, 0], x_rng_decoded[:, 1], c="green", linewidth=2)

plt.title("Sample Path Trajectories Comparison")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(["Ground Truth", "AE-SDE", "Euclidean-SDE"])
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('curve_plots/trajectory_comparison.png')
plt.show()

#=============================================================================
# Comparison of drift vector fields in ambient space
#=============================================================================
# Define the domain [a,b]^2.
a, b = large_bounds[0][0], large_bounds[0][1]  # you may change these values as needed


# Generate a uniform grid of points in [a, b]^2.
x_vals = np.linspace(a, b, num_grid)
y_vals = np.linspace(a, b, num_grid)
X, Y = np.meshgrid(x_vals, y_vals)
# Each point is a 2D coordinate (x, y)
points = np.column_stack([X.ravel(), Y.ravel()])
# True curve:
true_curve = np.array([point_cloud.np_phi(x_vals[i]) for i in range(num_grid)]).squeeze()

# -------------------------------------------------------------------
# Compute the vector fields.
# -------------------------------------------------------------------

# 1. True Ambient Drift:
# Note: point_cloud.np_extrinsic_drift is not vectorized and expects only the x-coordinate.
# We compute it for each grid point separately.
true_drift = np.array([point_cloud.np_extrinsic_drift(pt[0]) for pt in points])
# each call returns a 2D vector.

# 2. NN Ambient Drift: vectorized, so we call it directly on the (n,2) array.
torch_grid_pts = torch.tensor(points, dtype=torch.float32)
drift_nn = ambient_drift_model(torch_grid_pts).detach().numpy()
print(points.shape)
# 3. AE Ambient Drift: also vectorized.
drift_ae = local_drift_model(torch_grid_pts[:, 0].unsqueeze(1)).detach()
print("Local drift size")
print(drift_ae.size())
print(drift_ae.unsqueeze(1).size())
dphi = np.zeros((num_grid**2, extrinsic_dim, intrinsic_dim))
q = np.zeros((num_grid**2, extrinsic_dim))
for i in range(num_grid**2):
    dphi[i] = point_cloud.np_chart_jacobian(points[i, 0])
    q[i] = point_cloud.np_q(points[i, 0]).squeeze()
print("hey!")
print(dphi.shape)
print(q.shape)
dphi = torch.tensor(dphi, dtype=torch.float32)
q = torch.tensor(q, dtype=torch.float32)
drift_ae = torch.bmm(dphi, drift_ae.unsqueeze(1)).squeeze(2)+0.5*q
drift_ae = drift_ae.detach().numpy()
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
# Create a figure with three subplots side by side.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# True Ambient Drift Quiver Plot
axes[0].quiver(X, Y,
               true_drift[:, 0].reshape(X.shape),
               true_drift[:, 1].reshape(Y.shape),
               angles='xy', scale_units='xy', scale=1)
axes[0].plot(true_curve[:, 0], true_curve[:, 1], c="blue")
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
axes[2].set_title("AE Ambient Drift")
axes[2].set_xlim(a, b)
axes[2].set_ylim(a, b)
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].set_aspect('equal')

plt.tight_layout()
plt.savefig('curve_plots/drift_fields.png')
plt.show()


#================================================================================
# Checking the growth condition
#================================================================================
x_norm = np.linalg.vector_norm(points, axis=1, ord=2, keepdims=True)
true_growth = np.linalg.vector_norm(true_drift, axis=1, ord=2)/(1+x_norm)
true_growth = np.squeeze(true_growth)
nn_growth = np.linalg.vector_norm(drift_nn, axis=1, ord=2)/(1+x_norm)[:, 0]
ae_growth = np.linalg.vector_norm(drift_ae, axis=1, ord=2)/(1+x_norm)[:, 0]
# Reshape the growth arrays to the grid shape (assuming points was built from X.ravel(), Y.ravel())
print(X.shape)
print(points.shape)
print(true_growth.shape)
print(nn_growth.shape)
print(ae_growth.shape)
true_growth_field = true_growth.reshape(X.shape)
nn_growth_field   = nn_growth.reshape(X.shape)
ae_growth_field   = ae_growth.reshape(X.shape)

# Create a figure with three subplots side by side.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot True Drift Growth Ratio as a scalar field
im0 = axes[0].imshow(true_growth_field, extent=(a, b, a, b), origin='lower', cmap='viridis')
axes[0].plot(true_curve[:, 0], true_curve[:, 1], c="blue", label="True Curve")
axes[0].set_title("True Drift Growth Ratio")
axes[0].set_xlim(a, b)
axes[0].set_ylim(a, b)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_aspect('equal')
fig.colorbar(im0, ax=axes[0], shrink=0.8)

# Plot NN Drift Growth Ratio as a scalar field
im1 = axes[1].imshow(nn_growth_field, extent=(a, b, a, b), origin='lower', cmap='viridis')
axes[1].plot(true_curve[:, 0], true_curve[:, 1], c="blue", label="True Curve")
axes[1].set_title("NN Drift Growth Ratio")
axes[1].set_xlim(a, b)
axes[1].set_ylim(a, b)
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_aspect('equal')
fig.colorbar(im1, ax=axes[1], shrink=0.8)

# Plot AE Drift Growth Ratio as a scalar field
im2 = axes[2].imshow(ae_growth_field, extent=(a, b, a, b), origin='lower', cmap='viridis')
axes[2].plot(true_curve[:, 0], true_curve[:, 1], c="blue", label="True Curve")
axes[2].set_title("AE Drift Growth Ratio")
axes[2].set_xlim(a, b)
axes[2].set_ylim(a, b)
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].set_aspect('equal')
fig.colorbar(im2, ax=axes[2], shrink=0.8)

plt.tight_layout()
plt.savefig('curve_plots/growth_ratio_grid.png')
plt.show()



# #================================================================================
# # Checking the growth condition for the diffusion
# #================================================================================
# # 1. True Ambient Drift:
# # Note: point_cloud.np_extrinsic_drift is not vectorized and expects only the x-coordinate.
# # We compute it for each grid point separately.
# true_diffusion = np.array([point_cloud.np_extrinsic_diffusion(pt[0]) for pt in points])
# # each call returns a 2D vector.
#
# # 2. NN Ambient Drift: vectorized, so we call it directly on the (n,2) array.
# diffusion_nn =  ambient_diff_model(torch.tensor(points, dtype=torch.float32)).detach().numpy()
#
# # 3. AE Ambient Drift: also vectorized.
# diffusion_ae = local_diff_model(torch.tensor(points[0,:], dtype=torch.float32)).detach().numpy()
# x_norm = np.linalg.vector_norm(points, axis=1, ord=2, keepdims=True)
# true_growth = np.linalg.matrix_norm(true_diffusion, ord=2)/(1+x_norm)[:,0]
# # true_growth = np.squeeze(true_growth)
# nn_growth = np.linalg.matrix_norm(diffusion_nn, ord=2)/(1+x_norm)[:, 0]
# ae_growth = np.linalg.matrix_norm(diffusion_ae, ord=2)/(1+x_norm)[:, 0]
# # Reshape the growth arrays to the grid shape (assuming points was built from X.ravel(), Y.ravel())
# print("Diffusion growth shapes")
# print(X.shape)
# print(points.shape)
# print(true_growth.shape)
# print(nn_growth.shape)
# print(ae_growth.shape)
# true_growth_field = true_growth.reshape(X.shape)
# nn_growth_field   = nn_growth.reshape(X.shape)
# ae_growth_field   = ae_growth.reshape(X.shape)
#
# # Create a figure with three subplots side by side.
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
# # Plot True Drift Growth Ratio as a scalar field
# im0 = axes[0].imshow(true_growth_field, extent=(a, b, a, b), origin='lower', cmap='viridis')
# axes[0].plot(true_curve[:, 0], true_curve[:, 1], c="blue", label="True Curve")
# axes[0].set_title("True Diffusion Growth Ratio")
# axes[0].set_xlim(a, b)
# axes[0].set_ylim(a, b)
# axes[0].set_xlabel("x")
# axes[0].set_ylabel("y")
# axes[0].set_aspect('equal')
# fig.colorbar(im0, ax=axes[0], shrink=0.8)
#
# # Plot NN Drift Growth Ratio as a scalar field
# im1 = axes[1].imshow(nn_growth_field, extent=(a, b, a, b), origin='lower', cmap='viridis')
# axes[1].plot(true_curve[:, 0], true_curve[:, 1], c="blue", label="True Curve")
# axes[1].set_title("NN Diffusion Growth Ratio")
# axes[1].set_xlim(a, b)
# axes[1].set_ylim(a, b)
# axes[1].set_xlabel("x")
# axes[1].set_ylabel("y")
# axes[1].set_aspect('equal')
# fig.colorbar(im1, ax=axes[1], shrink=0.8)
#
# # Plot AE Drift Growth Ratio as a scalar field
# im2 = axes[2].imshow(ae_growth_field, extent=(a, b, a, b), origin='lower', cmap='viridis')
# axes[2].plot(true_curve[:, 0], true_curve[:, 1], c="blue", label="True Curve")
# axes[2].set_title("AE Diffusion Growth Ratio")
# axes[2].set_xlim(a, b)
# axes[2].set_ylim(a, b)
# axes[2].set_xlabel("x")
# axes[2].set_ylabel("y")
# axes[2].set_aspect('equal')
# fig.colorbar(im2, ax=axes[2], shrink=0.8)
#
# plt.tight_layout()
# plt.savefig('curve_plots/growth_ratio_grid.png')
# plt.show()

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

def chart_error_vectorized_sq_term(paths):
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
            errors[i, t] = expected_z**2+y_coords[i, t].detach().numpy()**2
    errors = torch.tensor(errors, dtype=torch.float32, device=paths.device)
    return errors

def chart_error_vectorized_cross_term(paths):
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
            errors[i, t] = -2 * expected_z * y_coords[i, t].detach().numpy()
    errors = torch.tensor(errors, dtype=torch.float32, device=paths.device)
    return errors


# Define a set of test functions for Feynman-Kac analysis
def test_functions(x):
    """
    Compute multiple test functions for Feynman-Kac analysis.
    Returns a dictionary of function values.
    """
    results = {
        'x^2': x[..., 0] ** 2,
        'x^3': x[..., 0] ** 3,
        'x^4': x[..., 0] ** 4,
        'xy': x[..., 0]*x[..., 1],
        'x over (1+y)': x[..., 0]/(1+x[..., 1]**2),
        'arctan(y,x)': torch.atan2(x[..., 1], x[..., 0]),
        'xy-x^3': x[..., 0]*x[..., 1]-x[..., 0]**3,
        'log(1+x^2+y^2)': torch.log(1+x[..., 0] ** 2 + x[..., 1] ** 2),
        'x+y': (x[..., 0]+x[..., 1]),
        'norm': torch.sqrt(x[..., 0]**2 + x[..., 1]**2),
        'sin(x)+sin(y)': (torch.sin(x[..., 0]) + torch.sin(x[..., 1])),
        'normal ext': 0.5 * (torch.sqrt(1+x[..., 0]**2)+ torch.arctanh(x[..., 0]/torch.sqrt(1+x[...,0]**2))),
        'manifold_constr': chart_error_vectorized(x),
        'manifold_constr sq': chart_error_vectorized(x)**2,
        'manifold_constr sq term': chart_error_vectorized_sq_term(x),
        'manifold_constr cross term': chart_error_vectorized_cross_term(x)
    }
    return results


# Compute test function values for all three models
gt_fk = test_functions(ambient_paths.detach())
aesde_fk = test_functions(model_ambient_paths.detach())
# aesde_direct_fk = test_functions(model_direct_ambient_paths.detach())
eucl_fk = test_functions(euclidean_model_paths.detach())

# Compute mean values and errors
gt_means = {k: v.mean(dim=0) for k, v in gt_fk.items()}
aesde_means = {k: v.mean(dim=0) for k, v in aesde_fk.items()}
# aesde_direct_means = {k: v.mean(dim=0) for k, v in aesde_direct_fk.items()}
eucl_means = {k: v.mean(dim=0) for k, v in eucl_fk.items()}

aesde_errors = {k: np.abs(gt_means[k].detach().numpy() - aesde_means[k].detach().numpy()) for k in gt_means}
# aesde_direct_errors = {k: np.abs(gt_means[k].detach().numpy() - aesde_direct_means[k].detach().numpy()) for k in gt_means}
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


# Create plots for each test function
# for func_name in gt_means.keys():
#     fig = plt.figure(figsize=(10, 6))
#     plt.plot(time_grid, eucl_errors[func_name], c="blue", label="Euclidean SDE")
#     plt.plot(time_grid, aesde_errors[func_name], c="red", label="AE-SDE")
#     # plt.plot(time_grid, aesde_direct_errors[func_name], c="purple", label="AE-SDE Direct")
#     plt.title(f"FK Error for {func_name}: $|E(f(X_t))-E(f(\hat{{X}}_t))|$")
#     plt.xlabel("Time")
#     plt.ylabel("Absolute Error")
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.savefig(f'curve_plots/fk_error_{func_name}.png')
#     plt.show()
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


# ============================================================================
# Distribution Comparison Analysis
# ============================================================================

# Select time points for distribution analysis
time_indices = [0, int(ntime / 4), int(ntime / 2), int(3 * ntime / 4), ntime]
time_points = [time_grid[i] for i in time_indices]


# Compute KL divergence between distributions using histogram-based method
def estimate_kl_divergence(p_samples, q_samples, bins=20, epsilon=1e-10):
    """
    Estimate KL divergence between distributions using histograms
    This is more robust for lower-dimensional manifolds than KDE
    """
    # Define common bin edges for both distributions
    x_min = min(p_samples[:, 0].min(), q_samples[:, 0].min())
    x_max = max(p_samples[:, 0].max(), q_samples[:, 0].max())
    y_min = min(p_samples[:, 1].min(), q_samples[:, 1].min())
    y_max = max(p_samples[:, 1].max(), q_samples[:, 1].max())

    # Add small padding to avoid edge issues
    x_range = [x_min - 0.1 * abs(x_min), x_max + 0.1 * abs(x_max)]
    y_range = [y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max)]

    # Create histograms
    p_hist, x_edges, y_edges = np.histogram2d(p_samples[:, 0], p_samples[:, 1],
                                              bins=bins, range=[x_range, y_range], density=True)
    q_hist, _, _ = np.histogram2d(q_samples[:, 0], q_samples[:, 1],
                                  bins=[x_edges, y_edges], density=True)

    # Add small epsilon to avoid log(0)
    p_hist = p_hist + epsilon
    q_hist = q_hist + epsilon

    # Normalize
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    # Compute KL divergence: sum(p * log(p/q))
    kl_div = np.sum(p_hist * np.log(p_hist / q_hist))

    return kl_div


# Compute distribution distances using alternative metrics
def compute_distribution_distances(p_samples, q_samples):
    """
    Compute distances between empirical distributions using multiple metrics
    that are robust to low-dimensional manifolds
    """

    # 1. Energy distance (a robust alternative to Wasserstein)
    # Based on pairwise distances between and within samples
    def energy_distance(x, y):
        nx, ny = len(x), len(y)

        # Calculate pairwise distances
        xx_dist = np.sum([np.linalg.norm(xi - xj) for i, xi in enumerate(x)
                          for j, xj in enumerate(x) if i < j])
        yy_dist = np.sum([np.linalg.norm(yi - yj) for i, yi in enumerate(y)
                          for j, yj in enumerate(y) if i < j])
        xy_dist = np.sum([np.linalg.norm(xi - yj) for xi in x for yj in y])

        # Normalize by number of pairs
        xx_term = 2 * xx_dist / (nx * (nx - 1)) if nx > 1 else 0
        yy_term = 2 * yy_dist / (ny * (ny - 1)) if ny > 1 else 0
        xy_term = 2 * xy_dist / (nx * ny)

        return np.sqrt(2 * xy_term - xx_term - yy_term)

    # 2. Maximum Mean Discrepancy with Gaussian kernel
    def mmd_rbf(x, y, sigma=1.0):
        """Maximum Mean Discrepancy with RBF kernel"""
        nx, ny = len(x), len(y)

        # Calculate kernel matrices
        def rbf_kernel(x1, x2, sigma):
            return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))

        # Compute all pairwise kernel evaluations
        k_xx = np.sum([rbf_kernel(xi, xj, sigma) for i, xi in enumerate(x)
                       for j, xj in enumerate(x) if i < j]) * 2 / (nx * (nx - 1)) if nx > 1 else 0
        k_yy = np.sum([rbf_kernel(yi, yj, sigma) for i, yi in enumerate(y)
                       for j, yj in enumerate(y) if i < j]) * 2 / (ny * (ny - 1)) if ny > 1 else 0
        k_xy = np.sum([rbf_kernel(xi, yj, sigma) for xi in x for yj in y]) / (nx * ny)

        return k_xx + k_yy - 2 * k_xy

    # Take a subsample if datasets are large to make computation faster
    max_samples = min(100, len(p_samples), len(q_samples))
    p_subsample = p_samples[np.random.choice(len(p_samples), max_samples, replace=False)]
    q_subsample = q_samples[np.random.choice(len(q_samples), max_samples, replace=False)]

    # Compute both metrics
    energy_dist = energy_distance(p_subsample, q_subsample)
    mmd = mmd_rbf(p_subsample, q_subsample)

    return energy_dist, mmd


# Initialize arrays to store distribution metrics
kl_aesde = np.zeros(len(time_indices))
kl_eucl = np.zeros(len(time_indices))
# energy_aesde = np.zeros(len(time_indices))
# energy_eucl = np.zeros(len(time_indices))
# mmd_aesde = np.zeros(len(time_indices))
# mmd_eucl = np.zeros(len(time_indices))

# Add random noise to samples to ensure robustness
np.random.seed(42)  # For reproducibility
# noise_scale = 1e-9

# Compute metrics for selected time points
for i, t_idx in enumerate(time_indices):
    # Extract samples at this time point
    gt_samples = ambient_paths[:, t_idx, :].detach()
    aesde_samples = model_ambient_paths[:, t_idx, :].detach()
    eucl_samples = euclidean_model_paths[:, t_idx, :].detach()

    # Add tiny noise to avoid perfect collinearity
    # gt_samples_noisy = gt_samples + noise_scale * np.random.randn(*gt_samples.shape)
    # aesde_samples_noisy = aesde_samples + noise_scale * np.random.randn(*aesde_samples.shape)
    # eucl_samples_noisy = eucl_samples + noise_scale * np.random.randn(*eucl_samples.shape)
    gt_samples_noisy = gt_samples
    aesde_samples_noisy = aesde_samples
    eucl_samples_noisy = eucl_samples

    # Compute KL divergence using histogram method
    try:
        kl_aesde[i] = estimate_kl_divergence(gt_samples_noisy, aesde_samples_noisy)
        kl_eucl[i] = estimate_kl_divergence(gt_samples_noisy, eucl_samples_noisy)
    except Exception as e:
        print(f"KL divergence calculation failed at time {time_grid[t_idx]}: {e}")
        kl_aesde[i] = np.nan
        kl_eucl[i] = np.nan

    # # Compute robust distribution distances
    # try:
    #     energy_aesde[i], mmd_aesde[i] = compute_distribution_distances(gt_samples_noisy, aesde_samples_noisy)
    #     energy_eucl[i], mmd_eucl[i] = compute_distribution_distances(gt_samples_noisy, eucl_samples_noisy)
    # except Exception as e:
    #     print(f"Distribution distance calculation failed at time {time_grid[t_idx]}: {e}")
    #     energy_aesde[i], mmd_aesde[i] = np.nan, np.nan
    #     energy_eucl[i], mmd_eucl[i] = np.nan, np.nan

    # Create scatter plots to visually compare distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].scatter(gt_samples[:, 0], gt_samples[:, 1], alpha=0.6, c='black')
    axes[0].set_title(f'Ground Truth at t={time_grid[t_idx]:.3f}')

    axes[1].scatter(aesde_samples[:, 0], aesde_samples[:, 1], alpha=0.6, c='red')
    # axes[1].set_title(f'AE-SDE (KL={kl_aesde[i]:.4f}, Energy={energy_aesde[i]:.4f})')
    axes[1].set_title(f'AE-SDE (KL={kl_aesde[i]:.4f})')

    axes[2].scatter(eucl_samples[:, 0], eucl_samples[:, 1], alpha=0.6, c='blue')
    # axes[2].set_title(f'Euclidean SDE (KL={kl_eucl[i]:.4f}, Energy={energy_eucl[i]:.4f})')
    axes[2].set_title(f'Euclidean SDE (KL={kl_eucl[i]:.4f})')

    for ax in axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, linestyle='--', alpha=0.7)
        # Set consistent axis limits
        ax.set_xlim([min(gt_samples[:, 0].min(), aesde_samples[:, 0].min(), eucl_samples[:, 0].min()) - 0.5,
                     max(gt_samples[:, 0].max(), aesde_samples[:, 0].max(), eucl_samples[:, 0].max()) + 0.5])
        ax.set_ylim([min(gt_samples[:, 1].min(), aesde_samples[:, 1].min(), eucl_samples[:, 1].min()) - 0.5,
                     max(gt_samples[:, 1].max(), aesde_samples[:, 1].max(), eucl_samples[:, 1].max()) + 0.5])

    plt.tight_layout()
    plt.savefig(f'curve_plots/distribution_comparison_t{t_idx}.png')
    plt.show()

# Plot summary of distribution metrics over time
fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))

# KL divergence plot
ax1.plot(time_points[1:], kl_eucl[1:], 'bo-', label='Euclidean SDE')
ax1.plot(time_points[1:], kl_aesde[1:], 'ro-', label='AE-SDE')
ax1.set_title('KL Divergence from Ground Truth Over Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('Estimated KL Divergence')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# # Energy distance plot
# ax2.plot(time_points, energy_eucl, 'bo-', label='Euclidean SDE')
# ax2.plot(time_points, energy_aesde, 'ro-', label='AE-SDE')
# ax2.set_title('Energy Distance from Ground Truth Over Time')
# ax2.set_xlabel('Time')
# ax2.set_ylabel('Energy Distance')
# ax2.legend()
# ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('curve_plots/distribution_metrics_summary.png')
plt.show()

# Additional plot for MMD distance
# plt.figure(figsize=(10, 6))
# plt.plot(time_points, mmd_eucl, 'bo-', label='Euclidean SDE')
# plt.plot(time_points, mmd_aesde, 'ro-', label='AE-SDE')
# plt.title('Maximum Mean Discrepancy from Ground Truth Over Time')
# plt.xlabel('Time')
# plt.ylabel('MMD (RBF Kernel)')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig('mmd_comparison.png')
# plt.show()

# Print summary statistics
print("\n=== Distribution Comparison Summary ===")
print("Time points analyzed:", time_points)


# Calculate mean metrics, ignoring NaN values
def safe_mean(x):
    return np.nanmean(x) if np.any(~np.isnan(x)) else float('nan')


mean_kl_aesde = safe_mean(kl_aesde)
mean_kl_eucl = safe_mean(kl_eucl)
# mean_energy_aesde = safe_mean(energy_aesde)
# mean_energy_eucl = safe_mean(energy_eucl)
# mean_mmd_aesde = safe_mean(mmd_aesde)
# mean_mmd_eucl = safe_mean(mmd_eucl)

print("\nMean(over time) KL Divergence:")
print(f"  AE-SDE: {mean_kl_aesde:.6f}")
print(f"  Euclidean SDE: {mean_kl_eucl:.6f}")
if not np.isnan(mean_kl_aesde) and not np.isnan(mean_kl_eucl) and mean_kl_aesde > 0:
    print(f"  Ratio (Eucl/AE-SDE): {mean_kl_eucl / mean_kl_aesde:.2f}x")

# print("\nMean Energy Distance:")
# print(f"  AE-SDE: {mean_energy_aesde:.6f}")
# print(f"  Euclidean SDE: {mean_energy_eucl:.6f}")
# if not np.isnan(mean_energy_aesde) and not np.isnan(mean_energy_eucl) and mean_energy_aesde > 0:
#     print(f"  Ratio (Eucl/AE-SDE): {mean_energy_eucl / mean_energy_aesde:.2f}x")
#
# print("\nMean Maximum Mean Discrepancy:")
# print(f"  AE-SDE: {mean_mmd_aesde:.6f}")
# print(f"  Euclidean SDE: {mean_mmd_eucl:.6f}")
# if not np.isnan(mean_mmd_aesde) and not np.isnan(mean_mmd_eucl) and mean_mmd_aesde > 0:
#     print(f"  Ratio (Eucl/AE-SDE): {mean_mmd_eucl / mean_mmd_aesde:.2f}x")