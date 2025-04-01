import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from sdes import SDE

from ae.toydata.curves import *
from ae.toydata.local_dynamics import *
from ae.toydata import RiemannianManifold, PointCloud
from ae.utils import process_data
from ae.models import AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion, fit_model, ThreeStageFit
from ae.models import LossWeights, AmbientDriftNetwork, AmbientDiffusionNetwork
from ae.models.losses import AmbientDriftLoss, AmbientDiffusionLoss
from scipy import stats

# Model configuration parameters
train_seed = None  # Set fixed seeds for reproducibility
test_seed = None
n_train = 30
n_test = 2000
batch_size = 25
intrinsic_dim = 1
extrinsic_dim = 2
epsilon = 0.2
hidden_dims = [32]
drift_layers = [16]
diff_layers = [16]

# Training parameters
lr = 0.001
epochs_ae = 9000
epochs_diffusion = 9000
epochs_drift = 9000
weight_decay = 0.
print_freq = 1000
first_order_weight = 0.05
second_order_weight = 0.005
diffeo_weight = 0.05

# Sample path input
tn = 1.
ntime = 100
npaths = 100

# Activation functions
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
drift_act = nn.CELU()
diffusion_act = nn.CELU()

# ============================================================================
# Generate data and train models
# ============================================================================

# Pick the manifold and dynamics
curve = RationalCurve() # If you change this, you need to hard-code the chart constraint
dynamics = LangevinHarmonicOscillator()
manifold = RiemannianManifold(curve.local_coords(), curve.equation())
local_drift = dynamics.drift(manifold)
local_diffusion = dynamics.diffusion(manifold)

# Generate point cloud and process the data
point_cloud = PointCloud(manifold, curve.bounds(), local_drift, local_diffusion, True)
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
ambient_sde = SDE(ambient_drift_model.drift_numpy, ambient_diff_model.diffusion_numpy)

# ============================================================================
# Model performance evaluation
# ============================================================================

# Encode the train data
x_encoded = aedf.autoencoder.encoder.forward(x)
x_recon = aedf.autoencoder.decoder.forward(x_encoded)

# Generate test data with expanded bounds
train_bounds = curve.bounds()[0]
large_bounds = [(train_bounds[0] - epsilon, train_bounds[1] + epsilon)]
point_cloud = PointCloud(manifold, large_bounds, local_drift, local_diffusion, True)
x_test, _, mu_test, cov_test, local_x_test = point_cloud.generate(n=n_test, seed=test_seed)
x_test, mu_test, cov_test, p_test, n_test, h_test = process_data(x_test, mu_test, cov_test, d=intrinsic_dim)

# 1. Reconstruction loss
x_test_encoded = aedf.autoencoder.encoder.forward(x_test)
x_test_recon = aedf.autoencoder.decoder.forward(x_test_encoded)
test_recon_error = torch.linalg.vector_norm(x_test_recon - x_test, ord=2, dim=1) ** 2
test_recon_mse = torch.mean(test_recon_error).numpy()
print("Reconstruction MSE on test data = " + str(test_recon_mse))

# 2. Tangent space reconstruction loss
p_model_train = aedf.autoencoder.neural_orthogonal_projection(x_encoded).detach()
tangent_space_train_mse = torch.mean(torch.linalg.matrix_norm(p_model_train - p, ord="fro") ** 2).detach().numpy()
print("Tangent space MSE on train data = " + str(tangent_space_train_mse))

p_model_test = aedf.autoencoder.neural_orthogonal_projection(x_test_encoded).detach()
tangent_space_test_mse = torch.mean(torch.linalg.matrix_norm(p_model_test - p_test, ord="fro") ** 2).detach().numpy()
print("Tangent space MSE on test data = " + str(tangent_space_test_mse))

model_grid = 90
# Generate points along the manifold for visualization
local_x_rng = np.linspace(large_bounds[0][0], large_bounds[0][1], model_grid)
x_rng = np.zeros((model_grid, extrinsic_dim))
for i in range(model_grid):
    x_rng[i, :] = point_cloud.np_phi(local_x_rng[i]).squeeze()
x_rng = torch.tensor(x_rng, dtype=torch.float32)
x_rng_encoded = aedf.autoencoder.encoder.forward(x_rng)
x_rng_decoded = aedf.autoencoder.decoder.forward(x_rng_encoded)

# Plot the test scatter plot against the model curve
fig = plt.figure(figsize=(10, 6))
plt.scatter(x_test[:, 0], x_test[:, 1], alpha=0.5, label='Test Data')
plt.plot(x_rng_decoded[:, 0], x_rng_decoded[:, 1], c="red", linewidth=2, label='AE-SDE Manifold')
plt.title('Test Data vs Learned Manifold')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig('curve_plots/manifold_reconstruction.png')
plt.show()

# ============================================================================
# Sample Path Generation and Analysis
# ============================================================================

# Initialize for path generation
x0 = x[0, :].detach().numpy()  # Starting point
d = intrinsic_dim  # Intrinsic dimension
time_grid = np.linspace(0, tn, ntime + 1)  # Time grid for path evolution

# 1. Generate Ground Truth paths
z0_true = local_x[0, :]  # Use the true latent coordinates for the starting point
ambient_paths = np.zeros((npaths, ntime + 1, extrinsic_dim))
latent_paths = point_cloud.latent_sde.sample_ensemble(z0_true, tn, ntime, npaths)

for j in range(npaths):
    for i in range(ntime + 1):
        ambient_paths[j, i, :] = np.squeeze(point_cloud.np_phi(latent_paths[j, i, :]))

# 2. Generate AE-SDE Model paths
z0 = x_encoded[0, :].detach().numpy()  # Encoded starting point
model_local_paths = aedf.latent_sde.sample_paths(z0, tn, ntime, npaths)
model_ambient_paths = aedf.lift_sample_paths(model_local_paths)

# 3. Generate Euclidean Ambient SDE model paths
euclidean_model_paths = ambient_sde.sample_ensemble(x0, tn, ntime, npaths)

# ============================================================================
# Trajectory Visualization
# ============================================================================

# Plot sample paths from all three models
fig = plt.figure(figsize=(12, 8))
for j in range(min(10, npaths)):  # Plot only a subset for clarity
    plt.plot(ambient_paths[j, :, 0], ambient_paths[j, :, 1], c="black", alpha=0.5)
    plt.plot(model_ambient_paths[j, :, 0], model_ambient_paths[j, :, 1], c="red", alpha=0.5)
    plt.plot(euclidean_model_paths[j, :, 0], euclidean_model_paths[j, :, 1], c="blue", alpha=0.5)

# Add the manifold for reference
# plt.plot(x_rng_decoded[:, 0], x_rng_decoded[:, 1], c="green", linewidth=2)

plt.title("Sample Path Trajectories Comparison")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(["Ground Truth", "AE-SDE", "Euclidean SDE"])
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('curve_plots/trajectory_comparison.png')
plt.show()

# ============================================================================
# Statistical Analysis of Model Performance
# ============================================================================

# 1. Comparing Euclidean distance of means over time
true_mean = np.mean(ambient_paths, axis=0)
model_mean = np.mean(model_ambient_paths, axis=0)
euclidean_model_mean = np.mean(euclidean_model_paths, axis=0)

deviation_of_means_ae_sde = np.linalg.norm(model_mean - true_mean, axis=1)
deviation_of_means_euclidean_model = np.linalg.norm(euclidean_model_mean - true_mean, axis=1)

fig = plt.figure(figsize=(10, 6))
plt.plot(time_grid, deviation_of_means_euclidean_model, c="blue", label="Euclidean SDE")
plt.plot(time_grid, deviation_of_means_ae_sde, c="red", label="AE-SDE")
plt.title("Deviation of Means: $\\|E(X_t)-E(\hat{X}_t)\\|_2$")
plt.xlabel("Time")
plt.ylabel("Euclidean Distance")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('curve_plots/mean_deviation.png')
plt.show()

# 2. Comparing average Euclidean distance between paths
deviation_aesde = np.linalg.norm(ambient_paths - model_ambient_paths, axis=2)
deviation_eucl = np.linalg.norm(ambient_paths - euclidean_model_paths, axis=2)
mean_of_deviation_aesde = deviation_aesde.mean(axis=0)
mean_of_deviation_eucl = deviation_eucl.mean(axis=0)

fig = plt.figure(figsize=(10, 6))
plt.plot(time_grid, mean_of_deviation_eucl, c="blue", label="Euclidean SDE")
plt.plot(time_grid, mean_of_deviation_aesde, c="red", label="AE-SDE")
plt.title("Mean of Path Deviations: $E\\|X_t-\hat{X}_t\\|_2$")
plt.xlabel("Time")
plt.ylabel("Average Euclidean Distance")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('curve_plots/path_deviation.png')
plt.show()


# Compare variance
var_of_deviation_aesde = deviation_aesde.var(axis=0, ddof=1)
var_of_deviation_eucl = deviation_eucl.var(axis=0, ddof=1)

fig = plt.figure(figsize=(10, 6))
plt.plot(time_grid, var_of_deviation_eucl, c="blue", label="Euclidean SDE")
plt.plot(time_grid, var_of_deviation_aesde, c="red", label="AE-SDE")
plt.title("Mean of Path Deviations: $Var\\|X_t-\hat{X}_t\\|_2$")
plt.xlabel("Time")
plt.ylabel("Variance of Euclidean Distance")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('curve_plots/path_var_deviation.png')
plt.show()


# ============================================================================
# Feynman-Kac Analysis with Multiple Test Functions
# ============================================================================
def chart_error_vectorized(paths):
    """
    Vectorized version of chart_error for ensemble paths

    Args:
        paths (numpy.ndarray): Ensemble paths of shape (n_ensemble, n_time, n_dim)

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
            expected_z = point_cloud.np_phi(x_coords[i, t])[intrinsic_dim].squeeze()
            errors[i, t] = np.abs(expected_z - y_coords[i, t])
    return errors


# Define a set of test functions for Feynman-Kac analysis
def test_functions(x):
    """
    Compute multiple test functions for Feynman-Kac analysis.
    Returns a dictionary of function values.
    """
    results = {
        'sin_x': np.sin(x[..., 0]),
        'sin_y': np.sin(x[..., 1]),
        'exp_neg_x2': np.exp(-x[..., 0] ** 2),
        'tanh_xy': np.tanh(x[..., 0] * x[..., 1]),
        'poly_x2y': x[..., 0] ** 2 * x[..., 1],
        'log_1p_x2y2': np.log(1 + x[..., 0] ** 2 + x[..., 1] ** 2),
        # 'manifold_constr': np.abs(np.sin(x[..., 0])-x[..., 1])
        'manifold_constr': chart_error_vectorized(x)
    }
    return results


# Compute test function values for all three models
gt_fk = test_functions(ambient_paths)
aesde_fk = test_functions(model_ambient_paths)
eucl_fk = test_functions(euclidean_model_paths)

# Compute mean values and errors
gt_means = {k: v.mean(axis=0) for k, v in gt_fk.items()}
aesde_means = {k: v.mean(axis=0) for k, v in aesde_fk.items()}
eucl_means = {k: v.mean(axis=0) for k, v in eucl_fk.items()}

aesde_errors = {k: np.abs(gt_means[k] - aesde_means[k]) for k in gt_means}
eucl_errors = {k: np.abs(gt_means[k] - eucl_means[k]) for k in gt_means}

# Create plots for each test function
for func_name in gt_means.keys():
    fig = plt.figure(figsize=(10, 6))
    plt.plot(time_grid, eucl_errors[func_name], c="blue", label="Euclidean SDE")
    plt.plot(time_grid, aesde_errors[func_name], c="red", label="AE-SDE")
    plt.title(f"FK Error for {func_name}: $|E(f(X_t))-E(f(\hat{{X}}_t))|$")
    plt.xlabel("Time")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'curve_plots/fk_error_{func_name}.png')
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
    gt_samples = ambient_paths[:, t_idx, :]
    aesde_samples = model_ambient_paths[:, t_idx, :]
    eucl_samples = euclidean_model_paths[:, t_idx, :]

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
ax1.plot(time_points, kl_eucl, 'bo-', label='Euclidean SDE')
ax1.plot(time_points, kl_aesde, 'ro-', label='AE-SDE')
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