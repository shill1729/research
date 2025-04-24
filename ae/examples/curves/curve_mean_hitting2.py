import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from ae.sdes import SDEtorch
from ae.toydata.curves import Parabola
from ae.toydata.local_dynamics import RiemannianBrownianMotion
from ae.toydata import RiemannianManifold, PointCloud
from ae.utils import process_data
from ae.models import (
    AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion, ThreeStageFit, LossWeights,
)

# === Configuration ===
intrinsic_dim = 1
extrinsic_dim = 2
hidden_dims = [32]
drift_layers = [32]
diff_layers = [32]
lr = 0.001
weight_decay = 0.
batch_size = 20
print_freq = 1000
epochs_ae = 9000
epochs_diffusion = 9000
epochs_drift = 9000
n_train = 30
tn = 1.5
ntime = 1000
npaths = 1000
# Penalties
tangent_angle_weight = 0.01
tangent_drift_weight = 0.01
diffeo_weight = 0.1
bd = 0.5
# Time grid
time_grid = np.linspace(0, tn, ntime+1)

# === Activations ===
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
drift_act = nn.Tanh()
diffusion_act = nn.Tanh()

# === Data Generation ===
curve = Parabola()
dynamics = RiemannianBrownianMotion()
manifold = RiemannianManifold(curve.local_coords(), curve.equation())
local_drift = dynamics.drift(manifold)
local_diffusion = dynamics.diffusion(manifold)
point_cloud = PointCloud(manifold, curve.bounds(), local_drift, local_diffusion, True)
x, _, mu, cov, local_x = point_cloud.generate(n=n_train)
x, mu, cov, p, n, h = process_data(x, mu, cov, d=intrinsic_dim)

# === AE-SDE Vanilla (zero regularization) ===
ae_vanilla = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
latent_sde_vanilla = LatentNeuralSDE(intrinsic_dim, drift_layers, diff_layers, drift_act, diffusion_act)
aedf_vanilla = AutoEncoderDiffusion(latent_sde_vanilla, ae_vanilla)
weights_zero = LossWeights(
    tangent_angle_weight=0.,
    tangent_drift_weight=0.,
    diffeomorphism_reg=0.
)

fit3 = ThreeStageFit(lr, epochs_ae, epochs_diffusion, epochs_drift, weight_decay, batch_size, print_freq)
print("Training vanilla AE")
fit3.three_stage_fit(aedf_vanilla, weights_zero, x, mu, cov, p, h)

# === AE-SDE Penalized ===
ae_penalized = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
latent_sde_penalized = LatentNeuralSDE(intrinsic_dim, drift_layers, diff_layers, drift_act, diffusion_act)
aedf_penalized = AutoEncoderDiffusion(latent_sde_penalized, ae_penalized)
weights_penalized = LossWeights(
    tangent_angle_weight=tangent_angle_weight,
    tangent_drift_weight=tangent_drift_weight,
    diffeomorphism_reg=diffeo_weight
)
print("\nTraining penalized AE")
fit3.three_stage_fit(aedf_penalized, weights_penalized, x, mu, cov, p, h)

# === Ground Truth Path Sampling ===
# x0 = x[0, :]
# z0_true = local_x[0, :]
z0_true = np.array([0.01])
x0 = torch.tensor(point_cloud.np_phi(z0_true).squeeze().T, dtype=torch.float32, device=x.device)

ambient_paths = np.zeros((npaths, ntime + 1, extrinsic_dim))
latent_paths = point_cloud.latent_sde.sample_ensemble(z0_true, tn, ntime, npaths)
for j in range(npaths):
    for i in range(ntime + 1):
        ambient_paths[j, i, :] = np.squeeze(point_cloud.np_phi(latent_paths[j, i, :]))
ambient_paths = torch.from_numpy(ambient_paths)

# === Model Sampling ===
def sample_model_paths(aedf, x0):
    z0 = aedf.autoencoder.encoder(x0).detach().squeeze()
    latent_paths = aedf.latent_sde.sample_paths(z0, tn, ntime, npaths)
    return aedf.lift_sample_paths(latent_paths)

ambient_paths_vanilla = sample_model_paths(aedf_vanilla, x0)
ambient_paths_penalized = sample_model_paths(aedf_penalized, x0)
bd_vec = point_cloud.np_phi(bd).squeeze().T

# === Hitting Time Estimation ===
def boundary_func(x):
    return np.linalg.vector_norm(x-bd_vec) < 0.01

def estimate_hitting_times(paths, boundary_func, time_grid):
    npaths, ntime, _ = paths.shape
    hit_times = np.full(npaths, np.nan)
    for i in range(npaths):
        for t in range(ntime):
            if boundary_func(paths[i, t, :]):
                hit_times[i] = time_grid[t]
                break
    return hit_times, ~np.isnan(hit_times)

gt_hit, gt_mask = estimate_hitting_times(ambient_paths.numpy(), boundary_func, time_grid)
van_hit, van_mask = estimate_hitting_times(ambient_paths_vanilla.detach().numpy(), boundary_func, time_grid)
pen_hit, pen_mask = estimate_hitting_times(ambient_paths_penalized.detach().numpy(), boundary_func, time_grid)

# === Plot Hitting Time Distributions ===
def plot_hitting_time_histogram(hit_times, hit_mask, label, color):
    plt.hist(hit_times[hit_mask], bins=20, alpha=0.4, label=label, color=color)

plt.figure(figsize=(10, 6))
plot_hitting_time_histogram(gt_hit, gt_mask, "Ground Truth", "green")
plot_hitting_time_histogram(van_hit, van_mask, "Vanilla AE-SDE", "red")
plot_hitting_time_histogram(pen_hit, pen_mask, "Penalized AE-SDE", "blue")
plt.xlabel("Hitting Time")
plt.ylabel("Number of Paths")
plt.legend()
plt.title("Distribution of First Hitting Times to Boundary")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("curve_plots/hitting_time_comparison_parabola.png")
plt.show()




def plot_kde(hit_times, hit_mask, label, color):
    values = hit_times[hit_mask]
    if len(values) > 1:  # KDE requires at least two data points
        kde = gaussian_kde(values)
        xs = np.linspace(values.min() - 0.05, values.max() + 0.05, 200)
        plt.plot(xs, kde(xs), label=label, color=color, lw=2)
    else:
        plt.axvline(0, label=f"{label} (single hit)", color=color, linestyle="--")

plt.figure(figsize=(10, 6))
plot_kde(gt_hit, gt_mask, "Ground Truth", "green")
plot_kde(van_hit, van_mask, "Vanilla AE-SDE", "red")
plot_kde(pen_hit, pen_mask, "Penalized AE-SDE", "blue")
plt.xlabel("Hitting Time")
plt.ylabel("Density")
plt.legend()
plt.title("Kernel Density Estimation of First Hitting Times")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("curve_plots/hitting_time_kde_comparison_parabola2.png")
plt.show()

# === Summary Statistics ===
print(f"Mean Hitting Time (GT):        {np.nanmean(gt_hit):.4f}, paths hit: {gt_mask.sum()}/{len(gt_mask)}")
print(f"Mean Hitting Time (Vanilla):   {np.nanmean(van_hit):.4f}, paths hit: {van_mask.sum()}/{len(van_mask)}")
print(f"Mean Hitting Time (Penalized): {np.nanmean(pen_hit):.4f}, paths hit: {pen_mask.sum()}/{len(pen_mask)}")