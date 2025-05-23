import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time

from ae.sdes import SDEtorch
from ae.toydata.curves import *
from ae.toydata.surfaces import *
from ae.toydata.local_dynamics import RiemannianBrownianMotion
from ae.toydata import RiemannianManifold, PointCloud, LangevinHarmonicOscillator
from ae.utils import process_data
from ae.models import (
    AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion, ThreeStageFit, LossWeights,
)

# === Configuration ===
intrinsic_dim = 2
extrinsic_dim = 3
hidden_dims = [32]
drift_layers = [32]
diff_layers = [32]
lr = 0.001
weight_decay = 0.
batch_size = 20
n_train = 50
# training epochs
print_freq = 1000
epochs_ae = 10000
epochs_diffusion = 2000
epochs_drift = 2000
# Paths
tn = 1
ntime = 500
npaths = 2000
# Penalties
tangent_angle_weight = 0.02
tangent_drift_weight = 0.02
diffeo_weight = 0.2
bd = np.array([0.95, -0.95])
hit_radius = 0.018
# Time grid
time_grid = np.linspace(0, tn, ntime+1)

# === Activations ===
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
drift_act = nn.Tanh()
diffusion_act = nn.Tanh()

# === Data Generation ===
curve = ProductSurface()
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
z0_true = np.array([0.01, 0.01])
x0 = torch.tensor(point_cloud.np_phi(*z0_true).squeeze().T, dtype=torch.float32, device=x.device)
start_time = time.time()
ambient_paths = np.zeros((npaths, ntime + 1, extrinsic_dim))
latent_paths = point_cloud.latent_sde.sample_ensemble(z0_true, tn, ntime, npaths)
for j in range(npaths):
    for i in range(ntime + 1):
        ambient_paths[j, i, :] = np.squeeze(point_cloud.np_phi(*latent_paths[j, i, :]))
ambient_paths = torch.from_numpy(ambient_paths)

# === Model Sampling ===
def sample_model_paths(aedf, x0):
    z0 = aedf.autoencoder.encoder(x0).detach().squeeze()
    latent_paths = aedf.latent_sde.sample_paths(z0, tn, ntime, npaths)
    return aedf.lift_sample_paths(latent_paths)

ambient_paths_vanilla = sample_model_paths(aedf_vanilla, x0)
ambient_paths_penalized = sample_model_paths(aedf_penalized, x0)
bd_vec = point_cloud.np_phi(*bd).squeeze().T
print("Sample path generation time:")
print(time.time()-start_time)
# === Hitting Time Estimation ===
def boundary_func(x):
    return np.linalg.vector_norm(x-bd_vec) < hit_radius

def estimate_hitting_times(paths, boundary_func, time_grid):
    npaths, ntime, _ = paths.shape
    hit_times = np.full(npaths, np.nan)
    for i in range(npaths):
        for t in range(ntime):
            if boundary_func(paths[i, t, :]):
                hit_times[i] = time_grid[t]
                break
    return hit_times, ~np.isnan(hit_times)

def estimate_hitting_times_torch(paths, bd_vec, hit_radius, time_grid):
    diffs = paths - bd_vec[None, None, :]
    dists = np.linalg.norm(diffs, axis=-1)
    hits = dists < hit_radius
    first_hits = hits.argmax(axis=1)
    has_hit = hits.any(axis=1)
    hit_times = np.full(paths.shape[0], np.nan)
    hit_times[has_hit] = time_grid[first_hits[has_hit]]
    return hit_times, has_hit

# gt_hit, gt_mask = estimate_hitting_times(ambient_paths.numpy(), boundary_func, time_grid)
# van_hit, van_mask = estimate_hitting_times(ambient_paths_vanilla.detach().numpy(), boundary_func, time_grid)
# pen_hit, pen_mask = estimate_hitting_times(ambient_paths_penalized.detach().numpy(), boundary_func, time_grid)

# ensure bd_vec is torch tensor on the same device as paths
bd_vec_tensor = torch.tensor(bd_vec, dtype=torch.float32, device=ambient_paths.device)
start_time = time.time()
gt_hit, gt_mask = estimate_hitting_times_torch(ambient_paths, bd_vec_tensor, hit_radius, time_grid)
van_hit, van_mask = estimate_hitting_times_torch(ambient_paths_vanilla, bd_vec_tensor, hit_radius, time_grid)
pen_hit, pen_mask = estimate_hitting_times_torch(ambient_paths_penalized, bd_vec_tensor, hit_radius, time_grid)
print("Hitting time analysis:")
print(time.time()-start_time)
# # === Plot Hitting Time Distributions ===
# def plot_hitting_time_histogram(hit_times, hit_mask, label, color):
#     plt.hist(hit_times[hit_mask], bins=20, alpha=0.4, label=label, color=color)
#
# plt.figure(figsize=(10, 6))
# plot_hitting_time_histogram(gt_hit, gt_mask, "Ground Truth", "green")
# plot_hitting_time_histogram(van_hit, van_mask, "Vanilla AE-SDE", "red")
# plot_hitting_time_histogram(pen_hit, pen_mask, "Penalized AE-SDE", "blue")
# plt.xlabel("Hitting Time")
# plt.ylabel("Number of Paths")
# plt.legend()
# plt.title("Distribution of First Hitting Times to Boundary")
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.savefig("curve_plots/hitting_time_comparison_bellcurve5.png")
# plt.show()
#
#
#
#
# def plot_kde(hit_times, hit_mask, label, color):
#     values = hit_times[hit_mask]
#     if len(values) > 1:  # KDE requires at least two data points
#         kde = gaussian_kde(values)
#         xs = np.linspace(values.min() - 0.05, values.max() + 0.05, 200)
#         plt.plot(xs, kde(xs), label=label, color=color, lw=2)
#     else:
#         plt.axvline(0, label=f"{label} (single hit)", color=color, linestyle="--")
#
# plt.figure(figsize=(10, 6))
# plot_kde(gt_hit, gt_mask, "Ground Truth", "green")
# plot_kde(van_hit, van_mask, "Vanilla AE-SDE", "red")
# plot_kde(pen_hit, pen_mask, "Penalized AE-SDE", "blue")
# plt.xlabel("Hitting Time")
# plt.ylabel("Density")
# plt.legend()
# plt.title("Kernel Density Estimation of First Hitting Times")
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.savefig("curve_plots/hitting_time_kde_comparison_bellcurve5.png")
# plt.show()
#
# # === Summary Statistics ===
# print(f"Mean Hitting Time (GT):        {np.nanmean(gt_hit):.4f}, paths hit: {gt_mask.sum()}/{len(gt_mask)}")
# print(f"Mean Hitting Time (Vanilla):   {np.nanmean(van_hit):.4f}, paths hit: {van_mask.sum()}/{len(van_mask)}")
# print(f"Mean Hitting Time (Penalized): {np.nanmean(pen_hit):.4f}, paths hit: {pen_mask.sum()}/{len(pen_mask)}")

# --- Compute additional summary statistics ---
def summarize(hit_times, mask):
    vals = hit_times[mask]
    return {
        "mean":    np.nanmean(hit_times),
        "median":  np.nanmedian(hit_times),
        "max":     np.nanmax(hit_times),
        "std":     np.nanstd(hit_times),
        "25%":     np.percentile(vals, 25) if len(vals)>0 else np.nan,
        "75%":     np.percentile(vals, 75) if len(vals)>0 else np.nan,
        "n_hits":  mask.sum(),
        "pct_hit": 100*mask.mean()
    }

stats = {
    "GT":        summarize(gt_hit, gt_mask),
    "Vanilla":   summarize(van_hit, van_mask),
    "Penalized": summarize(pen_hit, pen_mask),
}

# print them
for label, s in stats.items():
    print(f"{label:10s}  mean={s['mean']:.4f}, median={s['median']:.4f}, max={s['max']:.4f} "
          f"std={s['std']:.4f}, IQR=[{s['25%']:.4f},{s['75%']:.4f}], "
          f"hits={s['n_hits']}/{len(gt_hit)} ({s['pct_hit']:.1f}%)")

# --- Plotting side by side ---
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


# # left panel: sample paths + boundary
# for path in ambient_paths.numpy():
#     ax1.plot(path[:, 0], path[:, 1], color="green", alpha=0.05, linewidth=0.5)
# for path in ambient_paths_vanilla.detach().numpy():
#     ax1.plot(path[:, 0], path[:, 1], color="red", alpha=0.05, linewidth=0.5)
# for path in ambient_paths_penalized.detach().numpy():
#     ax1.plot(path[:, 0], path[:, 1], color="blue", alpha=0.05, linewidth=0.5)
# # mark the center and the hitting‐radius circle
# circle = plt.Circle((bd_vec[0], bd_vec[1]), hit_radius,
#                     edgecolor="black", fill=False, linewidth=1)
# ax1.add_patch(circle)
# ax1.scatter(bd_vec[0], bd_vec[1], color="black", s=30)
# ax1.set_title("Ensembles of Paths and Hitting Boundary")
# ax1.set_xlabel("x₁");  ax1.set_ylabel("x₂")
# ax1.set_aspect("equal", "box")
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
# plot each ensemble of paths in 3D
for path in ambient_paths.numpy():
    ax1.plot(path[:, 0], path[:, 1], path[:, 2], color="green", alpha=0.05, linewidth=0.5)
for path in ambient_paths_vanilla.detach().numpy():
    ax1.plot(path[:, 0], path[:, 1], path[:, 2], color="red", alpha=0.05, linewidth=0.5)
for path in ambient_paths_penalized.detach().numpy():
    ax1.plot(path[:, 0], path[:, 1], path[:, 2], color="blue", alpha=0.05, linewidth=0.5)

# generate a wireframe sphere for the hitting boundary
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 30)
x = bd_vec[0] + hit_radius * np.outer(np.cos(u), np.sin(v))
y = bd_vec[1] + hit_radius * np.outer(np.sin(u), np.sin(v))
z = bd_vec[2] + hit_radius * np.outer(np.ones_like(u), np.cos(v))
ax1.plot_wireframe(x, y, z, color="black", linewidth=0.5, rstride=2, cstride=2)

# mark the center point
ax1.scatter([bd_vec[0]], [bd_vec[1]], [bd_vec[2]], color="black", s=30)

ax1.set_title("3D Ensembles of Paths and Hitting Boundary")
ax1.set_xlabel("x₁"); ax1.set_ylabel("x₂"); ax1.set_zlabel("x₃")

# enforce equal aspect ratio
ax1.set_box_aspect([1, 1, 1])

# right panel: KDEs of hitting times
def plot_kde(times, mask, label, color):
    data = times[mask]
    if len(data)>1:
        kde = gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 200)
        ax2.plot(xs, kde(xs), label=label, color=color, linewidth=2)
    else:
        ax2.axvline(np.nanmedian(data), linestyle="--", color=color,
                    label=f"{label} (single hit)")

plot_kde(gt_hit,      gt_mask,  "Ground Truth",    "green")
plot_kde(van_hit,     van_mask, "Vanilla AE-SDE",  "red")
plot_kde(pen_hit,     pen_mask, "Penalized AE-SDE","blue")

ax2.set_title("KDE of First Hitting Times")
ax2.set_xlabel("Hitting Time");  ax2.set_ylabel("Density")
ax2.legend()
ax2.grid(linestyle="--", alpha=0.6)
plt.tight_layout()
# plt.savefig("curve_plots/hitting_time_kde_comparison_bellcurve5.png")
plt.show()

# --- Plot Empirical Survival Function ---
fig, ax = plt.subplots(figsize=(8, 5))

def plot_survival(hit_times, hit_mask, label, color):
    times = np.sort(hit_times[hit_mask])
    if len(times) > 0:
        survival = 1 - np.arange(1, len(times) + 1) / len(times)
        ax.step(times, survival, where='post', label=label, color=color, linewidth=2)
    else:
        ax.axhline(1.0, linestyle="--", color=color, label=f"{label} (no hits)")

plot_survival(gt_hit, gt_mask, "Ground Truth", "green")
plot_survival(van_hit, van_mask, "Vanilla AE-SDE", "red")
plot_survival(pen_hit, pen_mask, "Penalized AE-SDE", "blue")

ax.set_xlabel("Hitting Time")
ax.set_ylabel("Survival Probability")
ax.set_title("Empirical Survival Function of Hitting Times")
ax.grid(linestyle="--", alpha=0.6)
ax.legend()
plt.tight_layout()
plt.show()