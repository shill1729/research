import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

from ae.sdes import SDE, SDEtorch
from ae.toydata.curves import *
from ae.toydata.local_dynamics import *
from ae.toydata import RiemannianManifold, PointCloud
from ae.utils import process_data
from ae.models import AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion, fit_model, ThreeStageFit
from ae.models import LossWeights


# Configuration
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
diffeo_weight = 0.
first_order_weight = 0.
second_order_weight = 0.
n_train = 30
tn = 0.1
ntime = 200
npaths = 200

# Activations
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
drift_act = nn.Tanh()
diffusion_act = nn.Tanh()

# Data and models
curve = Parabola()
dynamics = RiemannianBrownianMotion()
manifold = RiemannianManifold(curve.local_coords(), curve.equation())
local_drift = dynamics.drift(manifold)
local_diffusion = dynamics.diffusion(manifold)
point_cloud = PointCloud(manifold, curve.bounds(), local_drift, local_diffusion, True)
x, _, mu, cov, local_x = point_cloud.generate(n=n_train)
x, mu, cov, p, n, h = process_data(x, mu, cov, d=intrinsic_dim)

ae = AutoEncoder(extrinsic_dim=extrinsic_dim, intrinsic_dim=intrinsic_dim,
                 hidden_dims=hidden_dims, encoder_act=encoder_act, decoder_act=decoder_act)
latent_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diff_layers, drift_act, diffusion_act)
aedf = AutoEncoderDiffusion(latent_sde, ae)
weights = LossWeights(tangent_angle_weight=first_order_weight,
                      tangent_drift_weight=second_order_weight,
                      diffeomorphism_reg=diffeo_weight)

fit3 = ThreeStageFit(lr, epochs_ae, epochs_diffusion, epochs_drift, weight_decay, batch_size, print_freq)
fit3.three_stage_fit(aedf, weights, x, mu, cov, p, h)



# Path generation
x0 = x[0, :]
z0_true = local_x[0, :]

time_grid = np.linspace(0, tn, ntime + 1)
ambient_paths = np.zeros((npaths, ntime + 1, extrinsic_dim))
latent_paths = point_cloud.latent_sde.sample_ensemble(z0_true, tn, ntime, npaths)
for j in range(npaths):
    for i in range(ntime + 1):
        ambient_paths[j, i, :] = np.squeeze(point_cloud.np_phi(latent_paths[j, i, :]))
ambient_paths = torch.from_numpy(ambient_paths)

x_encoded = aedf.autoencoder.encoder.forward(x)
z0 = x_encoded[0, :].detach()
model_local_paths = aedf.latent_sde.sample_paths(z0, tn, ntime, npaths)
model_ambient_paths = aedf.lift_sample_paths(model_local_paths)

# Hitting time estimation
def estimate_hitting_times(paths, boundary_func, time_grid):
    npaths, ntime, _ = paths.shape
    hit_times = np.full(npaths, np.nan)
    for i in range(npaths):
        for t in range(ntime):
            if boundary_func(paths[i, t, :]):
                hit_times[i] = time_grid[t]
                break
    hit_mask = ~np.isnan(hit_times)
    return hit_times, hit_mask

# Define boundary: vertical line at x = 0.02
def boundary_func(x):
    return x[0] > 1.

# Compute
gt_hit_times, gt_mask = estimate_hitting_times(ambient_paths.numpy(), boundary_func, time_grid)
ae_hit_times, ae_mask = estimate_hitting_times(model_ambient_paths.detach().numpy(), boundary_func, time_grid)

print(f"Mean Hitting Time (GT): {np.nanmean(gt_hit_times):.4f}, paths hit: {gt_mask.sum()}/{len(gt_mask)}")
print(f"Mean Hitting Time (AE-SDE): {np.nanmean(ae_hit_times):.4f}, paths hit: {ae_mask.sum()}/{len(ae_mask)}")

# Plotting
def plot_hitting_time_histogram(hit_times, hit_mask, label, color):
    plt.hist(hit_times[hit_mask], bins=10, alpha=0.6, label=label, color=color)

plt.figure(figsize=(10, 6))
plot_hitting_time_histogram(gt_hit_times, gt_mask, "Ground Truth", "green")
plot_hitting_time_histogram(ae_hit_times, ae_mask, "AE-SDE", "red")
plt.xlabel("Hitting Time")
plt.ylabel("Number of Paths")
plt.legend()
plt.title("Distribution of First Hitting Times to Boundary")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("curve_plots/hitting_time_distribution_parabola2.png")
plt.show()
