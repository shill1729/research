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
train_seed = None
test_seed = None
n_train = 30
n_test = 2000
batch_size = 15
intrinsic_dim = 1
extrinsic_dim = 2
epsilon = 0.5
hidden_dims = [16]
drift_layers = [16]
diff_layers = [16]
# Training parameters
lr = 0.001
epochs_ae = 6000
epochs_diffusion = 6000
epochs_drift = 6000
weight_decay = 0.001
print_freq = 1000
first_order_weight = 0.001
second_order_weight = 0.01
diffeo_weight = 0.1
# Sample path input
tn = 0.25
ntime = 1000
npaths = 100


encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
drift_act = nn.Tanh()
diffusion_act = nn.Tanh()

# Pick the manifold and dynamics
curve = Parabola(2.)
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
                 encoder_act = encoder_act,
                 decoder_act=decoder_act)
# Declare an instance of the local coordinate SDE networks
latent_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diff_layers, drift_act, diffusion_act)
# Finally initialize the AE-SDE object
aedf = AutoEncoderDiffusion(latent_sde, ae)
weights = LossWeights(tangent_angle_weight=first_order_weight,
                      tangent_drift_weight=second_order_weight,
                      diffeomorphism_reg=diffeo_weight)
# Train the model:
fit3 = ThreeStageFit(lr, epochs_ae, epochs_diffusion, epochs_drift, weight_decay, batch_size, print_freq)
fit3.three_stage_fit(aedf, weights, x, mu, cov, p, h)

# Instantiate the Euclidean ambient SDE model
ambient_drift_model = AmbientDriftNetwork(extrinsic_dim, extrinsic_dim, drift_layers, drift_act)
ambient_diff_model = AmbientDiffusionNetwork(extrinsic_dim, extrinsic_dim, diff_layers, diffusion_act)
ambient_drift_loss = AmbientDriftLoss()
ambient_diff_loss = AmbientDiffusionLoss()
# Train the ambient model
print("Training ambient diffusion model")
fit_model(ambient_diff_model, ambient_diff_loss, x, cov, lr, epochs_diffusion, print_freq, weight_decay, batch_size)
print("\nTraining ambient diffusion model")
fit_model(ambient_drift_model, ambient_drift_loss, x, mu, lr, epochs_drift, print_freq, weight_decay, batch_size)
ambient_sde = SDE(ambient_drift_model.drift_numpy, ambient_diff_model.diffusion_numpy)
#============================================================================
# Model performance
#============================================================================
# Encode the train data
x_encoded = aedf.autoencoder.encoder.forward(x)
x_recon = aedf.autoencoder.decoder.forward(x_encoded)

# generate test data:
train_bounds = curve.bounds()[0]
large_bounds = [(train_bounds[0] - epsilon, train_bounds[1] + epsilon)]
point_cloud = PointCloud(manifold, large_bounds, local_drift, local_diffusion, True)
x_test, _, mu_test, cov_test, local_x_test = point_cloud.generate(n=n_test, seed=test_seed)
x_test, mu_test, cov_test, p_test, n_test, h_test = process_data(x_test, mu_test, cov_test, d=intrinsic_dim)

# 1. Reconstruction loss
x_test_encoded = aedf.autoencoder.encoder.forward(x_test)
x_test_recon = aedf.autoencoder.decoder.forward(x_test_encoded)
test_recon_error = torch.linalg.vector_norm(x_test_recon-x_test, ord=2, dim=1)**2
test_recon_mse = torch.mean(test_recon_error).numpy()
print("Reconstruction MSE on test data = "+str(test_recon_mse))
# 2. Tangent space reconstruction loss :
# TODO: other geometric measures?
p_model_train = aedf.autoencoder.neural_orthogonal_projection(x_encoded).detach()
tangent_space_train_mse = torch.mean(torch.linalg.matrix_norm(p_model_train-p, ord="fro")**2).detach().numpy()
print("Tangent space MSE on train data = "+str(tangent_space_train_mse))
p_model_test = aedf.autoencoder.neural_orthogonal_projection(x_test_encoded).detach()
tangent_space_test_mse = torch.mean(torch.linalg.matrix_norm(p_model_test-p_test, ord="fro")**2).detach().numpy()
print("Tangent space MSE on test data = "+str(tangent_space_test_mse))

# Range data for plotting the model's curve.
# To be sure, we generate a large local coordinate range, map it up to the ambient space
# using the ground-truth parameterization
# then reconstruct it with the model.
# Perhaps we should compare this to directly decoding from the large local coord. range.
local_x_rng = np.linspace(large_bounds[0][0], large_bounds[0][1], 50)
x_rng = np.zeros((50, extrinsic_dim))
for i in range(50):
    x_rng[i, :] = point_cloud.np_phi(local_x_rng[i]).squeeze()
x_rng = torch.tensor(x_rng, dtype=torch.float32)
x_rng_encoded = aedf.autoencoder.encoder.forward(x_rng)
x_rng_decoded = aedf.autoencoder.decoder.forward(x_rng_encoded)

# Plot the test scatter plot against the model curve
fig = plt.figure()
plt.scatter(x_test[:, 0], x_test[:, 1])
plt.plot(x_rng_decoded[:, 0], x_rng_decoded[:, 1], c="red")
plt.show()

#
# TODO: Sample path statistics
# Generate ground truth local and ambient models
# TODO consider implementing phi^{-1} in diff_geo.py to avoid this limitation
x0 = x[0, :].detach().numpy()
d = manifold.local_coordinates.shape[0]
z0_true = x0[:d]
# TODO: currently only hypersurfaces are implemented
ambient_paths = np.zeros((npaths, ntime + 1, d+1))
latent_paths = point_cloud.latent_sde.sample_ensemble(z0_true, tn, ntime, npaths)
for j in range(npaths):
    for i in range(ntime + 1):
        ambient_paths[j, i, :] = np.squeeze(point_cloud.np_phi(*latent_paths[j, i, :]))

# AE-SDE Model paths:
z0 = x_encoded[0, :].detach().numpy()
model_local_paths = aedf.latent_sde.sample_paths(z0, tn, ntime, npaths)
model_ambient_paths = aedf.lift_sample_paths(model_local_paths)

# Euclidean Ambient SDE model
euclidean_model_paths = ambient_sde.sample_ensemble(x0, tn, ntime, npaths)
#======================================================================
# 1. Comparing trajectories
#======================================================================
# Plot the test scatter plot against the model curve
fig = plt.figure()
for j in range(npaths):
    plt.plot(ambient_paths[j, :, 0], ambient_paths[j, :, 1], c="black", alpha=0.5)
    plt.plot(model_ambient_paths[j, :, 0], model_ambient_paths[j, :, 1], c="red", alpha=0.5)
    plt.plot(euclidean_model_paths[j, :, 0], euclidean_model_paths[j, :, 1], c="blue", alpha=0.5)
plt.title("Sample path trajectories")
plt.show()

time_grid = np.linspace(0, tn, ntime+1)
#======================================================================
# 2. Comparing Euclidean distance of means
#======================================================================
true_mean = np.mean(ambient_paths, axis=0)
model_mean = np.mean(model_ambient_paths, axis=0)
euclidean_model_mean = np.mean(euclidean_model_paths, axis=0)
deviation_of_means_ae_sde = np.linalg.vector_norm(model_mean-true_mean, axis=1, ord=2)
deviation_of_means_euclidean_model = np.linalg.vector_norm(euclidean_model_mean-true_mean, axis=1, ord=2)
fig = plt.figure()
plt.plot(time_grid, deviation_of_means_euclidean_model, c="blue", label="Euclidean")
plt.plot(time_grid, deviation_of_means_ae_sde, c="red", label="AE-SDE")
plt.title("Deviation of means $\\|E(X_t)-E(\hat{X}_t)\\|_2$")
plt.legend()
plt.show()

#======================================================================
# 3. Comparing average Euclidean distance of GT from model
#======================================================================
mean_of_deviation_aesde = np.linalg.vector_norm(ambient_paths-model_ambient_paths, axis=2, ord=2).mean(axis=0)
mean_of_deviation_eucl = np.linalg.vector_norm(ambient_paths-euclidean_model_paths, axis=2, ord=2).mean(axis=0)
fig = plt.figure()
plt.plot(time_grid, mean_of_deviation_eucl, c="blue", label="Euclidean")
plt.plot(time_grid, mean_of_deviation_aesde, c="red", label="AE-SDE")
plt.title("Mean of deviation $E\\|X_t-\hat{X}_t\\|_2$")

plt.show()

#======================================================================
# 4. Comparing Feynman-Kac statistics
#======================================================================
sine_paths = np.sin(ambient_paths[:, :, 0]).mean(axis=0)
sine_model_paths_aesde = np.sin(model_ambient_paths[:, :, 0]).mean(axis=0)
sine_model_paths_eucl = np.sin(euclidean_model_paths[:, :, 0]).mean(axis=0)
fk_error_sine_aesde = np.abs(sine_paths-sine_model_paths_aesde)
fk_error_sine_eucl = np.abs(sine_paths-sine_model_paths_eucl)

fig = plt.figure()
plt.plot(time_grid, fk_error_sine_eucl, c="blue", label="Euclidean")
plt.plot(time_grid, fk_error_sine_aesde, c="red", label="AE-SDE")
plt.title("Abs error $|E(\sin(X_t^1))-E(\sin(\hat{X}_t^1))|$")
plt.legend()
plt.show()

#======================================================================
# 4. Comparing Feynman-Kac statistics
#======================================================================
sine_paths = np.sin(ambient_paths[:, :, 1]).mean(axis=0)
sine_model_paths_aesde = np.sin(model_ambient_paths[:, :, 1]).mean(axis=0)
sine_model_paths_eucl = np.sin(euclidean_model_paths[:, :, 1]).mean(axis=0)
fk_error_sine_aesde = np.abs(sine_paths-sine_model_paths_aesde)
fk_error_sine_eucl = np.abs(sine_paths-sine_model_paths_eucl)

fig = plt.figure()
plt.plot(time_grid, fk_error_sine_aesde, c="red", label="AE-SDE")
plt.plot(time_grid, fk_error_sine_eucl, c="blue", label="Euclidean")
plt.title("Abs error $|E(\sin(X_t^2))-E(\sin(\hat{X}_t^2))|$")
plt.legend()
plt.show()

#======================================================================
# 4. Comparing Feynman-Kac statistics
#======================================================================
sine_paths = np.tanh(ambient_paths[:, :, 1]**2).mean(axis=0)
sine_model_paths_aesde = np.tanh(model_ambient_paths[:, :, 1]**2).mean(axis=0)
sine_model_paths_eucl = np.tanh(euclidean_model_paths[:, :, 1]**2).mean(axis=0)
fk_error_sine_aesde = np.abs(sine_paths-sine_model_paths_aesde)
fk_error_sine_eucl = np.abs(sine_paths-sine_model_paths_eucl)

fig = plt.figure()
plt.plot(time_grid, fk_error_sine_aesde, c="red", label="AE-SDE")
plt.plot(time_grid, fk_error_sine_eucl, c="blue", label="Euclidean")
plt.title("Abs error $|E(\\tanh(X_2^2(t)))-E(\\tanh(\hat{X}_2^2(t)))|$")
plt.show()



