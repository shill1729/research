# This module trains a AE-SDE pair.
import torch

from ae.toydata import LangevinHarmonicOscillator, SpherePatch, PointCloud, RiemannianManifold
from ae.utils import process_data
from ae.models import AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion, ThreeStageFit, fit_model, LossWeights

# Data generation parameters
n_train = 50
n_test = 10000
batch_size = int(n_train/2)
train_seed = 17
test_seed = 42
intrinsic_dim = 2
extrinsic_dim = 3
embedding_dim = 5

# Newtork arch
hidden_dims = [16]
drift_layers = [16]
diff_layers = [16]
encoder_act = torch.nn.Tanh()
decoder_act = torch.nn.Tanh()
final_act = torch.nn.Tanh()
drift_act = torch.nn.Tanh()
diffusion_act = torch.nn.Tanh()
spectral_normalize = False
weight_normalize = False
fro_normalize = False
fro_max_norm = 5.
tie_weights = True

# Training parameters
lr = 0.001
weight_decay = 0.
epochs_ae = 9000
epochs_diffusion = 200
epochs_drift = 200
print_freq = 1000
use_ambient_cov_mse = False
use_ambient_drift_mse = False
diffeo_reg = 0.01
tangent_space_reg = 0.01
normal_drift_reg = 0.01


# Define the parameterization
sphere_patch = SpherePatch()
# Initialize the manifold
manifold = RiemannianManifold(sphere_patch.local_coords(), sphere_patch.equation())
# Initialize the dynamics
dynamics = LangevinHarmonicOscillator()

# Compute the Martingale problem drift and covariance
local_drift = dynamics.drift(manifold)
local_diffusion = dynamics.diffusion(manifold)

# Initialize point cloud object
point_cloud = PointCloud(manifold, sphere_patch.bounds(), local_drift, local_diffusion, True)
# Generate point cloud plus dynamics as tensor fields:
x, _, mu, cov, local_x = point_cloud.generate(n_train, seed=train_seed)
x, mu, cov, p, n, h = process_data(x, mu, cov, d=intrinsic_dim)
# Now we embed this into a higher dimension randomly and isometrically.
# Generate random Gaussian matrix.
embedding_matrix = torch.randn(size=(embedding_dim, extrinsic_dim))
embedding_matrix, _ = torch.linalg.qr(embedding_matrix)
# Embed every object
x_embed = x@embedding_matrix.T
mu_embed = mu@embedding_matrix.T
cov_embed = embedding_matrix@cov@embedding_matrix.T
p_embed = embedding_matrix@p@embedding_matrix.T
n_embed = embedding_matrix@n@embedding_matrix.T
h_embed = torch.einsum("ij,njk->nik", embedding_matrix, h)
# Check sizes to be sure
print(x_embed.size())
print(mu_embed.size())
print(cov_embed.size())
print(p_embed.size())
print(n_embed.size())
print(h_embed.size())

# Initialize model
autoencoder = AutoEncoder(extrinsic_dim=embedding_dim,
                          intrinsic_dim=intrinsic_dim,
                          hidden_dims=hidden_dims,
                          encoder_act=encoder_act,
                          decoder_act=decoder_act,
                          final_act=final_act,
                          spectral_normalize=spectral_normalize,
                          weight_normalize=weight_normalize,
                          fro_normalize=fro_normalize,
                          fro_max_norm=fro_max_norm,
                          tie_weights=tie_weights
                          )
latent_sde = LatentNeuralSDE(intrinsic_dim=intrinsic_dim,
                             drift_layers=drift_layers,
                             diff_layers=diff_layers,
                             drift_act = drift_act,
                             diffusion_act = diffusion_act,
                             encoder_act = final_act,
                             spectral_normalize=spectral_normalize,
                             weight_normalize=weight_normalize,
                             fro_normalize=fro_normalize,
                             fro_max_norm=fro_max_norm
                             )
aedf = AutoEncoderDiffusion(latent_sde, autoencoder)
fit3 = ThreeStageFit(lr, epochs_ae, epochs_diffusion, epochs_drift, weight_decay, batch_size, print_freq)
# Penalty weights:
weights = LossWeights(diffeomorphism_reg=diffeo_reg,
                      tangent_space_error_weight=tangent_space_reg,
                      tangent_drift_weight=normal_drift_reg)

# Fit the 3-stage model
fit3.three_stage_fit(aedf, weights, x_embed, mu_embed, cov_embed, p_embed, h_embed,
                     ambient_cov_mse=use_ambient_cov_mse,
                     ambient_drift_mse=use_ambient_drift_mse)
# TODO: Assess the performance of our fit.
# TODO: make a wrapper for generating new test data and embedding it:
# Generate point cloud plus dynamics as tensor fields:
x_test, _, mu_test, cov_test, local_x_test = point_cloud.generate(n_test, seed=test_seed)
x_test, mu_test, cov_test, p_test, n_test, h_test = process_data(x_test, mu_test, cov_test, d=intrinsic_dim)
# Now we embed this into a higher dimension randomly and isometrically.
# Embed every object
x_test_embed = x_test@embedding_matrix.T
mu_test_embed = mu_test@embedding_matrix.T
cov_test_embed = embedding_matrix@cov_test@embedding_matrix.T
p_test_embed = embedding_matrix@p_test@embedding_matrix.T
n_test_embed = embedding_matrix@n_test@embedding_matrix.T
h_test_embed = torch.einsum("ij,njk->nik", embedding_matrix, h_test)
# Test reconstruction loss:
z_test_embed = aedf.autoencoder.encoder(x_test_embed)
x_test_embed_hat = aedf.autoencoder.decoder(z_test_embed)
print(torch.mean(torch.linalg.vector_norm(x_test_embed_hat-x_test_embed, ord=2, dim=1)**2))
