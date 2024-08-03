# Experiment to test where contractive/tangent bundle AE beats vanilla AE
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# from shillml.autoencoders import *
from shillml.losses import fit_model
from shillml.diffgeo import RiemannianManifold
from shillml.newptcld import PointCloud
from shillml.scorebased import ScoreBasedDiffusion, ScoreBasedMatchingDiffusionLoss, mala_sampling

from data_processing import process_data
from sde_coefficients import *
from surfaces import *
from torch.utils.data import TensorDataset, DataLoader

surface = "paraboloid"
num_pts = 30
num_test = 100
seed = 17
bd_epsilon = 0.25
# Encoder region and quiver length
alpha = -1
beta = 1
cov_weight = 0.
stationary_weight = 0.
# Regularization: contract and tangent
lr = 0.0001
epochs = 1000
print_freq = 1000
weight_decay = 0.
# Network structure
extrinsic_dim = 3
intrinsic_dim = 2
h1 = [2]
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
if seed is not None:
    torch.manual_seed(seed)
u, v = sp.symbols("u v", real=True)
local_coordinates = sp.Matrix([u, v])
# Assume that 'surface' is the user's input
if surface in surface_bounds:
    # Set the chart dynamically
    chart = globals()[surface]
    # Set the bounds from the dictionary
    bounds = surface_bounds[surface]
    # Set large_bounds (if needed)
    large_bounds = [(b[0] - bd_epsilon, b[1] + bd_epsilon) for b in bounds]
else:
    raise ValueError("Invalid surface")

# Initialize the manifold and choose the dynamics
manifold = RiemannianManifold(local_coordinates, chart)
true_drift = manifold.local_bm_drift()
true_diffusion = manifold.local_bm_diffusion()

# Initialize the point cloud
point_cloud = PointCloud(manifold, bounds=bounds, local_drift=true_drift, local_diffusion=true_diffusion)
x, w, mu, cov, _ = point_cloud.generate(n=num_pts, seed=seed)
x, mu, cov, P = process_data(x, mu, cov, d=intrinsic_dim)

# Instantiate a score based diffusion model
sb = ScoreBasedDiffusion(2, h1)
sb_loss = ScoreBasedMatchingDiffusionLoss(cov_weight, stationary_weight)

# Fit model
fit_model(sb, sb_loss, x, (mu, cov), epochs=epochs, print_freq=100)
samples = mala_sampling(sb, 10, 0.01, num_steps=5000, burn_in=2500, thinning=50)
samples = samples.detach()
fig = plt.figure()
ax = plt.subplot(111, projection="3d")
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])
plt.show()