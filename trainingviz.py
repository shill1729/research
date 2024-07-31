# Experiment to test where contractive/tangent bundle AE beats vanilla AE
import os

import matplotlib.pyplot as plt

from shillml.diffgeo import RiemannianManifold
from shillml.autoencoders import *
from shillml.losses import CAELoss, TBAELoss, CTBAELoss, CurvatureCTBAELoss, fit_model
from shillml.newptcld import PointCloud

from data_processing import process_data
from sde_coefficients import *
from surfaces import *
from shillml.trainviz import PlotSurfaceCallback, VectorFieldCallback


surface = "paraboloid"
num_pts = 30
num_test = 500
seed = 17
bd_epsilon = 0.5
# Encoder region and quiver length
alpha = -1
beta = 1
# Regularization: contract and tangent
contractive_weight = 0.001
tangent_bundle_weight = 0.001
curvature_weight = 0.001
lr = 0.0001
epochs = 10000
print_freq = 1000
weight_decay = 0.01
# Network structure
extrinsic_dim = 3
intrinsic_dim = 2
h1 = [16]
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
true_drift = manifold.local_bm_drift() + manifold.metric_tensor().inv() * double_well_potential
true_diffusion = manifold.local_bm_diffusion()

# Initialize the point cloud
point_cloud = PointCloud(manifold, bounds=bounds, local_drift=true_drift, local_diffusion=true_diffusion)
x, w, mu, cov, _ = point_cloud.generate(n=num_pts, seed=seed)
x, mu, cov, P = process_data(x, mu, cov, d=intrinsic_dim)

# Interpolation testing point cloud
x_interp, _, mu_interp, cov_interp, _ = point_cloud.generate(n=num_test, seed=None)
x_interp, mu_interp, cov_interp, P_interp = process_data(x_interp, mu_interp, cov_interp, d=intrinsic_dim)

# Extrapolation testing point cloud
point_cloud = PointCloud(manifold, bounds=large_bounds, local_drift=true_drift, local_diffusion=true_diffusion)
x_extrap, _, mu_extrap, cov_extrap, _ = point_cloud.generate(n=num_test, seed=None)
x_extrap, mu_extrap, cov_extrap, P_extrap = process_data(x_extrap, mu_extrap, cov_extrap, d=intrinsic_dim)

# Define models
ae = AutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
cae = ContractiveAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
tbae = TangentBundleAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
ctbae = ContractiveTangentBundleAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
curve_ae = CurvatureCTBAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
# Define the loss functions for each auto encoder
mse_loss = nn.MSELoss()
cae_loss = CAELoss(contractive_weight)
tbae_loss = TBAELoss(tangent_bundle_weight, P)
ctbae_loss = CTBAELoss(contractive_weight, tangent_bundle_weight, P)
curve_loss = CurvatureCTBAELoss(contractive_weight, tangent_bundle_weight, curvature_weight, P)
# Run the program.
if __name__ == "__main__":
    plot_callback = PlotSurfaceCallback(num_frames=100, a=-1, b=1, grid_size=50, save_dir="./plots/")
    normal_drift_callback = VectorFieldCallback(100, x, mu, cov, P, save_dir="./plots/")
    fit_model(curve_ae, curve_loss, x, (x, mu, cov), lr, epochs, print_freq, weight_decay, [normal_drift_callback])
