# This is the generic prototype from earlier.
from shillml.dynae import *
from shillml.newptcld import PointCloud
from shillml.diffgeo import RiemannianManifold
import matplotlib.pyplot as plt
from data_processing import process_data
import torch
from surfaces import *
from sde_coefficients import *
surface = "paraboloid"
num_pts = 30
num_test = 100
seed = 17
# Encoder region and quiver length
alpha = -1
beta = 1
quiver_length = 0.25
# Regularization: contract and tangent
contractive_weight = 0.0001
tangent_bundle_weight = 1.
normal_bundle_weight = 1.
lr = 0.0001
epochs = [10000] * 3
print_freq = 1000
weight_decay = 0.
# Network structure
extrinsic_dim = 3
intrinsic_dim = 2
h1 = [32]
h2 = [32]
h3 = [32]
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
drift_act = nn.Tanh()
diffusion_act = nn.Tanh()
final_coef_act = None
if seed is not None:
    torch.manual_seed(seed)
u, v = sp.symbols("u v", real=True)
local_coordinates = sp.Matrix([u, v])
if surface == "paraboloid":
    chart = paraboloid
bounds = [(-1, 1), (-1, 1)]
large_bounds = [(-2, 2), (-2, 2)]

manifold = RiemannianManifold(local_coordinates, chart)
drift = manifold.local_bm_drift() + manifold.metric_tensor().inv() * 0.5 * double_well_potential
diffusion = manifold.local_bm_diffusion()

# Initialize the point cloud
point_cloud = PointCloud(manifold, bounds=bounds, local_drift=drift, local_diffusion=diffusion)
x, w, mu, cov, _ = point_cloud.generate(n=num_pts, seed=seed)
x, mu, cov, P = process_data(x, mu, cov, d=intrinsic_dim)
# Define models
ctbae = ContractiveTangentBundleAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
ctbae_loss = CTBAELoss(contractive_weight, tangent_bundle_weight, P)
fit_model(ctbae, ctbae_loss, x, x, lr, epochs[0], print_freq, weight_decay)
[toggle_model(model, False) for model in [ctbae]]
# Define SDEs
latent_sde = LatentNeuralSDE(intrinsic_dim, h2, h3, drift_act, diffusion_act, final_coef_act)
model_diffusion = AutoEncoderDiffusion(latent_sde, ctbae)
diffusion_loss = DiffusionLoss(normal_bundle_weight)
dpi = ctbae.encoder.jacobian_network(x)
encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
fit_model(model_diffusion, diffusion_loss, x, (cov, mu, encoded_cov), lr, epochs[1], print_freq, weight_decay)
# Fit drift
toggle_model(model_diffusion.latent_sde.diffusion_net, False)
model_drift = AutoEncoderDrift(latent_sde, ctbae)
drift_loss = DriftMSELoss()
fit_model(model_drift, drift_loss, x, mu, lr, epochs[2], print_freq, weight_decay)


def compute_losses(point_cloud: PointCloud):
    x_test, _, mu_test, cov_test, _ = point_cloud.generate(n=num_test, seed=None)
    x_test, mu_test, cov_test, P_test = process_data(x_test, mu_test, cov_test, d=intrinsic_dim)
    # Compute in-bound test loss
    mse_loss = nn.MSELoss()
    tbloss = TangentBundleLoss()
    contraction = ContractiveRegularization()
    ctbae_loss_test = CTBAELoss(contractive_weight, tangent_bundle_weight, P_test)
    x_test_reconstructed, dpi, P_model = ctbae(x_test)
    mse = mse_loss(x_test, x_test_reconstructed).item()
    tb_loss = tbloss.forward(P_model, P_test).item()
    contraction_value = contraction.forward(dpi).item()
    ctbae_loss_test_value = ctbae_loss_test(ctbae(x_test), x_test).item()
    covariance_loss = CovarianceMSELoss()
    normal_bundle_loss = NormalBundleLoss()
    model_mu_test = model_drift(x_test)
    model_cov, N, q, bbt = model_diffusion.forward(x_test)
    tangent_vector = mu_test - 0.5 * q
    normal_proj_vector = torch.bmm(N, tangent_vector.unsqueeze(2))
    normal_bundle_loss_value = normal_bundle_loss(normal_proj_vector).item()
    cov_mse_loss = covariance_loss.forward(model_cov, cov_test).item()
    total_diffusion_loss = diffusion_loss.forward(model_diffusion(x_test), (cov_test, mu_test, None)).item()
    drift_mse = drift_loss.forward(model_mu_test, mu_test).item()
    print("Test Reconstruction Error = " + str(mse))
    print("Test Tangent Bundle Error = " + str(tb_loss))
    print("Test Contraction Error = " + str(contractive_weight * contraction_value))
    print("Test Total CTBAE Error = " + str(ctbae_loss_test_value))
    print("\nTest Ambient Covariance Fro-sq Error = " + str(cov_mse_loss))
    print("Test Drift-Curvature Error = " + str(normal_bundle_loss_value))
    print("Test Total diffusion Error = " + str(total_diffusion_loss))
    print("Total drift Error = " + str(drift_mse))
    return x_test, mu_test, model_mu_test


# Test set within bounds:
print("Test loss within training bounds===========================")
compute_losses(point_cloud)
point_cloud = PointCloud(manifold, bounds=large_bounds, local_drift=drift, local_diffusion=diffusion)
print("\nTest loss outside of training bounds=====================")
x_test, mu_test, model_mu_test = compute_losses(point_cloud)
x = x.detach()
x_test = x_test.detach()
mu = mu.detach()
model_mu_test = model_mu_test.detach()

# Point clouds with drifts vector fields
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.scatter3D(x_test[:, 0], x_test[:, 1], x_test[:, 2])
ctbae.plot_surface(alpha, beta, 30, ax, "True Drift Field")
ax.quiver3D(x_test[:, 0], x_test[:, 1], x_test[:, 2],
            mu_test[:, 0], mu_test[:, 1], mu_test[:, 2],
            length=quiver_length, normalize=True, color="red")

ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.scatter3D(x_test[:, 0], x_test[:, 1], x_test[:, 2])
ctbae.plot_surface(alpha, beta, 30, ax, "Model Drift Field")
ax.quiver3D(x_test[:, 0], x_test[:, 1], x_test[:, 2],
            model_mu_test[:, 0], model_mu_test[:, 1], model_mu_test[:, 2],
            length=quiver_length, normalize=True, color="red")
plt.show()


