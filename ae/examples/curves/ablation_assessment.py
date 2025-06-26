import importlib
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from ae.models import AutoEncoder
from ae.models.sdes_latent import ambient_quadratic_variation_drift
from ae.toydata import RiemannianManifold, PointCloud
from ae.utils import process_data

n_test = 10000
# Resolution for comparing true curve to model curve
# Grid for the box [a,b]^2 for L^2 error of coefficients in ambient space
num_grid = 100
# Epsilon for boundary extension
epsilon = 1


# =======================
# 1. Load Configuration
# =======================
save_dir = "saved_models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(save_dir, "config.json"), "r") as f:
    config = json.load(f)


train_seed = config["train_seed"]
test_seed = config["test_seed"]
n_train = config["n_train"]
batch_size = config["batch_size"]
intrinsic_dim = config["intrinsic_dim"]
extrinsic_dim = config["extrinsic_dim"]
hidden_dims = config["hidden_dims"]
lr = config["lr"]
weight_decay = config["weight_decay"]
epochs_ae = config["epochs_ae"]
print_freq = config["print_freq"]
diffeo_weight = config["diffeo_weight"]
first_order_weight = config["first_order_weight"]
second_order_weight = config["second_order_weight"]
manifold_name = config["manifold"]
dynamics_name = config["dynamics"]

# =======================
# 2. Dynamically Load Manifold and Dynamics
# =======================
curves_mod = importlib.import_module("ae.toydata.curves")
ManifoldClass = getattr(curves_mod, manifold_name)
curve = ManifoldClass()
large_bounds = curve.bounds()
large_bounds = [(large_bounds[0][0]-epsilon, large_bounds[0][1]+epsilon)]

dynamics_mod = importlib.import_module("ae.toydata.local_dynamics")
DynamicsClass = getattr(dynamics_mod, dynamics_name)
dynamics = DynamicsClass()

# Instantiate a manifold object
manifold = RiemannianManifold(curve.local_coords(), curve.equation())
local_drift = dynamics.drift(manifold)
local_diffusion = dynamics.diffusion(manifold)

# Pass it to a point cloud object for test data.
point_cloud_interp = PointCloud(manifold, curve.bounds(), local_drift, local_diffusion, True)
point_cloud_extrap = PointCloud(manifold, large_bounds, local_drift, local_diffusion, True)
print(f"Loaded manifold: {manifold_name}, dynamics: {dynamics_name}.")

# =======================
# 3. Load Neural Network Models Iteratively
# =======================
model_names = ["vanilla", "diffeo", "first_order", "second_order"]
loaded_models = {}


for name in model_names:
    model_path = os.path.join(save_dir, f"ae_{name}.pth")
    ae = AutoEncoder(
        extrinsic_dim=extrinsic_dim,
        intrinsic_dim=intrinsic_dim,
        hidden_dims=hidden_dims,
        encoder_act=torch.nn.Tanh(),
        decoder_act=torch.nn.Tanh()
    )
    ae.load_state_dict(torch.load(model_path, map_location=device))
    ae.eval()
    loaded_models[name] = ae

print("All models loaded successfully. Ready for evaluation.")
# =======================
# 4. Generate Test data for assessing interpolation and extrapolation performance
# =======================
x, _, mu, cov, local_x = point_cloud_interp.generate(n=n_test, seed=test_seed)
x, mu, cov, p, n, h = process_data(x, mu, cov, d=intrinsic_dim)
curve_grid, _ = point_cloud_interp.get_curve(num_grid)
curve_grid = torch.tensor(curve_grid, dtype=torch.float32).detach()
# TODO: add extrapolation test data


#=====================================================================================================================
# 5. Compute losses over test data in interpolation region
#=====================================================================================================================
# TODO: 1. interpolation error analysis on a test sample in training region
reconstructions = {}
reconstruction_losses = {}
tangent_losses = {}
curvature_losses = {}
mininum_smallest_decoder_sv = {}
maximum_largest_decoder_sv = {}
mininum_smallest_encoder_sv = {}
maximum_largest_encoder_sv = {}
penrose_errors = {}
diffeo_errors = {}

# with torch.no_grad():
for name, ae in loaded_models.items():
    z = ae.encoder(x)
    x_recon = ae.decoder(z)
    reconstructions[name] = x_recon
    reconstruction_losses[name] = torch.mean(torch.linalg.vector_norm(x_recon-x, ord=2, dim=1) ** 2, dim=0).detach().numpy()

    # First order loss:
    p_model = ae.neural_orthogonal_projection(z)
    tangent_loss = torch.mean(torch.linalg.matrix_norm(p_model-p, ord="fro")**2, dim=0)
    tangent_losses[name] = tangent_loss.detach().numpy()

    # Second order loss:
    dpi = ae.encoder_jacobian(x)
    decoder_hessian = ae.decoder_hessian(z)
    latent_covariance = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
    model_tangent_drift = mu-0.5*ambient_quadratic_variation_drift(latent_covariance, decoder_hessian)
    normal_component = model_tangent_drift-torch.bmm(p, model_tangent_drift.unsqueeze(-1)).squeeze(-1)
    curvature_loss = torch.linalg.vector_norm(normal_component, ord=2, dim=1)**2
    curvature_losses[name] = torch.mean(curvature_loss, dim=0).detach().numpy()

    # Minimum smallest decoder SV:
    dphi = ae.decoder_jacobian(z)
    smallest_decoder_sv = torch.linalg.matrix_norm(dphi, ord=-2)
    mininum_smallest_decoder_sv[name] = torch.min(smallest_decoder_sv).detach().numpy()
    # Maximum largest decoder SV
    largest_decoder_sv = torch.linalg.matrix_norm(dphi, ord=2)
    maximum_largest_decoder_sv[name] = torch.max(largest_decoder_sv).detach().numpy()

    # Minimum smallest encoder SV:
    smallest_encoder_sv = torch.linalg.matrix_norm(dpi, ord=-2)
    mininum_smallest_encoder_sv[name] = torch.min(smallest_encoder_sv).detach().numpy()
    # Maximum largest encoder SV
    largest_encoder_sv = torch.linalg.matrix_norm(dpi, ord=2)
    maximum_largest_encoder_sv[name] = torch.max(largest_encoder_sv).detach().numpy()

    # Moore-Penroose inverse distance
    g = ae.neural_metric_tensor(z)
    g_inv = torch.linalg.inv(g)
    dphi_inverse = torch.bmm(g_inv, dphi.mT)
    penrose_error = torch.mean(torch.linalg.matrix_norm(dpi-dphi_inverse, ord="fro")**2).detach().numpy()
    penrose_errors[name] = penrose_error

    # Diffeomorphism error
    jacob_prod = torch.bmm(dpi, dphi)
    n, d = dpi.shape[0], dpi.shape[1]
    eye_d = torch.eye(d, device=dpi.device, dtype=dpi.dtype).expand(n, -1, -1)
    diffeo_error = torch.mean(torch.linalg.matrix_norm(jacob_prod-eye_d, ord="fro")**2).detach().numpy()
    diffeo_errors[name] = diffeo_error
interp_losses = pd.DataFrame(data=[reconstruction_losses,
                                   tangent_losses,
                                   curvature_losses,
                                   mininum_smallest_decoder_sv,
                                   maximum_largest_decoder_sv,
                                   mininum_smallest_encoder_sv,
                                   maximum_largest_encoder_sv,
                                   penrose_errors,
                                   diffeo_errors],
                             index=["Reconstruction",
                                    "Tangent penalty",
                                    "Ito penalty",
                                    "Min Smallest Decoder SV",
                                    "Max Largest Decoder SV",
                                    "Min Smallest Encoder SV",
                                    "Max Largest Encoder SV",
                                    "Moore-Penrose error",
                                    "Diffeomorphism Error"])
print("Interpolation losses (new test data sampled from training region)")
pd.set_option('display.max_columns', None)
interp_losses = interp_losses.astype(float)
print(interp_losses.round(6))

# # =======================
# # TODO: what to do with this block? It's just plotting the reconstructions over a grid.
# # =======================
with torch.no_grad():
    manifold_decodings = {
        name: model.decoder(model.encoder(curve_grid))
        for name, model in loaded_models.items()
    }

# Plotting example: compare manifold decodings
plt.figure(figsize=(10, 6))
plt.plot(curve_grid[:, 0], curve_grid[:, 1], label="GT", color="black")
for name, decoded in manifold_decodings.items():
    decoded = decoded.numpy()
    plt.plot(decoded[:, 0], decoded[:, 1], label=name)
plt.title("Autoencoder Manifold Reconstructions")
plt.legend()
plt.axis('equal')
plt.show()


# TODO: 2. extrapolation error analysis on test sample in boundary thickening.

