import json
import torch
import os


from ae.models import AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion
from ae.toydata import RiemannianManifold, PointCloud
from ae.toydata.local_dynamics import *
from ae.toydata.surfaces import *
from ae.toydata.curves import *
from ae.utils import process_data
import torch.nn as nn

import matplotlib.pyplot as plt

n_test = 100

# === Load paths ===
save_dir = "saved_models/"
model_path = os.path.join(save_dir, "aedf_model.pt")
config_path = os.path.join(save_dir, "config.json")

# === Load config ===
with open(config_path, "r") as f:
    config = json.load(f)

# === Instantiate surface and dynamics from class names ===
curve_class = globals()[config["manifold_class"]]
dynamics_class = globals()[config["dynamics_class"]]
curve = curve_class()
dynamics = dynamics_class()

# === Build manifold and local dynamics ===
manifold = RiemannianManifold(curve.local_coords(), curve.equation())
local_drift = dynamics.drift(manifold)
local_diffusion = dynamics.diffusion(manifold)

large_bounds = config["point_cloud_bounds"]
large_bounds = [(bounds[0]-0.1, bounds[1]+0.1) for bounds in large_bounds]
# === Generate new test data ===
point_cloud = PointCloud(manifold, large_bounds, local_drift, local_diffusion, True)
test_x, _, test_mu, test_cov, local_test_x = point_cloud.generate(n=n_test, seed=None)
test_x, test_mu, test_cov, p, n, h = process_data(test_x, test_mu, test_cov, d=config["intrinsic_dim"])

# === Reconstruct model architecture ===
act_map = {
    "Tanh": nn.Tanh(),
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(),
    "Sigmoid": nn.Sigmoid(),
    "ELU": nn.ELU()
}
encoder_act = act_map[config["activation_functions"]["encoder"]]
decoder_act = act_map[config["activation_functions"]["decoder"]]
drift_act = act_map[config["activation_functions"]["drift"]]
diffusion_act = act_map[config["activation_functions"]["diffusion"]]

ae = AutoEncoder(extrinsic_dim=config["extrinsic_dim"],
                 intrinsic_dim=config["intrinsic_dim"],
                 hidden_dims=config["hidden_dims"],
                 encoder_act=encoder_act,
                 decoder_act=decoder_act)

latent_sde = LatentNeuralSDE(config["intrinsic_dim"],
                              config["drift_layers"],
                              config["diff_layers"],
                              drift_act,
                              diffusion_act)

# === Compose AE-SDE model ===
aedf = AutoEncoderDiffusion(latent_sde, ae)

# === Load model weights ===
state = torch.load(model_path)
aedf.autoencoder.load_state_dict(state['ae_state_dict'])
aedf.latent_sde.load_state_dict(state['sde_state_dict'])
aedf.eval()

# === Output ===
print("Model and test data loaded.")

model_cov = aedf.compute_ambient_covariance(test_x)
model_local_cov = aedf.compute_local_covariance(test_x)

dpi = aedf.autoencoder.encoder_jacobian(test_x)

true_local_cov = torch.bmm(dpi, torch.bmm(test_cov, dpi.mT))

local_covariance_error = torch.linalg.matrix_norm(model_local_cov-true_local_cov).detach()
ambient_covariance_error = torch.linalg.matrix_norm(model_cov-test_cov).detach()
z = aedf.autoencoder.encoder(test_x)
dphi = aedf.autoencoder.decoder_jacobian(z)
identity_matrix = torch.eye(config["intrinsic_dim"])
batch_identity = identity_matrix.repeat(n_test, 1, 1)
diffeo_error = torch.linalg.matrix_norm(torch.bmm(dpi, dphi)-batch_identity).detach()


g = aedf.autoencoder.neural_metric_tensor(z)
trg = torch.einsum("bii -> b", g).detach()

q = torch.bmm(dphi, dpi)
qsq = torch.bmm(q, q)
print("Q^2 -Q Fro error"+str(torch.mean(torch.linalg.matrix_norm(q-qsq))))

fig = plt.figure()
ax = plt.subplot(221)
ax.plot(ambient_covariance_error, label="Ambient Cov Error")
ax.plot(trg*local_covariance_error, label="Upper bound")
ax.legend()
ax = plt.subplot(222)
ax.plot(diffeo_error, label="Diffeo error")
ax.legend()
ax = plt.subplot(223)
ax.plot(trg, label="Trace of metric tensor")
ax.legend()
ax = plt.subplot(224)
ax.plot(local_covariance_error, label="localcov error")
ax.legend()
plt.show()