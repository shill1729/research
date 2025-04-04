# Train our second-order model for various values of the penalty-weight and plot the boundary error
# as a function of this weight.
from ae.toydata.datagen import ToyData
from ae.toydata.surfaces import *
from ae.toydata.local_dynamics import *
from ae.models.fitting import fit_model
from ae.models.autoencoder import AutoEncoder
from ae.models.local_neural_sdes import AutoEncoderDiffusion, LatentNeuralSDE
from ae.models.losses import TotalLoss, LossWeights
from ae.utils.performance_analysis import compute_test_losses
import torch.nn as nn
import matplotlib.pyplot as plt
lr = 0.0001
epochs = 9000
print_freq = 1000
weight_decay = 0.
n_train = 30
batch_size = 20

ae_layers = [32, 32]
drift_layers = [1]
diffusion_layers = [1]
weights_drift = [0.001, 0.02, 0.05, 0.08, 0.1, 0.25, 0.5]
test_losses = []

# Initialize data
surface = WaveSurface()
dynamics = LangevinHarmonicOscillator()
toy_data = ToyData(surface, dynamics)
data_dict = toy_data.generate_data(n_train, 2)
toy_data.set_point_cloud(0.5)
test_data_dict = toy_data.generate_data(n_train, 2)
loss_weights = LossWeights()
loss_weights.tangent_angle_weight = 0.01
loss_weights.diffeomorphism_reg = 0.01

for i in range(len(weights_drift)):
    loss_weights.tangent_drift_weight = weights_drift[i]
    total_loss = TotalLoss(loss_weights)
    # Initialize model
    latent_sde = LatentNeuralSDE(2, drift_layers, diffusion_layers, nn.Tanh(), nn.Tanh(), nn.Tanh())
    ae = AutoEncoder(3, 2, ae_layers, nn.Tanh(), nn.Tanh())
    aedf = AutoEncoderDiffusion(latent_sde, ae)
    x = data_dict["x"]
    mu = data_dict["mu"]
    cov = data_dict["cov"]
    p = data_dict["p"]
    orthonormal_frame = data_dict["orthonormal_frame"]
    targets = (p, orthonormal_frame, cov, mu)
    # Fit the model
    fit_model(aedf.autoencoder, total_loss, x, targets, lr, epochs, print_freq, weight_decay, batch_size, None)
    # Compute test loss

    x_test = test_data_dict["x"]
    mu_test = test_data_dict["mu"]
    cov_test = test_data_dict["cov"]
    p_test = test_data_dict["p"]
    frame_test = test_data_dict["orthonormal_frame"]
    losses = compute_test_losses(aedf, total_loss, x_test, p_test, frame_test, cov_test, mu_test)
    test_losses.append(losses["reconstruction loss"])


fig = plt.figure()
plt.plot(weights_drift, test_losses)
plt.show()


