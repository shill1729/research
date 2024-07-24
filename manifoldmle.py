import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shillml.dynae import AutoEncoder
import torch.nn as nn


def generate_sphere_samples(n_samples, dim=3):
    """Generate uniform samples from a unit sphere."""
    samples = np.random.randn(n_samples, dim)
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    return samples


def compute_losses(model, x, z, a, b, n):
    """
    Compute both reconstruction loss and neural uniform NLL.

    :param model: AutoEncoder instance
    :param x: input data
    :param z: encoded data
    :param a: lower bound of the box
    :param b: upper bound of the box
    :param n: number of points for normalization
    """
    # Reconstruction loss
    x_recon = model(x)
    recon_loss = torch.nn.MSELoss()(x, x_recon)

    # Neural uniform NLL
    g = model.neural_metric_tensor(z)
    det_g = torch.linalg.det(g)
    log_det_g = torch.log(det_g)
    h = (b - a) / n
    Z = torch.sum(torch.sqrt(det_g)) * (h ** 2)
    nll = -0.5 * log_det_g + torch.log(Z)
    nll_loss = torch.sum(nll)

    return recon_loss, nll_loss


def train_autoencoder(model, data, epochs, lr, a, b, alpha=0.):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        x = torch.tensor(data, dtype=torch.float32)
        z = model.encoder(x)

        recon_loss, nll_loss = compute_losses(model, x, z, a, b, len(data))
        total_loss = alpha * recon_loss + (1 - alpha) * torch.exp(nll_loss)

        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Total Loss: {total_loss.item():.4f}, "
                  f"Recon Loss: {recon_loss.item():.4f}, NLL Loss: {nll_loss.item():.4f}")

    return losses


# Set up parameters
n_samples = 50
extrinsic_dim = 3
intrinsic_dim = 2
hidden_dims = [64, 64]
a, b = -1, 1
epochs = 10000
lr = 0.001

# Generate data
sphere_data = generate_sphere_samples(n_samples, extrinsic_dim)

# Create and train the model
model = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims,
                    nn.Tanh(), nn.Tanh())
losses = train_autoencoder(model, sphere_data, epochs, lr, a, b)

# Plotting the loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.show()

# Visualize the learned manifold
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Plot original sphere
ax1.scatter(*sphere_data.T, c='b', alpha=0.1)
ax1.set_title("Original Sphere")

# Plot learned manifold
model.plot_surface(a, b, 20, ax=ax2, title="Learned Manifold")

plt.tight_layout()
plt.show()

# Encode and decode a few points to check reconstruction
sample_points = torch.tensor(sphere_data[:5], dtype=torch.float32)
encoded = model.encoder(sample_points)
reconstructed = model.decoder(encoded)

print("Sample points:")
print(sample_points)
print("\nReconstructions:")
print(reconstructed)
