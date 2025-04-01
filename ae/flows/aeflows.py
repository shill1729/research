import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from ae.flows.flows import flow, generate_brownian_motion

 ##############################
# # Flow (SDE) parameters:
# ##############################
alpha = 0.01      # drift (relaxation) rate in normal direction
sigma = 0.01       # noise strength (in normal direction)
tn = 1.  # total flow time
ntime = 9000  # number of steps for the flow
seed = None
N_train = 30
N_test = 100
num_epochs = 5000
# ---------------------------
# 1. Import Flow and Brownian Motion utilities
# ---------------------------


# ---------------------------
# 2. Define the Manifold and Flow Fields
# ---------------------------
# Our manifold: points (x, x^2) in ℝ³.
def curve(x):
    return np.sin(x)


# Generate training and test data (points in ℝ^2)

x_train = np.linspace(-3.2, 3.2, N_train)
train_points = np.stack([x_train, curve(x_train)], axis=1)


x_test = np.linspace(-4, 4, N_test)
test_points = np.stack([x_test, curve(x_test)], axis=1)

# Choose a basepoint on the training data (e.g. the middle point)
base_index = N_train // 2
basepoint_np = train_points[base_index]  # shape (3,)

# Compute the tangent vector at the basepoint.
# For our curve, derivative f'(x)=2x. We define tangent as [1, 2*x, 0] (normalized).
x0 = basepoint_np[0]
tangent_vec_np = np.array([1.0, np.cos(x0)])
tangent_vec_np = tangent_vec_np / np.linalg.norm(tangent_vec_np)


# Define the drift and diffusion functions.
# They "push" the point toward the tangent line at the basepoint.
def drift(x):
    """
    Compute drift at point x.
    We shift x by basepoint_np and remove the tangent component.
    """
    y = x - basepoint_np
    proj = np.dot(y, tangent_vec_np) * tangent_vec_np
    # Push the normal component to zero.
    return -alpha * (y - proj)


def diffusion(x):
    """
    Return a 3x3 matrix that applies noise only in the normal space.
    """
    P_tangent = np.outer(tangent_vec_np, tangent_vec_np)
    return sigma * (np.eye(2) - P_tangent)


# ---------------------------
# 3. Preconditioning via the Flow (Forward and Reverse)
# ---------------------------


# Generate a Brownian motion path in ℝ³
W = generate_brownian_motion(tn, ntime, d=2, seed=seed)

# Compute forward flow on training data
paths_train = flow(train_points, tn, W, drift, diffusion, num_steps=ntime, reverse=False)
x_flow_train = paths_train[-1]  # Final positions after flow (shape: (N_train, 3))

# ---------------------------
# 4. Define the Autoencoder Model
# ---------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim=2, latent_dim=1, hidden_dim=10):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ---------------------------
# 5. Train Vanilla Autoencoder
# ---------------------------
# Convert the original training data to a PyTorch tensor
x_train_tensor = torch.tensor(train_points, dtype=torch.float)

# Initialize vanilla autoencoder
vanilla_model = Autoencoder().to('cpu')
criterion = nn.MSELoss()
optimizer_vanilla = optim.Adam(vanilla_model.parameters(), lr=1e-3)

print("Training Vanilla Autoencoder...")
for epoch in range(num_epochs):
    optimizer_vanilla.zero_grad()
    output = vanilla_model(x_train_tensor)
    loss = criterion(output, x_train_tensor)
    loss.backward()
    optimizer_vanilla.step()
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

# ---------------------------
# 6. Train Flow-Preconditioned Autoencoder
# ---------------------------
# Convert the flowed training data to a PyTorch tensor.
x_train_flow_tensor = torch.tensor(x_flow_train, dtype=torch.float)

# Initialize flow-preconditioned autoencoder
flow_model = Autoencoder().to('cpu')
optimizer_flow = optim.Adam(flow_model.parameters(), lr=1e-3)

print("\nTraining Flow-Preconditioned Autoencoder...")
for epoch in range(num_epochs):
    optimizer_flow.zero_grad()
    output = flow_model(x_train_flow_tensor)
    loss = criterion(output, x_train_flow_tensor)
    loss.backward()
    optimizer_flow.step()
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

# ---------------------------
# 7. Testing: Vanilla Autoencoder
# ---------------------------
# Test vanilla autoencoder on test data
x_test_tensor = torch.tensor(test_points, dtype=torch.float)
with torch.no_grad():
    rec_test_tensor = vanilla_model(x_test_tensor)
rec_test = rec_test_tensor.cpu().numpy()

# Calculate reconstruction error for vanilla autoencoder
vanilla_error = np.mean(np.sum((test_points - rec_test)**2, axis=1))
print(f"\nVanilla Autoencoder Test Reconstruction Error: {vanilla_error:.6f}")

# ---------------------------
# 8. Testing: Flow, Autoencode, and Reverse Flow (Deflow)
# ---------------------------
# Forward flow on test data.
paths_test = flow(test_points, tn, W, drift, diffusion, num_steps=ntime, reverse=False)
x_flow_test = paths_test[-1]  # (N_test, 3)

# Autoencoder reconstruction in the flowed space.
x_flow_test_tensor = torch.tensor(x_flow_test, dtype=torch.float)
with torch.no_grad():
    rec_flow_test_tensor = flow_model(x_flow_test_tensor)
rec_flow_test = rec_flow_test_tensor.cpu().numpy()

# Calculate reconstruction error in flow space
flow_space_error = np.mean(np.sum((x_flow_test - rec_flow_test)**2, axis=1))
print(f"Flow-Preconditioned Autoencoder Flow-Space Reconstruction Error: {flow_space_error:.6f}")

# Reverse the flow: use the same Brownian path but in reverse order.
paths_reverse = flow(rec_flow_test, tn, W[::-1, :], drift, diffusion, num_steps=ntime, reverse=True)
x_reflow_test = paths_reverse[-1]  # reconstructed points in original coordinates

# Calculate reconstruction error for flow-preconditioned autoencoder in original space
flow_error = np.mean(np.sum((test_points - x_reflow_test)**2, axis=1))
print(f"Flow-Preconditioned Autoencoder Original-Space Reconstruction Error: {flow_error:.6f}")

# ---------------------------
# 9. Plotting the Results
# ---------------------------
plt.figure(figsize=(12, 8))

# Plot the original curve
plt.plot(test_points[:, 0], test_points[:, 1], 'k--', label='Original Curve', linewidth=1.5)
plt.scatter(test_points[:, 0], test_points[:, 1], color='black', s=20, label='Test Points')

# Plot the vanilla autoencoder reconstruction
plt.scatter(rec_test[:, 0], rec_test[:, 1], color='blue', s=20, alpha=0.7,
            label='Vanilla Autoencoder Reconstruction')

# Plot the flow-preconditioned autoencoder reconstruction
plt.scatter(x_reflow_test[:, 0], x_reflow_test[:, 1], color='red', s=20, alpha=0.7,
            label='Flow-Preconditioned Reconstruction')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Autoencoder Reconstructions')
plt.legend()
plt.grid(True)

# Add text with reconstruction errors
plt.text(0.02, 0.02,
         f"Vanilla Error: {vanilla_error:.6f}\nFlow Error: {flow_error:.6f}",
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# # Additional visualization: flowed space
# plt.figure(figsize=(10, 6))
# plt.scatter(x_flow_test[:, 0], x_flow_test[:, 1], color='black', s=20,
#             label='Flowed Test Points')
# plt.scatter(rec_flow_test[:, 0], rec_flow_test[:, 1], color='red', s=20,
#             label='Reconstructed in Flow Space')
# plt.xlabel('x (flow space)')
# plt.ylabel('y (flow space)')
# plt.title('Reconstruction in Flow Space')
# plt.legend()
# plt.grid(True)
# plt.show()