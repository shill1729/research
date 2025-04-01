import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# Flow (SDE) parameters
alpha = 1.  # drift (relaxation) rate in normal direction
sigma = 0.01  # noise strength (in normal direction)
tn = 2.0  # total flow time
ntime = 2000  # number of steps for the flow
seed = 42  # for reproducibility
N_train = 30
N_test = 100
num_epochs = 5000
animation_epochs = 10  # Epochs to save for animation (to avoid memory issues)
frames_per_epoch = 1
flow_frames = 10  # Number of frames to use for flow animation


# ---------------------------
# 1. Define Brownian Motion and Flow Functions
# ---------------------------
def generate_brownian_motion(T, N, d=2, seed=None):
    """Generate a d-dimensional Brownian motion path."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    dW = np.sqrt(dt) * np.random.randn(N, d)
    W = np.cumsum(dW, axis=0)
    W = np.insert(W, 0, np.zeros(d), axis=0)  # Add the origin point
    return W


def flow(X0, T, W, drift_fn, diffusion_fn, num_steps=1000, reverse=False):
    """
    Simulate SDEs using the Euler-Maruyama method.
    Args:
        X0: Initial positions (N, d)
        T: Total flow time
        W: Brownian motion, shape (num_steps+1, d)
        drift_fn: Function computing the drift
        diffusion_fn: Function computing the diffusion matrix
        num_steps: Number of time steps
        reverse: Whether to run the flow in reverse
    Returns:
        Paths: Array of shape (num_steps+1, N, d) with all points along the flow
    """
    N, d = X0.shape
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    # Initialize the array to store the paths
    X = np.zeros((num_steps + 1, N, d))
    X[0] = X0

    # Loop through time
    for i in range(num_steps):
        if reverse:
            # For reverse flow, we use the negative drift and read the Brownian increments backwards
            dW = W[-(i + 1)] - W[-(i + 2)]
            for j in range(N):
                drift = -drift_fn(X[i, j])
                diffusion = diffusion_fn(X[i, j])
                X[i + 1, j] = X[i, j] + drift * dt + np.dot(diffusion, dW)
        else:
            # For forward flow
            dW = W[i + 1] - W[i]
            for j in range(N):
                drift = drift_fn(X[i, j])
                diffusion = diffusion_fn(X[i, j])
                X[i + 1, j] = X[i, j] + drift * dt + np.dot(diffusion, dW)

    return X


# ---------------------------
# 2. Define the Manifold and Flow Fields
# ---------------------------
def curve(x):
    return np.sin(x)


# Generate training and test data (points in ℝ^2)
x_train = np.linspace(-3.2, 3.2, N_train)
train_points = np.stack([x_train, curve(x_train)], axis=1)

x_test = np.linspace(-4, 4, N_test)
test_points = np.stack([x_test, curve(x_test)], axis=1)

# Choose a basepoint on the training data
base_index = N_train // 2
basepoint_np = train_points[base_index]

# Compute the tangent vector at the basepoint
x0 = basepoint_np[0]
tangent_vec_np = np.array([1.0, np.cos(x0)])
tangent_vec_np = tangent_vec_np / np.linalg.norm(tangent_vec_np)


# Define the drift and diffusion functions
def drift(x):
    """Compute drift at point x."""
    y = x - basepoint_np
    proj = np.dot(y, tangent_vec_np) * tangent_vec_np
    # Push the normal component to zero
    return -alpha * (y - proj)


def diffusion(x):
    """Return a matrix that applies noise only in the normal space."""
    P_tangent = np.outer(tangent_vec_np, tangent_vec_np)
    return sigma * (np.eye(2) - P_tangent)


# ---------------------------
# 3. Generate Brownian Motion and Compute Flows
# ---------------------------
# Generate a Brownian motion path in ℝ²
np.random.seed(seed)  # Set random seed for reproducibility
W = generate_brownian_motion(tn, ntime, d=2, seed=seed)

# Compute forward flow on training data at selected frames
flow_indices = np.linspace(0, ntime, flow_frames, dtype=int)
paths_train_full = flow(train_points, tn, W, drift, diffusion, num_steps=ntime, reverse=False)
paths_train = paths_train_full[flow_indices]
x_flow_train = paths_train_full[-1]  # Final positions after flow

# Compute forward flow on test data
paths_test_full = flow(test_points, tn, W, drift, diffusion, num_steps=ntime, reverse=False)
paths_test = paths_test_full[flow_indices]
x_flow_test = paths_test_full[-1]  # Final positions after flow


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
# 5. Train both autoencoders and save states for animation
# ---------------------------
# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(train_points, dtype=torch.float)
x_train_flow_tensor = torch.tensor(x_flow_train, dtype=torch.float)
x_test_tensor = torch.tensor(test_points, dtype=torch.float)
x_flow_test_tensor = torch.tensor(x_flow_test, dtype=torch.float)

# Initialize models
vanilla_model = Autoencoder()
flow_model = Autoencoder()
criterion = nn.MSELoss()
optimizer_vanilla = optim.Adam(vanilla_model.parameters(), lr=1e-3)
optimizer_flow = optim.Adam(flow_model.parameters(), lr=1e-3)

# Lists to store reconstruction progress for animation
vanilla_recon_history = []
flow_recon_history = []
deflow_recon_history = []
vanilla_loss_history = []
flow_loss_history = []

# Training loop
print("Training both autoencoders...")
sample_epochs = np.linspace(0, num_epochs - 1, animation_epochs, dtype=int)

for epoch in range(num_epochs):
    # Train vanilla autoencoder
    optimizer_vanilla.zero_grad()
    output_vanilla = vanilla_model(x_train_tensor)
    loss_vanilla = criterion(output_vanilla, x_train_tensor)
    loss_vanilla.backward()
    optimizer_vanilla.step()

    # Train flow-preconditioned autoencoder
    optimizer_flow.zero_grad()
    output_flow = flow_model(x_train_flow_tensor)
    loss_flow = criterion(output_flow, x_train_flow_tensor)
    loss_flow.backward()
    optimizer_flow.step()

    # Record losses
    vanilla_loss_history.append(loss_vanilla.item())
    flow_loss_history.append(loss_flow.item())

    # Save reconstructions for animation at selected epochs
    if epoch in sample_epochs:
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Vanilla Loss: {loss_vanilla.item():.6f}, "
              f"Flow Loss: {loss_flow.item():.6f}")

        # Vanilla autoencoder reconstruction
        with torch.no_grad():
            rec_test_vanilla = vanilla_model(x_test_tensor).numpy()
        vanilla_recon_history.append(rec_test_vanilla)

        # Flow autoencoder reconstruction
        with torch.no_grad():
            rec_flow_test = flow_model(x_flow_test_tensor).numpy()
        flow_recon_history.append(rec_flow_test)

        # Reverse flow to get reconstructions in original space
        paths_reverse = flow(rec_flow_test, tn, W[::-1, :], drift, diffusion,
                             num_steps=ntime, reverse=True)
        deflow_recon_history.append(paths_reverse[-1])

# ---------------------------
# 6. Create animations
# ---------------------------
# Create a figure with three subplots
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, height_ratios=[2, 1])

# Original space subplots
ax1 = fig.add_subplot(gs[0, 0])  # Flow animation
ax2 = fig.add_subplot(gs[0, 1])  # Training animation
ax3 = fig.add_subplot(gs[1, :])  # Loss curves

# Plot the original sine curve (static)
x_curve = np.linspace(-4, 4, 1000)
y_curve = curve(x_curve)
ax1.plot(x_curve, y_curve, 'k-', linewidth=1, alpha=0.3, label='True Manifold')
ax2.plot(x_curve, y_curve, 'k-', linewidth=1, alpha=0.3, label='True Manifold')

# Draw basepoint and tangent vector
ax1.scatter(basepoint_np[0], basepoint_np[1], color='red', s=50,
            edgecolor='black', label='Basepoint')
tangent_scale = 0.5
ax1.arrow(basepoint_np[0], basepoint_np[1],
          tangent_vec_np[0] * tangent_scale, tangent_vec_np[1] * tangent_scale,
          color='red', width=0.02, head_width=0.1, length_includes_head=True)

# Initialize plots
train_points_flow_plot, = ax1.plot([], [], 'bo', markersize=4, alpha=0.7, label='Train Points')
test_points_flow_plot, = ax1.plot([], [], 'go', markersize=2, alpha=0.5, label='Test Points')

vanilla_recon_plot, = ax2.plot([], [], 'bo', markersize=4, alpha=0.7, label='Vanilla Reconstruction')
flow_recon_plot, = ax2.plot([], [], 'ro', markersize=4, alpha=0.7, label='Flow Reconstruction')
test_points_plot, = ax2.plot([], [], 'go', markersize=2, alpha=0.5, label='Test Points')

# Setting up legends and labels
ax1.set_title('Flow Animation')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend(loc='upper right')
ax1.set_xlim(-4.5, 4.5)
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True)

ax2.set_title('Reconstruction Animation')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend(loc='upper right')
ax2.set_xlim(-4.5, 4.5)
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True)

# Epoch indicator text
epoch_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes,
                      bbox=dict(facecolor='white', alpha=0.7))

# Plotting loss curves (updated during animation)
vanilla_loss_line, = ax3.plot([], [], 'b-', label='Vanilla AE Loss')
flow_loss_line, = ax3.plot([], [], 'r-', label='Flow AE Loss')
ax3.set_title('Training Loss')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True)


# Animation function for the flow
def animate_flow(frame):
    # Animation of the flow
    if frame < len(paths_train):
        train_points_flow_plot.set_data(paths_train[frame][:, 0], paths_train[frame][:, 1])
        test_points_flow_plot.set_data(paths_test[frame][:, 0], paths_test[frame][:, 1])
        return train_points_flow_plot, test_points_flow_plot
    return train_points_flow_plot, test_points_flow_plot


# Animation function for the training
def animate_training(frame):
    epoch_frame = frame % len(vanilla_recon_history)
    flow_frame = frame // len(vanilla_recon_history)

    if flow_frame < len(paths_train):
        # Display current epoch
        current_epoch = sample_epochs[epoch_frame]
        epoch_text.set_text(f'Epoch: {current_epoch + 1}')

        # Update reconstructions
        vanilla_recon_plot.set_data(vanilla_recon_history[epoch_frame][:, 0],
                                    vanilla_recon_history[epoch_frame][:, 1])
        flow_recon_plot.set_data(deflow_recon_history[epoch_frame][:, 0],
                                 deflow_recon_history[epoch_frame][:, 1])
        test_points_plot.set_data(test_points[:, 0], test_points[:, 1])

        # Update loss curves
        epochs_to_plot = np.arange(current_epoch + 1)
        losses_to_plot_vanilla = vanilla_loss_history[:current_epoch + 1]
        losses_to_plot_flow = flow_loss_history[:current_epoch + 1]

        vanilla_loss_line.set_data(epochs_to_plot, losses_to_plot_vanilla)
        flow_loss_line.set_data(epochs_to_plot, losses_to_plot_flow)

        ax3.relim()
        ax3.autoscale_view()

        return vanilla_recon_plot, flow_recon_plot, test_points_plot, epoch_text, vanilla_loss_line, flow_loss_line
    return vanilla_recon_plot, flow_recon_plot, test_points_plot, epoch_text, vanilla_loss_line, flow_loss_line


# Create animations
flow_anim = animation.FuncAnimation(fig, animate_flow, frames=len(paths_train),
                                    interval=50, blit=True)

training_anim = animation.FuncAnimation(fig, animate_training,
                                        frames=len(vanilla_recon_history) * len(paths_train),
                                        interval=100, blit=True)

plt.tight_layout()

# ---------------------------
# 7. Final results comparison
# ---------------------------
# Calculate final reconstruction errors
with torch.no_grad():
    final_rec_vanilla = vanilla_model(x_test_tensor).numpy()

with torch.no_grad():
    final_rec_flow = flow_model(x_flow_test_tensor).numpy()

final_paths_reverse = flow(final_rec_flow, tn, W[::-1, :], drift, diffusion,
                           num_steps=ntime, reverse=True)
final_rec_deflow = final_paths_reverse[-1]

vanilla_error = np.mean(np.sum((test_points - final_rec_vanilla) ** 2, axis=1))
flow_error = np.mean(np.sum((test_points - final_rec_deflow) ** 2, axis=1))

print(f"\nFinal Vanilla Autoencoder Test Reconstruction Error: {vanilla_error:.6f}")
print(f"Final Flow-Preconditioned Autoencoder Original-Space Reconstruction Error: {flow_error:.6f}")

# Create a final comparison plot
plt.figure(figsize=(10, 6))
plt.plot(x_curve, y_curve, 'k--', label='Original Curve', linewidth=1.5)
plt.scatter(test_points[:, 0], test_points[:, 1], color='black', s=20, label='Test Points')
plt.scatter(final_rec_vanilla[:, 0], final_rec_vanilla[:, 1], color='blue', s=20, alpha=0.7,
            label='Vanilla Autoencoder Reconstruction')
plt.scatter(final_rec_deflow[:, 0], final_rec_deflow[:, 1], color='red', s=20, alpha=0.7,
            label='Flow-Preconditioned Reconstruction')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Final Comparison of Autoencoder Reconstructions')
plt.legend()
plt.grid(True)
plt.text(0.02, 0.02,
         f"Vanilla Error: {vanilla_error:.6f}\nFlow Error: {flow_error:.6f}",
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()

# Save animations
flow_anim.save('flow_animation.gif', writer='pillow', fps=15)
training_anim.save('training_animation.gif', writer='pillow', fps=10)

print("Animations saved as 'flow_animation.mp4' and 'training_animation.mp4'")

# Display final plot
plt.show()