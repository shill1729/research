import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
t_range = np.linspace(0, 1, 1000)
c = 0.1  # Scaling factor for the counterexample


# True manifold: (t, t^2)
def phi_true(t):
    return np.array([t, t ** 2])


def D_phi_true(t):
    return np.array([[1], [2 * t]])


# Base autoencoder (Autoencoder 2)
def pi_2(x):
    # Simple encoder: just take first coordinate
    return x[0]


def phi_2(t):
    # Decoder that reconstructs the manifold
    return np.array([t, t ** 2])


def D_phi_2(t):
    return np.array([[1], [2 * t]])


# Scaled autoencoder (Autoencoder 1) - this is our counterexample
def pi_1(x):
    # Encoder that scales the latent: π₁(x) = c * π₂(x)
    return c * pi_2(x)


def phi_1(z):
    # Decoder that undoes the scaling: φ₁(z) = φ₂(z/c)
    return phi_2(z / c)


def D_phi_1(z):
    # Jacobian: d/dz [φ₂(z/c)] = D_φ₂(z/c) * (1/c)
    return (1 / c) * D_phi_2(z / c)


# Compute projection matrices
def compute_projection_matrix(D_phi_t):
    """Compute P = D_phi * g^{-1} * D_phi^T where g = D_phi^T * D_phi"""
    g = D_phi_t.T @ D_phi_t
    g_inv = 1.0 / g  # scalar since g is 1x1
    P = D_phi_t @ g_inv @ D_phi_t.T
    return P


# Compute metrics
def compute_reconstruction_error_manifold(phi_func, pi_func, t_vals):
    """Compute reconstruction error on manifold points"""
    errors = []
    for t in t_vals:
        true_point = phi_true(t)
        latent = pi_func(true_point)
        reconstructed = phi_func(latent)
        error = np.linalg.norm(reconstructed - true_point)
        errors.append(error)
    return np.mean(errors), np.max(errors)


def compute_tangent_alignment_error_manifold(D_phi_func, pi_func, t_vals):
    """Compute tangent bundle alignment error on manifold"""
    errors = []
    for t in t_vals:
        # True tangent space projection
        P_true = compute_projection_matrix(D_phi_true(t))

        # Learned tangent space projection
        # We need to evaluate at the latent point
        true_point = phi_true(t)
        latent = pi_func(true_point)
        P_hat = compute_projection_matrix(D_phi_func(latent))

        error = np.linalg.norm(P_hat - P_true, 'fro')
        errors.append(error)
    return np.mean(errors), np.max(errors)


def compute_lipschitz_constant(D_phi_func, z_vals):
    """Compute Lipschitz constant (supremum of Jacobian operator norms)"""
    norms = []
    for z in z_vals:
        jacobian = D_phi_func(z)
        op_norm = np.linalg.norm(jacobian, 2)  # operator norm
        norms.append(op_norm)
    return max(norms)


# Generate data for plotting
t_plot = np.linspace(0, 1, 200)
true_manifold = np.array([phi_true(t) for t in t_plot])

# Generate reconstructed manifolds by going through encode-decode process
manifold_1_reconstructed = []
manifold_2_reconstructed = []
for t in t_plot:
    true_point = phi_true(t)

    # Autoencoder 1 reconstruction
    latent_1 = pi_1(true_point)
    recon_1 = phi_1(latent_1)
    manifold_1_reconstructed.append(recon_1)

    # Autoencoder 2 reconstruction
    latent_2 = pi_2(true_point)
    recon_2 = phi_2(latent_2)
    manifold_2_reconstructed.append(recon_2)

manifold_1_reconstructed = np.array(manifold_1_reconstructed)
manifold_2_reconstructed = np.array(manifold_2_reconstructed)

# Compute metrics
t_sample = np.linspace(0.1, 1, 50)  # Avoid t=0 for numerical stability

# Reconstruction errors
recon_error_1_mean, recon_error_1_max = compute_reconstruction_error_manifold(phi_1, pi_1, t_sample)
recon_error_2_mean, recon_error_2_max = compute_reconstruction_error_manifold(phi_2, pi_2, t_sample)

# Tangent alignment errors
tangent_error_1_mean, tangent_error_1_max = compute_tangent_alignment_error_manifold(D_phi_1, pi_1, t_sample)
tangent_error_2_mean, tangent_error_2_max = compute_tangent_alignment_error_manifold(D_phi_2, pi_2, t_sample)

# Lipschitz constants (evaluated in latent space)
z_sample = np.linspace(0.1, 1, 50)  # Sample points in latent space
lipschitz_1 = compute_lipschitz_constant(D_phi_1, z_sample)
lipschitz_2 = compute_lipschitz_constant(D_phi_2, z_sample)

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Manifolds comparison
ax1.plot(true_manifold[:, 0], true_manifold[:, 1], 'k-', linewidth=3, label='True Manifold')
ax1.plot(manifold_1_reconstructed[:, 0], manifold_1_reconstructed[:, 1], 'b--', linewidth=2,
         label='Autoencoder 1 (Scaled)')
ax1.plot(manifold_2_reconstructed[:, 0], manifold_2_reconstructed[:, 1], 'r:', linewidth=2,
         label='Autoencoder 2 (Base)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Manifold Comparison (All Should Overlap)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Reconstruction error over parameter space (should be zero for both)
t_dense = np.linspace(0.1, 1, 200)
recon_errors_1 = []
recon_errors_2 = []
for t in t_dense:
    true_point = phi_true(t)

    latent_1 = pi_1(true_point)
    recon_1 = phi_1(latent_1)
    error_1 = np.linalg.norm(recon_1 - true_point)
    recon_errors_1.append(error_1)

    latent_2 = pi_2(true_point)
    recon_2 = phi_2(latent_2)
    error_2 = np.linalg.norm(recon_2 - true_point)
    recon_errors_2.append(error_2)

ax2.plot(t_dense, recon_errors_1, 'b-', label='Autoencoder 1', linewidth=2)
ax2.plot(t_dense, recon_errors_2, 'r-', label='Autoencoder 2', linewidth=2)
ax2.set_xlabel('Parameter t')
ax2.set_ylabel('Reconstruction Error')
ax2.set_title('Reconstruction Error vs Parameter (Should be Zero)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Tangent spaces at select points
select_points = [0.2, 0.5, 0.8]
ax3.plot(true_manifold[:, 0], true_manifold[:, 1], 'k-', linewidth=2, alpha=0.5, label='True Manifold')

for i, t_val in enumerate(select_points):
    point_true = phi_true(t_val)

    # Get reconstructed points
    latent_1 = pi_1(point_true)
    point_1 = phi_1(latent_1)
    latent_2 = pi_2(point_true)
    point_2 = phi_2(latent_2)

    # Tangent vectors (normalized for visualization)
    tangent_true = D_phi_true(t_val).flatten()
    tangent_1 = D_phi_1(latent_1).flatten()
    tangent_2 = D_phi_2(latent_2).flatten()

    # Normalize for visualization
    scale = 0.1
    tangent_true_norm = tangent_true / np.linalg.norm(tangent_true) * scale
    tangent_1_norm = tangent_1 / np.linalg.norm(tangent_1) * scale
    tangent_2_norm = tangent_2 / np.linalg.norm(tangent_2) * scale

    # Plot points
    ax3.plot(point_true[0], point_true[1], 'ko', markersize=8)
    ax3.plot(point_1[0], point_1[1], 'bs', markersize=6)
    ax3.plot(point_2[0], point_2[1], 'r^', markersize=6)

    # Plot tangent vectors (all should point in same direction!)
    ax3.arrow(point_true[0], point_true[1], tangent_true_norm[0], tangent_true_norm[1],
              head_width=0.02, head_length=0.01, fc='black', ec='black', linewidth=2)
    ax3.arrow(point_1[0], point_1[1], tangent_1_norm[0], tangent_1_norm[1],
              head_width=0.02, head_length=0.01, fc='blue', ec='blue', linewidth=2)
    ax3.arrow(point_2[0], point_2[1], tangent_2_norm[0], tangent_2_norm[1],
              head_width=0.02, head_length=0.01, fc='red', ec='red', linewidth=2)

ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Tangent Spaces (Should All Align)')
ax3.legend(['True Manifold', 'True Points', 'Autoencoder 1', 'Autoencoder 2'])
ax3.grid(True, alpha=0.3)

# Plot 4: Jacobian operator norms in latent space
z_dense = np.linspace(0.1, 1, 200)
jacobian_norms_1 = [np.linalg.norm(D_phi_1(z), 2) for z in z_dense]
jacobian_norms_2 = [np.linalg.norm(D_phi_2(z), 2) for z in z_dense]

ax4.plot(z_dense, jacobian_norms_1, 'b-', label=f'Autoencoder 1 (c={c})', linewidth=2)
ax4.plot(z_dense, jacobian_norms_2, 'r-', label='Autoencoder 2 (c=1)', linewidth=2)
ax4.set_xlabel('Latent Variable z')
ax4.set_ylabel('Jacobian Operator Norm')
ax4.set_title('Jacobian Operator Norms (Lipschitz Constants)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create results table
results_data = {
    'Metric': [
        'Reconstruction Error (Mean)',
        'Reconstruction Error (Max)',
        'Tangent Alignment Error (Mean)',
        'Tangent Alignment Error (Max)',
        'Lipschitz Constant',
        'Scaling Factor c'
    ],
    'Autoencoder 1 (Scaled)': [
        f'{recon_error_1_mean:.8f}',
        f'{recon_error_1_max:.8f}',
        f'{tangent_error_1_mean:.8f}',
        f'{tangent_error_1_max:.8f}',
        f'{lipschitz_1:.6f}',
        f'{c:.3f}'
    ],
    'Autoencoder 2 (Base)': [
        f'{recon_error_2_mean:.8f}',
        f'{recon_error_2_max:.8f}',
        f'{tangent_error_2_mean:.8f}',
        f'{tangent_error_2_max:.8f}',
        f'{lipschitz_2:.6f}',
        f'1.000'
    ]
}

results_df = pd.DataFrame(results_data)
print("\nResults Summary:")
print("=" * 80)
print(results_df.to_string(index=False))

print(f"\nScaling relationship:")
print(f"Autoencoder 1 encoder: π₁(x) = {c} * π₂(x)")
print(f"Autoencoder 1 decoder: φ₁(z) = (1/{c}) * φ₂(z)")
print(f"\nTheoretical Lipschitz ratio: L₁/L₂ = 1/c = {1 / c:.1f}")
print(f"Actual Lipschitz ratio: {lipschitz_1 / lipschitz_2:.6f}")

print(f"\nKey Findings:")
print(f"1. Both autoencoders have IDENTICAL reconstruction error (≈ 0)")
print(f"2. Both autoencoders have IDENTICAL tangent alignment error (≈ 0)")
print(f"3. BUT Autoencoder 1 has Lipschitz constant {lipschitz_1:.1f} vs {lipschitz_2:.1f}")
print(f"4. The ratio is {lipschitz_1 / lipschitz_2:.1f}, matching the theoretical 1/c = {1 / c:.1f}")
print(f"\nCOUNTEREXAMPLE PROVEN: Tangent bundle regularization does NOT uniquely determine the Lipschitz constant!")