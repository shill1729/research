from ae.sdes import SDE
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
x0 = np.array([0.2, 0.2])
tn = 4.
ntime = 10000
npaths = 100

def mu(t, x):
    r = np.linalg.vector_norm(x, axis=0, ord=2)
    return np.zeros(2)-0.5 * r**2 * x

def sigma(t, x):
    r = np.linalg.vector_norm(x, axis=0, ord=2)
    return np.diag([r,r])*0.01


sde = SDE(mu, sigma)
ensemble = sde.sample_ensemble(x0, tn, ntime, npaths)

# Plot of ensemble
fig = plt.figure()
for i in range(npaths):
    plt.plot(ensemble[i, :, 0], ensemble[i, :, 1], alpha=0.5, c="blue")
plt.title("Riemannian BM in $\mathbb{C}\setminus \{0\}$")
plt.xlabel("$\Re(z)$")
plt.ylabel("$\Im(z)$")
plt.grid()
plt.show()



# Extract all final states from the ensemble
final_states = ensemble[:, -1, :]  # shape (npaths, 2)
x, y = final_states[:, 0], final_states[:, 1]

# Define a grid for plotting the exact density
grid_size = 200
xgrid = np.linspace(-4, 4, grid_size)
ygrid = np.linspace(-4, 4, grid_size)
X, Y = np.meshgrid(xgrid, ygrid)
Z = X + 1j * Y
R = np.abs(Z)
exact_density = (1 / (np.sqrt(2) * np.pi ** (3/2))) * np.exp(-0.5 * R**2) / R
exact_density[R == 0] = 0  # avoid division by zero

# Empirical density using KDE
values = np.vstack([x, y])
kde = gaussian_kde(values)
emp_density = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot empirical
cf1 = axs[0].contourf(X, Y, emp_density, levels=100, cmap="viridis")
axs[0].set_title("Empirical Density from Ensemble")
axs[0].set_xlabel("$\Re(z)$")
axs[0].set_ylabel("$\Im(z)$")
fig.colorbar(cf1, ax=axs[0])

# Plot exact
cf2 = axs[1].contourf(X, Y, exact_density, levels=100, cmap="plasma")
axs[1].set_title("Exact Steady-State Density")
axs[1].set_xlabel("$\Re(z)$")
axs[1].set_ylabel("$\Im(z)$")
fig.colorbar(cf2, ax=axs[1])

plt.tight_layout()
plt.show()
