import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the functions for mean and variance
def mean_theta(t, theta0):
    return 0.5 * np.exp(-t / 2) * (2 * theta0 + np.exp(t / 2) - 1)

def var_theta(t, theta0):
    return (1/8) * np.exp(-2 * t) * (np.exp(t) - 1) * (8 * theta0 * (1 - theta0) + np.exp(t) - 1)

# Create a meshgrid for t and theta0
t_values = np.linspace(0, 5, 100)
theta0_values = np.linspace(0.01, 0.99, 100)  # Avoid 0 and 1 to prevent singularities
T, Theta0 = np.meshgrid(t_values, theta0_values)

# Compute mean and variance over the grid
Mean = mean_theta(T, Theta0)
Variance = var_theta(T, Theta0)

# Plot Mean
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(T, Theta0, Mean, cmap='viridis')
ax1.set_xlabel('t')
ax1.set_ylabel(r'$\theta_0$')
ax1.set_zlabel(r'$\mathbb{E}[\theta_t | \theta_0]$')
ax1.set_title('Mean of $\Theta_t$')

# Plot Variance
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(T, Theta0, Variance, cmap='plasma')
ax2.set_xlabel('t')
ax2.set_ylabel(r'$\theta_0$')
ax2.set_zlabel(r'$\mathrm{Var}(\theta_t | \theta_0)$')
ax2.set_title('Variance of $\Theta_t$')

plt.show()