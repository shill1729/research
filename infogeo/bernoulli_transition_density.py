import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Define the PDF function
def pdf(t, theta, W0):
    if t == 0:
        return np.inf if theta == W0 else 0  # Handle t=0 edge case
    else:
        term = (2 * (np.arcsin(np.sqrt(theta)) - W0)) / np.sqrt(t)
        return (1 / (np.sqrt(t * theta * (1 - theta)))) * norm.pdf(term)


# Define the grid for (t, theta)
t_values = np.linspace(0.001, 2*np.pi**2/4, 200)  # Avoid t=0 to prevent division by zero
theta_values = np.linspace(0.001, 0.999, 200)  # Avoid 0 and 1 due to singularities

T, Theta = np.meshgrid(t_values, theta_values)
W0 = np.arcsin(np.sqrt(0.5))  # Assume starting at theta_0 = 0.5

# Compute PDF values
Z = np.array([[pdf(t, theta, W0) for t in t_values] for theta in theta_values])

# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, Theta, Z, cmap='viridis', edgecolor='k')

# Labels and title
ax.set_xlabel('t')
ax.set_ylabel(r'$\theta$')
ax.set_zlabel(r'$f(t, \theta|\theta_0)$')
ax.set_title('Surface Plot of the PDF $f(t, \\theta|\\theta_0)$')

plt.show()
