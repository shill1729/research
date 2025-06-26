import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Define the closed curve
def closed_curve(t):
    r = 1.5 + 1.4 * np.cos(2 * t)**2
    return np.vstack((r * np.cos(t), r * np.sin(t))).T

# Derivative of the curve (for tangents)
def closed_curve_derivative(t):
    dr_dt = -2.8 * np.cos(2 * t) * np.sin(2 * t)
    r = 1.5 + 1.4 * np.cos(2 * t)**2
    dx_dt = dr_dt * np.cos(t) - r * np.sin(t)
    dy_dt = dr_dt * np.sin(t) + r * np.cos(t)
    return np.vstack((dx_dt, dy_dt)).T

# Select two t values to define the two points on the curve
t_vals = [0.0, np.pi]
points = closed_curve(np.array(t_vals))
tangents = closed_curve_derivative(np.array(t_vals))

# Normalize tangents and get normals
tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
normals = np.stack([tangents[:,1], -tangents[:,0]], axis=1)

# Create ellipse shape matrix from tangent-normal frame
def tangent_normal_matrix(tangent, normal, a, b):
    R = np.column_stack([tangent, normal])
    D = np.diag([a**2, b**2])
    return R @ D @ R.T

# Function to draw an ellipse given center and shape matrix
def draw_ellipse(ax, center, A, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(A)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigvals)
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# Adjusted MVEE shape matrices that still go through the two points
def get_adjusted_MVEE_matrix(p1, p2, a):
    center = (p1 + p2) / 2
    diff = p1 - center
    x_len = np.linalg.norm(diff)
    scale = x_len ** 2
    return center, np.diag([scale, (a * x_len)**2])

# Prepare plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

# Plot the full curve
t_full = np.linspace(0, 2 * np.pi, 500)
curve_pts = closed_curve(t_full)
ax.plot(curve_pts[:, 0], curve_pts[:, 1], 'gray', linewidth=1, label='Closed curve')

# Plot the data points
ax.plot(points[:, 0], points[:, 1], 'ko', label='Selected points')

# Plot enclosing ellipses centered at the midpoint, going through both points
a_values = [1.0, 0.5, 0.2, 0.1]
colors = ['red', 'green', 'blue', 'purple']
for a, color in zip(a_values, colors):
    center, A = get_adjusted_MVEE_matrix(points[0], points[1], a)
    draw_ellipse(ax, center=center, A=A, edgecolor=color, fill=False, label=f'a = {a:.1f}')

# Add tangent-normal ellipses at the two points
a_tn, b_tn = 0.5, 0.1
for pt, tan, nor in zip(points, tangents, normals):
    A_tn = tangent_normal_matrix(tan, nor, a_tn, b_tn)
    draw_ellipse(ax, center=pt, A=A_tn, edgecolor='black', linestyle='--', fill=False)

ax.legend()
plt.title("Tangential-Normal Ellipses and Degenerate Enclosing Ellipses")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
