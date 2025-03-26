import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Rotation: 60 degrees around x-axis
theta = np.radians(60)
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta),  np.cos(theta)]
])

# Reflection: across x-z plane (i.e., y -> -y)
Reflect_xz = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])

# Composite transformation: reflect AFTER rotation
Transform = Reflect_xz @ R_x

# Sample vectors to transform
vectors = np.array([
    [1, 1, 0],
    [0, 1, 1],
    [-1, 1, 0],
    [1, 0, 1]
]).T  # Shape (3, N)

# Apply transformations
rotated = R_x @ vectors
reflected = Reflect_xz @ vectors
composed = Transform @ vectors

# Origin for arrows
origin = np.zeros((3, vectors.shape[1]))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Function to draw arrows
def draw_arrows(start, vecs, color, label):
    for i in range(vecs.shape[1]):
        ax.quiver(
            start[0, i], start[1, i], start[2, i],
            vecs[0, i], vecs[1, i], vecs[2, i],
            color=color, arrow_length_ratio=0.1, linewidth=2,
            label=label if i == 0 else ""
        )

# Plot all sets
draw_arrows(origin, vectors, 'gray', 'Original')
draw_arrows(origin, rotated, 'blue', 'Rotated (60° around x)')
draw_arrows(origin, reflected, 'green', 'Reflected (xz-plane)')
draw_arrows(origin, composed, 'red', 'Composed')

# Plot settings
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.set_title("Rotation + Reflection (Problem 11)")

plt.tight_layout()
plt.show()

# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# --- Basis vectors ---
basis_vectors = {
    "e₁": np.array([1, 0, 0]),
    "e₂": np.array([0, 1, 0]),
    "e₃": np.array([0, 0, 1])
}

# Plot basis vectors
for name, vec in basis_vectors.items():
    ax.quiver(0, 0, 0, *vec, color='black', linewidth=2, arrow_length_ratio=0.1)
    ax.text(*vec * 1.2, name, color='black', fontsize=12)

# --- Plot the x–z plane (plane of reflection) ---
xz_plane = np.array([
    [-1.5, 0, -1.5],
    [-1.5, 0, 1.5],
    [1.5, 0, 1.5],
    [1.5, 0, -1.5]
])
ax.add_collection3d(Poly3DCollection([xz_plane], color='green', alpha=0.2, label="x–z plane (reflect)"))

# --- Plot the reflection plane from statement 1: perpendicular to √3 e₂ - e₃ ---
# Normal vector = √3 e₂ - e₃
n = np.sqrt(3) * np.array([0, 1, 0]) - np.array([0, 0, 1])
n = n / np.linalg.norm(n)

# To plot a plane, find two orthogonal vectors on the plane
u = np.array([1, 0, -(n[0] / n[2])]) if n[2] != 0 else np.array([1, 0, 0])
v = np.cross(n, u)
u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)

# Create a mesh grid of the new plane
plane_center = np.array([0, 0, 0])
plane_size = 1.5
plane_pts = [plane_center + a * u + b * v for a, b in [(-plane_size, -plane_size), (-plane_size, plane_size),
                                                       (plane_size, plane_size), (plane_size, -plane_size)]]
reflection_plane = np.array(plane_pts)
ax.add_collection3d(Poly3DCollection([reflection_plane], color='purple', alpha=0.2, label='Plane ⟂ (√3 e₂ − e₃)'))

# --- Plot original and transformed vectors ---
draw_arrows(origin, vectors, 'gray', 'Original')
draw_arrows(origin, composed, 'red', 'Composed (rotate+reflect)')

# Plot settings
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend(loc='upper left')
ax.set_title("Geometric Interpretation of Problem 11: Planes and Transformations")

plt.tight_layout()
plt.show()

# Rebuild plot without trying to auto-label Poly3DCollections in legend
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot basis vectors with labels
for name, vec in basis_vectors.items():
    ax.quiver(0, 0, 0, *vec, color='black', linewidth=2, arrow_length_ratio=0.1)
    ax.text(*vec * 1.2, name, color='black', fontsize=12)

# Plot xz-plane
ax.add_collection3d(Poly3DCollection([xz_plane], color='green', alpha=0.2))

# Plot the reflection plane perpendicular to sqrt(3)e2 - e3
ax.add_collection3d(Poly3DCollection([reflection_plane], color='purple', alpha=0.2))

# Plot vectors
draw_arrows(origin, vectors, 'gray', 'Original')
draw_arrows(origin, composed, 'red', 'Composed (rotate+reflect)')

# Plot settings
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Manually create legend
legend_elements = [
    Line2D([0], [0], color='gray', lw=2, label='Original'),
    Line2D([0], [0], color='red', lw=2, label='Composed (rotate+reflect)'),
    Patch(facecolor='green', edgecolor='green', alpha=0.2, label='x–z plane'),
    Patch(facecolor='purple', edgecolor='purple', alpha=0.2, label='Plane ⟂ √3 e₂ − e₃')
]
ax.legend(handles=legend_elements, loc='upper left')

ax.set_title("Geometric Interpretation of Problem 11: Planes and Transformations")
plt.tight_layout()
plt.show()
