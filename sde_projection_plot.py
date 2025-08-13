import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
# np.random.seed(42)


def sphere_surface(u, v, radius=1):
    """Parametric representation of a sphere"""
    x = radius * np.sin(u) * np.cos(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(u)
    return x, y, z


def normal_vector(point, radius=1):
    """Normal vector to sphere at given point (outward pointing)"""
    return point / radius


def mean_curvature_sphere(radius=1):
    """Mean curvature of a sphere (constant)"""
    return 1 / radius


def projection_matrix(n):
    """Projection matrix P = I - nn^T"""
    n = n.reshape(-1, 1)
    return np.eye(3) - n @ n.T


def tangent_plane_points(center, normal, size=0.5, grid_size=10):
    """Generate points for tangent plane visualization"""
    # Create two orthogonal vectors in the tangent plane
    if abs(normal[2]) < 0.9:
        v1 = np.cross(normal, [0, 0, 1])
    else:
        v1 = np.cross(normal, [1, 0, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)

    # Create grid
    u = np.linspace(-size, size, grid_size)
    v = np.linspace(-size, size, grid_size)
    U, V = np.meshgrid(u, v)

    # Generate plane points
    plane_points = (center[:, np.newaxis, np.newaxis] +
                    U[np.newaxis, :, :] * v1[:, np.newaxis, np.newaxis] +
                    V[np.newaxis, :, :] * v2[:, np.newaxis, np.newaxis])

    return plane_points[0], plane_points[1], plane_points[2]


def sde_on_sphere(X0, brownian_increments, T, radius=1):
    """
    Solve SDE: dX = c(x)n(x)dt + P(x)dB
    where c(x) is mean curvature, n(x) is normal vector, P(x) = I - nn^T
    """
    n_steps = len(brownian_increments)
    dt = T / n_steps
    X = np.zeros((n_steps + 1, 3))
    X[0] = X0

    c = mean_curvature_sphere(radius)

    for i in range(n_steps):
        x_curr = X[i]
        n_curr = normal_vector(x_curr, radius)
        P_curr = projection_matrix(n_curr)

        # SDE step
        drift = c * n_curr * dt
        diffusion = P_curr @ brownian_increments[i]

        X[i + 1] = x_curr + drift + diffusion

        # Project back to sphere to maintain constraint
        X[i + 1] = radius * X[i + 1] / np.linalg.norm(X[i + 1])

    return X


def create_visualization():
    # Parameters
    radius = 1.0
    T = 2.2
    n_steps = 15000
    dt = T / n_steps

    # Create sphere surface
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    x_sphere, y_sphere, z_sphere = sphere_surface(U, V, radius)

    # Initial point on sphere
    theta0, phi0 = np.pi / 3, np.pi / 4
    X0 = np.array([
        radius * np.sin(theta0) * np.cos(phi0),
        radius * np.sin(theta0) * np.sin(phi0),
        radius * np.cos(theta0)
    ])

    # Generate Brownian increments
    brownian_increments = np.random.normal(0, np.sqrt(dt), (n_steps, 3))

    # Create ambient Brownian path
    brownian_path = np.zeros((n_steps + 1, 3))
    brownian_path[1:] = np.cumsum(brownian_increments, axis=0)  # Scale for visibility

    # Generate SDE solution on sphere using same Brownian increments
    sde_path = sde_on_sphere(X0, brownian_increments, T, radius)

    # Key points for tangent planes
    mid_idx = n_steps // 2
    X_mid = sde_path[mid_idx]
    X_final = sde_path[-1]

    # Create plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot sphere
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='lightblue')

    # Plot initial point and tangent plane
    ax.scatter(*X0, color='red', s=100, label='Initial point $x_0$')
    n0 = normal_vector(X0, radius)
    plane_x0, plane_y0, plane_z0 = tangent_plane_points(X0, n0)
    ax.plot_surface(plane_x0, plane_y0, plane_z0, alpha=0.5, color='red')

    # Plot ambient Brownian motion
    ax.plot(brownian_path[:, 0], brownian_path[:, 1], brownian_path[:, 2],
            'g-', alpha=0.7, linewidth=1, label='Ambient Brownian motion')
    ax.scatter(*brownian_path[0], color='green', s=50)
    ax.scatter(*brownian_path[-1], color='darkgreen', s=50)

    # Plot SDE solution on sphere
    ax.plot(sde_path[:, 0], sde_path[:, 1], sde_path[:, 2],
            'orange', linewidth=2, label='SDE solution on sphere')

    # Plot key points and their tangent planes
    # ax.scatter(*X_mid, color='blue', s=80, label='$X_{T/2}$')
    ax.scatter(*X_final, color='purple', s=80, label='$X_T$')

    # # Tangent plane at T/2
    # n_mid = normal_vector(X_mid, radius)
    # plane_x_mid, plane_y_mid, plane_z_mid = tangent_plane_points(X_mid, n_mid)
    # ax.plot_surface(plane_x_mid, plane_y_mid, plane_z_mid, alpha=0.5, color='blue')

    # Tangent plane at T
    n_final = normal_vector(X_final, radius)
    plane_x_final, plane_y_final, plane_z_final = tangent_plane_points(X_final, n_final)
    ax.plot_surface(plane_x_final, plane_y_final, plane_z_final, alpha=0.5, color='purple')

    # Formatting
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sphere with SDE and Brownian Motion')
    ax.legend()
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    # Save and show
    plt.savefig('sphere_brownian_sde_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


# Run the visualization
if __name__ == "__main__":
    fig = create_visualization()