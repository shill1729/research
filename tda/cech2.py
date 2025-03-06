import numpy as np
import matplotlib.pyplot as plt
from tda.cech import compute_0d_persistence, build_cech_filtration, plot_barcode

def generate_curve_samples(n=50, D=3):
    """Generate points on a smooth curve in R^D."""
    t = np.linspace(0, 2 * np.pi, n)
    points = np.vstack([np.cos(t), np.sin(t), 0.1 * t]).T  # Example 3D spiral
    return points


def construct_covariance_matrices(points, a1=0.02, a2=2):
    """Construct covariance matrices A(x) based on projection onto tangent space."""
    n, D = points.shape
    A_list = []

    for i in range(n):
        if i == 0:
            tangent = points[i + 1] - points[i]
        elif i == n - 1:
            tangent = points[i] - points[i - 1]
        else:
            tangent = (points[i + 1] - points[i - 1]) / 2

        tangent = tangent / np.linalg.norm(tangent)  # Normalize

        # Compute tangent projection matrix
        P = np.outer(tangent, tangent)
        U, _, _ = np.linalg.svd(P)

        # Construct modified A(x) using eigen decomposition
        Lambda = np.eye(D)
        Lambda[:-1, :-1] *= a1  # Inflate tangent directions
        Lambda[-1, -1] *= a2  # Deflate normal direction

        A = U @ Lambda @ U.T  # Transform back to original basis
        A_list.append(A)

    return A_list


def test_cech_complex():
    """Test the Čech complex with ellipsoidal balls."""
    np.random.seed(0)

    # Generate a synthetic point cloud from a curve
    n, D = 50, 3
    points = generate_curve_samples(n, D)

    # Construct covariance matrices A(x)
    A_list = construct_covariance_matrices(points)

    # Build Čech filtration
    filtration = build_cech_filtration(points, A_list, k_nn=10, max_dim=2)

    print(f"Filtration computed with {len(filtration)} simplices")

    # Compute 0D persistence
    persistence_0d = compute_0d_persistence(filtration)

    # Plot persistence barcode
    plot_barcode(persistence_0d)

    # Plot the point cloud with ellipsoidal shapes (2D projection for visualization)
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c='black', label='Curve Samples')
    plt.title("Point Cloud with Covariance-Based Ellipsoidal Balls")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_cech_complex()
