# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
#
# def generate_lambda_grid(num_points=50):
#     # Generate 2D grid for lambda1 and lambda2
#     lambda1_vals = np.linspace(0, 1, num_points)
#     lambda2_vals = np.linspace(0, 1, num_points)
#     Lambda1, Lambda2 = np.meshgrid(lambda1_vals, lambda2_vals)
#
#     # Compute lambda3 from the constraint lambda1 + lambda2 + lambda3 = 1
#     Lambda3 = 1 - Lambda1 - Lambda2
#
#     # Create mask for valid points (where lambda3 >= 0)
#     mask = Lambda3 >= 0
#
#     # Apply mask to all arrays
#     Lambda1_valid = Lambda1 * mask
#     Lambda2_valid = Lambda2 * mask
#     Lambda3_valid = Lambda3 * mask
#
#     return Lambda1_valid, Lambda2_valid, Lambda3_valid, mask
#
#
# def plot_ellipses_and_K(centers, A_invs, epsilon2, title):
#     # Convert centers to numpy arrays
#     centers = [np.array(c) for c in centers]
#
#     # Create figure with two subplots side by side
#     fig = plt.figure(figsize=(15, 6))
#
#     # 2D plot of ellipses
#     ax1 = fig.add_subplot(121)
#
#     # Plot each ellipse
#     t = np.linspace(0, 2 * np.pi, 100)
#     for i, (center, A_inv) in enumerate(zip(centers, A_invs)):
#         # Generate points on unit circle
#         circle = np.array([np.cos(t), np.sin(t)])
#
#         # Transform circle to ellipse
#         A = np.linalg.inv(A_inv)
#         L, V = np.linalg.eigh(A)
#
#         # Scale by epsilon
#         ellipse = np.sqrt(epsilon2) * V @ np.diag(1 / np.sqrt(L)) @ circle
#
#         # Translate to center
#         ellipse = ellipse + center.reshape(2, 1)
#
#         # Plot
#         ax1.plot(ellipse[0], ellipse[1], label=f'Ellipse {i + 1}')
#
#     ax1.set_aspect('equal')
#     ax1.grid(True)
#     ax1.legend()
#     ax1.set_title(f'Ellipse Configuration\n{title}')
#
#     # 3D surface plot of K(λ)
#     ax2 = fig.add_subplot(122, projection='3d')
#
#     # Generate grid points
#     Lambda1, Lambda2, Lambda3, mask = generate_lambda_grid()
#
#     # Initialize K values array with same shape as Lambda1
#     K_vals = np.zeros_like(Lambda1)
#
#     # Compute K(λ) values for each point in the grid
#     for i in range(Lambda1.shape[0]):
#         for j in range(Lambda1.shape[1]):
#             if mask[i, j]:  # Only compute for valid points
#                 l1, l2, l3 = Lambda1[i, j], Lambda2[i, j], Lambda3[i, j]
#
#                 # Compute E_lambda_inv
#                 E_lambda_inv = l1 * A_invs[0] + l2 * A_invs[1] + l3 * A_invs[2]
#
#                 # Compute m_lambda
#                 m_lambda = np.linalg.inv(E_lambda_inv) @ (
#                         l1 * A_invs[0] @ centers[0] +
#                         l2 * A_invs[1] @ centers[1] +
#                         l3 * A_invs[2] @ centers[2]
#                 )
#
#                 # Compute C_lambda
#                 C_lambda = (
#                         l1 * (centers[0] @ A_invs[0] @ centers[0]) +
#                         l2 * (centers[1] @ A_invs[1] @ centers[1]) +
#                         l3 * (centers[2] @ A_invs[2] @ centers[2]) -
#                         m_lambda @ E_lambda_inv @ m_lambda
#                 )
#
#                 K_vals[i, j] = epsilon2 - C_lambda
#             else:
#                 K_vals[i, j] = np.nan  # Set invalid points to NaN
#
#     # Plot the surface
#     surf = ax2.plot_surface(Lambda1, Lambda2, K_vals, cmap='viridis')
#     ax2.set_xlabel('λ₁')
#     ax2.set_ylabel('λ₂')
#     ax2.set_zlabel('K(λ)')
#     ax2.set_title(f'K(λ) Surface\n{title}')
#     fig.colorbar(surf)
#
#     plt.tight_layout()
#     return fig
#
#
# # Define test cases with adjusted centers and epsilon values
# cases = [
#     {
#         "centers": [(0, 0), (2.5, 0), (1.25, 2)],  # No intersections
#         "A_invs": [np.linalg.inv(np.array([[2, 0], [0, 1]])),
#                    np.linalg.inv(np.array([[1.5, 0.5], [0.5, 1]])),
#                    np.linalg.inv(np.array([[1, -0.3], [-0.3, 1.2]]))],
#         "epsilon2": 0.5,
#         "title": "No Ellipses Intersect"
#     },
#     {
#         "centers": [(0, 0), (1.5, 0), (0.75, 2)],  # First two ellipses intersect
#         "A_invs": [np.linalg.inv(np.array([[2, 0], [0, 1]])),
#                    np.linalg.inv(np.array([[1.5, 0.5], [0.5, 1]])),
#                    np.linalg.inv(np.array([[1, -0.3], [-0.3, 1.2]]))],
#         "epsilon2": 1.0,
#         "title": "Pairwise Intersection"
#     },
#     {
#         "centers": [(0, 0), (1, 0), (0.5, 1)],  # All three intersect
#         "A_invs": [np.linalg.inv(np.array([[2, 0], [0, 1]])),
#                    np.linalg.inv(np.array([[1.5, 0.5], [0.5, 1]])),
#                    np.linalg.inv(np.array([[1, -0.3], [-0.3, 1.2]]))],
#         "epsilon2": 1.5,
#         "title": "All Ellipses Intersect"
#     }
# ]
#
# # Plot each case
# for case in cases:
#     fig = plot_ellipses_and_K(
#         case["centers"],
#         case["A_invs"],
#         case["epsilon2"],
#         case["title"]
#     )
#     plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_lambda_grid(num_points=50):
    lambda1_vals = np.linspace(0, 1, num_points)
    lambda2_vals = np.linspace(0, 1, num_points)
    Lambda1, Lambda2 = np.meshgrid(lambda1_vals, lambda2_vals)
    Lambda3 = 1 - Lambda1 - Lambda2
    mask = Lambda3 >= 0
    Lambda1_valid = Lambda1 * mask
    Lambda2_valid = Lambda2 * mask
    Lambda3_valid = Lambda3 * mask
    return Lambda1_valid, Lambda2_valid, Lambda3_valid, mask


def plot_ellipses_and_K(centers, A_invs, epsilon2, title):
    # Convert centers to numpy arrays
    centers = [np.array(c) for c in centers]

    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(15, 6))

    # 2D plot of ellipses
    ax1 = fig.add_subplot(121)

    # Plot each ellipse
    t = np.linspace(0, 2 * np.pi, 100)
    for i, (center, A_inv) in enumerate(zip(centers, A_invs)):
        circle = np.array([np.cos(t), np.sin(t)])
        A = np.linalg.inv(A_inv)
        L, V = np.linalg.eigh(A)
        ellipse = np.sqrt(epsilon2) * V @ np.diag(1 / np.sqrt(L)) @ circle
        ellipse = ellipse + center.reshape(2, 1)
        ax1.plot(ellipse[0], ellipse[1], label=f'Ellipse {i + 1}')

    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.legend()
    ax1.set_title(f'Ellipse Configuration\n{title}')

    # 3D surface plot of K(λ)
    ax2 = fig.add_subplot(122, projection='3d')

    # Generate grid points
    Lambda1, Lambda2, Lambda3, mask = generate_lambda_grid()

    # Initialize K values array
    K_vals = np.zeros_like(Lambda1)

    # Compute K(λ) values
    for i in range(Lambda1.shape[0]):
        for j in range(Lambda1.shape[1]):
            if mask[i, j]:
                l1, l2, l3 = Lambda1[i, j], Lambda2[i, j], Lambda3[i, j]
                E_lambda_inv = l1 * A_invs[0] + l2 * A_invs[1] + l3 * A_invs[2]
                m_lambda = np.linalg.inv(E_lambda_inv) @ (
                        l1 * A_invs[0] @ centers[0] +
                        l2 * A_invs[1] @ centers[1] +
                        l3 * A_invs[2] @ centers[2]
                )
                C_lambda = (
                        l1 * (centers[0] @ A_invs[0] @ centers[0]) +
                        l2 * (centers[1] @ A_invs[1] @ centers[1]) +
                        l3 * (centers[2] @ A_invs[2] @ centers[2]) -
                        m_lambda @ E_lambda_inv @ m_lambda
                )
                K_vals[i, j] = epsilon2 - C_lambda
            else:
                K_vals[i, j] = np.nan

    # Plot the surface
    surf = ax2.plot_surface(Lambda1, Lambda2, K_vals, cmap='viridis')

    # Add transparent plane at z=0
    xx, yy = np.meshgrid(np.linspace(0, 1, 2), np.linspace(0, 1, 2))
    zz = np.zeros_like(xx)
    ax2.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

    ax2.set_xlabel('λ₁')
    ax2.set_ylabel('λ₂')
    ax2.set_zlabel('K(λ)')
    ax2.set_title(f'K(λ) Surface\n{title}')
    fig.colorbar(surf)

    plt.tight_layout()
    return fig


# Define test cases
cases = [
    {
        "centers": [(0, 0), (2.5, 0), (1.25, 2)],  # No intersections
        "A_invs": [np.linalg.inv(np.array([[2, 0], [0, 1]])),
                   np.linalg.inv(np.array([[1.5, 0.5], [0.5, 1]])),
                   np.linalg.inv(np.array([[1, -0.3], [-0.3, 1.2]]))],
        "epsilon2": 0.5,
        "title": "No Ellipses Intersect"
    },
    {
        "centers": [(0, 0), (1.5, 0), (0.75, 2)],  # First two ellipses intersect
        "A_invs": [np.linalg.inv(np.array([[2, 0], [0, 1]])),
                   np.linalg.inv(np.array([[1.5, 0.5], [0.5, 1]])),
                   np.linalg.inv(np.array([[1, -0.3], [-0.3, 1.2]]))],
        "epsilon2": 1.0,
        "title": "Pairwise Intersection"
    },
    {
        "centers": [(0, 0), (1, 0), (0.5, 1)],  # All three intersect
        "A_invs": [np.linalg.inv(np.array([[2, 0], [0, 1]])),
                   np.linalg.inv(np.array([[1.5, 0.5], [0.5, 1]])),
                   np.linalg.inv(np.array([[1, -0.3], [-0.3, 1.2]]))],
        "epsilon2": 1.5,
        "title": "All Ellipses Intersect"
    }
]

# Plot each case
for case in cases:
    fig = plot_ellipses_and_K(
        case["centers"],
        case["A_invs"],
        case["epsilon2"],
        case["title"]
    )
    plt.show()