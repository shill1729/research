from ae.toydata.datagen import ToyData
from ae.models import AutoEncoderDiffusion

import matplotlib.pyplot as plt

import numpy as np
import torch

def plot_interior_boundary_highlight(epsilon, toydata: ToyData, aedf: AutoEncoderDiffusion, title=None):
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection="3d")

    # Set the point cloud domain to [a - ε, b + ε]^2

    toydata.set_point_cloud(epsilon)

    a = toydata.surface.bounds()[0][0]
    b = toydata.surface.bounds()[0][1]

    # Plot the learned surface
    aedf.autoencoder.plot_surface(a-epsilon, b+epsilon, 30, ax, title)

    # Get data from the point cloud
    data = toydata.point_cloud.generate()
    x = data[0]         # Embedded 3D points on the learned surface
    local_x = data[4]   # Original 2D input points

    # Check which local_x are in the interior box [a, b]^2
    is_interior = np.all((local_x >= a) & (local_x <= b), axis=1)

    # Separate into interior and exterior
    interior_local = local_x[is_interior]
    boundary_local = local_x[~is_interior]
    interior_x = x[is_interior]
    boundary_x = x[~is_interior]

    z_floor1 = np.zeros_like(interior_local[:, 0])-10
    z_floor2 = np.zeros_like(boundary_local[:, 0])-10
    # Plot the original 2D inputs at z = 0
    ax.scatter(interior_local[:, 0], interior_local[:, 1], z_floor1,
               color='green', label='z (interior)')
    ax.scatter(boundary_local[:, 0], boundary_local[:, 1], z_floor2,
               color='red', label='z (boundary)')

    # Plot the corresponding points on the learned surface
    ax.scatter(interior_x[:, 0], interior_x[:, 1], interior_x[:, 2],
               color='green', alpha=0.6, label='x (from interior)')
    ax.scatter(boundary_x[:, 0], boundary_x[:, 1], boundary_x[:, 2],
               color='red', alpha=0.6, label='x (from boundary)')

    ax.set_title(title or "Interior and Boundary Points Highlighted")
    ax.legend()
    plt.show()
    return fig


def plot_interior_boundary_recon(epsilon, toydata: ToyData, aedf: AutoEncoderDiffusion, title=None):
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection="3d")

    # Set the point cloud domain to [a - ε, b + ε]^2

    toydata.set_point_cloud(epsilon)

    a = toydata.surface.bounds()[0][0]
    b = toydata.surface.bounds()[0][1]

    # Plot the learned surface
    aedf.autoencoder.plot_surface(toydata.point_cloud.bounds[0][0], toydata.point_cloud.bounds[0][1],
                                  30, ax, title)

    # Get data from the point cloud
    data = toydata.point_cloud.generate()

    x = data[0]         # Embedded 3D points on the learned surface
    model_local_x = aedf.autoencoder.encoder.forward(torch.tensor(x, dtype=torch.float32))
    model_x = aedf.autoencoder.decoder.forward(model_local_x).detach().numpy()
    local_x = data[4] # True local x

    # Check which local_x are in the interior box [a, b]^2
    is_interior = np.all((local_x >= a) & (local_x <= b), axis=1)

    # Separate into interior and exterior
    interior_local = local_x[is_interior]
    boundary_local = local_x[~is_interior]
    interior_x = model_x[is_interior]
    boundary_x = model_x[~is_interior]

    z_floor1 = np.zeros_like(interior_local[:, 0])-10
    z_floor2 = np.zeros_like(boundary_local[:, 0])-10
    # Plot the original 2D inputs at z = 0
    ax.scatter(interior_local[:, 0], interior_local[:, 1], z_floor1,
               color='green', label='z (interior)')
    ax.scatter(boundary_local[:, 0], boundary_local[:, 1], z_floor2,
               color='red', label='z (boundary)')

    # Plot the corresponding points on the learned surface
    ax.scatter(interior_x[:, 0], interior_x[:, 1], interior_x[:, 2],
               color='green', alpha=0.6, label='$\hat{x}$ (from interior)')
    ax.scatter(boundary_x[:, 0], boundary_x[:, 1], boundary_x[:, 2],
               color='red', alpha=0.6, label='$\hat{x}$ (from boundary)')

    ax.set_title(title or "Interior and Boundary Points Highlighted")
    ax.legend()
    plt.show()
    return fig



