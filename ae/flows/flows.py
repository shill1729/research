import numpy as np
import torch

def generate_brownian_motion(t, num_steps, d=3, seed=None):
    # Generate Brownian motion
    dt = t / num_steps
    rng = np.random.default_rng(seed=seed)
    z = rng.normal(size=(num_steps, d)) * np.sqrt(dt)
    W = np.cumsum(z, axis=0)
    W = np.vstack((np.zeros(d), W))
    return W


def flow(points, t, W, mu, sigma, num_steps=1000, reverse=False):
    """
        Compute the stochastic flow of a point cloud under a given SDE.
    """
    dt = t / num_steps
    d = points.shape[1]
    paths = np.zeros((num_steps + 1, len(points), d))
    paths[0] = points
    for i in range(num_steps):
        dW = W[i + 1] - W[i]
        for j in range(len(points)):
            x = paths[i, j]
            sigma1 = sigma(x)
            if reverse:
                mu1 = -mu(x)
            else:
                mu1 = mu(x)
            paths[i + 1, j] = paths[i, j] + mu1 * dt + (sigma1 @ dW)
    return paths



def generate_brownian_motion_torch(t, num_steps, d=3, seed=None):
    # Generate Brownian motion
    dt = t / num_steps

    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Generate random normal increments
    z = torch.randn(num_steps, d) * torch.sqrt(torch.tensor(dt))

    # Cumulative sum to get Brownian motion
    W = torch.cumsum(z, dim=0)

    # Prepend zero vector for initial state
    W = torch.cat([torch.zeros(1, d), W], dim=0)

    return W


def flow_torch(points, t, W, mu, sigma, num_steps=1000, reverse=False):
    """
    Compute the stochastic flow of a point cloud under a given SDE.
    """
    dt = t / num_steps

    # Convert points to tensor if not already
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)

    # Initialize tensor to store paths
    paths = torch.zeros(num_steps + 1, len(points), 3)
    paths[0] = points

    for i in range(num_steps):
        dW = W[i + 1] - W[i]

        for j in range(len(points)):
            x = paths[i, j]
            sigma1 = sigma(x)
            if reverse:
                mu1 = -mu(x)
            else:
                mu1 = mu(x)
            paths[i + 1, j] = paths[i, j] + mu1 * dt + torch.matmul(sigma1, dW)

    return paths